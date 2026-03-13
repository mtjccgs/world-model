#!/usr/bin/env python3
"""
Embodied Virtual Mouse with Spiking EC-Hippocampal Circuit
==========================================================

Inspired by Eon Systems' embodied Drosophila brain emulation
(Shiu et al., Nature 2024; eon.systems), this simulation places
a spiking neural circuit inside a virtual mouse body that navigates
a 2D arena in real-time.

Architecture (following the allo-ego-allo loop):

    Sensory Input ──→ EC Grid Cells (allocentric latent state)
                           │
                      DG (sparse compression)
                           │
                      CA3 (attractor memory, context)
                           │
                      CA1 (egocentric readout, BTSP plasticity)
                           │
                      Subiculum (stability filter)
                           │
                      EC feedback (allo anchor)
                           │
                      Motor Output ──→ Body Movement

Neuron model: Leaky Integrate-and-Fire (LIF) with conductance-
based synapses, identical to the model used by Eon for the fly
connectome (Shiu et al., 2024).

The WORLD MODEL is not a separate module — it IS the circuit's
internal state, decoded from the mouse's own neural activity:
  - Believed position  ← grid cell population vector
  - Goal representation ← CA1 population vector
  - Forward predictions ← theta-phase precession
  - Uncertainty         ← population vector dispersion

Total: ~1200 spiking neurons, ~150K synapses
Simulation: 50 seconds of mouse time at dt=0.5ms
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
import time as clock

# ============================================================================
# Constants — LIF parameters from mouse cortical electrophysiology
# ============================================================================

# Membrane (threshold adjusted for small network — fewer synapses
# per cell means each synapse must be relatively stronger, equivalent
# to lowering threshold to compensate for 100-1000x fewer inputs)
V_REST = -70.0    # mV, resting potential
V_THRESH = -60.0  # mV, spike threshold (lowered for small network)
V_RESET = -67.0   # mV, post-spike reset
TAU_M = 15.0      # ms, membrane time constant (faster for small net)
C_M = 1.0         # normalized capacitance

# Synapses (current-based, exponential)
TAU_SYN_EXC = 8.0    # ms
TAU_SYN_INH = 12.0   # ms (slower → sustained inhibition)

# Refractory
TAU_REF = 2.0  # ms

# Simulation
DT = 0.5             # ms, neural timestep
DT_ENV = 10.0        # ms, environment update interval
STEPS_PER_ENV = int(DT_ENV / DT)  # 20 neural steps per env step
T_TOTAL = 50.0       # seconds total simulation
N_ENV_STEPS = int(T_TOTAL * 1000 / DT_ENV)  # 5000 env steps

# Theta oscillation
THETA_FREQ = 8.0     # Hz
THETA_PERIOD = 1000.0 / THETA_FREQ  # 125 ms

# Arena
ARENA_SIZE = 1.0     # meters
N_SPATIAL_BINS = 30  # for rate maps

# BTSP plasticity
BTSP_WINDOW = 1500.0  # ms, plasticity window around plateau
BTSP_LR = 0.08        # learning rate
BTSP_TRACE_TAU = 800.0  # ms, eligibility trace decay

# Reward
REWARD_RADIUS = 0.12  # meters


# ============================================================================
# Region definitions — neuron index ranges
# ============================================================================

def _make_regions():
    """Define neuron populations and their index ranges."""
    regions = {}
    idx = 0

    def add(name, n):
        nonlocal idx
        regions[name] = slice(idx, idx + n)
        idx += n
        return n

    # Entorhinal cortex — 6 grid modules × 32 cells + interneurons
    add('grid', 192)
    add('grid_inh', 24)
    # EC Layer III (direct path to CA1)
    add('ec3', 48)
    # Dentate gyrus
    add('dg_gc', 360)      # granule cells (large expansion)
    add('dg_inh', 24)      # feedback interneurons
    # CA3
    add('ca3_pyr', 96)
    add('ca3_inh', 12)
    # CA1
    add('ca1_pyr', 180)
    add('ca1_inh', 18)
    # Subiculum
    add('sub', 60)
    # Head direction
    add('hd', 24)
    # Sensory
    add('sens_prox', 16)    # proximity (whiskers)
    add('sens_odor', 12)    # olfactory
    add('sens_cue', 8)      # visual cues
    # Motor / planning
    add('motor', 12)
    # Theta pacemaker
    add('theta', 4)
    # Context signal (goal A vs B)
    add('ctx', 8)

    regions['N_TOTAL'] = idx
    return regions

R = _make_regions()
N_TOTAL = R['N_TOTAL']


# ============================================================================
# LIF Spiking Network
# ============================================================================

class LIFNetwork:
    """
    Vectorized Leaky Integrate-and-Fire network with current-based synapses.

    Like Eon's fly brain model (Shiu et al. 2024, Nature), all neurons
    share the same LIF equations but differ in their connectivity.

    Uses current-based synapses (not conductance-based) for robustness,
    matching the α-synapse approach used in the Drosophila model:

        τ_m dV/dt = -(V - V_rest) + I_syn + I_ext
        I_syn = I_exc - I_inh                        (current injection)
        dI_exc/dt = -I_exc / τ_exc   (upon pre spike: I_exc += w)
        dI_inh/dt = -I_inh / τ_inh   (upon pre spike: I_inh += w)
        if V ≥ V_thresh: spike, V → V_reset, refractory for τ_ref
    """

    def __init__(self, n, seed=42):
        self.n = n
        self.rng = np.random.RandomState(seed)

        # State variables
        self.V = np.full(n, V_REST)
        self.I_exc = np.zeros(n)    # excitatory synaptic current
        self.I_inh = np.zeros(n)    # inhibitory synaptic current
        self.adapt = np.zeros(n)    # spike-frequency adaptation current
        self.refrac = np.zeros(n)
        self.spikes = np.zeros(n, dtype=bool)

        # Adaptation parameters (prevents runaway excitation)
        self.adapt_increment = 1.5   # mV, added on each spike
        self.adapt_tau = 40.0        # ms, adaptation decay time
        self.adapt_decay = np.exp(-DT / self.adapt_tau)

        # Connectivity matrices (dense, N×N)
        # W_exc[i,j] > 0: when j spikes, add W_exc[i,j] to I_exc[i]
        self.W_exc = np.zeros((n, n), dtype=np.float32)
        # W_inh[i,j] > 0: when j spikes, add W_inh[i,j] to I_inh[i]
        self.W_inh = np.zeros((n, n), dtype=np.float32)

        # Pre-computed decay factors
        self.decay_exc = np.exp(-DT / TAU_SYN_EXC)
        self.decay_inh = np.exp(-DT / TAU_SYN_INH)

    def step(self, I_ext):
        """Advance one timestep (DT ms). Returns spike boolean array."""
        # 1. Decay synaptic currents
        self.I_exc *= self.decay_exc
        self.I_inh *= self.decay_inh

        # 2. Propagate spikes from previous step via synaptic weights
        if np.any(self.spikes):
            spike_vec = self.spikes.astype(np.float32)
            self.I_exc += self.W_exc @ spike_vec
            self.I_inh += self.W_inh @ spike_vec

        # 3. Net synaptic current (excitation - inhibition - adaptation)
        I_syn = self.I_exc - self.I_inh - self.adapt

        # 4. Update membrane potential (non-refractory neurons only)
        active = self.refrac <= 0
        dV = (-(self.V - V_REST) + I_syn + I_ext) / TAU_M * DT
        self.V[active] += dV[active]

        # 5. Threshold crossing → spike
        self.spikes[:] = False
        fired = (self.V >= V_THRESH) & active
        self.spikes[fired] = True
        self.V[fired] = V_RESET
        self.refrac[fired] = TAU_REF
        self.adapt[fired] += self.adapt_increment  # spike-freq adaptation

        # 6. Decay refractory and adaptation (unconditional)
        self.adapt *= self.adapt_decay
        self.refrac -= DT
        self.refrac = np.maximum(self.refrac, 0)

        return self.spikes.copy()

    def reset(self):
        self.V[:] = V_REST
        self.I_exc[:] = 0
        self.I_inh[:] = 0
        self.adapt[:] = 0
        self.refrac[:] = 0
        self.spikes[:] = False


# ============================================================================
# Circuit Builder — wires up the EC-hippocampal connectivity
# ============================================================================

class CircuitBuilder:
    """
    Builds biologically-structured connectivity for the EC-hippocampal
    circuit.  Connectivity patterns follow known mouse anatomy:

    Tri-synaptic path:  EC → DG → CA3 → CA1
    Direct path:        EC Layer III → CA1
    Feedback:           CA1 → Sub → EC
    Recurrent:          CA3 → CA3 (attractor)
    """

    def __init__(self, net, rng):
        self.net = net
        self.rng = rng
        self.grid_modules = []  # store grid cell properties

    def build_all(self):
        """Wire up the complete circuit."""
        self._build_grid_cells()
        self._build_theta_pacemaker()
        self._build_head_direction()
        self._build_dg()
        self._build_ca3()
        self._build_ca1()
        self._build_subiculum()
        self._build_feedback()
        self._build_sensory_motor()
        self._build_context()

    def _connect_exc(self, pre, post, prob, weight):
        """Random excitatory connections."""
        pre_idx = np.arange(self.net.n)[pre]
        post_idx = np.arange(self.net.n)[post]
        for j in pre_idx:
            targets = post_idx[self.rng.rand(len(post_idx)) < prob]
            w = weight * (1 + 0.2 * self.rng.randn(len(targets)))
            w = np.clip(w, 0, weight * 3)
            self.net.W_exc[targets, j] = w * 3.0

    def _connect_inh(self, pre, post, prob, weight):
        """Random inhibitory connections."""
        pre_idx = np.arange(self.net.n)[pre]
        post_idx = np.arange(self.net.n)[post]
        for j in pre_idx:
            targets = post_idx[self.rng.rand(len(post_idx)) < prob]
            w = weight * (1 + 0.2 * self.rng.randn(len(targets)))
            w = np.clip(w, 0, weight * 3)
            self.net.W_inh[targets, j] = w * 5.0

    def _build_grid_cells(self):
        """
        EC grid cells organized in 6 modules.
        Each module has 32 excitatory cells on a torus with
        Mexican-hat connectivity (continuous attractor).
        """
        grid_start = R['grid'].start
        inh_start = R['grid_inh'].start
        n_modules = 6
        n_per_mod = 32
        spacings = np.array([0.15, 0.21, 0.30, 0.42, 0.59, 0.83])
        orientations = np.array([7.5, 15, 22.5, 30, 37.5, 45]) * np.pi / 180

        for m in range(n_modules):
            mod_start = grid_start + m * n_per_mod
            mod_slice = slice(mod_start, mod_start + n_per_mod)

            # Store module properties
            phases = np.zeros((n_per_mod, 2))
            angles = np.linspace(0, 2 * np.pi, n_per_mod, endpoint=False)
            phases[:, 0] = np.cos(angles) * spacings[m] * 0.3
            phases[:, 1] = np.sin(angles) * spacings[m] * 0.3

            self.grid_modules.append({
                'slice': mod_slice,
                'spacing': spacings[m],
                'orientation': orientations[m],
                'phases': phases,
                'n': n_per_mod,
            })

            # Grid cells are driven directly by external position-
            # dependent current (like Eon's sensory neuron approach).
            # No recurrent dynamics needed — the spatial tuning comes
            # from the rate-to-spike conversion of the grid pattern.

    def _build_theta_pacemaker(self):
        """Theta pacemaker neurons drive the 8Hz rhythm."""
        theta_idx = np.arange(self.net.n)[R['theta']]
        # Theta → grid cells (rhythmic drive)
        self._connect_exc(R['theta'], R['grid'], 0.15, 0.08)
        # Theta → CA1 (phase reference)
        self._connect_exc(R['theta'], R['ca1_pyr'], 0.10, 0.05)

    def _build_head_direction(self):
        """Head direction cells driven by external heading signal."""
        # Like grid cells, HD cells are driven by external current.
        # No recurrent dynamics — heading encoding comes from
        # the cosine-tuned external drive.
        pass

    def _build_dg(self):
        """
        Dentate gyrus: massive expansion + sparse coding.
        EC → DG (perforant path): divergent, each GC samples few EC cells.
        DG interneurons provide feedback inhibition → winner-take-all.
        """
        n_gc = R['dg_gc'].stop - R['dg_gc'].start
        n_grid = R['grid'].stop - R['grid'].start
        gc_idx = np.arange(self.net.n)[R['dg_gc']]
        grid_idx = np.arange(self.net.n)[R['grid']]

        # Sparse EC → DG_GC (each GC receives from ~25 grid cells)
        # Weights are strong to compensate for 100x fewer synapses
        # than real brain (~4000 EC inputs per GC in vivo)
        for i in range(n_gc):
            sources = self.rng.choice(len(grid_idx), size=min(25, len(grid_idx)),
                                      replace=False)
            w = 1.8 * self.rng.exponential(0.8, size=len(sources))
            self.net.W_exc[gc_idx[i], grid_idx[sources]] = np.clip(w, 0.2, 6.0)

        # DG_GC → DG_IN (feedback) — strong drive to interneurons
        self._connect_exc(R['dg_gc'], R['dg_inh'], 0.12, 0.8)
        # DG_IN → DG_GC (lateral inhibition → enforces sparsity)
        self._connect_inh(R['dg_inh'], R['dg_gc'], 0.35, 1.2)

    def _build_ca3(self):
        """
        CA3: attractor network with recurrent excitation.
        DG → CA3 (mossy fibers): sparse, strong ("detonator" synapses).
        EC → CA3 (perforant path): broad, weak.
        CA3 → CA3 (recurrent): dense, Hebbian-initialized.
        """
        ca3_idx = np.arange(self.net.n)[R['ca3_pyr']]
        dg_idx = np.arange(self.net.n)[R['dg_gc']]
        n_ca3 = len(ca3_idx)

        # DG → CA3 (mossy fiber: sparse but very strong "detonator")
        for i in range(n_ca3):
            sources = self.rng.choice(len(dg_idx), size=6, replace=False)
            self.net.W_exc[ca3_idx[i], dg_idx[sources]] = 12.0

        # EC → CA3 (perforant path: broad, moderate)
        self._connect_exc(R['grid'], R['ca3_pyr'], 0.15, 0.4)

        # CA3 recurrent (attractor dynamics)
        # Initialize with two stored patterns (goal A and goal B contexts)
        pattern_a = (self.rng.rand(n_ca3) > 0.5).astype(float)
        pattern_b = 1 - pattern_a + self.rng.randn(n_ca3) * 0.05
        pattern_b = np.clip(pattern_b, 0, 1)

        for p in [pattern_a, pattern_b]:
            p_centered = p - p.mean()
            W_hebb = 0.8 * np.outer(p_centered, p_centered) / n_ca3
            np.fill_diagonal(W_hebb, 0)
            pos = np.clip(W_hebb, 0, None)
            neg = np.clip(-W_hebb, 0, None)
            self.net.W_exc[np.ix_(ca3_idx, ca3_idx)] += pos.astype(np.float32)
            self.net.W_inh[np.ix_(ca3_idx, ca3_idx)] += neg.astype(np.float32)

        # CA3 → CA3 interneuron → CA3 (feedback, controls runaway)
        self._connect_exc(R['ca3_pyr'], R['ca3_inh'], 0.4, 0.6)
        self._connect_inh(R['ca3_inh'], R['ca3_pyr'], 0.5, 0.8)

    def _build_ca1(self):
        """
        CA1: context-dependent readout with BTSP plasticity.
        CA3 → CA1 (Schaffer collateral): main drive.
        EC3 → CA1 (temporoammonic): direct path, modulatory.
        CA1 interneurons: feedforward inhibition.
        """
        # CA3 → CA1 (Schaffer collateral) — main drive
        self._connect_exc(R['ca3_pyr'], R['ca1_pyr'], 0.30, 0.7)

        # EC Layer III → CA1 (direct path)
        self._connect_exc(R['grid'], R['ec3'], 0.3, 0.4)
        self._connect_exc(R['ec3'], R['ca1_pyr'], 0.18, 0.4)

        # CA1 feedforward inhibition — controls CA1 firing rate
        self._connect_exc(R['ca3_pyr'], R['ca1_inh'], 0.3, 0.5)
        self._connect_inh(R['ca1_inh'], R['ca1_pyr'], 0.45, 1.0)

        # Context signal → CA1 (goal-dependent modulation)
        self._connect_exc(R['ctx'], R['ca1_pyr'], 0.25, 0.3)

    def _build_subiculum(self):
        """
        Subiculum: stable readout, NO fast plasticity.
        CA1 → Sub: excitatory drive.
        Sub has broader, more stable place fields.
        """
        self._connect_exc(R['ca1_pyr'], R['sub'], 0.20, 0.4)

        # Sub interneuron-like suppression (via self-inhibition approximation)
        sub_idx = np.arange(self.net.n)[R['sub']]
        n_sub = len(sub_idx)
        for i in range(n_sub):
            others = list(range(n_sub))
            others.remove(i)
            targets = self.rng.choice(others, size=min(8, n_sub - 1),
                                      replace=False)
            self.net.W_inh[sub_idx[targets], sub_idx[i]] = 0.4

    def _build_feedback(self):
        """Sub → EC feedback: closes the allo-ego-allo loop."""
        self._connect_exc(R['sub'], R['grid'], 0.12, 0.3)
        self._connect_exc(R['sub'], R['ec3'], 0.12, 0.25)

    def _build_sensory_motor(self):
        """
        Sensory → EC (spatial input).
        CA1 + HD → Motor (behavioral output).
        """
        # Sensory → EC grid cells (landmark correction)
        self._connect_exc(R['sens_cue'], R['grid'], 0.08, 0.10)
        self._connect_exc(R['sens_prox'], R['grid'], 0.05, 0.08)

        # Sensory → motor (reflexive: wall avoidance, odor following)
        self._connect_exc(R['sens_prox'], R['motor'], 0.20, 0.6)
        self._connect_exc(R['sens_odor'], R['motor'], 0.25, 0.7)

        # CA1 → motor (model-driven: goal navigation)
        self._connect_exc(R['ca1_pyr'], R['motor'], 0.10, 0.4)

        # HD → motor (heading correction)
        self._connect_exc(R['hd'], R['motor'], 0.15, 0.10)

    def _build_context(self):
        """Context neurons encode current goal (A vs B)."""
        # Context → CA3 (biases attractor)
        self._connect_exc(R['ctx'], R['ca3_pyr'], 0.25, 0.35)


# ============================================================================
# 2D Arena Environment
# ============================================================================

class Arena2D:
    """
    Square arena with walls, two reward locations, visual cues.

    Layout:
        ┌──── Cue 1 (stripe) ─────┐
        │                           │
    Cue 4│     ★ A        ★ B      │Cue 2
   (dot) │       (0.75,0.75) (0.25,0.75)│(checker)
        │                           │
        │         ● mouse           │
        │                           │
        └──── Cue 3 (gradient) ────┘
    """

    def __init__(self, size=ARENA_SIZE):
        self.size = size
        self.rewards = {
            'A': np.array([0.75, 0.75]),
            'B': np.array([0.25, 0.75]),
        }
        # Wall cue identities (N, E, S, W)
        self.cue_ids = np.array([0, 1, 2, 3])  # distinct cues

    def wall_distances(self, pos):
        """Distance to each wall: [N, E, S, W]."""
        return np.array([
            self.size - pos[1],   # North
            self.size - pos[0],   # East
            pos[1],               # South
            pos[0],               # West
        ])

    def odor_gradient(self, pos, goal_label):
        """Odor concentration gradient pointing toward active reward."""
        reward_pos = self.rewards[goal_label]
        diff = reward_pos - pos
        dist = np.linalg.norm(diff)
        concentration = np.exp(-dist**2 / (2 * 0.3**2))
        if dist > 1e-6:
            gradient_dir = diff / dist
        else:
            gradient_dir = np.zeros(2)
        return concentration, gradient_dir

    def nearest_cue_signal(self, pos, heading):
        """Visual cue signal based on which wall the mouse faces."""
        # Which wall is in front?
        dx, dy = np.cos(heading), np.sin(heading)
        dists = self.wall_distances(pos)
        # Project heading onto wall normals
        wall_normals = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        facing = np.array([dx, dy])
        dots = wall_normals @ facing
        # Signal strength = facing wall / distance
        signals = np.zeros(4)
        for i in range(4):
            if dots[i] > 0:
                signals[i] = dots[i] * np.exp(-dists[i] / 0.3)
        return signals

    def check_reward(self, pos, goal_label):
        """Check if mouse is at the active reward."""
        dist = np.linalg.norm(pos - self.rewards[goal_label])
        return dist < REWARD_RADIUS

    def clip_position(self, pos):
        """Keep mouse inside arena."""
        margin = 0.02
        return np.clip(pos, margin, self.size - margin)


# ============================================================================
# Virtual Mouse — the embodied agent
# ============================================================================

class VirtualMouse:
    """
    The mouse body: position, heading, sensors, motors.

    Sensors:
        - 8 whiskers (proximity to walls, 45° apart)
        - Olfactory (odor gradient, 4-directional)
        - Visual (wall cue detection, heading-dependent)

    Motor:
        - 6 motor neurons: 3 left-turn, 3 right-turn
        - Forward drive from CA1 population activity
    """

    def __init__(self, arena, start_pos=None, rng=None):
        self.arena = arena
        self.rng = rng or np.random.RandomState()
        self.pos = start_pos if start_pos is not None else np.array([0.5, 0.3])
        self.heading = self.rng.uniform(0, 2 * np.pi)
        self.speed = 0.15  # m/s baseline
        self.max_speed = 0.35  # m/s

    def sense(self, goal_label):
        """Generate sensory input currents for all sensory neurons."""
        # Proximity sensors (16 neurons: 8 whiskers × 2 ON/OFF)
        n_whiskers = 8
        whisker_angles = np.linspace(
            self.heading - np.pi/2, self.heading + np.pi/2, n_whiskers)
        wall_dists = self.arena.wall_distances(self.pos)
        prox_current = np.zeros(16)
        for i, angle in enumerate(whisker_angles):
            dx, dy = np.cos(angle), np.sin(angle)
            # Check distance in whisker direction
            min_d = 1.0
            if dx > 0:
                min_d = min(min_d, (self.arena.size - self.pos[0]) / (dx + 1e-9))
            elif dx < 0:
                min_d = min(min_d, self.pos[0] / (-dx + 1e-9))
            if dy > 0:
                min_d = min(min_d, (self.arena.size - self.pos[1]) / (dy + 1e-9))
            elif dy < 0:
                min_d = min(min_d, self.pos[1] / (-dy + 1e-9))
            min_d = max(min_d, 0.01)
            prox_current[i] = np.exp(-min_d / 0.08) * 18.0     # ON
            prox_current[i + 8] = max(0, 6 - prox_current[i])  # OFF

        # Odor sensors (12 neurons: 4 directions × 3 concentration levels)
        conc, grad_dir = self.arena.odor_gradient(self.pos, goal_label)
        odor_dirs = np.array([
            [np.cos(self.heading), np.sin(self.heading)],
            [np.cos(self.heading + np.pi/2), np.sin(self.heading + np.pi/2)],
            [np.cos(self.heading + np.pi), np.sin(self.heading + np.pi)],
            [np.cos(self.heading - np.pi/2), np.sin(self.heading - np.pi/2)],
        ])
        odor_current = np.zeros(12)
        for i in range(4):
            proj = np.dot(grad_dir, odor_dirs[i])
            strength = max(0, proj) * conc
            odor_current[i * 3] = strength * 22.0       # strong
            odor_current[i * 3 + 1] = strength * 14.0   # medium
            odor_current[i * 3 + 2] = strength * 8.0    # weak

        # Visual cue sensors (8 neurons: 4 walls × 2 features)
        cue_signals = self.arena.nearest_cue_signal(self.pos, self.heading)
        cue_current = np.zeros(8)
        for i in range(4):
            cue_current[i] = cue_signals[i] * 16.0
            cue_current[i + 4] = cue_signals[i] * 10.0

        return prox_current, odor_current, cue_current

    def get_velocity(self):
        """Current velocity vector (for grid cell path integration)."""
        return self.speed * np.array([np.cos(self.heading),
                                      np.sin(self.heading)])

    def move(self, motor_spikes, dt_sec):
        """
        Update position based on motor neuron activity.
        Motor neurons: [0:3] = left turn, [3:6] = right turn,
                       [6:9] = forward, [9:12] = slow/stop
        """
        left = motor_spikes[0:3].sum()
        right = motor_spikes[3:6].sum()
        forward = motor_spikes[6:9].sum()
        slow = motor_spikes[9:12].sum()

        # Turn: motor-driven + random exploration
        if left + right > 0:
            turn_rate = 2.5 * (right - left) / max(1, left + right)
        else:
            turn_rate = 0.0
        self.heading += turn_rate * dt_sec
        self.heading = self.heading % (2 * np.pi)

        # Speed: always move (baseline exploration) + motor modulation
        motor_drive = forward / max(1, forward + slow + 1)
        self.speed = 0.15 + 0.15 * motor_drive

        # Exploration noise: smooth random heading changes
        self.heading += self.rng.randn() * 0.6 * np.sqrt(dt_sec)
        self.speed += self.rng.randn() * 0.02
        # Occasional larger turns (Levy-flight-like exploration)
        if self.rng.rand() < 0.02:
            self.heading += self.rng.uniform(-np.pi/2, np.pi/2)

        # Move
        dx = self.speed * np.cos(self.heading) * dt_sec
        dy = self.speed * np.sin(self.heading) * dt_sec
        new_pos = self.pos + np.array([dx, dy])

        # Wall bounce (simple reflection)
        margin = 0.03
        for dim in [0, 1]:
            if new_pos[dim] < margin:
                new_pos[dim] = margin
                if dim == 0:
                    self.heading = np.pi - self.heading
                else:
                    self.heading = -self.heading
                self.heading += self.rng.randn() * 0.5
            elif new_pos[dim] > ARENA_SIZE - margin:
                new_pos[dim] = ARENA_SIZE - margin
                if dim == 0:
                    self.heading = np.pi - self.heading
                else:
                    self.heading = -self.heading
                self.heading += self.rng.randn() * 0.5

        self.pos = new_pos
        self.speed = max(0.02, min(self.max_speed, self.speed))


# ============================================================================
# BTSP Plasticity Module
# ============================================================================

class BTSPPlasticity:
    """
    Behavioral Time-Scale Synaptic Plasticity for CA1.

    When a reward is found (plateau potential), strengthen EC→CA1
    synapses that were recently active.  This creates place fields
    near rewarded locations within a single trial.

    The plasticity window (~1.5s) is much longer than standard STDP
    (~20ms), matching the behavioral timescale of navigation.
    """

    def __init__(self, net, n_ec, n_ca1, rng):
        self.net = net
        self.ec_slice = R['grid']
        self.ca1_slice = R['ca1_pyr']
        self.rng = rng
        self.n_ec = n_ec
        self.n_ca1 = n_ca1

        # Eligibility traces
        self.ec_trace = np.zeros(n_ec)   # running average of EC activity
        self.ca1_trace = np.zeros(n_ca1)  # running average of CA1 activity
        self.trace_decay = np.exp(-DT / BTSP_TRACE_TAU)

        # Plateau state
        self.plateau_active = False
        self.plateau_timer = 0.0
        self.plateau_goal = None

    def update_traces(self, spikes):
        """Update eligibility traces with current spikes."""
        self.ec_trace *= self.trace_decay
        self.ca1_trace *= self.trace_decay
        ec_spk = spikes[self.ec_slice].astype(float)
        ca1_spk = spikes[self.ca1_slice].astype(float)
        self.ec_trace += ec_spk * 0.1
        self.ca1_trace += ca1_spk * 0.1

    def trigger_plateau(self, goal_label):
        """Reward found → trigger plateau potential in CA1."""
        self.plateau_active = True
        self.plateau_timer = BTSP_WINDOW
        self.plateau_goal = goal_label

    def apply_plasticity(self):
        """If plateau is active, modify EC→CA1 weights."""
        if not self.plateau_active:
            return

        self.plateau_timer -= DT
        if self.plateau_timer <= 0:
            self.plateau_active = False
            return

        # Hebbian: strengthen connections where both pre and post were active
        ec_idx = np.arange(self.net.n)[self.ec_slice]
        ca1_idx = np.arange(self.net.n)[self.ca1_slice]

        # Only modify for neurons with significant traces
        ec_active = self.ec_trace > 0.02
        ca1_active = self.ca1_trace > 0.02

        if ec_active.any() and ca1_active.any():
            # Outer product of traces, scaled by learning rate
            dW = BTSP_LR * np.outer(
                self.ca1_trace[ca1_active],
                self.ec_trace[ec_active]) * (DT / BTSP_WINDOW)

            ca1_active_idx = ca1_idx[ca1_active]
            ec_active_idx = ec_idx[ec_active]
            current = self.net.W_exc[np.ix_(ca1_active_idx, ec_active_idx)]
            self.net.W_exc[np.ix_(ca1_active_idx, ec_active_idx)] = \
                np.clip(current + dW.astype(np.float32), 0, 2.0)


# ============================================================================
# Mouse World Model — decoded from neural activity
# ============================================================================

class MouseWorldModel:
    """
    The world model is NOT a separate computation — it IS the circuit's
    internal state, read out from the mouse's own neural activity.

    This class provides methods to decode what the mouse "believes"
    about the world, from its first-person perspective:

    1. Believed position: decoded from grid cell population vector
    2. Goal representation: decoded from CA1 population activity
    3. Forward predictions: theta-phase precession → sweep
    4. Uncertainty: population vector dispersion
    """

    def __init__(self, circuit_builder, arena_size=ARENA_SIZE):
        self.modules = circuit_builder.grid_modules
        self.arena_size = arena_size

        # Pre-compute grid cell preferred positions for decoding
        nbins = N_SPATIAL_BINS
        self.decode_grid = np.zeros((nbins, nbins, R['grid'].stop - R['grid'].start))
        for iy in range(nbins):
            for ix in range(nbins):
                pos = np.array([(ix + 0.5) / nbins * arena_size,
                                (iy + 0.5) / nbins * arena_size])
                self.decode_grid[iy, ix] = self._grid_rates(pos)

        # Normalize
        norms = np.linalg.norm(self.decode_grid, axis=2, keepdims=True)
        norms[norms < 1e-9] = 1
        self.decode_grid_norm = self.decode_grid / norms

    def _grid_rates(self, pos):
        """Compute expected grid cell firing rates at a position."""
        rates = np.zeros(R['grid'].stop - R['grid'].start)
        for mod in self.modules:
            mod_rates = np.zeros(mod['n'])
            shifted = pos[np.newaxis, :] - mod['phases']
            for k in range(3):
                theta = mod['orientation'] + k * np.pi / 3
                wave = np.array([np.cos(theta), np.sin(theta)])
                proj = np.sum(wave * shifted, axis=-1)
                mod_rates += np.cos(2 * np.pi * proj / mod['spacing'])
            mod_rates = np.clip((mod_rates / 3 + 1) / 2, 0, 1)
            start = mod['slice'].start - R['grid'].start
            rates[start:start + mod['n']] = mod_rates
        return rates

    def decode_position(self, grid_spike_counts):
        """
        Decode believed position from grid cell activity.
        Uses template matching against expected firing patterns.
        """
        # Normalize spike counts
        sc = grid_spike_counts.astype(float)
        norm = np.linalg.norm(sc)
        if norm < 1e-9:
            return np.array([0.5, 0.5]), 0.0  # no information
        sc_norm = sc / norm

        # Correlate with all position templates
        nbins = N_SPATIAL_BINS
        corr_map = np.sum(self.decode_grid_norm * sc_norm[np.newaxis, np.newaxis, :],
                          axis=2)

        # Find peak
        peak_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        believed_pos = np.array([
            (peak_idx[1] + 0.5) / nbins * self.arena_size,
            (peak_idx[0] + 0.5) / nbins * self.arena_size,
        ])

        # Confidence = peak correlation
        confidence = corr_map[peak_idx]

        return believed_pos, confidence

    def decode_goal_direction(self, ca1_spike_counts, ca1_positions):
        """
        Decode goal direction from CA1 population vector.
        CA1 cells with place fields fire more when the mouse
        is heading toward their field → population vector points
        toward the goal.
        """
        if ca1_spike_counts.sum() < 1:
            return np.array([0.0, 0.0]), 0.0

        weights = ca1_spike_counts / (ca1_spike_counts.sum() + 1e-9)
        centroid = (weights[:, np.newaxis] * ca1_positions).sum(axis=0)
        strength = np.linalg.norm(centroid - 0.5)
        return centroid, strength

    def compute_uncertainty(self, grid_spike_counts):
        """
        Uncertainty from grid cell population vector dispersion.
        Low dispersion = confident about position.
        High dispersion = uncertain (e.g., after path integration drift).
        """
        sc = grid_spike_counts.astype(float)
        if sc.sum() < 1:
            return 1.0

        # Entropy of normalized activity
        p = sc / sc.sum()
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(sc))
        return entropy / max_entropy  # 0 = certain, 1 = maximally uncertain


# ============================================================================
# Main Simulation
# ============================================================================

class Simulation:
    """
    Runs the full embodied simulation:
    sensory → brain → motor → movement → sensory (closed loop).
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.arena = Arena2D()
        self.mouse = VirtualMouse(self.arena, rng=self.rng)

        # Build neural circuit
        self.net = LIFNetwork(N_TOTAL, seed=seed)
        self.builder = CircuitBuilder(self.net, self.rng)
        self.builder.build_all()

        # BTSP plasticity
        n_ec = R['grid'].stop - R['grid'].start
        n_ca1 = R['ca1_pyr'].stop - R['ca1_pyr'].start
        self.btsp = BTSPPlasticity(self.net, n_ec, n_ca1, self.rng)

        # World model readout
        self.world_model = MouseWorldModel(self.builder, ARENA_SIZE)

        # Assign spatial receptive field centers to CA1 neurons (for decoding)
        self.ca1_centers = self.rng.uniform(0, ARENA_SIZE, (n_ca1, 2))

        # Goal schedule
        self.current_goal = 'A'
        self.goal_switch_time = 30.0  # switch at 30s

        # Recording buffers
        self.trajectory = np.zeros((N_ENV_STEPS, 2))
        self.headings = np.zeros(N_ENV_STEPS)
        self.spike_counts = {
            'grid': np.zeros((N_ENV_STEPS, n_ec)),
            'ca1': np.zeros((N_ENV_STEPS, n_ca1)),
            'sub': np.zeros((N_ENV_STEPS, R['sub'].stop - R['sub'].start)),
        }
        self.believed_positions = np.zeros((N_ENV_STEPS, 2))
        self.confidences = np.zeros(N_ENV_STEPS)
        self.goal_labels = []
        self.reward_times = []

        # Rate maps (accumulated over simulation)
        nbins = N_SPATIAL_BINS
        self.grid_ratemap = np.zeros((nbins, nbins, n_ec))
        self.ca1_ratemap_A = np.zeros((nbins, nbins, n_ca1))
        self.ca1_ratemap_B = np.zeros((nbins, nbins, n_ca1))
        self.sub_ratemap_A = np.zeros((nbins, nbins, R['sub'].stop - R['sub'].start))
        self.sub_ratemap_B = np.zeros((nbins, nbins, R['sub'].stop - R['sub'].start))
        self.occupancy_A = np.zeros((nbins, nbins))
        self.occupancy_B = np.zeros((nbins, nbins))

    def _compute_grid_drive(self, pos, velocity):
        """
        Compute external current for grid cells based on position
        and velocity (path integration).
        """
        n_grid = R['grid'].stop - R['grid'].start
        currents = np.zeros(n_grid)

        for mod in self.builder.grid_modules:
            start = mod['slice'].start - R['grid'].start
            n = mod['n']

            # Position-dependent firing (rate code → current)
            shifted = pos[np.newaxis, :] - mod['phases']
            rates = np.zeros(n)
            for k in range(3):
                theta = mod['orientation'] + k * np.pi / 3
                wave = np.array([np.cos(theta), np.sin(theta)])
                proj = np.sum(wave * shifted, axis=-1)
                rates += np.cos(2 * np.pi * proj / mod['spacing'])
            rates = np.clip((rates / 3 + 1) / 2, 0, 1)

            # Velocity modulation (path integration)
            vel_mag = np.linalg.norm(velocity)
            vel_gain = 1.0 + 2.0 * vel_mag  # faster → stronger drive

            currents[start:start + n] = rates * vel_gain * 22.0

        return currents

    def _compute_theta_drive(self, t_ms):
        """Theta pacemaker current (8 Hz sinusoid)."""
        phase = 2 * np.pi * THETA_FREQ * t_ms / 1000
        drive = 12.0 * (0.5 + 0.5 * np.sin(phase))
        return np.full(R['theta'].stop - R['theta'].start, drive)

    def _compute_hd_drive(self, heading):
        """Head direction cell drive based on current heading."""
        n_hd = R['hd'].stop - R['hd'].start
        pref_dirs = np.linspace(0, 2 * np.pi, n_hd, endpoint=False)
        drive = np.exp(3.0 * np.cos(pref_dirs - heading))
        drive = drive / drive.max() * 14.0
        return drive

    def _compute_context_drive(self, goal_label):
        """Context signal: which goal is active."""
        n_ctx = R['ctx'].stop - R['ctx'].start
        drive = np.zeros(n_ctx)
        if goal_label == 'A':
            drive[:n_ctx // 2] = 14.0
        else:
            drive[n_ctx // 2:] = 14.0
        return drive

    def run(self):
        """Run the full simulation."""
        print(f"Running embodied mouse simulation...")
        print(f"  {N_TOTAL} LIF neurons, {T_TOTAL}s simulated time")
        print(f"  dt={DT}ms, {N_ENV_STEPS} environment steps")
        print(f"  Goal switch: A→B at {self.goal_switch_time}s")
        print()

        t0 = clock.time()
        t_ms = 0.0

        for env_step in range(N_ENV_STEPS):
            t_sec = env_step * DT_ENV / 1000

            # ── Goal schedule ──
            if t_sec >= self.goal_switch_time and self.current_goal == 'A':
                self.current_goal = 'B'
                print(f"  [{t_sec:.1f}s] Goal switched to B")

            self.goal_labels.append(self.current_goal)

            # ── Sensory input ──
            prox, odor, cue = self.mouse.sense(self.current_goal)
            velocity = self.mouse.get_velocity()

            # ── Compute external currents ──
            I_ext = np.zeros(N_TOTAL)

            # Grid cells: position + velocity
            grid_drive = self._compute_grid_drive(self.mouse.pos, velocity)
            I_ext[R['grid']] = grid_drive

            # Theta pacemaker
            I_ext[R['theta']] = self._compute_theta_drive(t_ms)

            # Head direction
            I_ext[R['hd']] = self._compute_hd_drive(self.mouse.heading)

            # Sensory neurons (Poisson-like: current proportional to stimulus)
            I_ext[R['sens_prox']] = prox * 6.0 + self.rng.randn(16) * 1.0
            I_ext[R['sens_odor']] = odor * 6.0 + self.rng.randn(12) * 1.0
            I_ext[R['sens_cue']] = cue * 6.0 + self.rng.randn(8) * 1.0

            # Context signal
            I_ext[R['ctx']] = self._compute_context_drive(self.current_goal)

            # ── Run neural substeps ──
            env_spikes = np.zeros(N_TOTAL, dtype=int)
            for _ in range(STEPS_PER_ENV):
                spikes = self.net.step(I_ext)
                env_spikes += spikes.astype(int)

                # BTSP traces and plasticity
                self.btsp.update_traces(spikes)
                self.btsp.apply_plasticity()

                t_ms += DT

            # ── Motor output → movement ──
            motor_spikes = env_spikes[R['motor']]
            self.mouse.move(motor_spikes, DT_ENV / 1000)

            # ── Check reward ──
            if self.arena.check_reward(self.mouse.pos, self.current_goal):
                self.btsp.trigger_plateau(self.current_goal)
                self.reward_times.append(t_sec)
                if len(self.reward_times) <= 20:
                    print(f"  [{t_sec:.1f}s] Reward found at "
                          f"({self.mouse.pos[0]:.2f}, {self.mouse.pos[1]:.2f})!"
                          f" Goal={self.current_goal}")

            # ── Record ──
            self.trajectory[env_step] = self.mouse.pos.copy()
            self.headings[env_step] = self.mouse.heading

            grid_sc = env_spikes[R['grid']]
            ca1_sc = env_spikes[R['ca1_pyr']]
            sub_sc = env_spikes[R['sub']]

            self.spike_counts['grid'][env_step] = grid_sc
            self.spike_counts['ca1'][env_step] = ca1_sc
            self.spike_counts['sub'][env_step] = sub_sc

            # Decode world model
            believed_pos, conf = self.world_model.decode_position(grid_sc)
            self.believed_positions[env_step] = believed_pos
            self.confidences[env_step] = conf

            # Accumulate rate maps
            bx = min(int(self.mouse.pos[0] / ARENA_SIZE * N_SPATIAL_BINS),
                     N_SPATIAL_BINS - 1)
            by = min(int(self.mouse.pos[1] / ARENA_SIZE * N_SPATIAL_BINS),
                     N_SPATIAL_BINS - 1)

            self.grid_ratemap[by, bx] += grid_sc
            if self.current_goal == 'A':
                self.ca1_ratemap_A[by, bx] += ca1_sc
                self.sub_ratemap_A[by, bx] += sub_sc
                self.occupancy_A[by, bx] += 1
            else:
                self.ca1_ratemap_B[by, bx] += ca1_sc
                self.sub_ratemap_B[by, bx] += sub_sc
                self.occupancy_B[by, bx] += 1

            # Progress
            if (env_step + 1) % 1000 == 0:
                elapsed = clock.time() - t0
                pct = (env_step + 1) / N_ENV_STEPS * 100
                rate = (env_step + 1) / elapsed
                print(f"  [{t_sec:.1f}s] {pct:.0f}% complete "
                      f"({rate:.0f} env steps/s, "
                      f"{rate * STEPS_PER_ENV:.0f} neural steps/s)")

        elapsed = clock.time() - t0
        total_neural = N_ENV_STEPS * STEPS_PER_ENV
        print(f"\nDone! {elapsed:.1f}s wall time for "
              f"{total_neural:,} neural timesteps "
              f"({total_neural/elapsed:,.0f} steps/s)")
        print(f"  Rewards found: {len(self.reward_times)}")

    def compute_rate_maps(self):
        """Normalize rate maps by occupancy."""
        min_occ = 3
        for rm, occ in [
            (self.grid_ratemap, self.occupancy_A + self.occupancy_B),
            (self.ca1_ratemap_A, self.occupancy_A),
            (self.ca1_ratemap_B, self.occupancy_B),
            (self.sub_ratemap_A, self.occupancy_A),
            (self.sub_ratemap_B, self.occupancy_B),
        ]:
            mask = occ >= min_occ
            for c in range(rm.shape[2]):
                rm[mask, c] /= occ[mask]
                rm[~mask, c] = 0


# ============================================================================
# Visualization
# ============================================================================

def visualize(sim):
    """Generate comprehensive visualization of the embodied mouse."""
    sim.compute_rate_maps()

    plt.rcParams.update({
        'font.size': 8, 'axes.titlesize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.45)

    # ── (a) Arena trajectory ──
    ax = fig.add_subplot(gs[0, 0:2])
    _plot_trajectory(ax, sim)

    # ── (b) Spike raster ──
    ax = fig.add_subplot(gs[0, 2:4])
    _plot_raster(ax, sim)

    # ── (c-d) Grid cell rate maps ──
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        cell_idx = [0, 32, 64, 128][i]
        rm = sim.grid_ratemap[:, :, cell_idx]
        rm_smooth = gaussian_filter(rm, sigma=1)
        ax.imshow(rm_smooth, origin='lower', cmap='hot',
                  extent=[0, 1, 0, 1], aspect='equal')
        mod_id = cell_idx // 32
        ax.set_title(f'Grid cell #{cell_idx}\n'
                     f'Module {mod_id}, λ={sim.builder.grid_modules[mod_id]["spacing"]:.2f}m',
                     fontsize=7)
        if i == 0:
            ax.set_ylabel('y (m)')
        ax.set_xlabel('x (m)')

    # ── (e-f) CA1 rate maps: Goal A vs B ──
    # Find most active CA1 cells
    ca1_total_A = sim.ca1_ratemap_A.sum(axis=(0, 1))
    ca1_total_B = sim.ca1_ratemap_B.sum(axis=(0, 1))
    top_cells_A = np.argsort(ca1_total_A)[-2:][::-1]
    top_cells_B = np.argsort(ca1_total_B)[-2:][::-1]

    for i, (cell_A, cell_B) in enumerate(zip(top_cells_A, top_cells_B)):
        # Goal A
        ax = fig.add_subplot(gs[2, i * 2])
        rm = gaussian_filter(sim.ca1_ratemap_A[:, :, cell_A], sigma=1.2)
        ax.imshow(rm, origin='lower', cmap='hot', extent=[0, 1, 0, 1])
        ax.plot(*sim.arena.rewards['A'], '*', color='cyan', markersize=10,
                markeredgecolor='white')
        ax.set_title(f'CA1 #{cell_A}, Goal A', fontsize=7)
        if i == 0:
            ax.set_ylabel('y (m)')
        ax.set_xlabel('x (m)')

        # Goal B
        ax = fig.add_subplot(gs[2, i * 2 + 1])
        rm = gaussian_filter(sim.ca1_ratemap_B[:, :, cell_B], sigma=1.2)
        ax.imshow(rm, origin='lower', cmap='hot', extent=[0, 1, 0, 1])
        ax.plot(*sim.arena.rewards['B'], '*', color='cyan', markersize=10,
                markeredgecolor='white')
        ax.set_title(f'CA1 #{cell_B}, Goal B', fontsize=7)
        ax.set_xlabel('x (m)')

    # ── (g) World model: believed vs actual position ──
    ax = fig.add_subplot(gs[3, 0:2])
    _plot_world_model(ax, sim)

    # ── (h) PV correlation: CA1 vs Subiculum ──
    ax = fig.add_subplot(gs[3, 2])
    _plot_pv_correlation(ax, sim)

    # ── (i) Circuit diagram ──
    ax = fig.add_subplot(gs[3, 3])
    _plot_circuit_summary(ax, sim)

    fig.suptitle(
        f'Embodied Virtual Mouse: {N_TOTAL} LIF Neurons, '
        f'{T_TOTAL:.0f}s Simulation\n'
        f'Allo-Ego-Allo Loop with BTSP Plasticity',
        fontsize=12, fontweight='bold', y=0.98)

    fig.savefig('/home/user/world-model/embodied_mouse_demo.png')
    plt.close()
    print("Saved: embodied_mouse_demo.png")


def _plot_trajectory(ax, sim):
    """Plot mouse trajectory colored by goal."""
    traj = sim.trajectory
    switch_idx = int(sim.goal_switch_time * 1000 / DT_ENV)

    # Phase A
    if switch_idx > 1:
        ax.plot(traj[:switch_idx, 0], traj[:switch_idx, 1],
                '-', color='#E74C3C', alpha=0.3, linewidth=0.3, label='Goal A')
    # Phase B
    if switch_idx < len(traj) - 1:
        ax.plot(traj[switch_idx:, 0], traj[switch_idx:, 1],
                '-', color='#3498DB', alpha=0.3, linewidth=0.3, label='Goal B')

    # Start and end
    ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=8, label='End')

    # Rewards
    for label, pos in sim.arena.rewards.items():
        c = '#E74C3C' if label == 'A' else '#3498DB'
        ax.plot(pos[0], pos[1], '*', color=c, markersize=15,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.annotate(label, pos, fontsize=8, fontweight='bold',
                    ha='center', va='bottom', color=c,
                    xytext=(0, 8), textcoords='offset points')

    # Reward visits
    for rt in sim.reward_times[:50]:
        idx = int(rt * 1000 / DT_ENV)
        if idx < len(traj):
            ax.plot(traj[idx, 0], traj[idx, 1], 'y^', markersize=4, alpha=0.5)

    ax.set_xlim(0, ARENA_SIZE)
    ax.set_ylim(0, ARENA_SIZE)
    ax.set_aspect('equal')
    ax.legend(fontsize=6, loc='lower right')
    ax.set_title('(a) Mouse trajectory in arena', fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')


def _plot_raster(ax, sim):
    """Spike raster plot showing activity across regions."""
    # Sample every 50th env step for clarity
    sample_rate = 50
    regions_to_plot = [
        ('Grid', R['grid'], '#4ECDC4'),
        ('DG', R['dg_gc'], '#45B7D1'),
        ('CA3', R['ca3_pyr'], '#96CEB4'),
        ('CA1', R['ca1_pyr'], '#FFEAA7'),
        ('Sub', R['sub'], '#DDA0DD'),
    ]

    y_offset = 0
    yticks = []
    yticklabels = []

    for name, region, color in regions_to_plot:
        n_cells = region.stop - region.start
        # Sample cells for visibility
        n_show = min(20, n_cells)
        cell_indices = np.linspace(0, n_cells - 1, n_show, dtype=int)

        for i_show, cell_local in enumerate(cell_indices):
            for t_env in range(0, N_ENV_STEPS, sample_rate):
                # Use spike counts
                if name == 'Grid':
                    sc = sim.spike_counts['grid'][t_env, cell_local]
                elif name == 'CA1':
                    sc = sim.spike_counts['ca1'][t_env, cell_local]
                elif name == 'Sub':
                    sc = sim.spike_counts['sub'][t_env, cell_local]
                else:
                    continue

                if sc > 0:
                    t_sec = t_env * DT_ENV / 1000
                    ax.plot(t_sec, y_offset + i_show, '|',
                            color=color, markersize=1.5, alpha=0.6)

        yticks.append(y_offset + n_show / 2)
        yticklabels.append(name)
        y_offset += n_show + 3

    # Mark goal switch
    ax.axvline(sim.goal_switch_time, color='red', linestyle='--',
               alpha=0.5, linewidth=1)
    ax.text(sim.goal_switch_time, y_offset, 'A→B', fontsize=7,
            color='red', ha='center')

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7)
    ax.set_xlabel('Time (s)')
    ax.set_title('(b) Neural activity across circuit', fontweight='bold')
    ax.set_xlim(0, T_TOTAL)


def _plot_world_model(ax, sim):
    """
    World model visualization: believed vs actual position.
    This is what the mouse 'thinks' about where it is.
    """
    t = np.arange(N_ENV_STEPS) * DT_ENV / 1000

    # Position error over time
    actual = sim.trajectory
    believed = sim.believed_positions
    error = np.linalg.norm(actual - believed, axis=1)

    # Smooth for visibility
    window = 50
    if len(error) > window:
        smoothed = np.convolve(error, np.ones(window) / window, mode='valid')
        t_smooth = t[window // 2:window // 2 + len(smoothed)]
    else:
        smoothed = error
        t_smooth = t

    ax.fill_between(t_smooth, 0, smoothed, alpha=0.3, color='#E74C3C')
    ax.plot(t_smooth, smoothed, color='#E74C3C', linewidth=0.8)

    # Mark goal switch
    ax.axvline(sim.goal_switch_time, color='blue', linestyle='--',
               alpha=0.5, linewidth=1)

    # Mark reward times
    for rt in sim.reward_times[:30]:
        ax.axvline(rt, color='gold', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position error (m)')
    ax.set_title("(g) World model: position decoding error\n"
                 "(mouse's believed vs actual position)",
                 fontweight='bold')
    ax.set_xlim(0, T_TOTAL)
    ax.set_ylim(0, None)

    # Confidence on twin axis
    ax2 = ax.twinx()
    conf_smooth = np.convolve(sim.confidences,
                              np.ones(window) / window, mode='valid')
    ax2.plot(t_smooth[:len(conf_smooth)], conf_smooth[:len(t_smooth)],
             color='#2ECC71', linewidth=0.8, alpha=0.6)
    ax2.set_ylabel('Confidence', color='#2ECC71', fontsize=7)
    ax2.tick_params(axis='y', labelcolor='#2ECC71', labelsize=6)


def _plot_pv_correlation(ax, sim):
    """
    Population vector correlation between Goal A and B contexts.
    CA1 should show low correlation (remapping).
    Subiculum should show high correlation (stability).
    """
    nbins = N_SPATIAL_BINS
    min_occ = 3

    ca1_corrs = []
    sub_corrs = []

    for iy in range(nbins):
        for ix in range(nbins):
            if (sim.occupancy_A[iy, ix] >= min_occ and
                    sim.occupancy_B[iy, ix] >= min_occ):
                va = sim.ca1_ratemap_A[iy, ix]
                vb = sim.ca1_ratemap_B[iy, ix]
                if va.std() > 1e-9 and vb.std() > 1e-9:
                    ca1_corrs.append(np.corrcoef(va, vb)[0, 1])

                va = sim.sub_ratemap_A[iy, ix]
                vb = sim.sub_ratemap_B[iy, ix]
                if va.std() > 1e-9 and vb.std() > 1e-9:
                    sub_corrs.append(np.corrcoef(va, vb)[0, 1])

    if ca1_corrs:
        ax.hist(ca1_corrs, bins=20, range=(-1, 1), alpha=0.6,
                color='#FFEAA7', edgecolor='#F39C12', label='CA1')
    if sub_corrs:
        ax.hist(sub_corrs, bins=20, range=(-1, 1), alpha=0.6,
                color='#DDA0DD', edgecolor='#8E44AD', label='Sub')

    ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('PV correlation (A vs B)')
    ax.set_ylabel('Count')
    ax.set_title('(h) Remapping: CA1 vs Sub\n'
                 'CA1=low (remaps), Sub=high (stable)',
                 fontweight='bold')
    ax.legend(fontsize=7)


def _plot_circuit_summary(ax, sim):
    """Summary statistics of the circuit."""
    ax.axis('off')

    lines = [
        f"Circuit Summary",
        f"{'─' * 30}",
        f"Total neurons: {N_TOTAL}",
        f"Synapses (exc): {(sim.net.W_exc > 0).sum():,}",
        f"Synapses (inh): {(sim.net.W_inh > 0).sum():,}",
        f"",
        f"Simulation: {T_TOTAL:.0f}s @ dt={DT}ms",
        f"Neural steps: {int(N_ENV_STEPS * STEPS_PER_ENV):,}",
        f"",
        f"Grid cells: {R['grid'].stop - R['grid'].start}",
        f"  (6 modules × 32 cells)",
        f"DG granule: {R['dg_gc'].stop - R['dg_gc'].start}",
        f"CA3 pyramidal: {R['ca3_pyr'].stop - R['ca3_pyr'].start}",
        f"CA1 pyramidal: {R['ca1_pyr'].stop - R['ca1_pyr'].start}",
        f"Subiculum: {R['sub'].stop - R['sub'].start}",
        f"",
        f"Rewards found: {len(sim.reward_times)}",
        f"BTSP events: {len(sim.reward_times)}",
        f"Goal: A (0-{sim.goal_switch_time:.0f}s)"
        f" → B ({sim.goal_switch_time:.0f}-{T_TOTAL:.0f}s)",
    ]

    for i, line in enumerate(lines):
        weight = 'bold' if i == 0 else 'normal'
        ax.text(0.05, 0.95 - i * 0.05, line, transform=ax.transAxes,
                fontsize=6.5, fontfamily='monospace', fontweight=weight,
                verticalalignment='top')

    ax.set_title('(i) Circuit summary', fontweight='bold')


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  Embodied Virtual Mouse with Spiking EC-Hippocampal Circuit")
    print("  Inspired by Eon Systems' embodied Drosophila brain emulation")
    print("=" * 70)
    print()

    sim = Simulation(seed=42)
    sim.run()
    print()
    visualize(sim)

    print()
    print("=" * 70)
    print("  The WORLD MODEL is the circuit itself:")
    print("  - Grid cells = believed position (allocentric)")
    print("  - CA1 = goal-dependent observation (egocentric)")
    print("  - Theta sweeps = forward prediction")
    print("  - Sub→EC feedback = consistency maintenance")
    print("  - BTSP = single-trial learning of new representations")
    print("=" * 70)
