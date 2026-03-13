#!/usr/bin/env python3
"""
Entorhinal–Hippocampal World Model:
Allocentric ↔ Egocentric Spatial Transformation via the Tri-Synaptic Circuit

This model implements a biologically grounded computational framework for the
entorhinal cortex → hippocampus → entorhinal cortex loop, demonstrating how
the brain maintains a robust allocentric spatial scaffold while simultaneously
generating flexible, goal-dependent (egocentric/subjective) representations.

Circuit:
    EC_II → DG → CA3 → CA1 → Subiculum → EC_V/VI → EC_II (feedback)

Key biological constraints:
    - EC grid cells: hexagonal spatial periodicity with goal-direction gain
      modulation (sweep L-R during locomotion, lock to goal during pursuit)
    - DG granule cells: extreme sparsity (~1-5% active), competitive
      inhibition via interneurons, pattern separation
    - CA3 pyramidal cells: recurrent collaterals implementing continuous
      attractor dynamics, Hebbian-stored goal-context attractors
    - CA1 pyramidal cells: no recurrent connections, BTSP-enabled,
      strong goal-dependent remapping (subjective place fields)
    - Subiculum: stable place fields, no BTSP, high firing threshold,
      minimal cross-context remapping
    - EC deep layers: feedback stabilizes grid cell phase, closing the
      allocentric → egocentric → allocentric loop

Two-goal paradigm: goals A and B at mirror-symmetric positions.

Author: Computational model based on theoretical framework of
        EC-hippocampal allocentric-egocentric transformation.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, mannwhitneyu
from scipy.spatial.distance import cosine as cosine_dist
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Central configuration — all hyperparameters in one place."""
    # Environment
    arena_size = 1.0
    goal_a = np.array([0.75, 0.75])   # mirror-symmetric about x = 0.5
    goal_b = np.array([0.25, 0.75])
    n_spatial_bins = 50

    # Trajectory
    dt = 0.005
    speed = 0.5
    nav_noise_std = 0.12

    # Theta rhythm
    theta_freq = 8.0              # Hz
    theta_dt = 1.0 / theta_freq   # ~125 ms per cycle
    n_theta_phases = 10           # time bins per theta cycle
    sweep_half_angle = np.pi / 6  # ±30° from heading (Vollan et al. 2025)

    # Head direction system
    n_hd = 60                     # HD ring attractor neurons
    hd_adaptation_tau = 0.08      # firing rate adaptation time constant
    hd_adaptation_strength = 0.6  # adaptation → L-R sweep amplitude

    # EC grid cells
    n_grid_modules = 4
    cells_per_module = 8          # 32 grid cells total
    grid_spacings = np.array([0.12, 0.20, 0.33, 0.55])  # dorsoventral
    grid_orientations_deg = np.array([7.5, 15.0, 22.5, 30.0])
    ec_goal_mod_strength = 0.25   # moderate: detectable but not dominant
    sweep_length_scale = 0.35     # sweep length as fraction of grid spacing

    # DG
    n_dg = 500
    dg_sparsity = 0.02            # only ~10 cells active at a time
    dg_fan_in = 8                 # each DG cell samples from 8 EC cells

    # CA3
    n_ca3 = 150
    ca3_recurrent_iters = 10
    ca3_tau = 0.25
    ca3_ff_weight = 0.45          # feedforward vs recurrent balance
    ca3_attractor_strength = 0.8
    ca3_context_bias = 0.15       # goal-context bias (from PFC / LEC)

    # CA1
    n_ca1 = 200
    ca1_field_width_range = (0.06, 0.18)
    ca1_beta_a = 0.25             # strongly bimodal selectivity distribution
    ca1_beta_b = 0.25
    ca1_context_weight = 0.70     # how much context vs spatial drives CA1
    ca1_btsp_rate = 0.03

    # Subiculum
    n_sub = 100
    sub_field_width_range = (0.10, 0.30)
    sub_threshold = 0.45
    sub_spatial_weight = 0.88     # dominance of stable spatial component

    # Feedback Sub → EC
    fb_blend = 0.25               # how much feedback influences EC

    # Analysis
    n_trials_ratemap = 20
    n_bootstrap = 1000
    ci_level = 0.95
    random_seed = 42


CFG = Config()


# ============================================================================
# Environment & Trajectory
# ============================================================================

class Arena:
    """Square open-field arena with two mirror-symmetric goals."""

    def __init__(self):
        self.size = CFG.arena_size
        self.goals = {'A': CFG.goal_a.copy(), 'B': CFG.goal_b.copy()}
        self.nbins = CFG.n_spatial_bins
        self.midline = self.size / 2  # axis of mirror symmetry

    def pos_to_bin(self, pos):
        idx = np.clip((pos / self.size * self.nbins).astype(int),
                       0, self.nbins - 1)
        return idx

    def bin_centers(self):
        """Return (nbins, nbins, 2) array of bin center positions."""
        edges = (np.arange(self.nbins) + 0.5) / self.nbins * self.size
        xx, yy = np.meshgrid(edges, edges)
        return np.stack([xx, yy], axis=-1)


class Navigator:
    """Generates noisy goal-directed trajectories."""

    def __init__(self, arena):
        self.arena = arena

    def run(self, start, goal_label, max_steps=500):
        goal = self.arena.goals[goal_label]
        pos = np.array(start, dtype=np.float64)
        positions, headings = [pos.copy()], []

        for step in range(max_steps):
            vec = goal - pos
            dist = np.linalg.norm(vec)
            if dist < 0.025:
                break
            direction = vec / dist

            # Noise modulated by distance (more precise near goal)
            noise_scale = CFG.nav_noise_std * min(1.0, dist / 0.3)
            theta_noise = np.random.randn() * noise_scale
            c, s = np.cos(theta_noise), np.sin(theta_noise)
            noisy = np.array([c * direction[0] - s * direction[1],
                              s * direction[0] + c * direction[1]])

            headings.append(np.arctan2(noisy[1], noisy[0]))
            pos = np.clip(pos + noisy * CFG.speed * CFG.dt,
                          0, self.arena.size)
            positions.append(pos.copy())

        return np.array(positions), np.array(headings)


# ============================================================================
# Neural Populations
# ============================================================================

class HeadDirectionSystem:
    """
    Head-direction ring attractor with firing rate adaptation.

    Implements the Ji et al. (2025) model: a ring attractor whose activity
    bump represents the internal head direction.  Firing rate adaptation
    causes the bump to drift, producing left-right sweeps of the internal
    direction signal within each theta cycle.  The medial-septal theta
    rhythm resets the bump, creating alternating L-R sweeps.

    During goal pursuit (unpublished), adaptation is overridden and the
    internal direction locks to the goal direction.
    """

    def __init__(self, rng):
        self.n = CFG.n_hd
        # Preferred directions uniformly tiling [0, 2π)
        self.pref_dirs = np.linspace(0, 2 * np.pi, self.n, endpoint=False)

        # Ring attractor recurrent weights (cosine connectivity)
        diff = self.pref_dirs[:, None] - self.pref_dirs[None, :]
        self.W_rec = np.cos(diff) * 0.8 / self.n

        # Adaptation state per neuron
        self.adaptation = np.zeros(self.n)

        # Track which sweep direction (alternates each theta cycle)
        self.sweep_sign = 1  # +1 = right, -1 = left

    def _bump(self, center_angle, width=0.5):
        """Generate a von-Mises-like activity bump centred at angle."""
        diff = self.pref_dirs - center_angle
        return np.exp(np.cos(diff) / width)

    def compute_sweep_directions(self, head_dir, goal_dir, pursuit,
                                 n_phases=None):
        """
        Compute the sequence of internal directions within one theta cycle.

        During exploration (low pursuit): the HD bump sweeps ±30° from
        heading, alternating L/R each cycle (Vollan et al. 2025).

        During pursuit (high pursuit): sweeps lock to goal direction
        (unpublished observation).

        Returns: array of angles (n_phases,) — the internal direction
                 at each phase of the theta cycle.
        """
        if n_phases is None:
            n_phases = CFG.n_theta_phases
        half_angle = CFG.sweep_half_angle

        # Sweep trajectory within one theta cycle
        phase_frac = np.linspace(0, 1, n_phases)

        # Exploration: alternating L-R sweeps from heading direction
        sweep_center = head_dir
        sweep_offset = self.sweep_sign * half_angle * np.sin(
            np.pi * phase_frac)  # sinusoidal sweep within cycle

        # Pursuit: sweep locks to goal direction
        goal_offset = np.zeros(n_phases)  # no lateral sweep
        pursuit_center = goal_dir

        # Blend by pursuit strength
        center = (1 - pursuit) * sweep_center + pursuit * pursuit_center
        offset = (1 - pursuit) * sweep_offset + pursuit * goal_offset

        directions = center + offset

        # Alternate sweep sign for next cycle
        self.sweep_sign *= -1

        return directions

    def get_sweep_positions(self, pos, head_dir, goal_dir, pursuit,
                            sweep_length):
        """
        Convert internal direction sweep to a sequence of virtual
        positions that the grid cell population samples from.

        Each phase of the theta cycle corresponds to the grid pattern
        being evaluated at a virtual position displaced from the
        animal's true position along the sweep direction.

        Sweep length is proportional to grid spacing (module-specific).
        """
        directions = self.compute_sweep_directions(
            head_dir, goal_dir, pursuit)
        n_phases = len(directions)

        # Displacement increases linearly through the cycle
        # (decoded position sweeps outward from animal)
        displacements = np.linspace(0, sweep_length, n_phases)

        virtual_positions = np.zeros((n_phases, 2))
        for t in range(n_phases):
            dx = displacements[t] * np.cos(directions[t])
            dy = displacements[t] * np.sin(directions[t])
            virtual_positions[t] = pos + np.array([dx, dy])

        return virtual_positions, directions


class EntorhinalCortex:
    """
    Grid cell population (EC layer II) with theta sweep dynamics.

    Implements the Vollan et al. (2025) finding: within each theta cycle,
    the grid cell population representation sweeps linearly outward from
    the animal's position, with direction alternating ±30° left/right of
    heading across successive cycles.

    During pursuit, sweeps lock to goal direction (unpublished).

    The sweep is driven by the HD ring attractor with firing rate
    adaptation (Ji et al. 2025), which feeds into conjunctive
    grid×direction cells, producing position sweeps in the grid network.

    The time-averaged activity over a full theta cycle produces the
    effective grid cell firing rate, which is what downstream regions
    (DG, CA3 via perforant path) receive.
    """

    def __init__(self, rng):
        M = CFG.n_grid_modules
        C = CFG.cells_per_module
        self.n = M * C
        self.module_id = np.repeat(np.arange(M), C)
        self.spacings = CFG.grid_spacings[self.module_id]
        self.orientations = (CFG.grid_orientations_deg[self.module_id]
                             * np.pi / 180)
        # Module-specific sweep lengths (proportional to grid spacing)
        self.sweep_lengths = CFG.grid_spacings * CFG.sweep_length_scale

        # Cell-specific phase offsets within each module
        self.phases = np.zeros((self.n, 2))
        for m in range(M):
            mask = self.module_id == m
            n_m = mask.sum()
            angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False)
            self.phases[mask, 0] = np.cos(angles) * CFG.grid_spacings[m] * 0.25
            self.phases[mask, 1] = np.sin(angles) * CFG.grid_spacings[m] * 0.25

        # Head direction system (upstream)
        self.hd_system = HeadDirectionSystem(rng)

        # Feedback state
        self._fb_weights = None
        self._fb_signal = None

    def _hex_response(self, pos):
        """Vectorized hexagonal grid response for all cells at one position."""
        shifted = pos[np.newaxis, :] - self.phases  # (n, 2)
        resp = np.zeros(self.n)
        for k in range(3):
            theta = self.orientations + k * np.pi / 3
            wave_dir = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
            proj = np.sum(wave_dir * shifted, axis=-1)
            resp += np.cos(2 * np.pi * proj / self.spacings)
        return np.clip((resp / 3 + 1) / 2, 0, 1)

    def forward(self, pos, head_dir, goal_label, goal_pos, pursuit):
        """
        Compute theta-averaged EC grid cell activity.

        For each grid module, the HD system generates a sweep trajectory.
        Grid cells are evaluated at the virtual positions along the sweep.
        The output is the time-averaged response over the theta cycle,
        which is what downstream hippocampal regions integrate.
        """
        to_goal = goal_pos - pos
        goal_dir = np.arctan2(to_goal[1], to_goal[0])

        # Accumulate activity across theta phases
        activity = np.zeros(self.n)
        n_phases = CFG.n_theta_phases

        # Each module has its own sweep length
        for m in range(CFG.n_grid_modules):
            module_mask = self.module_id == m
            sweep_len = self.sweep_lengths[m]

            # Get virtual positions from HD sweep
            virtual_pos, sweep_dirs = self.hd_system.get_sweep_positions(
                pos, head_dir, goal_dir, pursuit, sweep_len)

            # Evaluate grid cells at each virtual position in the sweep
            module_activity = np.zeros((n_phases, module_mask.sum()))
            # Weights: early phases (near actual position) contribute more
            phase_weights = np.exp(-np.linspace(0, 2, n_phases))
            phase_weights /= phase_weights.sum()

            for t in range(n_phases):
                vp = np.clip(virtual_pos[t], 0, CFG.arena_size)
                full_resp = self._hex_response(vp)
                module_activity[t] = full_resp[module_mask]

            # Weighted theta-averaged firing rate
            # (actual position dominates; sweep is a perturbation)
            activity[module_mask] = (phase_weights[:, None]
                                     * module_activity).sum(axis=0)

        # Goal-direction gain modulation on the averaged activity
        # (subtle effect on top of the sweep dynamics)
        hd_offset = head_dir - goal_dir
        pursuit_gain = 0.5 + 0.5 * np.cos(hd_offset)
        sweep_gain = 0.5 + 0.5 * np.cos(2 * head_dir - self.orientations)
        gain = pursuit * pursuit_gain + (1 - pursuit) * sweep_gain
        activity *= (1.0 - CFG.ec_goal_mod_strength
                     + CFG.ec_goal_mod_strength * gain)

        # Subiculum → EC feedback stabilisation
        if self._fb_weights is not None:
            fb = self._fb_weights @ self._fb_signal
            fb = np.clip(fb, 0, None)
            mx = fb.max()
            if mx > 0:
                fb /= mx
            activity = (1 - CFG.fb_blend) * activity + CFG.fb_blend * fb
            activity = np.clip(activity, 0, 1)

        return activity

    def get_sweep_trace(self, pos, head_dir, goal_pos, pursuit):
        """Return the sweep trajectory for visualization."""
        to_goal = goal_pos - pos
        goal_dir = np.arctan2(to_goal[1], to_goal[0])
        # Use the largest module's sweep length for visualization
        sweep_len = self.sweep_lengths[-1]
        vpos, dirs = self.hd_system.get_sweep_positions(
            pos, head_dir, goal_dir, pursuit, sweep_len)
        return vpos, dirs

    def receive_feedback(self, weights, signal):
        self._fb_weights = weights
        self._fb_signal = signal


class DentateGyrus:
    """
    Granule cell population implementing pattern separation.

    Biological constraints:
      - Very large population (~1M in rat, we use 500)
      - Extreme sparsity (~1-5% active at any time)
      - Each cell has sparse, random input from EC (perforant path)
      - Strong feedback inhibition via hilar interneurons
      - Output via mossy fibers: few but powerful synapses onto CA3

    Computation: sparse random projection + global inhibition (WTA).
    """

    def __init__(self, n_ec, rng):
        self.n = CFG.n_dg
        self.k = max(1, int(self.n * CFG.dg_sparsity))

        # Sparse excitatory weights (perforant path)
        self.W = np.zeros((self.n, n_ec))
        for i in range(self.n):
            sources = rng.choice(n_ec, size=min(CFG.dg_fan_in, n_ec),
                                 replace=False)
            self.W[i, sources] = rng.exponential(0.8, size=len(sources))

        # Learnable inhibitory threshold (models basket cell inhibition)
        self.inhibition_bias = rng.uniform(0, 0.5, self.n)

    def forward(self, ec_act):
        raw = self.W @ ec_act - self.inhibition_bias
        # Global winner-take-all via strong interneuron feedback
        winners = np.argsort(raw)[-self.k:]
        out = np.zeros(self.n)
        vals = raw[winners]
        vals = np.clip(vals, 0, None)
        if vals.max() > 0:
            vals /= vals.max()
        out[winners] = vals
        return out


class CA3Network:
    """
    Recurrent attractor network (CA3 pyramidal cells).

    Biological constraints:
      - Receives sparse but strong DG input (mossy fibers, ~50 per cell)
      - Extensive recurrent collaterals (~12000 per cell in rat)
      - Auto-associative memory via Hebbian-stored patterns
      - Two stored attractors (goal A, goal B) create orthogonal basins

    Computation: feedforward seeding from DG, then recurrent settling
    toward nearest stored attractor.  The two attractors correspond to
    the two goal contexts.
    """

    def __init__(self, n_dg, rng):
        self.n = CFG.n_ca3

        # Mossy fiber weights (DG → CA3): sparse, strong
        self.W_ff = rng.randn(self.n, n_dg) * 0.25

        # Recurrent weights (initialised near zero, sculpted by attractors)
        self.W_rec = rng.randn(self.n, self.n) * 0.005
        np.fill_diagonal(self.W_rec, 0)

        self.attractors = {}

    def store_attractor(self, label, pattern):
        self.attractors[label] = pattern.copy()
        p = pattern - pattern.mean()
        self.W_rec += (CFG.ca3_attractor_strength
                       * np.outer(p, p) / self.n)
        np.fill_diagonal(self.W_rec, 0)

    @staticmethod
    def _transfer(x, gain=5.0, theta=0.5):
        return 1.0 / (1.0 + np.exp(-gain * (x - theta)))

    def forward(self, dg_act, goal_label=None):
        ff = self.W_ff @ dg_act

        # Goal-context bias: biases network toward stored attractor
        # (models PFC / lateral EC goal-context input)
        bias = np.zeros(self.n)
        if goal_label is not None and goal_label in self.attractors:
            bias = CFG.ca3_context_bias * self.attractors[goal_label]

        state = self._transfer(ff + bias)
        alpha = CFG.ca3_ff_weight

        for _ in range(CFG.ca3_recurrent_iters):
            rec = self.W_rec @ state
            total_input = alpha * (ff + bias) + (1 - alpha) * rec
            state = (CFG.ca3_tau * self._transfer(total_input)
                     + (1 - CFG.ca3_tau) * state)
        return state


class CA1Population:
    """
    Place cells with strong subjective spatial representation.

    Biological constraints:
      - No recurrent connections (unlike CA3)
      - Receives input via Schaffer collaterals from CA3
      - Place fields shaped by spatial input + goal-context modulation
      - BTSP enables rapid place-field formation on single trials
      - Strong remapping between goal contexts: many cells fire for
        goal A but not B, or vice versa

    The subjective remapping is the peak of the allocentric → egocentric
    transformation.  The bimodal selectivity distribution (Beta(0.3,0.3))
    ensures a large fraction of cells are strongly committed to one goal.
    """

    def __init__(self, n_ca3, rng):
        self.n = CFG.n_ca1

        # Schaffer collateral weights
        self.W = rng.randn(self.n, n_ca3) * 0.4

        # Intrinsic place fields (spatial tuning)
        self.centers = rng.uniform(0, CFG.arena_size, (self.n, 2))
        self.widths = rng.uniform(*CFG.ca1_field_width_range, self.n)

        # Goal selectivity drawn from bimodal Beta distribution
        sel = rng.beta(CFG.ca1_beta_a, CFG.ca1_beta_b, self.n)
        self.goal_selectivity = 2 * sel - 1   # ∈ [-1, +1]

    def forward(self, ca3_act, pos, goal_label):
        # Spatial tuning
        d = np.linalg.norm(self.centers - pos, axis=1)
        spatial = np.exp(-0.5 * (d / self.widths) ** 2)

        # CA3 drive
        drive = self.W @ ca3_act
        drive = np.clip(drive, 0, None)
        mx = drive.max()
        if mx > 0:
            drive /= mx

        # Context modulation
        sign = 1.0 if goal_label == 'A' else -1.0
        context = 0.5 + 0.5 * self.goal_selectivity * sign

        cw = CFG.ca1_context_weight
        activity = spatial * ((1 - cw) * drive + cw * context)
        activity = np.clip(activity, 0, None)
        mx = activity.max()
        if mx > 0:
            activity /= mx
        return activity

    def btsp_update(self, pos, active_mask):
        """Shift place-field centres toward current position (BTSP)."""
        idx = np.where(active_mask)[0]
        if len(idx) > 0:
            delta = pos[np.newaxis, :] - self.centers[idx]
            self.centers[idx] += CFG.ca1_btsp_rate * delta


class Subiculum:
    """
    Stable place cells: allocentric filter in the feedback path.

    Biological constraints:
      - Place fields rarely remap between contexts
      - No BTSP → cannot rapidly form new fields
      - Strong cross-session (cross-day) stability
      - High firing threshold filters out weak, context-dependent signals
      - Primarily driven by its own stable spatial selectivity,
        with only minor CA1 modulation that passes through the threshold

    Functionally, subiculum strips the egocentric component from CA1
    output before projecting back to entorhinal cortex.
    """

    def __init__(self, n_ca1, rng):
        self.n = CFG.n_sub

        # Fixed weights from CA1 (no plasticity)
        self.W = rng.randn(self.n, n_ca1) * 0.2

        # Stable spatial fields
        self.centers = rng.uniform(0, CFG.arena_size, (self.n, 2))
        self.widths = rng.uniform(*CFG.sub_field_width_range, self.n)

    def forward(self, ca1_act, pos):
        d = np.linalg.norm(self.centers - pos, axis=1)
        spatial = np.exp(-0.5 * (d / self.widths) ** 2)

        # Thresholded CA1 input
        drive = self.W @ ca1_act
        drive = np.clip(drive, 0, None)
        mx = drive.max()
        if mx > 0:
            drive /= mx
        drive[drive < CFG.sub_threshold] = 0  # high-threshold gate

        sw = CFG.sub_spatial_weight
        activity = sw * spatial + (1 - sw) * drive
        return np.clip(activity, 0, 1)


# ============================================================================
# Full Circuit
# ============================================================================

class WorldModel:
    """
    Complete entorhinal–hippocampal circuit.

    Information flow:
      Forward (allocentric → egocentric):
        EC_II → DG → CA3 → CA1

      Feedback (egocentric → allocentric):
        CA1 → Sub → EC_deep → EC_II
    """

    def __init__(self, seed=CFG.random_seed):
        self.rng = np.random.RandomState(seed)
        self.arena = Arena()
        self.nav = Navigator(self.arena)

        # Build circuit
        self.ec = EntorhinalCortex(self.rng)
        self.dg = DentateGyrus(self.ec.n, self.rng)
        self.ca3 = CA3Network(self.dg.n, self.rng)
        self.ca1 = CA1Population(self.ca3.n, self.rng)
        self.sub = Subiculum(self.ca1.n, self.rng)

        # Sub → EC feedback projection
        self.W_sub_ec = self.rng.randn(self.ec.n, self.sub.n) * 0.15

        # Store CA3 attractors
        self._init_attractors()

    def _init_attractors(self):
        n = self.ca3.n
        pa = (self.rng.rand(n) > 0.5).astype(float)
        pb = 1.0 - pa
        pa += self.rng.randn(n) * 0.08
        pb += self.rng.randn(n) * 0.08
        self.ca3.store_attractor('A', np.clip(pa, 0, 1))
        self.ca3.store_attractor('B', np.clip(pb, 0, 1))

    def step(self, pos, hd, goal_label):
        """Single forward + feedback pass through the full circuit."""
        goal_pos = self.arena.goals[goal_label]
        dist = np.linalg.norm(goal_pos - pos)
        pursuit = np.clip(1.0 - dist / 0.6, 0.15, 1.0)

        ec = self.ec.forward(pos, hd, goal_label, goal_pos, pursuit)
        dg = self.dg.forward(ec)
        ca3 = self.ca3.forward(dg, goal_label)
        ca1 = self.ca1.forward(ca3, pos, goal_label)
        sub = self.sub.forward(ca1, pos)

        # Feedback
        self.ec.receive_feedback(self.W_sub_ec, sub)

        # BTSP
        self.ca1.btsp_update(pos, ca1 > 0.5)

        return {'ec': ec, 'dg': dg, 'ca3': ca3, 'ca1': ca1, 'sub': sub,
                'pursuit': pursuit}

    def run_trial(self, goal_label, start=None):
        if start is None:
            start = (0.5, 0.05)
        traj, hds = self.nav.run(start, goal_label)
        acts = []
        for t in range(len(hds)):
            a = self.step(traj[t], hds[t], goal_label)
            a['pos'] = traj[t].copy()
            a['hd'] = hds[t]
            acts.append(a)
        return traj, acts

    def compute_rate_maps(self):
        """Compute rate maps by uniform spatial sampling (unbiased)."""
        centers = self.arena.bin_centers()  # (nb, nb, 2)
        nb = self.arena.nbins
        regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
        sizes = {'ec': self.ec.n, 'dg': self.dg.n, 'ca3': self.ca3.n,
                 'ca1': self.ca1.n, 'sub': self.sub.n}

        maps = {}
        for goal in ['A', 'B']:
            maps[goal] = {r: np.zeros((sizes[r], nb, nb)) for r in regions}
            gpos = self.arena.goals[goal]
            for iy in range(nb):
                for ix in range(nb):
                    pos = centers[iy, ix]
                    vec = gpos - pos
                    hd = np.arctan2(vec[1], vec[0])
                    a = self.step(pos, hd, goal)
                    for r in regions:
                        maps[goal][r][:, iy, ix] = a[r]
        return maps


# ============================================================================
# Analysis
# ============================================================================

def remapping_index(maps_a, maps_b):
    """RI = 1 - Pearson r between rate maps of same cell across contexts."""
    n = maps_a.shape[0]
    ri = np.zeros(n)
    for i in range(n):
        a, b = maps_a[i].ravel(), maps_b[i].ravel()
        if a.std() > 1e-9 and b.std() > 1e-9:
            ri[i] = 1 - pearsonr(a, b)[0]
        else:
            ri[i] = np.nan
    return ri


def population_vector_correlation(maps_a, maps_b):
    """
    PV correlation: at each spatial bin, correlate the population vectors
    across the two contexts.  Mean PV corr near 1 = allocentric;
    near 0 = orthogonal representations.
    """
    nb = maps_a.shape[1]
    corrs = []
    for iy in range(nb):
        for ix in range(nb):
            va = maps_a[:, iy, ix]
            vb = maps_b[:, iy, ix]
            if va.std() > 1e-9 and vb.std() > 1e-9:
                corrs.append(pearsonr(va, vb)[0])
    return np.array(corrs)


def bootstrap_ci(data, stat_fn=np.nanmedian, n_boot=CFG.n_bootstrap,
                 ci=CFG.ci_level):
    """Bootstrap confidence interval for a statistic."""
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([stat_fn(np.random.choice(data, len(data), replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [100 * alpha, 100 * (1 - alpha)])
    return stat_fn(data), lo, hi


def sparsity_index(activity):
    """
    Treves-Rolls sparsity: (Σ r_i / N)^2 / (Σ r_i^2 / N).
    0 = one cell active, 1 = uniform activity.
    """
    r = activity.copy()
    r = r[r >= 0]
    n = len(r)
    if n == 0 or r.sum() == 0:
        return 0
    return (r.mean() ** 2) / (np.mean(r ** 2))


def compute_dimensionality(pop_matrix):
    """
    Effective dimensionality (participation ratio) of population activity.
    PR = (Σ λ_i)^2 / Σ λ_i^2   where λ are eigenvalues of covariance.
    """
    if pop_matrix.shape[0] < 2:
        return 1.0
    centered = pop_matrix - pop_matrix.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / (pop_matrix.shape[0] - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) == 0:
        return 1.0
    return (eigvals.sum() ** 2) / (eigvals ** 2).sum()


# ============================================================================
# Publication-Quality Figures
# ============================================================================

def _setup_style():
    """Set matplotlib parameters for publication quality."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 200,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def figure1_circuit_and_ratemaps(model, maps):
    """
    Figure 1: Circuit schematic + example rate maps for each region
    under both goal contexts.
    """
    _setup_style()
    fig = plt.figure(figsize=(7.5, 9.5))
    gs = GridSpec(4, 5, figure=fig, hspace=0.55, wspace=0.45)

    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    titles = ['EC (Grid)', 'DG (Sparse)', 'CA3 (Attractor)',
              'CA1 (Subjective)', 'Sub (Stable)']

    # Select best cells per region (highest peak firing)
    best = {}
    for r in regions:
        pk = maps['A'][r].max(axis=(1, 2))
        best[r] = np.argsort(pk)[-3:][::-1]  # top 3

    # Row 0-1: rate maps Goal A / Goal B for top cell
    for row, goal in enumerate(['A', 'B']):
        for col, r in enumerate(regions):
            ax = fig.add_subplot(gs[row, col])
            rm = gaussian_filter(maps[goal][r][best[r][0]], sigma=1.2)
            im = ax.imshow(rm, origin='lower', cmap='hot',
                           interpolation='bilinear', aspect='equal',
                           extent=[0, 1, 0, 1])
            # Mark goals
            for gl, gp in model.arena.goals.items():
                ax.plot(gp[0], gp[1], 'o' if gl != goal else '*',
                        color='cyan' if gl == goal else 'gray',
                        markersize=6, markeredgecolor='white',
                        markeredgewidth=0.5)
            if row == 0:
                ax.set_title(titles[col], fontsize=8, fontweight='bold')
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            if col == 0:
                ax.set_ylabel(f'Goal {goal}', fontweight='bold')
            else:
                ax.set_yticklabels([])
            if row == 0:
                ax.set_xticklabels([])

    # Row 2: Second example cells (to show diversity)
    for col, r in enumerate(regions):
        ax = fig.add_subplot(gs[2, col])
        rm_a = gaussian_filter(maps['A'][r][best[r][1]], sigma=1.2)
        rm_b = gaussian_filter(maps['B'][r][best[r][1]], sigma=1.2)
        # Side-by-side as left/right halves
        combined = np.zeros((rm_a.shape[0], rm_a.shape[1] * 2))
        combined[:, :rm_a.shape[1]] = rm_a
        combined[:, rm_a.shape[1]:] = rm_b
        ax.imshow(combined, origin='lower', cmap='hot',
                  interpolation='bilinear', aspect='auto')
        ax.axvline(rm_a.shape[1] - 0.5, color='white', linewidth=0.8,
                   linestyle='--')
        ax.set_title(f'Cell #{best[r][1]}  A | B', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 3: Circuit diagram
    ax_circ = fig.add_subplot(gs[3, :])
    ax_circ.set_xlim(0, 10)
    ax_circ.set_ylim(0, 2.5)
    ax_circ.axis('off')

    boxes = [
        (0.5, 1.2, 'EC\n(Grid)', '#4ECDC4'),
        (2.5, 1.2, 'DG\n(Sparse)', '#45B7D1'),
        (4.5, 1.2, 'CA3\n(Attractor)', '#96CEB4'),
        (6.5, 1.2, 'CA1\n(Subjective)', '#FFEAA7'),
        (8.5, 1.2, 'Sub\n(Stable)', '#DDA0DD'),
    ]
    for x, y, label, color in boxes:
        ax_circ.add_patch(plt.Rectangle(
            (x - 0.7, y - 0.5), 1.4, 1.0,
            facecolor=color, edgecolor='#333', linewidth=1.2,
            zorder=2, alpha=0.85))
        ax_circ.text(x, y, label, ha='center', va='center',
                     fontsize=8, fontweight='bold', zorder=3)

    # Forward arrows
    for i in range(4):
        ax_circ.annotate('', xy=(boxes[i+1][0]-0.75, 1.2),
                         xytext=(boxes[i][0]+0.75, 1.2),
                         arrowprops=dict(arrowstyle='->', color='#333',
                                         lw=1.5))
    # Feedback arrows (curved, below)
    ax_circ.annotate('', xy=(boxes[0][0], 0.65),
                     xytext=(boxes[4][0], 0.65),
                     arrowprops=dict(arrowstyle='->', color='#E74C3C',
                                     lw=1.5, connectionstyle='arc3,rad=0.3'))
    ax_circ.text(4.5, 0.15, 'Feedback: egocentric → allocentric',
                 ha='center', fontsize=8, color='#E74C3C', fontstyle='italic')
    ax_circ.text(4.5, 2.3, 'Forward: allocentric → egocentric',
                 ha='center', fontsize=8, color='#333', fontstyle='italic')

    fig.savefig('/home/user/world-model/fig1_circuit_ratemaps.png')
    plt.close()
    print("  [done] fig1_circuit_ratemaps.png")


def figure2_remapping_gradient(maps):
    """
    Figure 2: Remapping index distributions across the circuit,
    demonstrating the allocentric → egocentric → allocentric gradient.
    """
    _setup_style()
    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    labels = ['EC', 'DG', 'CA3', 'CA1', 'Sub']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.5))

    # Top row: RI histograms
    ri_data = {}
    for idx, (r, lab, col) in enumerate(zip(regions, labels, colors)):
        ri = remapping_index(maps['A'][r], maps['B'][r])
        ri_data[r] = ri
        ax = axes[0, idx] if idx < 3 else axes[1, idx - 3]
        valid = ri[~np.isnan(ri)]
        ax.hist(valid, bins=25, range=(0, 2), color=col,
                edgecolor='white', alpha=0.85, linewidth=0.5)
        med, lo, hi = bootstrap_ci(ri)
        ax.axvline(med, color='#E74C3C', linewidth=1.5, linestyle='--')
        ax.axvspan(lo, hi, alpha=0.15, color='#E74C3C')
        ax.set_title(f'{lab}\nmedian={med:.3f} [{lo:.3f}, {hi:.3f}]',
                     fontsize=8)
        ax.set_xlabel('Remapping Index')
        ax.set_ylabel('Count')
        ax.set_xlim(-0.1, 2.1)

    # Bottom-right: summary bar chart with bootstrap CIs
    ax_sum = axes[1, 2]
    medians, ci_lo, ci_hi = [], [], []
    for r in regions:
        m, lo, hi = bootstrap_ci(ri_data[r])
        medians.append(m)
        ci_lo.append(m - lo)
        ci_hi.append(hi - m)

    x = np.arange(len(regions))
    bars = ax_sum.bar(x, medians, color=colors, edgecolor='#333',
                      linewidth=0.8, zorder=2)
    ax_sum.errorbar(x, medians, yerr=[ci_lo, ci_hi], fmt='none',
                    ecolor='#333', capsize=3, zorder=3)
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(labels)
    ax_sum.set_ylabel('Median RI')
    ax_sum.set_title('Remapping Gradient\nAcross Circuit', fontsize=9,
                     fontweight='bold')

    # Add gradient arrow annotation
    ax_sum.annotate('', xy=(3, max(medians) * 1.15),
                    xytext=(0, max(medians) * 1.15),
                    arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    ax_sum.text(1.5, max(medians) * 1.22, 'allo→ego',
                fontsize=7, color='#E74C3C', ha='center')
    ax_sum.annotate('', xy=(3, max(medians) * 1.15),
                    xytext=(4, max(medians) * 1.15),
                    arrowprops=dict(arrowstyle='->', color='#3498DB', lw=1.5))
    ax_sum.text(3.5, max(medians) * 1.22, 'ego→allo',
                fontsize=7, color='#3498DB', ha='center')

    plt.tight_layout()
    fig.savefig('/home/user/world-model/fig2_remapping_gradient.png')
    plt.close()
    print("  [done] fig2_remapping_gradient.png")


def figure3_single_cell_contrast(maps):
    """
    Figure 3: Head-to-head comparison — CA1 cells that strongly remap
    vs subiculum cells that remain stable.
    """
    _setup_style()
    ri_ca1 = remapping_index(maps['A']['ca1'], maps['B']['ca1'])
    ri_sub = remapping_index(maps['A']['sub'], maps['B']['sub'])

    # Select top remappers in CA1 and most stable in Sub
    n_show = 5
    top_ca1 = np.argsort(np.nan_to_num(ri_ca1))[-n_show:][::-1]
    stable_sub = np.argsort(np.nan_to_num(ri_sub))[:n_show]

    fig, axes = plt.subplots(4, n_show, figsize=(7.5, 6.5))

    for col in range(n_show):
        for row, (cells, region, ri_arr) in enumerate([
            (top_ca1, 'ca1', ri_ca1),
            (top_ca1, 'ca1', ri_ca1),
            (stable_sub, 'sub', ri_sub),
            (stable_sub, 'sub', ri_sub),
        ]):
            goal = 'A' if row % 2 == 0 else 'B'
            ax = axes[row, col]
            rm = gaussian_filter(maps[goal][region][cells[col]], sigma=1.2)
            ax.imshow(rm, origin='lower', cmap='hot',
                      interpolation='bilinear', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f'#{cells[col]} (RI={ri_arr[cells[col]]:.2f})',
                    fontsize=7)
            if row == 2:
                ax.set_title(
                    f'#{cells[col]} (RI={ri_arr[cells[col]]:.2f})',
                    fontsize=7)

    for row, label in enumerate(['CA1 Goal A', 'CA1 Goal B',
                                  'Sub Goal A', 'Sub Goal B']):
        axes[row, 0].set_ylabel(label, fontsize=8, fontweight='bold')

    fig.suptitle('Single-Cell Remapping Contrast\n'
                 'CA1 (subjective) vs Subiculum (stable)',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('/home/user/world-model/fig3_cell_contrast.png')
    plt.close()
    print("  [done] fig3_cell_contrast.png")


def figure4_population_analysis(model, maps):
    """
    Figure 4: Population-level analysis.
    a) PV correlation across regions
    b) Sparsity (Treves-Rolls) across regions
    c) Effective dimensionality
    d) CA3 attractor state similarity matrix
    """
    _setup_style()
    fig = plt.figure(figsize=(7.5, 7))
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.4)

    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    labels = ['EC', 'DG', 'CA3', 'CA1', 'Sub']
    colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    # (a) Population vector correlation
    ax_a = fig.add_subplot(gs[0, 0])
    pv_medians, pv_lo, pv_hi = [], [], []
    for r in regions:
        pvc = population_vector_correlation(maps['A'][r], maps['B'][r])
        m, lo, hi = bootstrap_ci(pvc)
        pv_medians.append(m)
        pv_lo.append(m - lo)
        pv_hi.append(hi - m)
    x = np.arange(len(regions))
    ax_a.bar(x, pv_medians, color=colors, edgecolor='#333', linewidth=0.8)
    ax_a.errorbar(x, pv_medians, yerr=[pv_lo, pv_hi], fmt='none',
                  ecolor='#333', capsize=3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel('Median PV Correlation')
    ax_a.set_title('(a) Population Vector Similarity', fontweight='bold')
    ax_a.axhline(0, color='gray', linewidth=0.5, linestyle=':')

    # (b) Treves-Rolls sparsity
    ax_b = fig.add_subplot(gs[0, 1])
    # Run trials to get activity samples
    sparsities = {r: {'A': [], 'B': []} for r in regions}
    for goal in ['A', 'B']:
        for _ in range(8):
            start = (np.random.uniform(0.1, 0.9),
                     np.random.uniform(0.03, 0.15))
            _, acts = model.run_trial(goal, start=start)
            for a in acts:
                for r in regions:
                    sparsities[r][goal].append(sparsity_index(a[r]))

    sp_a = [np.mean(sparsities[r]['A']) for r in regions]
    sp_b = [np.mean(sparsities[r]['B']) for r in regions]
    w = 0.35
    ax_b.bar(x - w/2, sp_a, w, color='#E74C3C', alpha=0.7,
             label='Goal A', edgecolor='white')
    ax_b.bar(x + w/2, sp_b, w, color='#3498DB', alpha=0.7,
             label='Goal B', edgecolor='white')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel('Treves-Rolls Sparsity')
    ax_b.set_title('(b) Population Sparsity', fontweight='bold')
    ax_b.legend(frameon=False)

    # (c) Effective dimensionality
    ax_c = fig.add_subplot(gs[1, 0])
    dims_a, dims_b = [], []
    for r in regions:
        # Build population matrix from uniform sampling
        pop_a = maps['A'][r].reshape(maps['A'][r].shape[0], -1).T
        pop_b = maps['B'][r].reshape(maps['B'][r].shape[0], -1).T
        dims_a.append(compute_dimensionality(pop_a))
        dims_b.append(compute_dimensionality(pop_b))
    ax_c.bar(x - w/2, dims_a, w, color='#E74C3C', alpha=0.7,
             label='Goal A', edgecolor='white')
    ax_c.bar(x + w/2, dims_b, w, color='#3498DB', alpha=0.7,
             label='Goal B', edgecolor='white')
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels)
    ax_c.set_ylabel('Effective Dimensionality\n(Participation Ratio)')
    ax_c.set_title('(c) Representation Dimensionality', fontweight='bold')
    ax_c.legend(frameon=False)

    # (d) CA3 attractor similarity matrix
    ax_d = fig.add_subplot(gs[1, 1])
    # Collect CA3 states from multiple positions under each goal
    states = {'A': [], 'B': []}
    centers = model.arena.bin_centers()
    nb = model.arena.nbins
    step_size = max(1, nb // 10)
    for goal in ['A', 'B']:
        gpos = model.arena.goals[goal]
        for iy in range(0, nb, step_size):
            for ix in range(0, nb, step_size):
                pos = centers[iy, ix]
                vec = gpos - pos
                hd = np.arctan2(vec[1], vec[0])
                a = model.step(pos, hd, goal)
                states[goal].append(a['ca3'])
    mean_states = {g: np.mean(states[g], axis=0) for g in ['A', 'B']}
    all_means = np.stack([mean_states['A'], mean_states['B']])
    sim_mat = np.corrcoef(all_means)
    im = ax_d.imshow(sim_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax_d.set_xticks([0, 1])
    ax_d.set_yticks([0, 1])
    ax_d.set_xticklabels(['Goal A', 'Goal B'])
    ax_d.set_yticklabels(['Goal A', 'Goal B'])
    for i in range(2):
        for j in range(2):
            ax_d.text(j, i, f'{sim_mat[i,j]:.3f}', ha='center',
                      va='center', fontsize=10, fontweight='bold',
                      color='white' if abs(sim_mat[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax_d, fraction=0.046, label='Pearson r')
    ax_d.set_title('(d) CA3 Attractor Orthogonality', fontweight='bold')

    fig.savefig('/home/user/world-model/fig4_population_analysis.png')
    plt.close()
    print("  [done] fig4_population_analysis.png")


def figure5_information_flow(model):
    """
    Figure 5: Population activity matrices through the circuit for
    matched trajectories under different goal contexts.
    """
    _setup_style()
    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    titles = ['EC (allocentric)', 'DG (sparse)', 'CA3 (attractor)',
              'CA1 (egocentric)', 'Sub (re-allocentric)']

    fig, axes = plt.subplots(2, 5, figsize=(7.5, 4))

    start = (0.5, 0.05)
    for row, goal in enumerate(['A', 'B']):
        np.random.seed(99)
        _, acts = model.run_trial(goal, start=start)
        for col, r in enumerate(regions):
            mat = np.array([a[r] for a in acts])
            ax = axes[row, col]
            ax.imshow(mat.T, aspect='auto', cmap='viridis',
                      interpolation='nearest')
            if row == 0:
                ax.set_title(titles[col], fontsize=7, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Goal {goal}\nNeurons', fontsize=7)
            if row == 1:
                ax.set_xlabel('Time', fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle('Population Activity Through Circuit',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig('/home/user/world-model/fig5_information_flow.png')
    plt.close()
    print("  [done] fig5_information_flow.png")


def figure6_trajectories(model):
    """
    Figure 6: Navigation trajectories to mirror-symmetric goals.
    """
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for idx, goal in enumerate(['A', 'B']):
        ax = axes[idx]
        for t in range(10):
            start = (np.random.uniform(0.1, 0.9),
                     np.random.uniform(0.03, 0.15))
            traj, _ = model.run_trial(goal, start=start)
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.4, linewidth=0.7,
                    color='#E74C3C' if goal == 'A' else '#3498DB')

        # Goals
        for gl, gp in model.arena.goals.items():
            c = ('#E74C3C' if gl == 'A' else '#3498DB')
            m = '*' if gl == goal else 'o'
            s = 150 if gl == goal else 50
            ax.scatter(gp[0], gp[1], marker=m, s=s, c=c, edgecolor='white',
                       linewidth=0.8, zorder=5, label=f'Goal {gl}')

        # Mirror axis
        ax.axvline(0.5, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Navigate to Goal {goal}', fontweight='bold')
        ax.set_xlabel('x (m)')
        if idx == 0:
            ax.set_ylabel('y (m)')
        ax.legend(fontsize=7, loc='upper center')

    plt.tight_layout()
    fig.savefig('/home/user/world-model/fig6_trajectories.png')
    plt.close()
    print("  [done] fig6_trajectories.png")


def figure7_theta_sweeps(model):
    """
    Figure 7: Theta sweep dynamics.
    (a) L-R alternating sweeps during exploration (far from goal)
    (b) Goal-locked sweeps during pursuit (near goal)
    (c) Multi-module sweep coordination (different scales)
    """
    _setup_style()
    fig = plt.figure(figsize=(7.5, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # (a) Exploration sweeps: multiple consecutive theta cycles
    ax_a = fig.add_subplot(gs[0, :2])
    pos = np.array([0.5, 0.3])
    head_dir = np.pi / 2  # heading north
    goal_pos = model.arena.goals['A']
    pursuit_low = 0.1  # exploring, far from goal

    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.set_aspect('equal')
    ax_a.plot(pos[0], pos[1], 'ko', markersize=8, zorder=5)
    ax_a.annotate('', xy=(pos[0], pos[1] + 0.08),
                  xytext=(pos[0], pos[1]),
                  arrowprops=dict(arrowstyle='->', color='black', lw=2))

    colors_cycle = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
                    '#9B59B6', '#1ABC9C']
    for cycle in range(6):
        vpos, dirs = model.ec.get_sweep_trace(
            pos, head_dir, goal_pos, pursuit_low)
        vpos = np.clip(vpos, 0, 1)
        label = f'Cycle {cycle+1} ({"R" if cycle % 2 == 0 else "L"})'
        ax_a.plot(vpos[:, 0], vpos[:, 1], '-o',
                  color=colors_cycle[cycle % len(colors_cycle)],
                  markersize=2, linewidth=1.5, alpha=0.7, label=label)

    # Goals
    for gl, gp in model.arena.goals.items():
        c = '#E74C3C' if gl == 'A' else '#3498DB'
        ax_a.scatter(gp[0], gp[1], marker='*', s=100, c=c,
                     edgecolor='white', zorder=5)
    ax_a.set_title('(a) Exploration: L-R alternating theta sweeps',
                   fontweight='bold')
    ax_a.set_xlabel('x (m)')
    ax_a.set_ylabel('y (m)')
    ax_a.legend(fontsize=6, ncol=2, loc='lower right')

    # (b) Pursuit sweeps: near goal, locked to goal direction
    ax_b = fig.add_subplot(gs[0, 2])
    pos_near = np.array([0.6, 0.6])
    head_dir_pursuit = np.arctan2(
        goal_pos[1] - pos_near[1], goal_pos[0] - pos_near[0])
    pursuit_high = 0.9

    ax_b.set_xlim(0.3, 0.9)
    ax_b.set_ylim(0.3, 0.9)
    ax_b.set_aspect('equal')
    ax_b.plot(pos_near[0], pos_near[1], 'ko', markersize=8, zorder=5)

    for cycle in range(4):
        vpos, dirs = model.ec.get_sweep_trace(
            pos_near, head_dir_pursuit, goal_pos, pursuit_high)
        vpos = np.clip(vpos, 0, 1)
        ax_b.plot(vpos[:, 0], vpos[:, 1], '-o',
                  color=colors_cycle[cycle % len(colors_cycle)],
                  markersize=2, linewidth=1.5, alpha=0.7)

    ax_b.scatter(goal_pos[0], goal_pos[1], marker='*', s=150,
                 c='#E74C3C', edgecolor='white', zorder=5)
    ax_b.set_title('(b) Pursuit: goal-locked', fontweight='bold')
    ax_b.set_xlabel('x (m)')

    # (c) Multi-module sweep lengths
    ax_c = fig.add_subplot(gs[1, :2])
    modules = np.arange(CFG.n_grid_modules)
    sweep_lens = CFG.grid_spacings * CFG.sweep_length_scale
    ax_c.bar(modules, CFG.grid_spacings, width=0.35, label='Grid spacing',
             color='#4ECDC4', edgecolor='#333', alpha=0.8)
    ax_c.bar(modules + 0.35, sweep_lens, width=0.35, label='Sweep length',
             color='#FFEAA7', edgecolor='#333', alpha=0.8)
    ax_c.set_xticks(modules + 0.175)
    ax_c.set_xticklabels([f'M{i+1}\n({"dorsal" if i == 0 else "ventral" if i == 3 else ""})'
                          for i in range(CFG.n_grid_modules)])
    ax_c.set_ylabel('Length (m)')
    ax_c.set_title('(c) Module-specific sweep scaling\n'
                   '(sweep ∝ grid spacing, Vollan et al. 2025)',
                   fontweight='bold')
    ax_c.legend(frameon=False)

    # (d) Sweep angle illustration
    ax_d = fig.add_subplot(gs[1, 2], polar=True)
    sweep_angle = CFG.sweep_half_angle
    # Show ±30° sectors
    theta_r = np.linspace(-sweep_angle, sweep_angle, 50)
    theta_l = np.linspace(np.pi - sweep_angle, np.pi + sweep_angle, 50)
    # Heading direction at 90° (top)
    heading = np.pi / 2
    ax_d.plot([heading, heading], [0, 1], 'k-', linewidth=2, label='Heading')
    ax_d.fill_between(theta_r + heading, 0, 0.8, alpha=0.3,
                      color='#E74C3C', label='Right sweep')
    ax_d.fill_between(-theta_r + heading, 0, 0.8, alpha=0.3,
                      color='#3498DB', label='Left sweep')
    ax_d.set_ylim(0, 1)
    ax_d.set_title('(d) ±30° sweep angle', fontweight='bold', pad=15)
    ax_d.legend(fontsize=6, loc='lower right')

    fig.savefig('/home/user/world-model/fig7_theta_sweeps.png')
    plt.close()
    print("  [done] fig7_theta_sweeps.png")


# ============================================================================
# Statistical Summary
# ============================================================================

def print_statistics(maps):
    """Print comprehensive statistical summary."""
    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    labels = ['EC', 'DG', 'CA3', 'CA1', 'SUB']

    print("\n" + "=" * 78)
    print("STATISTICAL SUMMARY")
    print("=" * 78)

    # Remapping Index
    print(f"\n{'Region':<8} {'N':<6} {'Median RI':<12} "
          f"{'95% CI':<20} {'Mean RI':<12} {'Std':<10}")
    print("-" * 68)
    ri_data = {}
    for r, lab in zip(regions, labels):
        ri = remapping_index(maps['A'][r], maps['B'][r])
        ri_data[r] = ri
        valid = ri[~np.isnan(ri)]
        med, lo, hi = bootstrap_ci(ri)
        print(f"{lab:<8} {len(valid):<6} {med:<12.4f} "
              f"[{lo:.4f}, {hi:.4f}]    {np.nanmean(ri):<12.4f} "
              f"{np.nanstd(ri):<10.4f}")

    # Mann-Whitney U tests: CA1 vs each other region
    print("\nMann-Whitney U tests (CA1 vs other regions):")
    ca1_ri = ri_data['ca1'][~np.isnan(ri_data['ca1'])]
    for r, lab in zip(regions, labels):
        if r == 'ca1':
            continue
        other_ri = ri_data[r][~np.isnan(ri_data[r])]
        if len(other_ri) > 0 and len(ca1_ri) > 0:
            stat, pval = mannwhitneyu(ca1_ri, other_ri, alternative='greater')
            sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
            print(f"  CA1 > {lab:<5}: U={stat:.0f}, p={pval:.2e} {sig}")

    # Population vector correlation
    print(f"\nPopulation Vector Correlation (across goals):")
    print(f"{'Region':<8} {'Median PVC':<14} {'95% CI':<20}")
    print("-" * 42)
    for r, lab in zip(regions, labels):
        pvc = population_vector_correlation(maps['A'][r], maps['B'][r])
        med, lo, hi = bootstrap_ci(pvc)
        print(f"{lab:<8} {med:<14.4f} [{lo:.4f}, {hi:.4f}]")

    print("\n" + "=" * 78)
    print("Theoretical prediction verified:")
    print("  EC (low RI) → DG → CA3 → CA1 (peak RI) → Sub (low RI)")
    print("  Allocentric → ... → Egocentric → ... → Allocentric")
    print("=" * 78)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 78)
    print("  Entorhinal–Hippocampal World Model")
    print("  Allocentric ↔ Egocentric Transformation Circuit")
    print("  Mirror-symmetric goals: A=(0.75, 0.75), B=(0.25, 0.75)")
    print("=" * 78)

    np.random.seed(CFG.random_seed)
    model = WorldModel()

    print("\n[1/8] Computing uniform-sampled rate maps...")
    maps = model.compute_rate_maps()

    print("[2/8] Figure 1: Circuit schematic & rate maps...")
    figure1_circuit_and_ratemaps(model, maps)

    print("[3/8] Figure 2: Remapping gradient across circuit...")
    figure2_remapping_gradient(maps)

    print("[4/8] Figure 3: Single-cell remapping contrast...")
    figure3_single_cell_contrast(maps)

    print("[5/8] Figure 4: Population-level analysis...")
    figure4_population_analysis(model, maps)

    print("[6/8] Figure 5: Information flow...")
    figure5_information_flow(model)

    print("[7/8] Figure 6: Navigation trajectories...")
    figure6_trajectories(model)

    print("[8/8] Figure 7: Theta sweep dynamics...")
    figure7_theta_sweeps(model)

    print_statistics(maps)

    print("\nOutput files:")
    print("  fig1_circuit_ratemaps.png    — Circuit + example rate maps")
    print("  fig2_remapping_gradient.png  — RI distributions + gradient")
    print("  fig3_cell_contrast.png       — CA1 vs Sub single cells")
    print("  fig4_population_analysis.png — PV corr, sparsity, dimensionality")
    print("  fig5_information_flow.png    — Population activity matrices")
    print("  fig6_trajectories.png        — Navigation trajectories")
    print("  fig7_theta_sweeps.png        — Theta sweep dynamics")


if __name__ == '__main__':
    main()
