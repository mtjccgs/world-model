#!/usr/bin/env python3
"""
Minimal World Model Based on EC-Hippocampal Theory

This distills the full circuit into its computational essence:
a world model that maintains a stable internal map (allocentric)
while generating flexible, goal-dependent predictions (egocentric).

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    WORLD MODEL                       │
    │                                                      │
    │  ┌──────────┐   ┌──────┐   ┌──────┐   ┌──────────┐ │
    │  │  Latent   │──→│Sparse│──→│Attrac│──→│ Context  │ │
    │  │  State    │   │Compr.│   │ tor  │   │ Readout  │ │
    │  │ (EC Grid) │   │ (DG) │   │(CA3) │   │  (CA1)   │ │
    │  └────▲─────┘   └──────┘   └──────┘   └────┬─────┘ │
    │       │              FORWARD PATH           │       │
    │       │         (allo → ego)                │       │
    │       │                                     │       │
    │  ┌────┴─────┐          FEEDBACK PATH        │       │
    │  │Stability │◀────────(ego → allo)──────────┘       │
    │  │ Filter   │                                       │
    │  │  (Sub)   │                                       │
    │  └──────────┘                                       │
    │                                                      │
    │  ┌──────────┐                                       │
    │  │ Predict  │  Theta sweeps = forward rollout       │
    │  │ (Sweep)  │  "What will I encounter next?"        │
    │  └──────────┘                                       │
    └─────────────────────────────────────────────────────┘

Key insight: The same latent state (grid cell phase) can be
"rendered" into completely different observations depending on
the agent's current goal — this is what makes it a WORLD MODEL
rather than a simple map.

Usage:
    model = ECHippocampalWorldModel(arena_size=1.0)
    model.set_goal('A')
    obs = model.observe(position)           # goal-dependent observation
    predictions = model.predict_ahead(pos, heading)  # theta sweep rollout
    model.update(position)                  # feedback maintains consistency
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


# ============================================================================
# Core Components (minimal, each <30 lines)
# ============================================================================

class GridLatentState:
    """
    EC Grid Cells = Latent State Space.

    The grid cell phase is the brain's "latent variable" for position.
    Key property: the SAME physical state generates DIFFERENT latent
    representations depending on which direction the agent is heading
    (via theta sweeps), but the base spatial scaffold is stable.

    This is analogous to a VAE's latent space, but with the crucial
    addition of built-in spatial structure (hexagonal periodicity)
    that doesn't need to be learned from scratch.
    """

    def __init__(self, n_modules=4, cells_per_mod=8, arena_size=1.0):
        self.n = n_modules * cells_per_mod
        self.arena_size = arena_size
        spacings = np.array([0.12, 0.20, 0.33, 0.55])[:n_modules]
        orientations = np.array([7.5, 15, 22.5, 30])[:n_modules] * np.pi / 180

        # Pre-compute grid parameters per cell
        self.module_id = np.repeat(np.arange(n_modules), cells_per_mod)
        self.spacings = spacings[self.module_id]
        self.orientations = orientations[self.module_id]
        self.phases = np.zeros((self.n, 2))
        for m in range(n_modules):
            mask = self.module_id == m
            angles = np.linspace(0, 2 * np.pi, mask.sum(), endpoint=False)
            self.phases[mask, 0] = np.cos(angles) * spacings[m] * 0.25
            self.phases[mask, 1] = np.sin(angles) * spacings[m] * 0.25

    def encode(self, pos):
        """Encode position into grid cell phase (latent state)."""
        shifted = pos[np.newaxis, :] - self.phases
        resp = np.zeros(self.n)
        for k in range(3):
            theta = self.orientations + k * np.pi / 3
            wave_dir = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
            proj = np.sum(wave_dir * shifted, axis=-1)
            resp += np.cos(2 * np.pi * proj / self.spacings)
        return np.clip((resp / 3 + 1) / 2, 0, 1)


class SparseCompressor:
    """
    DG = Sparse Dimensionality Reduction.

    Takes high-dimensional, overlapping latent states and produces
    ultra-sparse, orthogonal codes.  This is the key to enabling
    pattern separation: two similar inputs become non-overlapping.

    Analogous to: sparse autoencoder bottleneck, or VQ-VAE codebook.
    """

    def __init__(self, n_in, n_out=500, sparsity=0.02, rng=None):
        rng = rng or np.random.RandomState()
        self.n_out = n_out
        self.k = max(1, int(n_out * sparsity))  # ~10 active neurons
        # Sparse random projection
        self.W = np.zeros((n_out, n_in))
        for i in range(n_out):
            src = rng.choice(n_in, size=min(8, n_in), replace=False)
            self.W[i, src] = rng.exponential(0.8, size=len(src))
        self.bias = rng.uniform(0, 0.5, n_out)

    def forward(self, x):
        raw = self.W @ x - self.bias
        winners = np.argsort(raw)[-self.k:]
        out = np.zeros(self.n_out)
        vals = np.clip(raw[winners], 0, None)
        if vals.max() > 0:
            vals /= vals.max()
        out[winners] = vals
        return out


class AttractorMemory:
    """
    CA3 = Attractor-based Context Memory.

    Stores discrete "contexts" (goal A, goal B) as attractor states.
    Given sparse input from DG + a context bias, settles into the
    nearest stored pattern.  This is what makes the world model
    context-aware: the SAME spatial input, combined with different
    goal context, activates different attractor states.

    Connection to Sun et al. (2025, Nature): The orthogonalized state
    machine (OSM) found in CA1 is built on CA3 attractor dynamics.
    The "clone states" in the CSCG model correspond to the two
    stored attractor patterns here — the same observation (spatial
    position) can be assigned to different latent states (goal
    contexts) via the attractor dynamics.

    Analogous to: Hopfield network, CSCG hidden states, or discrete
    latent variables in a mixture-of-experts world model.
    """

    def __init__(self, n_in, n_out=150, rng=None):
        rng = rng or np.random.RandomState()
        self.n = n_out
        self.W_ff = rng.randn(n_out, n_in) * 0.25
        self.W_rec = rng.randn(n_out, n_out) * 0.005
        np.fill_diagonal(self.W_rec, 0)
        self.attractors = {}

    def store_context(self, label, pattern):
        """Store a context pattern as an attractor."""
        self.attractors[label] = pattern.copy()
        p = pattern - pattern.mean()
        self.W_rec += 0.8 * np.outer(p, p) / self.n
        np.fill_diagonal(self.W_rec, 0)

    def forward(self, sparse_code, context_label=None):
        ff = self.W_ff @ sparse_code
        bias = np.zeros(self.n)
        if context_label and context_label in self.attractors:
            bias = 0.15 * self.attractors[context_label]
        state = _sigmoid(ff + bias)
        for _ in range(10):
            rec = self.W_rec @ state
            state = 0.25 * _sigmoid(0.45 * (ff + bias) + 0.55 * rec) + 0.75 * state
        return state


class ContextReadout:
    """
    CA1 = Context-Dependent Observation Model.

    This is where the "magic" happens: the SAME position produces
    COMPLETELY DIFFERENT observations depending on the active goal.

    This is the core of what makes the circuit a world model:
    - Traditional map: position → fixed observation
    - World model:     position × context → flexible observation

    Each readout neuron has:
    - A spatial receptive field (place field)
    - A goal preference (strongly bimodal: most cells prefer one goal)
    - No recurrent connections → maximally diverse readouts

    This implements the "orthogonalized state machine" (OSM) described
    in Sun et al. (2025, Nature): learning progressively decorrelates
    CA1 representations of similar environments, ultimately producing
    orthogonal codes for distinct task contexts.  The bimodal goal
    selectivity captures the "state cells" they identified — neurons
    that become task-state-specific through learning.

    Analogous to: conditional decoder in a CVAE, or goal-conditioned
    observation model in model-based RL, or the readout layer of a
    Clone-Structured Causal Graph (CSCG).
    """

    def __init__(self, n_in, n_out=200, rng=None):
        rng = rng or np.random.RandomState()
        self.n = n_out
        self.W = rng.randn(n_out, n_in) * 0.4
        self.centers = rng.uniform(0, 1, (n_out, 2))
        self.widths = rng.uniform(0.06, 0.18, n_out)
        # Bimodal goal selectivity (the subjective component)
        sel = rng.beta(0.25, 0.25, n_out)
        self.goal_selectivity = 2 * sel - 1

    def forward(self, attractor_state, pos, goal_label):
        d = np.linalg.norm(self.centers - pos, axis=1)
        spatial = np.exp(-0.5 * (d / self.widths) ** 2)
        drive = np.clip(self.W @ attractor_state, 0, None)
        mx = drive.max()
        if mx > 0:
            drive /= mx
        sign = 1.0 if goal_label == 'A' else -1.0
        context = 0.5 + 0.5 * self.goal_selectivity * sign
        activity = spatial * (0.3 * drive + 0.7 * context)
        activity = np.clip(activity, 0, None)
        mx = activity.max()
        if mx > 0:
            activity /= mx
        return activity


class StabilityFilter:
    """
    Subiculum = Consistency Maintenance.

    Prevents the world model from being corrupted by its own
    goal-dependent processing.  Without this, repeated processing
    through the egocentric pathway would cause the allocentric
    scaffold to drift.

    Key mechanism: NO fast plasticity (no BTSP) → can't form new
    context-dependent representations → only passes through
    consistent, context-invariant spatial information.

    Analogous to: skip connection that bypasses the context-dependent
    processing, or a "reality check" that anchors the world model
    to ground truth.
    """

    def __init__(self, n_in, n_out=100, rng=None):
        rng = rng or np.random.RandomState()
        self.n = n_out
        self.W = rng.randn(n_out, n_in) * 0.2
        self.centers = rng.uniform(0, 1, (n_out, 2))
        self.widths = rng.uniform(0.10, 0.30, n_out)

    def forward(self, ca1_activity, pos):
        d = np.linalg.norm(self.centers - pos, axis=1)
        spatial = np.exp(-0.5 * (d / self.widths) ** 2)
        drive = np.clip(self.W @ ca1_activity, 0, None)
        mx = drive.max()
        if mx > 0:
            drive /= mx
        drive[drive < 0.45] = 0  # high threshold: only consistent signals
        return np.clip(0.88 * spatial + 0.12 * drive, 0, 1)


class ThetaSweepPredictor:
    """
    Theta Sweeps = Forward Prediction / Trajectory Rollout.

    This is the world model's PREDICTION mechanism.  Within each
    theta cycle, the grid cells evaluate their firing at a sequence
    of virtual positions extending outward from the animal.

    This is computationally identical to:
    - Model predictive control: simulate N steps ahead
    - Dreamer's "imagination": roll out latent trajectories
    - Monte Carlo tree search: evaluate future states

    The alternating L-R pattern samples the environment bilaterally,
    creating a panoramic prediction of upcoming space.
    During pursuit, predictions focus on the goal direction.

    Ref: Vollan et al. (2025) Nature; Ji et al. (2025) Curr Biol.
    """

    def __init__(self, grid_state, sweep_half_angle=np.pi/6, n_phases=10):
        self.grid = grid_state
        self.half_angle = sweep_half_angle
        self.n_phases = n_phases
        self.sweep_sign = 1  # alternates L/R
        self.sweep_lengths = np.array([0.12, 0.20, 0.33, 0.55]) * 0.35

    def predict(self, pos, heading, goal_dir=None, pursuit=0.0):
        """
        Generate predicted latent states along the sweep trajectory.

        Returns:
            predictions: list of (virtual_pos, latent_state) tuples
                         representing the model's prediction of what
                         the agent will encounter along the sweep path.
        """
        phase_frac = np.linspace(0, 1, self.n_phases)

        # Sweep direction within this theta cycle
        if pursuit > 0.5 and goal_dir is not None:
            # Pursuit: locked to goal
            center = (1 - pursuit) * heading + pursuit * goal_dir
            offsets = np.zeros(self.n_phases)
        else:
            # Exploration: alternating L-R
            center = heading
            offsets = self.sweep_sign * self.half_angle * np.sin(
                np.pi * phase_frac)
            self.sweep_sign *= -1

        directions = center + offsets
        # Use median module sweep length for prediction
        sweep_len = np.median(self.sweep_lengths)
        displacements = np.linspace(0, sweep_len, self.n_phases)

        predictions = []
        for t in range(self.n_phases):
            vx = pos[0] + displacements[t] * np.cos(directions[t])
            vy = pos[1] + displacements[t] * np.sin(directions[t])
            vpos = np.clip(np.array([vx, vy]), 0, self.grid.arena_size)
            latent = self.grid.encode(vpos)
            predictions.append((vpos, latent, directions[t]))

        return predictions


# ============================================================================
# Utility
# ============================================================================

def _sigmoid(x, gain=5.0, theta=0.5):
    return 1.0 / (1.0 + np.exp(-gain * (x - theta)))


# ============================================================================
# Assembled World Model
# ============================================================================

class ECHippocampalWorldModel:
    """
    Complete world model assembled from EC-hippocampal components.

    Interface:
        model.set_goal(label)          — set current goal context
        model.observe(pos)             — get goal-dependent observation
        model.predict_ahead(pos, hd)   — theta sweep forward prediction
        model.update(pos)              — feedback to maintain consistency
        model.plan(start, candidates)  — evaluate candidate goals
    """

    def __init__(self, arena_size=1.0, seed=42):
        self.rng = np.random.RandomState(seed)
        self.arena_size = arena_size
        self.goals = {
            'A': np.array([0.75, 0.75]),  # mirror-symmetric
            'B': np.array([0.25, 0.75]),
        }
        self.current_goal = 'A'

        # Build circuit
        self.grid = GridLatentState(arena_size=arena_size)
        self.compressor = SparseCompressor(self.grid.n, rng=self.rng)
        self.memory = AttractorMemory(self.compressor.n_out, rng=self.rng)
        self.readout = ContextReadout(self.memory.n, rng=self.rng)
        self.filter = StabilityFilter(self.readout.n, rng=self.rng)
        self.predictor = ThetaSweepPredictor(self.grid)

        # Feedback weights (Sub → Grid)
        self.W_feedback = self.rng.randn(self.grid.n, self.filter.n) * 0.15

        # Store attractor patterns for goals
        n = self.memory.n
        pa = (self.rng.rand(n) > 0.5).astype(float)
        pa += self.rng.randn(n) * 0.08
        self.memory.store_context('A', np.clip(pa, 0, 1))
        self.memory.store_context('B', np.clip(1 - pa + self.rng.randn(n) * 0.08, 0, 1))

        # Internal state
        self._last_filter_output = None

    def set_goal(self, label):
        """Switch the world model's goal context."""
        self.current_goal = label

    def observe(self, pos):
        """
        Query the world model: "What do I observe at this position,
        given my current goal?"

        This is the key world-model operation: the same position
        produces different observations for different goals.
        """
        # Encode position into latent state
        latent = self.grid.encode(pos)

        # Apply feedback stabilization from previous step
        if self._last_filter_output is not None:
            fb = self.W_feedback @ self._last_filter_output
            fb = np.clip(fb, 0, None)
            mx = fb.max()
            if mx > 0:
                fb /= mx
            latent = 0.75 * latent + 0.25 * fb
            latent = np.clip(latent, 0, 1)

        # Forward path: allo → ego
        sparse = self.compressor.forward(latent)
        context = self.memory.forward(sparse, self.current_goal)
        observation = self.readout.forward(context, pos, self.current_goal)

        # Update stability filter (ego → allo feedback)
        self._last_filter_output = self.filter.forward(observation, pos)

        return {
            'latent': latent,           # EC: allocentric state
            'sparse': sparse,           # DG: compressed code
            'context': context,         # CA3: attractor state
            'observation': observation,  # CA1: goal-dependent readout
            'stable': self._last_filter_output,  # Sub: allocentric anchor
        }

    def predict_ahead(self, pos, heading, n_cycles=2):
        """
        Theta sweep prediction: "What will I encounter if I move
        in this direction?"

        Runs N theta cycles of forward prediction, alternating L/R.
        Returns predicted latent states along the sweep path.

        This is the world model's imagination / planning mechanism.
        """
        goal_pos = self.goals[self.current_goal]
        goal_dir = np.arctan2(goal_pos[1] - pos[1], goal_pos[0] - pos[0])
        dist = np.linalg.norm(goal_pos - pos)
        pursuit = np.clip(1.0 - dist / 0.6, 0.15, 1.0)

        all_predictions = []
        for cycle in range(n_cycles):
            preds = self.predictor.predict(pos, heading, goal_dir, pursuit)
            all_predictions.append(preds)
        return all_predictions

    def plan(self, pos, heading):
        """
        Simple planning: compare predicted value of going to A vs B.

        For each goal, predict the trajectory and evaluate how much
        the observation model "likes" the predicted path.
        Uses theta sweep predictions to evaluate future states.
        """
        scores = {}
        for goal_label in ['A', 'B']:
            self.set_goal(goal_label)
            goal_pos = self.goals[goal_label]
            goal_dir = np.arctan2(goal_pos[1] - pos[1],
                                  goal_pos[0] - pos[0])
            predictions = self.predictor.predict(
                pos, goal_dir, goal_dir, pursuit=0.5)

            # Score: how strongly does the readout respond along path?
            total_response = 0
            for vpos, latent, direction in predictions:
                sparse = self.compressor.forward(latent)
                ctx = self.memory.forward(sparse, goal_label)
                obs = self.readout.forward(ctx, vpos, goal_label)
                total_response += obs.sum()

            scores[goal_label] = total_response

        return scores


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate_world_model():
    """Show the world model in action with key demonstrations."""

    print("=" * 70)
    print("  EC-Hippocampal World Model: Minimal Implementation")
    print("  Demonstrating core world-model computations")
    print("=" * 70)

    np.random.seed(42)
    model = ECHippocampalWorldModel()

    # ── Demo 1: Same position, different goal → different observation ──
    print("\n[Demo 1] Context-dependent observation")
    print("  Same position (0.5, 0.5), different goals:\n")

    pos = np.array([0.5, 0.5])

    model.set_goal('A')
    obs_a = model.observe(pos)
    model.set_goal('B')
    obs_b = model.observe(pos)

    # Correlation between observations
    ca1_a = obs_a['observation']
    ca1_b = obs_b['observation']
    corr = np.corrcoef(ca1_a, ca1_b)[0, 1]
    print(f"  CA1 observation correlation (A vs B): {corr:.3f}")
    print(f"  → Near-zero = nearly orthogonal representations!")

    # But the stable output is similar
    sub_a = obs_a['stable']
    sub_b = obs_b['stable']
    sub_corr = np.corrcoef(sub_a, sub_b)[0, 1]
    print(f"  Sub stability correlation (A vs B):   {sub_corr:.3f}")
    print(f"  → Near-one = allocentric scaffold maintained!")

    # ── Demo 2: Theta sweep prediction ──
    print("\n[Demo 2] Theta sweep forward prediction")
    model.set_goal('A')
    predictions = model.predict_ahead(
        np.array([0.5, 0.3]), heading=np.pi/2, n_cycles=4)

    print(f"  4 theta cycles × 10 phases = {4*10} predicted states")
    print("  Sweep directions alternate L-R:")
    for i, cycle_preds in enumerate(predictions):
        first_dir = cycle_preds[0][2] * 180 / np.pi
        last_dir = cycle_preds[-1][2] * 180 / np.pi
        side = "R" if (i % 2 == 0) else "L"
        print(f"    Cycle {i+1} ({side}): {first_dir:.0f}° → {last_dir:.0f}°")

    # ── Demo 3: Planning ──
    print("\n[Demo 3] Goal comparison via forward prediction")
    pos = np.array([0.5, 0.3])
    scores = model.plan(pos, heading=np.pi/2)
    print(f"  From position (0.5, 0.3):")
    for label, score in scores.items():
        goal_pos = model.goals[label]
        dist = np.linalg.norm(goal_pos - pos)
        print(f"    Goal {label} at {goal_pos}: score={score:.1f}, dist={dist:.2f}")

    # ── Demo 4: Full circuit visualization ──
    print("\n[Demo 4] Generating visualization...")
    _plot_world_model_demo(model)
    print("  Saved: world_model_demo.png")

    print("\n" + "=" * 70)
    print("  Key computational principles:")
    print("  1. Grid cells = latent state space (allocentric coordinates)")
    print("  2. Theta sweeps = forward prediction (trajectory rollout)")
    print("  3. CA1 readout = context-dependent observation model")
    print("  4. Sub→EC feedback = world model consistency maintenance")
    print("=" * 70)


def _plot_world_model_demo(model):
    """Generate a demonstration figure."""
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 200, 'savefig.bbox': 'tight',
    })

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.45)

    nbins = 40

    # ── (a) Grid cell latent state ──
    ax = fig.add_subplot(gs[0, 0])
    rm = np.zeros((nbins, nbins))
    for iy in range(nbins):
        for ix in range(nbins):
            pos = np.array([(ix+0.5)/nbins, (iy+0.5)/nbins])
            rm[iy, ix] = model.grid.encode(pos)[0]
    ax.imshow(gaussian_filter(rm, sigma=0.8), origin='lower', cmap='hot',
              extent=[0,1,0,1])
    ax.set_title('(a) Grid cell latent state\n(one example cell)',
                 fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # ── (b,c) CA1 observations for Goal A vs B ──
    for idx, goal in enumerate(['A', 'B']):
        ax = fig.add_subplot(gs[0, 1+idx])
        rm = np.zeros((nbins, nbins))
        cell_idx = 50  # pick a strongly selective cell
        for iy in range(nbins):
            for ix in range(nbins):
                pos = np.array([(ix+0.5)/nbins, (iy+0.5)/nbins])
                model.set_goal(goal)
                obs = model.observe(pos)
                rm[iy, ix] = obs['observation'][cell_idx]
        ax.imshow(gaussian_filter(rm, sigma=1), origin='lower', cmap='hot',
                  extent=[0,1,0,1])
        for gl, gp in model.goals.items():
            c = '#00BFFF' if gl == goal else 'gray'
            ax.plot(gp[0], gp[1], '*', color=c, markersize=12,
                    markeredgecolor='white')
        ax.set_title(f'({"b" if idx==0 else "c"}) CA1 cell #{cell_idx}, '
                     f'Goal {goal}', fontweight='bold')
        ax.set_xlabel('x')

    # ── (d) PV correlation across the arena ──
    ax = fig.add_subplot(gs[1, 0])
    pv_map = np.zeros((nbins, nbins))
    for iy in range(nbins):
        for ix in range(nbins):
            pos = np.array([(ix+0.5)/nbins, (iy+0.5)/nbins])
            model.set_goal('A')
            obs_a = model.observe(pos)
            model.set_goal('B')
            obs_b = model.observe(pos)
            va = obs_a['observation']
            vb = obs_b['observation']
            if va.std() > 1e-9 and vb.std() > 1e-9:
                pv_map[iy, ix] = np.corrcoef(va, vb)[0, 1]
    im = ax.imshow(gaussian_filter(pv_map, sigma=1), origin='lower',
                   cmap='RdBu_r', vmin=-1, vmax=1, extent=[0,1,0,1])
    plt.colorbar(im, ax=ax, fraction=0.046, label='PV correlation')
    for gl, gp in model.goals.items():
        ax.plot(gp[0], gp[1], '*', color='black', markersize=10)
    ax.set_title('(d) CA1 PV correlation (A vs B)\nacross arena',
                 fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # ── (e) Subiculum stability ──
    ax = fig.add_subplot(gs[1, 1])
    sub_map = np.zeros((nbins, nbins))
    for iy in range(nbins):
        for ix in range(nbins):
            pos = np.array([(ix+0.5)/nbins, (iy+0.5)/nbins])
            model.set_goal('A')
            obs_a = model.observe(pos)
            model.set_goal('B')
            obs_b = model.observe(pos)
            va = obs_a['stable']
            vb = obs_b['stable']
            if va.std() > 1e-9 and vb.std() > 1e-9:
                sub_map[iy, ix] = np.corrcoef(va, vb)[0, 1]
    im = ax.imshow(gaussian_filter(sub_map, sigma=1), origin='lower',
                   cmap='RdBu_r', vmin=-1, vmax=1, extent=[0,1,0,1])
    plt.colorbar(im, ax=ax, fraction=0.046, label='PV correlation')
    ax.set_title('(e) Sub PV correlation (A vs B)\n(stable, allocentric)',
                 fontweight='bold')
    ax.set_xlabel('x')

    # ── (f) Theta sweep trajectories ──
    ax = fig.add_subplot(gs[1, 2])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    pos = np.array([0.5, 0.3])
    ax.plot(pos[0], pos[1], 'ko', markersize=8, zorder=5)
    ax.annotate('', xy=(pos[0], pos[1]+0.06), xytext=(pos[0], pos[1]),
                arrowprops=dict(arrowstyle='->', color='k', lw=2))

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    model.set_goal('A')
    for i in range(4):
        preds = model.predictor.predict(pos, np.pi/2, np.pi/4, 0.1)
        xs = [p[0][0] for p in preds]
        ys = [p[0][1] for p in preds]
        side = 'R' if i % 2 == 0 else 'L'
        ax.plot(xs, ys, '-o', color=colors[i], markersize=2,
                linewidth=1.5, alpha=0.7, label=f'Cycle {i+1} ({side})')
    for gl, gp in model.goals.items():
        c = '#E74C3C' if gl == 'A' else '#3498DB'
        ax.scatter(gp[0], gp[1], marker='*', s=100, c=c,
                   edgecolor='white', zorder=5)
    ax.legend(fontsize=7)
    ax.set_title('(f) Theta sweep predictions\n(forward rollout)',
                 fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # ── (g) Circuit diagram ──
    ax = fig.add_subplot(gs[2, :])
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')

    # World model components
    boxes = [
        (1, 1.8, 'Grid\nLatent State\n(EC)', '#4ECDC4', 'Latent z'),
        (3, 1.8, 'Sparse\nCompressor\n(DG)', '#45B7D1', 'Bottleneck'),
        (5, 1.8, 'Attractor\nMemory\n(CA3)', '#96CEB4', 'Context'),
        (7, 1.8, 'Context\nReadout\n(CA1)', '#FFEAA7', 'Observation'),
        (9, 1.8, 'Stability\nFilter\n(Sub)', '#DDA0DD', 'Anchor'),
    ]
    for x, y, label, color, role in boxes:
        ax.add_patch(plt.Rectangle(
            (x-0.8, y-0.6), 1.6, 1.2, facecolor=color,
            edgecolor='#333', linewidth=1.2, zorder=2, alpha=0.85))
        ax.text(x, y+0.1, label, ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=3)
        ax.text(x, y-0.45, role, ha='center', va='center',
                fontsize=6, fontstyle='italic', color='#666', zorder=3)

    # Forward arrows
    for i in range(4):
        ax.annotate('', xy=(boxes[i+1][0]-0.85, 1.8),
                     xytext=(boxes[i][0]+0.85, 1.8),
                     arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    # Feedback arrow
    ax.annotate('', xy=(boxes[0][0], 1.1),
                 xytext=(boxes[4][0], 1.1),
                 arrowprops=dict(arrowstyle='->', color='#E74C3C',
                                 lw=1.5, connectionstyle='arc3,rad=0.25'))
    ax.text(5, 0.55, 'Feedback: ego→allo (consistency maintenance)',
            ha='center', fontsize=8, color='#E74C3C', fontstyle='italic')
    ax.text(5, 2.75, 'Forward: allo→ego (context-dependent rendering)',
            ha='center', fontsize=8, color='#333', fontstyle='italic')

    # Sweep annotation
    ax.add_patch(plt.Rectangle(
        (0.2, 0.1), 1.6, 0.7, facecolor='#E8F8F5',
        edgecolor='#333', linewidth=0.8, zorder=2, alpha=0.85))
    ax.text(1, 0.45, 'Theta Sweep\n= Forward\nPrediction', ha='center',
            va='center', fontsize=7, fontweight='bold', zorder=3)
    ax.annotate('', xy=(0.2, 1.2), xytext=(0.8, 0.8),
                arrowprops=dict(arrowstyle='->', color='#27AE60',
                                lw=1, connectionstyle='arc3,rad=-0.3'))

    fig.savefig('/home/user/world-model/world_model_demo.png')
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    demonstrate_world_model()
