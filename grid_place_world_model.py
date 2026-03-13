"""
Bioinspired World Model: Entorhinal Grid Cell – Hippocampal Place Cell Circuit

Implements the allocentric ↔ egocentric transformation loop:
    EC (grid cells) → DG (pattern separation) → CA3 (attractor dynamics)
    → CA1 (subjective place cells) → Subiculum (stable filter) → EC (feedback)

Theoretical framework:
- EC grid cells: sweep left-right during locomotion; lock to goal during pursuit
- DG: sparse coding compresses high-dimensional EC input, orthogonalizes
- CA3: recurrent connections create orthogonal attractor states per goal
- CA1: amplifies into diverse subjective place cell representations
- Subiculum: high-threshold filter, no BTSP → stable allocentric mapping
- EC feedback: reconstructs stable allocentric spatial frame

Two-goal paradigm: animal navigates to goal A or B in the same environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter


# ============================================================================
# Environment
# ============================================================================

class Environment:
    """2D arena with two alternative goals."""

    def __init__(self, size=1.0, goal_a=(0.8, 0.8), goal_b=(0.2, 0.8)):
        self.size = size
        self.goals = {'A': np.array(goal_a), 'B': np.array(goal_b)}
        self.n_spatial_bins = 40  # for rate map discretization

    def discretize(self, pos):
        """Convert continuous position to bin indices."""
        bins = np.clip(
            (pos / self.size * self.n_spatial_bins).astype(int),
            0, self.n_spatial_bins - 1
        )
        return bins


# ============================================================================
# Trajectory Generator
# ============================================================================

class TrajectoryGenerator:
    """Generate trajectories toward a specified goal with some noise."""

    def __init__(self, env, dt=0.01, speed=0.4, noise_std=0.15):
        self.env = env
        self.dt = dt
        self.speed = speed
        self.noise_std = noise_std

    def generate(self, start, goal_label, n_steps=300):
        """Generate a noisy trajectory from start toward goal."""
        goal = self.env.goals[goal_label]
        pos = np.array(start, dtype=float)
        trajectory = [pos.copy()]
        head_directions = []

        for _ in range(n_steps):
            to_goal = goal - pos
            dist = np.linalg.norm(to_goal)
            if dist < 0.03:
                break
            direction = to_goal / dist
            # Add navigational noise
            noise_angle = np.random.randn() * self.noise_std
            cos_a, sin_a = np.cos(noise_angle), np.sin(noise_angle)
            noisy_dir = np.array([
                cos_a * direction[0] - sin_a * direction[1],
                sin_a * direction[0] + cos_a * direction[1]
            ])
            hd = np.arctan2(noisy_dir[1], noisy_dir[0])
            head_directions.append(hd)
            pos = pos + noisy_dir * self.speed * self.dt
            pos = np.clip(pos, 0, self.env.size)
            trajectory.append(pos.copy())

        return np.array(trajectory), np.array(head_directions), goal_label


# ============================================================================
# Entorhinal Cortex — Grid Cells
# ============================================================================

class EntorhinalCortex:
    """
    Grid cell module.

    Grid cells produce periodic spatial firing. During locomotion they sweep
    left-right (alternating offset). During goal pursuit they lock firing
    direction toward the goal.

    The grid pattern is modulated by goal-direction tuning from upstream
    head-direction cells.
    """

    def __init__(self, n_modules=3, cells_per_module=6, env_size=1.0):
        self.n_modules = n_modules
        self.cells_per_module = cells_per_module
        self.n_cells = n_modules * cells_per_module
        self.env_size = env_size

        # Grid spacings increase across modules (dorsoventral gradient)
        self.spacings = np.array([0.15, 0.25, 0.40])
        # Grid orientations per module
        self.orientations = np.array([7.5, 15.0, 22.5]) * np.pi / 180
        # Phase offsets per cell within module
        self.phases = []
        for m in range(n_modules):
            for c in range(cells_per_module):
                angle = 2 * np.pi * c / cells_per_module
                self.phases.append(np.array([np.cos(angle), np.sin(angle)])
                                   * self.spacings[m] * 0.3)

        # Goal-direction modulation gains (learned via head-direction input)
        # Shape: (n_cells, 2) — one gain per goal context
        self.goal_modulation = np.random.randn(self.n_cells, 2) * 0.3

        # Feedback weights from subiculum (initialized later)
        self.feedback_weights = None

    def grid_response(self, pos, cell_idx):
        """Compute raw grid cell firing at position."""
        m = cell_idx // self.cells_per_module
        spacing = self.spacings[m]
        orientation = self.orientations[m]
        phase = self.phases[cell_idx]

        # Three plane waves at 60° intervals → hexagonal pattern
        cos_o, sin_o = np.cos(orientation), np.sin(orientation)
        rot = np.array([[cos_o, -sin_o], [sin_o, cos_o]])
        shifted = pos - phase

        response = 0.0
        for k in range(3):
            angle = k * np.pi / 3
            wave_dir = rot @ np.array([np.cos(angle), np.sin(angle)])
            response += np.cos(2 * np.pi * np.dot(wave_dir, shifted) / spacing)
        # Normalize to [0, 1]
        return np.clip((response / 3 + 1) / 2, 0, 1)

    def compute_activity(self, pos, head_direction, goal_label, goal_pos,
                         pursuit_strength=1.0):
        """
        Full grid cell activity with goal-direction modulation.

        During pursuit (high pursuit_strength): grid firing is modulated
        by alignment between head direction and goal direction.
        During locomotion (low pursuit_strength): left-right sweep modulation.
        """
        goal_idx = 0 if goal_label == 'A' else 1
        to_goal = goal_pos - pos
        goal_dist = np.linalg.norm(to_goal)
        goal_dir = to_goal / max(goal_dist, 1e-6)
        goal_angle = np.arctan2(goal_dir[1], goal_dir[0])

        activity = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            base = self.grid_response(pos, i)

            # Goal-direction modulation (from head-direction cells upstream)
            angle_diff = head_direction - goal_angle
            # Pursuit mode: modulate by alignment to goal
            pursuit_mod = 0.5 + 0.5 * np.cos(angle_diff)
            # Sweep mode: alternating left-right offset
            sweep_mod = 0.5 + 0.5 * np.cos(
                2 * head_direction + self.goal_modulation[i, goal_idx])

            modulation = (pursuit_strength * pursuit_mod +
                          (1 - pursuit_strength) * sweep_mod)

            # Grid cells are primarily spatial; goal modulation is subtle
            activity[i] = base * (0.75 + 0.25 * modulation)

        # Apply feedback from subiculum if available
        # Feedback pulls EC activity toward stable allocentric representation
        if self.feedback_weights is not None:
            feedback = self.feedback_signal
            fb_drive = self.feedback_weights @ feedback
            fb_drive = np.clip(fb_drive, 0, None)
            if fb_drive.max() > 0:
                fb_drive /= fb_drive.max()
            # Blend: feedback stabilizes grid representation
            activity = 0.7 * activity + 0.3 * fb_drive
            activity = np.clip(activity, 0, 1)

        return activity

    def set_feedback(self, weights, signal):
        self.feedback_weights = weights
        self.feedback_signal = signal


# ============================================================================
# Dentate Gyrus — Sparse Pattern Separation
# ============================================================================

class DentateGyrus:
    """
    Sparse coding layer implementing pattern separation.

    High-dimensional EC input is compressed and orthogonalized via:
    - Random sparse projection (mimicking mossy fiber divergence)
    - Winner-take-all competition (only top-k neurons active)
    - Strong inhibition (modeled as global threshold)

    This converts overlapping EC representations for goals A vs B
    into highly orthogonal sparse codes.
    """

    def __init__(self, n_ec, n_dg=200, sparsity=0.05):
        self.n_ec = n_ec
        self.n_dg = n_dg
        self.k = max(1, int(n_dg * sparsity))  # number of active neurons

        # Sparse random weights from EC → DG
        # Each DG cell samples from a small random subset of EC cells
        self.weights = np.zeros((n_dg, n_ec))
        for i in range(n_dg):
            fan_in = np.random.choice(n_ec, size=min(4, n_ec), replace=False)
            self.weights[i, fan_in] = np.random.exponential(1.0, size=len(fan_in))

    def compute_activity(self, ec_activity):
        """
        Compute DG activity: sparse, orthogonal codes via winner-take-all.
        """
        raw = self.weights @ ec_activity
        # Winner-take-all: only top-k survive
        threshold_idx = np.argsort(raw)[-self.k:]
        activity = np.zeros(self.n_dg)
        activity[threshold_idx] = raw[threshold_idx]
        # Normalize active units
        if activity.max() > 0:
            activity[threshold_idx] /= activity.max()
        return activity


# ============================================================================
# CA3 — Recurrent Attractor Network
# ============================================================================

class CA3:
    """
    Recurrent attractor network.

    Receives sparse DG input (mossy fibers, strong but sparse).
    Recurrent connections (associative weights) expand the sparse DG code
    into two orthogonal attractor states corresponding to goal A vs B.

    The recurrent dynamics implement pattern completion and create
    stable attractor basins — once pushed toward one goal's attractor,
    the network settles into a coherent state.
    """

    def __init__(self, n_dg, n_ca3=80, n_iterations=5, tau=0.3):
        self.n_dg = n_dg
        self.n_ca3 = n_ca3
        self.n_iterations = n_iterations
        self.tau = tau

        # DG → CA3 weights (mossy fibers: sparse, strong)
        self.ff_weights = np.random.randn(n_ca3, n_dg) * 0.3

        # Recurrent weights (initialized small, then sculpted by attractors)
        self.rec_weights = np.random.randn(n_ca3, n_ca3) * 0.01
        np.fill_diagonal(self.rec_weights, 0)  # no self-connections

        # Store attractor patterns for two goals
        self.attractors = {}

    def _sigmoid(self, x, gain=5.0, threshold=0.5):
        return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))

    def store_attractor(self, label, pattern):
        """Store a pattern as an attractor via Hebbian outer product."""
        self.attractors[label] = pattern
        # Hopfield-like storage
        centered = pattern - pattern.mean()
        self.rec_weights += 0.5 * np.outer(centered, centered) / self.n_ca3
        np.fill_diagonal(self.rec_weights, 0)

    def compute_activity(self, dg_activity):
        """
        Compute CA3 activity through recurrent settling.

        Feedforward input from DG seeds the state, then recurrent dynamics
        push it toward the nearest attractor.
        """
        ff_input = self.ff_weights @ dg_activity
        state = self._sigmoid(ff_input)

        # Recurrent settling
        for _ in range(self.n_iterations):
            recurrent = self.rec_weights @ state
            total = 0.6 * ff_input + 0.4 * recurrent
            state = self.tau * self._sigmoid(total) + (1 - self.tau) * state

        return state


# ============================================================================
# CA1 — Subjective Place Cells
# ============================================================================

class CA1:
    """
    Place cells with strong subjective (egocentric) spatial representation.

    CA1 receives from CA3 and transforms the attractor-state-tagged signal
    into diverse place fields that remap between goal contexts.

    Key property: same physical location can have completely different
    CA1 representations depending on which goal the animal pursues.
    This is the peak of allocentric → egocentric transformation.

    No recurrent connections (unlike CA3), so representations are shaped
    entirely by input weights and are highly diverse.

    Behavioral timescale synaptic plasticity (BTSP) allows rapid
    formation of new place fields.
    """

    def __init__(self, n_ca3, n_ca1=120):
        self.n_ca3 = n_ca3
        self.n_ca1 = n_ca1

        # CA3 → CA1 (Schaffer collaterals)
        self.weights = np.random.randn(n_ca1, n_ca3) * 0.5

        # Place field centers in 2D space (for each cell)
        self.place_centers = np.random.rand(n_ca1, 2)
        self.place_widths = np.random.uniform(0.08, 0.2, n_ca1)

        # Goal selectivity: each cell has different affinity for goal contexts
        # Drawn from bimodal distribution → many cells are strongly selective
        selectivity = np.random.beta(0.3, 0.3, n_ca1)  # strongly bimodal
        self.goal_selectivity = 2 * selectivity - 1  # range [-1, 1]

        # BTSP-like plasticity rate
        self.btsp_rate = 0.05

    def compute_activity(self, ca3_activity, pos, goal_label):
        """
        Compute CA1 place cell activity.

        Combines spatial tuning (place fields) with goal-context modulation
        (from CA3 attractor state) to produce subjective representations.
        """
        # Spatial component: Gaussian place fields
        dists = np.linalg.norm(self.place_centers - pos, axis=1)
        spatial = np.exp(-0.5 * (dists / self.place_widths) ** 2)

        # CA3 input component
        ca3_drive = self.weights @ ca3_activity
        ca3_drive = np.clip(ca3_drive, 0, None)
        if ca3_drive.max() > 0:
            ca3_drive /= ca3_drive.max()

        # Goal-context modulation (the key subjective component)
        goal_sign = 1.0 if goal_label == 'A' else -1.0
        context_mod = 0.5 + 0.5 * self.goal_selectivity * goal_sign

        # Combined activity — context modulation dominates in CA1
        activity = spatial * (0.3 * ca3_drive + 0.7 * context_mod)

        # Threshold and normalize
        activity = np.clip(activity, 0, None)
        if activity.max() > 0:
            activity /= activity.max()

        return activity

    def btsp_update(self, ca3_activity, pos, active_mask):
        """
        Behavioral timescale synaptic plasticity.
        Rapidly shifts place field centers toward current position
        for cells that are active.
        """
        for i in np.where(active_mask)[0]:
            self.place_centers[i] += self.btsp_rate * (pos - self.place_centers[i])


# ============================================================================
# Subiculum — Stable Allocentric Filter
# ============================================================================

class Subiculum:
    """
    Stable place cells that filter egocentric CA1 signal back toward
    allocentric coordinates.

    Key properties:
    - Place fields rarely remap across goal contexts
    - Strong cross-day stability
    - No behavioral timescale synaptic plasticity (BTSP)
    - Acts as high-threshold filter: only passes consistent spatial
      information, stripping away subjective/goal-dependent modulation

    This converts egocentric CA1 representations back toward allocentric
    coordinates before feeding back to EC.
    """

    def __init__(self, n_ca1, n_sub=60):
        self.n_ca1 = n_ca1
        self.n_sub = n_sub

        # CA1 → Subiculum weights (fixed, no BTSP)
        self.weights = np.random.randn(n_sub, n_ca1) * 0.3

        # Stable place field centers (do NOT remap)
        self.place_centers = np.random.rand(n_sub, 2)
        self.place_widths = np.random.uniform(0.12, 0.25, n_sub)

        # High firing threshold → only strong, consistent signals pass
        self.threshold = 0.5

    def compute_activity(self, ca1_activity, pos):
        """
        Compute subiculum activity.

        Combines stable spatial tuning with thresholded CA1 input.
        The threshold strips away weak, context-dependent signals,
        preserving only spatially consistent information.
        """
        # Stable spatial component (context-independent)
        dists = np.linalg.norm(self.place_centers - pos, axis=1)
        spatial = np.exp(-0.5 * (dists / self.place_widths) ** 2)

        # CA1 input (heavily thresholded — no BTSP means only strong,
        # spatially consistent signals survive)
        ca1_drive = self.weights @ ca1_activity
        ca1_drive = np.clip(ca1_drive, 0, None)
        if ca1_drive.max() > 0:
            ca1_drive /= ca1_drive.max()
        ca1_drive[ca1_drive < self.threshold] = 0

        # Subiculum activity: heavily dominated by stable spatial fields,
        # minimal CA1 influence (the key stability mechanism)
        activity = 0.85 * spatial + 0.15 * ca1_drive
        activity = np.clip(activity, 0, 1)

        return activity


# ============================================================================
# Full Circuit
# ============================================================================

class GridPlaceWorldModel:
    """
    Complete EC → DG → CA3 → CA1 → Sub → EC circuit.

    Allocentric → Egocentric → Allocentric loop.
    """

    def __init__(self):
        self.env = Environment()
        self.traj_gen = TrajectoryGenerator(self.env)

        # Instantiate circuit components
        self.ec = EntorhinalCortex()
        self.dg = DentateGyrus(n_ec=self.ec.n_cells)
        self.ca3 = CA3(n_dg=self.dg.n_dg)
        self.ca1 = CA1(n_ca3=self.ca3.n_ca3)
        self.sub = Subiculum(n_ca1=self.ca1.n_ca1)

        # Sub → EC feedback weights
        self.sub_to_ec = np.random.randn(self.ec.n_cells, self.sub.n_sub) * 0.2

        # Initialize CA3 attractors for the two goal contexts
        self._init_attractors()

    def _init_attractors(self):
        """Create two orthogonal attractor patterns in CA3 for goals A and B."""
        n = self.ca3.n_ca3
        # Create orthogonal random patterns
        pattern_a = np.random.rand(n)
        pattern_a = (pattern_a > 0.5).astype(float)
        # Pattern B is roughly orthogonal
        pattern_b = 1 - pattern_a
        # Add some noise to make it biologically realistic
        pattern_a += np.random.randn(n) * 0.1
        pattern_b += np.random.randn(n) * 0.1
        pattern_a = np.clip(pattern_a, 0, 1)
        pattern_b = np.clip(pattern_b, 0, 1)

        self.ca3.store_attractor('A', pattern_a)
        self.ca3.store_attractor('B', pattern_b)

    def run_step(self, pos, head_direction, goal_label):
        """
        Run one time step through the full circuit.
        Returns activity of each region.
        """
        goal_pos = self.env.goals[goal_label]
        dist_to_goal = np.linalg.norm(goal_pos - pos)
        # Pursuit strength increases as animal approaches goal
        pursuit_strength = np.clip(1.0 - dist_to_goal / 0.8, 0.2, 1.0)

        # Forward pass: EC → DG → CA3 → CA1 → Sub
        ec_act = self.ec.compute_activity(
            pos, head_direction, goal_label, goal_pos, pursuit_strength)
        dg_act = self.dg.compute_activity(ec_act)
        ca3_act = self.ca3.compute_activity(dg_act)
        ca1_act = self.ca1.compute_activity(ca3_act, pos, goal_label)
        sub_act = self.sub.compute_activity(ca1_act, pos)

        # Feedback: Sub → EC
        self.ec.set_feedback(self.sub_to_ec, sub_act)

        # BTSP in CA1 (only for sufficiently active cells)
        active = ca1_act > 0.5
        self.ca1.btsp_update(ca3_act, pos, active)

        return {
            'ec': ec_act, 'dg': dg_act, 'ca3': ca3_act,
            'ca1': ca1_act, 'sub': sub_act,
            'pursuit_strength': pursuit_strength
        }

    def run_trial(self, goal_label, start=None):
        """Run a full navigation trial to the specified goal."""
        if start is None:
            start = (0.5, 0.1)
        trajectory, head_dirs, _ = self.traj_gen.generate(start, goal_label)

        all_activities = []
        for t in range(len(head_dirs)):
            act = self.run_step(trajectory[t], head_dirs[t], goal_label)
            act['pos'] = trajectory[t]
            act['hd'] = head_dirs[t]
            all_activities.append(act)

        return trajectory, all_activities

    def compute_rate_maps(self, n_trials=15):
        """
        Compute spatial rate maps for all regions under both goal conditions.

        Uses uniform spatial sampling to isolate contextual modulation from
        trajectory-sampling bias. At each position, the animal is simulated
        as heading toward the current goal, and activity is computed.
        """
        nbins = self.env.n_spatial_bins
        regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
        sizes = {
            'ec': self.ec.n_cells, 'dg': self.dg.n_dg,
            'ca3': self.ca3.n_ca3, 'ca1': self.ca1.n_ca1,
            'sub': self.sub.n_sub
        }

        rate_maps = {}
        for goal in ['A', 'B']:
            rate_maps[goal] = {}
            for region in regions:
                rate_maps[goal][region] = np.zeros((sizes[region], nbins, nbins))

            goal_pos = self.env.goals[goal]

            # Sample positions uniformly, compute activity at each
            for bx in range(nbins):
                for by in range(nbins):
                    pos = np.array([(bx + 0.5) / nbins * self.env.size,
                                    (by + 0.5) / nbins * self.env.size])
                    # Head direction toward goal
                    to_goal = goal_pos - pos
                    hd = np.arctan2(to_goal[1], to_goal[0])
                    act = self.run_step(pos, hd, goal)
                    for region in regions:
                        rate_maps[goal][region][:, by, bx] = act[region]

        # Also run trajectory-based maps for visualization
        self._traj_rate_maps = {}
        for goal in ['A', 'B']:
            self._traj_rate_maps[goal] = {}
            occupancy = np.zeros((nbins, nbins))
            for region in regions:
                self._traj_rate_maps[goal][region] = np.zeros(
                    (sizes[region], nbins, nbins))
            for trial in range(n_trials):
                start = (np.random.uniform(0.1, 0.9),
                         np.random.uniform(0.05, 0.2))
                _, activities = self.run_trial(goal, start=start)
                for a in activities:
                    bx_, by_ = self.env.discretize(a['pos'])
                    occupancy[by_, bx_] += 1
                    for region in regions:
                        self._traj_rate_maps[goal][region][:, by_, bx_] += \
                            a[region]
            occupancy[occupancy == 0] = 1
            for region in regions:
                for i in range(sizes[region]):
                    self._traj_rate_maps[goal][region][i] /= occupancy

        return rate_maps


# ============================================================================
# Analysis & Metrics
# ============================================================================

def compute_remapping_index(maps_a, maps_b):
    """
    Compute remapping index between goal A and goal B rate maps.
    RI = 1 - correlation. High RI → strong subjective remapping.
    """
    n_cells = maps_a.shape[0]
    ri = np.zeros(n_cells)
    for i in range(n_cells):
        a = maps_a[i].flatten()
        b = maps_b[i].flatten()
        if a.std() > 1e-8 and b.std() > 1e-8:
            ri[i] = 1 - np.corrcoef(a, b)[0, 1]
        else:
            ri[i] = 0
    return ri


def compute_sparsity(activity_history, region):
    """Compute population sparsity for a region across time steps."""
    acts = np.array([a[region] for a in activity_history])
    # Fraction of cells active (>10% of max) at each time step
    sparsities = []
    for t in range(len(acts)):
        if acts[t].max() > 0:
            frac = np.mean(acts[t] > 0.1 * acts[t].max())
            sparsities.append(frac)
    return np.mean(sparsities) if sparsities else 0


# ============================================================================
# Visualization
# ============================================================================

def plot_full_analysis(model, rate_maps):
    """Generate comprehensive visualization of the circuit."""
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(6, 6, figure=fig, hspace=0.4, wspace=0.4)

    fig.suptitle(
        'Entorhinal–Hippocampal World Model\n'
        'Allocentric ↔ Egocentric Transformation Circuit',
        fontsize=16, fontweight='bold', y=0.98)

    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    region_names = {
        'ec': 'Entorhinal Cortex\n(Grid Cells)',
        'dg': 'Dentate Gyrus\n(Pattern Separation)',
        'ca3': 'CA3\n(Attractor Dynamics)',
        'ca1': 'CA1\n(Subjective Place Cells)',
        'sub': 'Subiculum\n(Stable Filter)'
    }

    # --- Row 0: Example rate maps for Goal A ---
    for col, region in enumerate(regions):
        ax = fig.add_subplot(gs[0, col])
        # Pick cell with highest peak rate
        peak_rates = rate_maps['A'][region].max(axis=(1, 2))
        best = np.argmax(peak_rates)
        rm = gaussian_filter(rate_maps['A'][region][best], sigma=1)
        im = ax.imshow(rm, origin='lower', cmap='hot', aspect='equal')
        ax.set_title(f'{region_names[region]}\nGoal A, cell {best}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- Row 1: Same cells, Goal B ---
    for col, region in enumerate(regions):
        ax = fig.add_subplot(gs[1, col])
        peak_rates = rate_maps['A'][region].max(axis=(1, 2))
        best = np.argmax(peak_rates)
        rm = gaussian_filter(rate_maps['B'][region][best], sigma=1)
        im = ax.imshow(rm, origin='lower', cmap='hot', aspect='equal')
        ax.set_title(f'Goal B, cell {best}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- Row 2: Remapping Index Distribution ---
    for col, region in enumerate(regions):
        ax = fig.add_subplot(gs[2, col])
        ri = compute_remapping_index(
            rate_maps['A'][region], rate_maps['B'][region])
        ax.hist(ri, bins=20, range=(0, 2), color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.axvline(np.median(ri), color='red', linestyle='--',
                   label=f'median={np.median(ri):.2f}')
        ax.set_title(f'Remapping Index\n{region.upper()}', fontsize=9)
        ax.set_xlabel('RI')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)

    # --- Row 3: Population sparsity comparison ---
    ax_sparsity = fig.add_subplot(gs[3, :3])
    _, acts_a = model.run_trial('A')
    _, acts_b = model.run_trial('B')
    sparsities_a = [compute_sparsity(acts_a, r) for r in regions]
    sparsities_b = [compute_sparsity(acts_b, r) for r in regions]
    x = np.arange(len(regions))
    w = 0.35
    ax_sparsity.bar(x - w/2, sparsities_a, w, label='Goal A',
                    color='coral', edgecolor='white')
    ax_sparsity.bar(x + w/2, sparsities_b, w, label='Goal B',
                    color='steelblue', edgecolor='white')
    ax_sparsity.set_xticks(x)
    ax_sparsity.set_xticklabels([r.upper() for r in regions])
    ax_sparsity.set_ylabel('Population Sparsity\n(fraction active)')
    ax_sparsity.set_title('Population Sparsity Across Regions', fontsize=10)
    ax_sparsity.legend()

    # --- Row 3 right: Attractor state similarity ---
    ax_attr = fig.add_subplot(gs[3, 3:])
    ca3_acts_a = np.array([a['ca3'] for a in acts_a])
    ca3_acts_b = np.array([a['ca3'] for a in acts_b])
    # Mean CA3 state for each goal
    mean_a = ca3_acts_a.mean(axis=0)
    mean_b = ca3_acts_b.mean(axis=0)
    corr = np.corrcoef(mean_a, mean_b)[0, 1]
    ax_attr.bar(['Goal A\nself-corr', 'A-B\ncross-corr', 'Goal B\nself-corr'],
                [1.0, corr, 1.0],
                color=['coral', 'gray', 'steelblue'], edgecolor='white')
    ax_attr.set_ylabel('Correlation')
    ax_attr.set_title(f'CA3 Attractor Orthogonality\n(cross-corr={corr:.3f})',
                      fontsize=10)
    ax_attr.set_ylim(-0.5, 1.1)

    # --- Row 4: Trajectory visualization ---
    for goal_idx, goal_label in enumerate(['A', 'B']):
        ax = fig.add_subplot(gs[4, goal_idx * 3:(goal_idx + 1) * 3])
        for trial in range(5):
            start = (np.random.uniform(0.1, 0.9), np.random.uniform(0.05, 0.2))
            traj, _ = model.run_trial(goal_label, start=start)
            ax.plot(traj[:, 0], traj[:, 1], alpha=0.5, linewidth=0.8)
        for label, pos in model.env.goals.items():
            marker = '*' if label == goal_label else 'o'
            color = 'red' if label == goal_label else 'gray'
            ax.plot(pos[0], pos[1], marker, markersize=15, color=color,
                    label=f'Goal {label}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'Trajectories to Goal {goal_label}', fontsize=10)
        ax.legend(fontsize=8)

    # --- Row 5: Circuit diagram (text-based) ---
    ax_circ = fig.add_subplot(gs[5, :])
    ax_circ.axis('off')
    circuit_text = (
        "CIRCUIT ARCHITECTURE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "ALLOCENTRIC → EGOCENTRIC (Forward Path):\n"
        "  EC (grid cells, goal-direction modulated)  →  "
        "DG (sparse coding, pattern separation)  →  "
        "CA3 (recurrent attractors, orthogonal states)  →  "
        "CA1 (subjective place cells, BTSP)\n\n"
        "EGOCENTRIC → ALLOCENTRIC (Feedback Path):\n"
        "  CA1  →  Subiculum (stable filter, no BTSP, high threshold)  →  "
        "EC (stabilized grid representation)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• EC grid cells sweep L-R during locomotion, lock to goal during pursuit\n"
        "• DG orthogonalizes overlapping EC representations via sparse winner-take-all\n"
        "• CA3 recurrent connections create stable attractor basins per goal context\n"
        "• CA1 lacks recurrence → diverse, subjective place fields that remap across goals\n"
        "• Subiculum: no BTSP → cross-day stable → filters egocentric back to allocentric\n"
        "• EC feedback closes the loop: maintains stable spatial reference frame"
    )
    ax_circ.text(0.5, 0.5, circuit_text, fontsize=9, fontfamily='monospace',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9))

    plt.savefig('/home/user/world-model/circuit_analysis.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: circuit_analysis.png")


def plot_single_cell_remapping(model, rate_maps):
    """
    Show individual CA1 cells that remap strongly between goals,
    contrasted with subiculum cells that remain stable.
    """
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    fig.suptitle(
        'Single-Cell Remapping: CA1 (subjective) vs Subiculum (stable)',
        fontsize=14, fontweight='bold')

    # Find CA1 cells with highest remapping
    ri_ca1 = compute_remapping_index(
        rate_maps['A']['ca1'], rate_maps['B']['ca1'])
    top_ca1 = np.argsort(ri_ca1)[-6:]

    # Find subiculum cells with lowest remapping
    ri_sub = compute_remapping_index(
        rate_maps['A']['sub'], rate_maps['B']['sub'])
    stable_sub = np.argsort(ri_sub)[:6]

    for col in range(6):
        # CA1 Goal A
        rm = gaussian_filter(rate_maps['A']['ca1'][top_ca1[col]], sigma=1)
        axes[0, col].imshow(rm, origin='lower', cmap='hot')
        axes[0, col].set_title(f'CA1 #{top_ca1[col]}\nGoal A', fontsize=8)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # CA1 Goal B
        rm = gaussian_filter(rate_maps['B']['ca1'][top_ca1[col]], sigma=1)
        axes[1, col].imshow(rm, origin='lower', cmap='hot')
        axes[1, col].set_title(f'Goal B (RI={ri_ca1[top_ca1[col]]:.2f})',
                               fontsize=8)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # Sub Goal A
        rm = gaussian_filter(rate_maps['A']['sub'][stable_sub[col]], sigma=1)
        axes[2, col].imshow(rm, origin='lower', cmap='hot')
        axes[2, col].set_title(f'Sub #{stable_sub[col]}\nGoal A', fontsize=8)
        axes[2, col].set_xticks([])
        axes[2, col].set_yticks([])

        # Sub Goal B
        rm = gaussian_filter(rate_maps['B']['sub'][stable_sub[col]], sigma=1)
        axes[3, col].imshow(rm, origin='lower', cmap='hot')
        axes[3, col].set_title(f'Goal B (RI={ri_sub[stable_sub[col]]:.2f})',
                               fontsize=8)
        axes[3, col].set_xticks([])
        axes[3, col].set_yticks([])

    axes[0, 0].set_ylabel('CA1\nGoal A', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('CA1\nGoal B', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('Sub\nGoal A', fontsize=10, fontweight='bold')
    axes[3, 0].set_ylabel('Sub\nGoal B', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/user/world-model/single_cell_remapping.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: single_cell_remapping.png")


def plot_information_flow(model):
    """
    Visualize how signal transforms through the circuit on a single trial.
    Show population vectors at each stage.
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(
        'Information Flow Through Circuit\n'
        'Top: Goal A | Bottom: Goal B | Same physical trajectory',
        fontsize=13, fontweight='bold')

    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    region_labels = ['EC\n(allocentric)', 'DG\n(sparse)', 'CA3\n(attractor)',
                     'CA1\n(egocentric)', 'Sub\n(re-allocentric)']

    start = (0.5, 0.1)
    np.random.seed(42)  # Same trajectory noise

    for row, goal in enumerate(['A', 'B']):
        np.random.seed(42)
        _, acts = model.run_trial(goal, start=start)
        # Collect population vectors over time
        for col, region in enumerate(regions):
            pop_matrix = np.array([a[region] for a in acts])
            axes[row, col].imshow(pop_matrix.T, aspect='auto', cmap='viridis',
                                  interpolation='nearest')
            if row == 0:
                axes[row, col].set_title(region_labels[col], fontsize=10)
            axes[row, col].set_ylabel(f'Goal {goal}\nNeurons' if col == 0
                                      else 'Neurons')
            if row == 1:
                axes[row, col].set_xlabel('Time steps')

    plt.tight_layout()
    plt.savefig('/home/user/world-model/information_flow.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: information_flow.png")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Entorhinal Grid Cell – Hippocampal Place Cell World Model")
    print("Allocentric ↔ Egocentric Transformation Circuit")
    print("=" * 70)

    np.random.seed(42)
    model = GridPlaceWorldModel()

    print("\n[1/4] Computing rate maps (multiple trials per goal)...")
    rate_maps = model.compute_rate_maps(n_trials=15)

    print("[2/4] Generating full circuit analysis...")
    plot_full_analysis(model, rate_maps)

    print("[3/4] Generating single-cell remapping comparison...")
    plot_single_cell_remapping(model, rate_maps)

    print("[4/4] Generating information flow visualization...")
    plot_information_flow(model)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    regions = ['ec', 'dg', 'ca3', 'ca1', 'sub']
    print(f"\n{'Region':<12} {'Median RI':<12} {'Mean RI':<12} "
          f"{'Std RI':<12} {'N cells':<10}")
    print("-" * 58)
    for region in regions:
        ri = compute_remapping_index(
            rate_maps['A'][region], rate_maps['B'][region])
        n = rate_maps['A'][region].shape[0]
        print(f"{region.upper():<12} {np.median(ri):<12.3f} "
              f"{np.mean(ri):<12.3f} {np.std(ri):<12.3f} {n:<10}")

    print("\nExpected pattern (from theory):")
    print("  EC:  Low-moderate RI  (allocentric, goal-direction modulated)")
    print("  DG:  Moderate RI      (sparse, orthogonalized)")
    print("  CA3: High RI          (orthogonal attractor states)")
    print("  CA1: Highest RI       (strong subjective remapping)")
    print("  Sub: Low RI           (stable, filtered back to allocentric)")

    print("\n" + "=" * 70)
    print("Output files:")
    print("  circuit_analysis.png       — Full circuit analysis")
    print("  single_cell_remapping.png  — CA1 vs Subiculum remapping")
    print("  information_flow.png       — Population activity through circuit")
    print("=" * 70)


if __name__ == '__main__':
    main()
