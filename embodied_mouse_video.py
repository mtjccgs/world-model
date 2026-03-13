#!/usr/bin/env python3
"""
Embodied Virtual Mouse — Real-Time Video Simulation
====================================================

Like Eon Systems' embodied Drosophila (Shiu et al. 2024, Nature),
this runs a continuous closed-loop simulation where:

    Sensory perception → Spiking neural circuit → Motor commands
         ↑                                            ↓
         └──── Physics body in 2D arena ◄─────────────┘

The video shows FOUR simultaneous views:

    ┌─────────────────────┬─────────────────────┐
    │   ARENA (top-down)  │  NEURAL ACTIVITY    │
    │   Mouse body +      │  Spike raster per   │
    │   trajectory +      │  region (live)       │
    │   heading arrow     │                     │
    ├─────────────────────┼─────────────────────┤
    │   MOUSE POV         │  WORLD MODEL        │
    │   Sensory input     │  Believed position  │
    │   (what the mouse   │  vs actual + goal   │
    │    perceives)       │  representation     │
    └─────────────────────┴─────────────────────┘

Each frame = 50ms of neural simulation (5 env steps × 20 neural
substeps = 100 LIF timesteps per frame).

Usage:
    python embodied_mouse_video.py
    # → produces embodied_mouse_video.gif (~30s animation)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle, Wedge
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from PIL import Image
import io
import time as clock

# Import the circuit from our main simulation
from embodied_mouse import (
    LIFNetwork, CircuitBuilder, Arena2D, VirtualMouse,
    BTSPPlasticity, MouseWorldModel,
    R, N_TOTAL, DT, DT_ENV, STEPS_PER_ENV,
    V_REST, V_THRESH, ARENA_SIZE, REWARD_RADIUS,
    THETA_FREQ, N_SPATIAL_BINS
)

# ============================================================================
# Video simulation parameters
# ============================================================================

T_SIM = 40.0            # seconds of simulated time
FPS = 12                 # video frames per second
ENV_STEPS_PER_FRAME = 5  # env steps between rendered frames
FRAME_DT = ENV_STEPS_PER_FRAME * DT_ENV / 1000  # seconds per frame

N_FRAMES = int(T_SIM / FRAME_DT)
N_ENV_TOTAL = int(T_SIM * 1000 / DT_ENV)

# Raster window
RASTER_WINDOW = 80  # frames of history to show in raster

# Reward cooldown
REWARD_COOLDOWN = 2.0  # seconds between reward triggers

# Task phases
PHASE_EXPLORE = 'explore'
PHASE_SEEK_A = 'seek_A'
PHASE_SEEK_B = 'seek_B'


# ============================================================================
# Enhanced mouse with task-driven behavior
# ============================================================================

class TaskMouse(VirtualMouse):
    """
    Mouse with task-dependent behavior:
    - Explore: random walk covering the arena
    - Seek: follow odor gradient toward active goal
    - At reward: pause briefly, then continue
    """

    def __init__(self, arena, rng=None):
        super().__init__(arena, start_pos=np.array([0.5, 0.25]), rng=rng)
        self.task_phase = PHASE_EXPLORE
        self.reward_timer = 0.0
        self.last_reward_time = -10.0
        self.rewards_collected = {'A': 0, 'B': 0}
        self.at_reward = False

    def update_task(self, t_sec, goal_label):
        """Update task phase based on time and rewards."""
        if t_sec < 8.0:
            self.task_phase = PHASE_EXPLORE
        elif t_sec < 25.0:
            self.task_phase = PHASE_SEEK_A
        else:
            self.task_phase = PHASE_SEEK_B

        # Determine active goal from task
        if self.task_phase == PHASE_SEEK_B:
            return 'B'
        elif self.task_phase == PHASE_SEEK_A:
            return 'A'
        return goal_label

    def move_with_task(self, motor_spikes, dt_sec, goal_label, t_sec):
        """Move with task-dependent behavior overlay."""
        self.at_reward = False

        # Check reward
        if (self.arena.check_reward(self.pos, goal_label) and
                t_sec - self.last_reward_time > REWARD_COOLDOWN):
            self.at_reward = True
            self.last_reward_time = t_sec
            self.rewards_collected[goal_label] = \
                self.rewards_collected.get(goal_label, 0) + 1
            self.reward_timer = 0.5  # pause 0.5s at reward

        # Pause at reward
        if self.reward_timer > 0:
            self.reward_timer -= dt_sec
            self.speed = 0.01
            return

        # Base motor drive
        left = motor_spikes[0:3].sum()
        right = motor_spikes[3:6].sum()
        forward = motor_spikes[6:9].sum()
        slow = motor_spikes[9:12].sum()

        # Task-dependent heading bias
        if self.task_phase in (PHASE_SEEK_A, PHASE_SEEK_B):
            # Follow odor gradient (sensory-driven goal seeking)
            conc, grad_dir = self.arena.odor_gradient(self.pos, goal_label)
            if conc > 0.01:
                goal_angle = np.arctan2(grad_dir[1], grad_dir[0])
                angle_diff = goal_angle - self.heading
                # Wrap to [-pi, pi]
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
                # Smooth turning toward goal
                turn_strength = 0.3 + 0.7 * conc  # stronger near goal
                self.heading += angle_diff * turn_strength * dt_sec * 8.0
            self.speed = 0.18 + 0.08 * conc
        else:
            # Exploration: random walk with motor modulation
            if left + right > 0:
                turn_rate = 2.0 * (right - left) / max(1, left + right)
                self.heading += turn_rate * dt_sec
            motor_drive = forward / max(1, forward + slow + 1)
            self.speed = 0.12 + 0.10 * motor_drive

        # Exploration noise
        self.heading += self.rng.randn() * 0.5 * np.sqrt(dt_sec)
        if self.rng.rand() < 0.015:
            self.heading += self.rng.uniform(-np.pi / 2, np.pi / 2)

        self.heading = self.heading % (2 * np.pi)
        self.speed = np.clip(self.speed + self.rng.randn() * 0.01,
                             0.03, self.max_speed)

        # Move
        dx = self.speed * np.cos(self.heading) * dt_sec
        dy = self.speed * np.sin(self.heading) * dt_sec
        new_pos = self.pos + np.array([dx, dy])

        # Wall bounce
        margin = 0.03
        for dim in [0, 1]:
            if new_pos[dim] < margin:
                new_pos[dim] = margin
                if dim == 0:
                    self.heading = np.pi - self.heading
                else:
                    self.heading = -self.heading
                self.heading += self.rng.randn() * 0.3
            elif new_pos[dim] > ARENA_SIZE - margin:
                new_pos[dim] = ARENA_SIZE - margin
                if dim == 0:
                    self.heading = np.pi - self.heading
                else:
                    self.heading = -self.heading
                self.heading += self.rng.randn() * 0.3
        self.pos = new_pos


# ============================================================================
# Video Renderer
# ============================================================================

class VideoRenderer:
    """Renders each frame of the simulation as a matplotlib figure."""

    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.set_facecolor('#1a1a2e')

        for ax in self.axes.flat:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#a0a0a0', labelsize=6)
            for spine in ax.spines.values():
                spine.set_color('#333')

        # Pre-allocate plot elements
        self._setup_arena(self.axes[0, 0])
        self._setup_raster(self.axes[0, 1])
        self._setup_sensory(self.axes[1, 0])
        self._setup_worldmodel(self.axes[1, 1])

        self.fig.tight_layout(pad=1.5)

        # Raster history buffer
        self.raster_history = {
            'grid': [], 'dg': [], 'ca3': [],
            'ca1': [], 'sub': [], 'motor': []
        }

    def _setup_arena(self, ax):
        """Top-down arena view."""
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title('Arena (top-down)', color='white', fontsize=9,
                      fontweight='bold')

        # Arena boundary
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], '-', color='#555', lw=2)

        # Wall cue markers
        for x, y, label, color in [
            (0.5, 1.02, 'N', '#e74c3c'), (1.02, 0.5, 'E', '#3498db'),
            (0.5, -0.03, 'S', '#2ecc71'), (-0.03, 0.5, 'W', '#f39c12')
        ]:
            ax.text(x, y, label, ha='center', va='center',
                    color=color, fontsize=7, fontweight='bold')

        self.ax_arena = ax

    def _setup_raster(self, ax):
        """Neural activity raster."""
        ax.set_title('Neural Activity (live)', color='white', fontsize=9,
                      fontweight='bold')
        ax.set_xlabel('Time (frames)', color='#a0a0a0', fontsize=7)
        self.ax_raster = ax

    def _setup_sensory(self, ax):
        """Mouse's sensory perspective."""
        ax.set_title("Mouse POV (sensory input)", color='white',
                      fontsize=9, fontweight='bold')
        self.ax_sensory = ax

    def _setup_worldmodel(self, ax):
        """World model visualization."""
        ax.set_title('World Model (internal state)', color='white',
                      fontsize=9, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        self.ax_wm = ax

    def render_frame(self, mouse, arena, spike_counts, believed_pos,
                     confidence, goal_label, t_sec, task_phase,
                     prox, odor, cue, grid_activity):
        """Render one frame. Returns PIL Image."""
        self._draw_arena(mouse, arena, goal_label, t_sec, task_phase)
        self._draw_raster(spike_counts, t_sec, goal_label)
        self._draw_sensory(prox, odor, cue, mouse, goal_label)
        self._draw_worldmodel(mouse, believed_pos, confidence,
                              goal_label, arena, grid_activity, t_sec)

        # Render to image
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', dpi=100,
                         facecolor=self.fig.get_facecolor(),
                         edgecolor='none')
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _draw_arena(self, mouse, arena, goal_label, t_sec, task_phase):
        ax = self.ax_arena
        # Remove old dynamic elements
        while len(ax.patches) > 0:
            ax.patches[-1].remove()
        # Remove old lines except boundary
        while len(ax.lines) > 1:
            ax.lines[-1].remove()
        # Remove old texts except wall labels
        while len(ax.texts) > 4:
            ax.texts[-1].remove()

        # Trajectory (last 200 positions)
        if hasattr(self, '_traj_history'):
            self._traj_history.append(mouse.pos.copy())
            if len(self._traj_history) > 300:
                self._traj_history = self._traj_history[-300:]
        else:
            self._traj_history = [mouse.pos.copy()]

        if len(self._traj_history) > 1:
            pts = np.array(self._traj_history)
            colors = np.linspace(0.1, 0.8, len(pts))
            for i in range(1, len(pts)):
                alpha = colors[i]
                ax.plot([pts[i - 1, 0], pts[i, 0]],
                        [pts[i - 1, 1], pts[i, 1]],
                        '-', color=(0.4, 0.8, 1.0, alpha), linewidth=0.5)

        # Reward locations
        for label, pos in arena.rewards.items():
            c = '#ff6b6b' if label == 'A' else '#4ecdc4'
            active = (goal_label == label)
            size = 0.04 if active else 0.025
            alpha = 0.9 if active else 0.3
            circle = Circle(pos, size, facecolor=c, edgecolor='white',
                            linewidth=1.5 if active else 0.5, alpha=alpha)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1] + 0.07, label, ha='center',
                    color=c, fontsize=8, fontweight='bold',
                    alpha=alpha)

        # Mouse body
        body = Circle(mouse.pos, 0.025, facecolor='#ffd93d',
                      edgecolor='white', linewidth=1.5, zorder=10)
        ax.add_patch(body)

        # Heading arrow
        hx = mouse.pos[0] + 0.05 * np.cos(mouse.heading)
        hy = mouse.pos[1] + 0.05 * np.sin(mouse.heading)
        ax.annotate('', xy=(hx, hy), xytext=mouse.pos,
                    arrowprops=dict(arrowstyle='->', color='#ffd93d',
                                    lw=2), zorder=11)

        # Whisker fan (field of view)
        fov_start = np.degrees(mouse.heading - np.pi / 3)
        fov_end = np.degrees(mouse.heading + np.pi / 3)
        fov = Wedge(mouse.pos, 0.12, fov_start, fov_end,
                     facecolor='#ffd93d', alpha=0.08, zorder=1)
        ax.add_patch(fov)

        # Status text
        phase_names = {
            PHASE_EXPLORE: 'EXPLORING',
            PHASE_SEEK_A: 'SEEKING GOAL A',
            PHASE_SEEK_B: 'SEEKING GOAL B',
        }
        phase_colors = {
            PHASE_EXPLORE: '#aaa',
            PHASE_SEEK_A: '#ff6b6b',
            PHASE_SEEK_B: '#4ecdc4',
        }
        ax.text(0.02, -0.03, f't={t_sec:.1f}s  {phase_names[task_phase]}',
                color=phase_colors[task_phase], fontsize=7,
                fontweight='bold')

        # Reward counter
        if hasattr(mouse, 'rewards_collected'):
            ax.text(0.98, -0.03,
                    f"A:{mouse.rewards_collected.get('A', 0)} "
                    f"B:{mouse.rewards_collected.get('B', 0)}",
                    color='#aaa', fontsize=7, ha='right')

        # Reward flash
        if hasattr(mouse, 'at_reward') and mouse.at_reward:
            flash = Circle(mouse.pos, 0.08, facecolor='#ffd93d',
                           alpha=0.4, zorder=0)
            ax.add_patch(flash)

    def _draw_raster(self, spike_counts, t_sec, goal_label):
        ax = self.ax_raster
        ax.clear()
        ax.set_facecolor('#16213e')

        regions = [
            ('Grid', R['grid'], '#4ecdc4', 'grid'),
            ('DG', R['dg_gc'], '#45b7d1', 'dg'),
            ('CA3', R['ca3_pyr'], '#96ceb4', 'ca3'),
            ('CA1', R['ca1_pyr'], '#ffeaa7', 'ca1'),
            ('Sub', R['sub'], '#dda0dd', 'sub'),
            ('Mtr', R['motor'], '#ff6b6b', 'motor'),
        ]

        # Accumulate history
        for name, region, _, key in regions:
            rate = spike_counts[region].sum() / max(1, region.stop - region.start)
            self.raster_history[key].append(rate)
            if len(self.raster_history[key]) > RASTER_WINDOW:
                self.raster_history[key] = self.raster_history[key][-RASTER_WINDOW:]

        # Plot activity bars
        y_offset = 0
        yticks, ylabels = [], []
        for name, region, color, key in regions:
            hist = self.raster_history[key]
            if len(hist) > 1:
                x = np.arange(len(hist))
                vals = np.array(hist)
                vals_norm = vals / max(vals.max(), 1) * 3
                ax.barh(y_offset + np.zeros(len(x)), vals_norm,
                        left=x, height=0.8, color=color, alpha=0.7)
            yticks.append(y_offset)
            ylabels.append(name)
            y_offset += 4

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7, color='#ccc')
        ax.set_xlim(0, RASTER_WINDOW)
        ax.set_ylim(-2, y_offset + 2)
        ax.tick_params(colors='#666', labelsize=6)
        ax.set_title('Neural Activity (live)', color='white',
                      fontsize=9, fontweight='bold')

        # Goal indicator
        gc = '#ff6b6b' if goal_label == 'A' else '#4ecdc4'
        ax.text(RASTER_WINDOW * 0.98, y_offset,
                f'Goal {goal_label}', color=gc, fontsize=8,
                fontweight='bold', ha='right', va='top')

    def _draw_sensory(self, prox, odor, cue, mouse, goal_label):
        ax = self.ax_sensory
        ax.clear()
        ax.set_facecolor('#16213e')

        # Radar-style sensory display
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')

        # Mouse body (center)
        body = Circle((0, 0), 0.15, facecolor='#ffd93d',
                       edgecolor='white', linewidth=1, zorder=5)
        ax.add_patch(body)
        ax.text(0, 0, '🐁' if False else 'M', ha='center', va='center',
                fontsize=8, color='#333', fontweight='bold', zorder=6)

        # Whisker proximity (8 whiskers in front semicircle)
        n_whiskers = 8
        for i in range(n_whiskers):
            angle = -np.pi / 2 + np.pi * i / (n_whiskers - 1)
            strength = prox[i] / 18.0  # normalize
            r = 0.3 + 0.8 * strength
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            color_val = min(1, strength)
            ax.plot([0.15 * np.cos(angle), x],
                    [0.15 * np.sin(angle), y],
                    '-', color=(1, color_val, 0.2), linewidth=2,
                    alpha=0.5 + 0.5 * strength)
            if strength > 0.3:
                ax.plot(x, y, 'o', color='#ff0000',
                        markersize=3 + 5 * strength, alpha=0.6)

        # Odor gradient (4 directions)
        odor_dirs = [np.pi / 2, 0, -np.pi / 2, np.pi]  # front, right, back, left
        odor_labels = ['F', 'R', 'B', 'L']
        for i in range(4):
            strength = odor[i * 3] / 22.0  # normalize by max
            if strength > 0.01:
                angle = odor_dirs[i]
                r = 0.5 + 0.6 * strength
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                circle = Circle((x, y), 0.08 + 0.15 * strength,
                                facecolor='#2ecc71', alpha=0.3 * strength,
                                edgecolor='none')
                ax.add_patch(circle)
                ax.text(x, y, odor_labels[i], ha='center', va='center',
                        color='#2ecc71', fontsize=6, alpha=0.5 + 0.5 * strength)

        # Heading indicator
        ax.annotate('', xy=(0, 0.9), xytext=(0, 0.2),
                    arrowprops=dict(arrowstyle='->', color='white',
                                    lw=1.5, alpha=0.4))
        ax.text(0, 1.05, f'HD: {np.degrees(mouse.heading):.0f}°',
                ha='center', color='#aaa', fontsize=6)

        # Speed indicator
        speed_bar = mouse.speed / mouse.max_speed
        ax.barh(-1.3, speed_bar * 2, height=0.12, left=-1,
                color='#ffd93d', alpha=0.6)
        ax.text(-1, -1.15, f'Speed: {mouse.speed:.2f} m/s',
                color='#aaa', fontsize=6)

        ax.set_title("Mouse POV (sensory input)", color='white',
                      fontsize=9, fontweight='bold')
        ax.axis('off')

    def _draw_worldmodel(self, mouse, believed_pos, confidence,
                         goal_label, arena, grid_activity, t_sec):
        ax = self.ax_wm
        ax.clear()
        ax.set_facecolor('#16213e')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')

        # Arena outline
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], '-', color='#333', lw=1)

        # Grid cell activity heatmap (what the mouse "thinks")
        if grid_activity is not None and grid_activity.sum() > 0:
            nbins = 20
            belief_map = np.zeros((nbins, nbins))
            # Simple population vector decode visualization
            for iy in range(nbins):
                for ix in range(nbins):
                    dx = (ix + 0.5) / nbins - believed_pos[0]
                    dy = (iy + 0.5) / nbins - believed_pos[1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    belief_map[iy, ix] = confidence * np.exp(-dist ** 2 / (2 * 0.08 ** 2))
            ax.imshow(belief_map, origin='lower', extent=[0, 1, 0, 1],
                      cmap='inferno', alpha=0.5, vmin=0,
                      vmax=max(0.01, belief_map.max()))

        # Believed position (circle)
        bp = Circle(believed_pos, 0.03, facecolor='none',
                    edgecolor='#ff6b6b', linewidth=2, linestyle='--',
                    zorder=5, alpha=0.8)
        ax.add_patch(bp)
        ax.text(believed_pos[0], believed_pos[1] + 0.06, 'believed',
                ha='center', color='#ff6b6b', fontsize=5, alpha=0.8)

        # Actual position (circle)
        ap = Circle(mouse.pos, 0.025, facecolor='#ffd93d',
                    edgecolor='white', linewidth=1, zorder=6)
        ax.add_patch(ap)
        ax.text(mouse.pos[0], mouse.pos[1] - 0.06, 'actual',
                ha='center', color='#ffd93d', fontsize=5)

        # Error line
        ax.plot([mouse.pos[0], believed_pos[0]],
                [mouse.pos[1], believed_pos[1]],
                '--', color='#e74c3c', linewidth=0.8, alpha=0.5)

        # Goals
        for label, pos in arena.rewards.items():
            c = '#ff6b6b' if label == 'A' else '#4ecdc4'
            active = (goal_label == label)
            ax.plot(pos[0], pos[1], '*', color=c,
                    markersize=12 if active else 6,
                    alpha=0.8 if active else 0.3)

        # Info
        error = np.linalg.norm(mouse.pos - believed_pos)
        ax.text(0.02, -0.03,
                f'Error: {error:.3f}m  Conf: {confidence:.2f}',
                color='#aaa', fontsize=6)

        ax.set_title('World Model (internal state)', color='white',
                      fontsize=9, fontweight='bold')
        ax.tick_params(colors='#666', labelsize=5)


# ============================================================================
# Main simulation loop
# ============================================================================

def run_video_simulation():
    print("=" * 65)
    print("  Embodied Virtual Mouse — Real-Time Video Simulation")
    print("  Closed-loop: sensory → 1098 LIF neurons → motor → body")
    print("=" * 65)
    print()

    rng = np.random.RandomState(42)

    # Build circuit
    arena = Arena2D()
    mouse = TaskMouse(arena, rng=rng)
    net = LIFNetwork(N_TOTAL, seed=42)
    builder = CircuitBuilder(net, rng)
    builder.build_all()

    # BTSP plasticity
    n_ec = R['grid'].stop - R['grid'].start
    n_ca1 = R['ca1_pyr'].stop - R['ca1_pyr'].start
    btsp = BTSPPlasticity(net, n_ec, n_ca1, rng)

    # World model
    world_model = MouseWorldModel(builder, ARENA_SIZE)

    # Renderer
    renderer = VideoRenderer()

    # Goal tracking
    current_goal = 'A'

    # Helper: compute external currents
    def compute_I_ext(pos, heading, velocity, prox, odor, cue, t_ms, goal):
        I = np.zeros(N_TOTAL)

        # Grid cells
        for mod in builder.grid_modules:
            start = mod['slice'].start - R['grid'].start
            n = mod['n']
            shifted = pos[np.newaxis, :] - mod['phases']
            rates = np.zeros(n)
            for k in range(3):
                theta = mod['orientation'] + k * np.pi / 3
                wave = np.array([np.cos(theta), np.sin(theta)])
                proj = np.sum(wave * shifted, axis=-1)
                rates += np.cos(2 * np.pi * proj / mod['spacing'])
            rates = np.clip((rates / 3 + 1) / 2, 0, 1)
            vel_mag = np.linalg.norm(velocity)
            vel_gain = 1.0 + 2.0 * vel_mag
            I[R['grid'].start + start:R['grid'].start + start + n] = \
                rates * vel_gain * 22.0

        # Theta
        phase = 2 * np.pi * THETA_FREQ * t_ms / 1000
        I[R['theta']] = 12.0 * (0.5 + 0.5 * np.sin(phase))

        # Head direction
        n_hd = R['hd'].stop - R['hd'].start
        pref_dirs = np.linspace(0, 2 * np.pi, n_hd, endpoint=False)
        hd_drive = np.exp(3.0 * np.cos(pref_dirs - heading))
        I[R['hd']] = hd_drive / hd_drive.max() * 14.0

        # Sensory
        I[R['sens_prox']] = prox * 6.0 + rng.randn(16) * 1.0
        I[R['sens_odor']] = odor * 6.0 + rng.randn(12) * 1.0
        I[R['sens_cue']] = cue * 6.0 + rng.randn(8) * 1.0

        # Context
        n_ctx = R['ctx'].stop - R['ctx'].start
        if goal == 'A':
            I[R['ctx'].start:R['ctx'].start + n_ctx // 2] = 14.0
        else:
            I[R['ctx'].start + n_ctx // 2:R['ctx'].stop] = 14.0

        return I

    # ── Main loop ──
    frames = []
    t_ms = 0.0
    frame_count = 0
    env_step_in_frame = 0

    # Accumulate spikes per frame
    frame_spike_counts = np.zeros(N_TOTAL, dtype=int)

    t_wall_start = clock.time()

    print(f"Simulating {T_SIM}s → {N_FRAMES} frames at {FPS} fps...")
    print(f"  {N_ENV_TOTAL} env steps, {N_ENV_TOTAL * STEPS_PER_ENV:,} "
          f"neural timesteps total")
    print()

    for env_step in range(N_ENV_TOTAL):
        t_sec = env_step * DT_ENV / 1000

        # Update task and goal
        current_goal = mouse.update_task(t_sec, current_goal)

        # Sensory input
        prox, odor, cue = mouse.sense(current_goal)
        velocity = mouse.get_velocity()

        # Compute external currents
        I_ext = compute_I_ext(mouse.pos, mouse.heading, velocity,
                              prox, odor, cue, t_ms, current_goal)

        # Run neural substeps
        env_spikes = np.zeros(N_TOTAL, dtype=int)
        for _ in range(STEPS_PER_ENV):
            spikes = net.step(I_ext)
            env_spikes += spikes.astype(int)
            btsp.update_traces(spikes)
            btsp.apply_plasticity()
            t_ms += DT

        frame_spike_counts += env_spikes

        # Motor output → movement
        motor_spikes = env_spikes[R['motor']]
        mouse.move_with_task(motor_spikes, DT_ENV / 1000,
                             current_goal, t_sec)

        # Check reward → BTSP
        if mouse.at_reward:
            btsp.trigger_plateau(current_goal)
            print(f"  [{t_sec:.1f}s] ★ Reward {current_goal} at "
                  f"({mouse.pos[0]:.2f}, {mouse.pos[1]:.2f})")

        env_step_in_frame += 1

        # ── Render frame? ──
        if env_step_in_frame >= ENV_STEPS_PER_FRAME:
            # Decode world model
            grid_sc = frame_spike_counts[R['grid']]
            believed_pos, confidence = world_model.decode_position(grid_sc)

            # Render
            img = renderer.render_frame(
                mouse, arena, frame_spike_counts,
                believed_pos, confidence, current_goal, t_sec,
                mouse.task_phase, prox, odor, cue, grid_sc)
            frames.append(img)

            frame_count += 1
            env_step_in_frame = 0
            frame_spike_counts[:] = 0

            # Progress
            if frame_count % 50 == 0:
                elapsed = clock.time() - t_wall_start
                pct = frame_count / N_FRAMES * 100
                fps_actual = frame_count / elapsed
                print(f"  [{t_sec:.1f}s] Frame {frame_count}/{N_FRAMES} "
                      f"({pct:.0f}%, {fps_actual:.1f} frames/s wall)")

    elapsed = clock.time() - t_wall_start
    print(f"\nSimulation done: {elapsed:.1f}s wall time, "
          f"{frame_count} frames rendered")

    # ── Save GIF ──
    print(f"\nSaving GIF ({frame_count} frames at {FPS} fps)...")
    gif_path = '/home/user/world-model/embodied_mouse_video.gif'

    # Optimize: reduce to reasonable size
    resized_frames = []
    for img in frames:
        # Resize to 800x640 for reasonable file size
        resized = img.resize((800, 640), Image.LANCZOS)
        # Convert to palette mode for smaller GIF
        resized = resized.quantize(colors=128, method=2)
        resized_frames.append(resized)

    resized_frames[0].save(
        gif_path,
        save_all=True,
        append_images=resized_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )

    file_size = len(open(gif_path, 'rb').read()) / (1024 * 1024)
    print(f"Saved: {gif_path} ({file_size:.1f} MB, "
          f"{frame_count} frames, {frame_count/FPS:.1f}s)")

    plt.close('all')

    print()
    print("=" * 65)
    print("  Closed-loop architecture (like Eon's embodied Drosophila):")
    print()
    print("  Sensory neurons → EC grid cells → DG → CA3 → CA1 → Sub")
    print("       ↑               (position)  (sparse) (attractor)")
    print("       │                                        ↓")
    print("  Arena physics ◄── Motor neurons ◄── CA1 + HD + Sensory")
    print("       │                                        ↓")
    print("       └── Wall/odor/cue signals ──► Sensory neurons")
    print()
    print("  BTSP plasticity at CA1 creates new place fields on")
    print("  single reward encounters (behavioral timescale ~1.5s)")
    print("=" * 65)


if __name__ == '__main__':
    run_video_simulation()
