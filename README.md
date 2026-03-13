# Entorhinal–Hippocampal World Model

A biologically grounded computational model of the entorhinal cortex → hippocampus → entorhinal cortex loop, demonstrating how the brain maintains a robust allocentric spatial scaffold while simultaneously generating flexible, goal-dependent (egocentric/subjective) representations.

## Theoretical Framework

The model implements the **allocentric ↔ egocentric transformation loop**:

```
Forward (allocentric → egocentric):
  EC_II → DG → CA3 → CA1

Feedback (egocentric → allocentric):
  CA1 → Subiculum → EC_deep → EC_II
```

### Two-Goal Paradigm

Goals A=(0.75, 0.75) and B=(0.25, 0.75) are placed at **mirror-symmetric** positions about the midline (x=0.5). The animal navigates to one goal per trial.

### Circuit Components

| Region | N | Mechanism | Role |
|--------|---|-----------|------|
| **HD System** | 60 | Ring attractor with firing rate adaptation (Ji et al. 2025) | Generates L-R alternating theta sweeps |
| **EC (Grid)** | 32 | Hexagonal spatial code + theta sweeps (Vollan et al. 2025) | Allocentric scaffold with goal-direction gain modulation |
| **DG** | 500 | Sparse winner-take-all (2% active) | Pattern separation: orthogonalizes overlapping EC patterns |
| **CA3** | 150 | Recurrent attractor (Hopfield-like) + PFC context bias | Amplifies goal-context separation via orthogonal attractors |
| **CA1** | 200 | No recurrence, BTSP, bimodal goal selectivity | Peak egocentric: strong subjective remapping |
| **Subiculum** | 100 | Stable fields, no BTSP, high threshold | Filters egocentric → allocentric; cross-day stable |

### Key Biological Mechanisms

**Theta sweep dynamics (Vollan et al. 2025; Ji et al. 2025)**:
- Within each theta cycle (~125ms), the grid cell population representation sweeps **linearly outward** from the animal's position
- Direction alternates **±30° left/right** of heading across successive cycles
- Driven by HD cell ring attractor with firing rate adaptation
- Sweep length scales with grid module spacing (dorsoventral gradient)
- During goal pursuit: sweeps lock to goal direction (unpublished)

**Tri-synaptic amplification**:
- EC provides mild goal-direction modulation (via sweep + gain fields)
- DG orthogonalizes the overlapping EC signals (sparse WTA)
- CA3 recurrent connections create orthogonal attractor basins per goal
- CA1 (no recurrence, BTSP) amplifies into strong subjective place cells

**Subiculum stability**:
- No BTSP → cannot rapidly form context-dependent fields
- High firing threshold filters out weak, context-dependent CA1 signals
- Place fields rarely remap → cross-day stable (unpublished)
- Feeds back to EC, maintaining stable allocentric grid representation

## Running

```bash
pip install numpy matplotlib scipy
python grid_place_world_model.py
```

## Results

### Population Vector Correlation (key metric)

| Region | PV Corr | Interpretation |
|--------|---------|----------------|
| EC | 0.936 | Allocentric (stable grid with subtle sweep modulation) |
| DG | 0.841 | Pattern separation beginning |
| CA3 | 0.813 | Attractor dynamics separating goals |
| **CA1** | **0.046** | **Near-orthogonal: peak egocentric representation** |
| Sub | 0.986 | **Most stable: filtered back to allocentric** |

### Remapping Index

| Region | Median RI | 95% CI |
|--------|-----------|--------|
| EC | 0.120 | [0.106, 0.129] |
| DG | 0.364 | [0.340, 0.415] |
| CA3 | 0.352 | [0.342, 0.361] |
| CA1 | 0.289 | [0.201, 0.352] |
| Sub | 0.010 | [0.008, 0.012] |

CA1 has a bimodal RI distribution (many cells strongly selective, others stable), consistent with experimental observations.

## Output Figures

| Figure | Contents |
|--------|----------|
| `fig1_circuit_ratemaps.png` | Circuit schematic + example rate maps per region |
| `fig2_remapping_gradient.png` | RI distributions + gradient across circuit |
| `fig3_cell_contrast.png` | CA1 (remapping) vs Subiculum (stable) single cells |
| `fig4_population_analysis.png` | PV correlation, sparsity, dimensionality, CA3 attractors |
| `fig5_information_flow.png` | Population activity matrices through circuit |
| `fig6_trajectories.png` | Navigation trajectories to mirror-symmetric goals |
| `fig7_theta_sweeps.png` | Theta sweep dynamics: L-R alternation vs goal-locking |

## References

- Vollan, A.Z., Gardner, R.J., Moser, M.-B. & Moser, E.I. Left–right-alternating theta sweeps in entorhinal–hippocampal maps of space. *Nature* 639, 995–1005 (2025).
- Ji, Z., Chu, T., Wu, S. & Burgess, N. A systems model of alternating theta sweeps via firing rate adaptation. *Current Biology* 35, 709–722 (2025).
