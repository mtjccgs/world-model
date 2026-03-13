# Entorhinal Grid Cell – Hippocampal Place Cell World Model

A bioinspired computational model of the entorhinal–hippocampal circuit implementing the **allocentric ↔ egocentric transformation loop** for spatial cognition.

## Theoretical Framework

This model implements the hypothesis that the EC → hippocampus → EC loop transforms stable external (allocentric) spatial representations into flexible subjective (egocentric) representations, then reconstructs the allocentric frame via feedback.

### Circuit Architecture

```
ALLOCENTRIC → EGOCENTRIC (Forward Path):
  EC (grid cells) → DG (pattern separation) → CA3 (attractors) → CA1 (subjective place cells)

EGOCENTRIC → ALLOCENTRIC (Feedback Path):
  CA1 → Subiculum (stable filter) → EC (stabilized grid representation)
```

### Key Mechanisms

| Region | Mechanism | Role |
|--------|-----------|------|
| **Entorhinal Cortex** | Grid cells with goal-direction modulation from head-direction cells | Allocentric spatial scaffold; sweeps L-R during locomotion, locks to goal during pursuit |
| **Dentate Gyrus** | Sparse winner-take-all coding | Compresses and orthogonalizes overlapping EC representations |
| **CA3** | Recurrent attractor network (Hopfield-like) | Expands DG sparse codes into orthogonal attractor basins per goal context |
| **CA1** | Diverse place cells with BTSP, no recurrence | Peak egocentric representation; strong remapping between goal contexts |
| **Subiculum** | Stable place cells, no BTSP, high threshold | Filters egocentric signal back to allocentric; cross-day stable |

### Two-Goal Paradigm

An animal navigates to one of two alternative goals (A or B) in the same environment. The model demonstrates:

- **CA1 place cells** show strong subjective remapping: firing at goal A location when pursuing A, silent when pursuing B (and vice versa)
- **Subiculum place cells** maintain stable firing fields regardless of which goal is pursued
- The **remapping index gradient** EC(low) → DG → CA3 → CA1(highest) → Sub(low) confirms the allocentric→egocentric→allocentric transformation

## Running

```bash
pip install numpy matplotlib scipy
python grid_place_world_model.py
```

### Outputs

- `circuit_analysis.png` — Full circuit analysis: rate maps, remapping distributions, sparsity, attractor orthogonality, trajectories
- `single_cell_remapping.png` — CA1 (strongly remapping) vs Subiculum (stable) single-cell comparison
- `information_flow.png` — Population activity vectors through the circuit for Goal A vs Goal B

## Remapping Index Results

| Region | Median RI | Interpretation |
|--------|-----------|----------------|
| EC | 0.02 | Low — allocentric, stable grid representation |
| DG | 0.00 | Very low (sparse) — most cells silent, active ones orthogonalized |
| CA3 | 0.15 | Moderate — attractor dynamics beginning to separate |
| CA1 | **0.38** | **Highest — strong subjective/egocentric remapping** |
| Sub | 0.02 | Low — filtered back to stable allocentric frame |
