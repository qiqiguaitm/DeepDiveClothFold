# LMWM Plan: Recurrence State World Model

## 1. Goal

Train a first CRAVE-native world model over recurrence milestone states.

The target is not pixel prediction and not a full VLA policy. The first target is
a compact model that consumes the current task-aware CRAVE state and predicts:

- Greedy next milestone: the one-step local maximum, `argmax P(stage_{t+1} | stage_t)`;
- Max-product next milestone: the next step on the finite-horizon highest-product path
  toward the terminal/completion milestone;
- a distribution over candidate future milestones;
- a path-level probability or confidence score that can later become VLA context.

This model is the first step toward:

```text
CRAVE recurrence states -> Latent Milestone World Model -> latent subgoal / planning prior -> VLA action model
```

### 1.1 Terminology Lock

The two next-milestone outputs are fixed as follows:

- `Greedy`: one-step local prediction, `argmax P(stage_{t+1} | stage_t)`.
  It answers: "from the current stage, what is the most likely immediate next
  stage?"
- `Max-product`: finite-horizon dynamic programming / max-product search toward
  the terminal milestone. It answers: "which immediate next stage lies on the
  highest-product path to completion?"

These names should not be swapped. Avoid `Max-Probability Milestone` in user-facing
text because it sounds like the one-step argmax and caused ambiguity.

## 2. LaWAM Reference Points

LaWAM is useful because it avoids direct pixel future prediction. It predicts
future observation features in a frozen visual feature space and injects those
features as latent visual subgoals for action generation.

From the LaWAM project page, the relevant design claims are:

- do not reconstruct future video for control;
- predict one compact latent visual subgoal in a frozen encoder space such as
  DINOv3;
- keep inference non-iterative and low-latency;
- learn the world model first, then let the policy predict/use its latent
  transition code.

Reference code copied locally:

```text
lmwm/vendor/LaWAM/
```

Parts worth reusing conceptually:

- `latent_action_model/`: latent future representation learning over frozen
  visual features.
- `starVLA/dataloader/latent_world_train_collator.py`: how latent world outputs
  are packaged together with VLA training samples.
- `starVLA/config/training/*.yaml`: run configuration style.
- `train_lawam.sh` and `train_lawam_distributed.sh`: environment setup, run
  directory, config snapshot, logging, and distributed launch pattern.

Parts we should not copy directly into the first LMWM prototype:

- Qwen/VLA training stack.
- LIBERO/RoboTwin simulator-specific adapters.
- LAM image-token reconstruction objective.

LMWM should borrow the interface idea, not the exact objective. Our latent is
CRAVE's task-aware milestone state, not a generic visual future token.

## 3. Local Project Layout

The `lmwm` directory follows the local `kai0` project style:

```text
lmwm/
  README.md
  pyproject.toml
  configs/
    datasets/
    models/
    training/
  data/
  checkpoints/
  logs/
  docs/
  scripts/
  src/lmwm/
  vendor/LaWAM/
```

Rules:

- Keep upstream LaWAM unmodified under `vendor/LaWAM`.
- Put first-party code only under `src/lmwm` and `scripts`.
- Keep large datasets and model checkpoints out of source control.
- Every training run should snapshot its YAML config into the run log directory.

## 4. CRAVE Data Interface

The first training dataset should be built from existing CRAVE artifacts.

Required per-frame fields:

- `episode_id`
- `frame_index` or feature timestep
- `milestone_id`
- `milestone_progress`
- `milestone_latent_prototype`
- optional decoded prototype image path
- optional DINO/state feature
- optional action chunk or robot state delta
- completion flag / endpoint quality flag if available

Initial source candidates:

- `crave` experiment outputs such as `_cache.npz`, `Pord`, `nm30`, milestone
  centers, decoded centers, and transition matrices.
- CRAVE scripts that already compute milestone sequences and Viterbi readouts.

The export format should be simple first:

```text
data/crave_sequences/<dataset_name>/
  manifest.jsonl
  sequences.npz
  prototypes.npz
  split.json
```

## 5. Model Scope: First Version

The first version should stay as close as possible to LaWAM Stage-1 LaWM.
Do not jump directly to the final graph/planner model. The first goal is to
reuse the same learning shape:

```text
LaWM:  z_t, z_{t+h} -> inverse transition code u_t
       z_t, u_t     -> predicted future visual latent z_hat_{t+h}

LMWM:  r_t, r_{t+h} -> inverse transition code u_t
       r_t, u_t     -> predicted future milestone latent r_hat_{t+h}
```

where `r_t` is a CRAVE recurrence-state latent, not a raw image feature.

This keeps input/output dimensions and training logic close to LaWM while still
using CRAVE's milestone abstraction. The final transition graph and Viterbi /
max-product planner should be added after this first LaWM-shaped model works.

### 5.1 Stage-1A: LaWM-Shaped LMWM

Use pairs of CRAVE states separated by a fixed horizon `h`:

```text
current state:  r_t
future state:   r_{t+h}
```

Recommended representation:

```text
r_t = concat(
  milestone_prototype_latent_t,   # projected to code_dim, e.g. 32 or 64
  milestone_id_embedding_t,       # optional, same code_dim scale
  progress_scalar_projection_t,   # optional small projection
  robot_state_projection_t        # optional, only if stable
)
```

For the first run, keep it minimal:

```text
r_t = projected milestone_prototype_latent_t
r_{t+h} = projected milestone_prototype_latent_{t+h}
code_dim = 32 or 64
num_queries = 1
```

This intentionally mirrors LaWAM's LAM config, where `code_dim=32` and
`num_queries=1`. If CRAVE prototype latents are high-dimensional, add a small
linear projector before the world model so the visible model interface remains
LaWM-like.

Architecture:

```text
r_t ------------------------------+
                                   v
r_t, r_{t+h} -> Inverse Encoder -> u_t
                                   |
                                   v
                      Forward Decoder(r_t, u_t)
                                   |
                                   v
                            r_hat_{t+h}

loss: distance(r_hat_{t+h}, r_{t+h})
      + optional CE(classifier(r_hat_{t+h}), milestone_{t+h})
      + optional state/progress auxiliary loss
```

Core modules:

- `StateProjector`: maps CRAVE prototype / id / progress to a fixed `code_dim`.
- `InverseTransitionEncoder`: predicts a compact transition code `u_t` from
  `(r_t, r_{t+h})`.
- `ForwardStateDecoder`: predicts `r_hat_{t+h}` from `(r_t, u_t)`.
- `MilestoneClassifier`: optional head that maps `r_hat` to milestone logits.

Why this first:

- It matches LaWM's inverse-plus-forward training recipe.
- It can reuse LaWAM's idea of a compact transition code.
- It avoids immediately solving graph planning, history modeling, and VLA
  integration at the same time.
- It produces a future latent subgoal that can later be injected into a policy
  similarly to LaWAM.

### 5.2 Stage-1B: Classification-Aligned Output

Because CRAVE milestones are discrete, pure latent regression is not enough. Add
a classifier over milestone ids:

```text
r_hat_{t+h} -> logits over M milestones
```

Loss:

```text
L = L_latent(r_hat_{t+h}, r_{t+h})
  + alpha * CE(logits, milestone_{t+h})
```

This handles the many-to-one property naturally. Multiple current states can
produce latents that classify to the same future milestone. The model is still
LaWM-shaped, but the evaluation can use milestone accuracy and graph behavior.

### 5.3 Stage-1C: From Fixed Horizon to Next Milestone

After fixed-horizon prediction works, train the same architecture with
`h = next unique milestone` instead of a fixed frame offset:

```text
r_t -> r_{next_unique_milestone}
```

This is the first point where LMWM starts to diverge from LaWM: the target is no
longer just a future time offset, but the next recurrence task stage.

### 5.4 Stage-2: Transition Distribution and Graph

Only after Stage-1A/B/C are stable, add the graph-style output:

```text
P(m_{t+1} | r_t, optional history/action)
P(done | r_t)
```

This stage supports:

- Greedy next milestone via one-step `argmax P(m_{t+1}|m_t)`;
- Max-product next milestone via finite-horizon Viterbi/max-product dynamic
  programming over the learned graph toward the terminal milestone;
- anomaly scoring via path likelihood.

Stage-2 is the route to the final recurrence state world model, but it should
not be the first implementation.

## 6. Stage-1 Input / Output Contract

### 6.1 Training Samples

Export pair samples from CRAVE episodes:

```text
episode_id
timestep_t
timestep_future
milestone_t
milestone_future
prototype_latent_t
prototype_latent_future
progress_t
progress_future
robot_state_t optional
action_chunk_t optional
```

Start with fixed horizon pairs:

```text
future = t + h
```

Then add next-unique-milestone pairs:

```text
future = first tau > t where milestone_tau != milestone_t
```

### 6.2 Model Inputs

Minimal first run:

```text
x_current = prototype_latent_t
x_future  = prototype_latent_future
```

After projection:

```text
r_t       : [code_dim]
r_future  : [code_dim]
```

Optional additions after the minimal run works:

```text
milestone_id_embedding_t
progress_projection_t
robot_state_projection_t
action_chunk_projection_t
```

### 6.3 Model Outputs

Stage-1A output:

```text
u_t              : transition code [code_dim]
r_hat_future     : predicted future CRAVE latent [code_dim]
```

Stage-1B output adds:

```text
logits_future_milestone : [num_milestones]
```

Output interpretation:

- `r_hat_future` is the LaWAM-compatible latent subgoal.
- `logits_future_milestone` is the CRAVE-specific discrete grounding.
- nearest prototype to `r_hat_future` gives a decoded center image for debugging.

### 6.4 Why Keep Dimensions Close to LaWM

Keeping the first model close to LaWM is useful because:

- LaWAM's LAM uses a compact code (`code_dim=32`, `num_queries=1`), so we can
  test whether CRAVE milestone dynamics also fit through a small transition
  bottleneck.
- A fixed `code_dim` interface makes later VLA conditioning easier.
- We can borrow LaWAM's inverse/forward training pattern before introducing
  CRAVE-specific graph planning.
- If the small bottleneck fails, the failure is informative: CRAVE recurrence
  dynamics may need history, action conditioning, or multi-query codes.

## 7. Evaluation

Use held-out CRAVE episodes.

Metrics:

- one-step top-1/top-3 next milestone accuracy;
- negative log likelihood of true next milestone;
- path likelihood of full held-out milestone sequence;
- completion-path accuracy: whether the max-completion path reaches observed
  later milestones;
- anomaly ranking: low probability for synthetic regress/reorder corruptions;
- subgoal usefulness: decoded next prototype is visually plausible and follows
  task order.

Important sanity checks:

- Compare against empirical transition graph.
- Compare against raw Viterbi milestone sequence without neural training.
- Report cases where CRAVE milestones are ambiguous or recurrent.
- Keep Viterbi smoothing / display smoothing separate from train labels.

## 8. Training Phases

### Phase 0: Repository and Data Audit

- Keep LaWAM code under `vendor/LaWAM`.
- Map CRAVE artifacts needed for sequence export.
- Decide canonical label names: `progress`, `milestone_id`, `prototype`.
- Write export script skeleton.

### Phase 1: Sequence Export

- Export milestone sequences for kai0_base and kai0_dagger.
- Save train/val/test splits by episode, not by frame.
- Save prototype latent and decoded prototype metadata.
- Recompute empirical transition graph from exported sequences.

### Phase 2: Non-Neural Graph Baseline

- Implement transition graph estimator.
- Implement greedy and max-completion planning.
- Evaluate on held-out episodes and synthetic corruptions.
- Produce visual report: current milestone -> predicted next milestones.

### Phase 3: Neural Recurrence State World Model

- Train MLP/GRU baseline.
- Add action/state conditioning if available.
- Compare against empirical graph.
- Export model checkpoint and inference API.

### Phase 4: VLA Interface

- Convert next milestone latent prototype to a policy-conditioning object.
- Export prompt/context fields:
  - current milestone;
  - greedy next milestone;
  - max-completion next milestone;
  - path probability;
  - decoded prototype image path if needed for debugging.
- Keep this interface compatible with a later RPVLA/LaWAM-style action model.

## 9. Risks and Decisions

- CRAVE progress is not a reward by definition. It can be converted into a
  progress-derived process reward, but the model should output `progress` and
  milestone probabilities.
- Viterbi is a decoder / planner, not the learned world model. It is useful for
  pseudo-labels, graph search, and max-product planning.
- Display smoothing must not be mixed into training labels. Train on milestone
  ids and graph transitions; use smoothed progress only for visualization.
- Recurrent milestones can make one-step prediction ambiguous. Evaluation should
  include top-k and path-level metrics, not only top-1.

## 10. Immediate Next Commands

Planned commands once scripts exist:

```bash
uv run python scripts/export_crave_sequences.py \
  --config configs/datasets/kai0_crave.yaml

uv run python scripts/train_state_world_model.py \
  --config configs/training/kai0_lmwm_gru.yaml

uv run python scripts/eval_state_world_model.py \
  --checkpoint checkpoints/kai0_lmwm_gru/latest.pt
```
