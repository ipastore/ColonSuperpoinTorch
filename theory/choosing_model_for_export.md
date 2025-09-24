# Choosing Model for Feature Export

## Context
- Target: export detector heatmaps and 256-D descriptors at 240×320 for fine-tuning on 960×1344 colonoscopy frames.
- Downstream matcher: nearest neighbour; needs ~300–400 spatially distributed correspondences per full-resolution image.
- Candidates: (A) `SuperPointNet_pretrained` (MagicLeap), (B) `SuperPointNet`, (C) `SuperPointNet_gauss2`.

## Model Profiles
### A. SuperPointNet_pretrained (`models/SuperPointNet_pretrained.py`)
- Original MagicLeap release, shallow VGG-style encoder without normalization.
- Lightweight (fast export) but tuned to indoor scenes; heatmaps saturate on specular colon highlights.
- Descriptor head is 256-D and compatible, yet gradients become noisy when fine-tuning on deformable tissue.

### B. SuperPointNet (`models/SuperPointNet.py`)
- Adds batch norm and optional subpixel branch to the classic VGG stem.
- More stable than A but still uses max-pooling encoder; weaker context causes clustered detections unless thresholds are retuned per sequence.
- Needs manual handling of BN statistics during export/fine-tune cycles.

### C. SuperPointNet_gauss2 (`models/SuperPointNet_gauss2.py`)
- U-Net encoder with double-conv blocks + batch norm, providing sharper semi heatmaps before any adaptation.
- Retains the 65-way detector head and 256-D descriptors, so downstream tooling stays unchanged.
- Includes `process_output` hook to align export-time NMS/thresholds with training-time settings.

## Decision Matrix
| Criteria | A: SuperPoint_pretrained | B: SuperPointNet | C: SuperPointNet_gauss2 |
| --- | --- | --- | --- |
| Detector head expressiveness | ✓ (3) | ✓ (4) | ✓ (5) |
| Descriptor dimensionality & quality | ✓ (4) | ✓ (3) | ✓ (5) |
| Training stability on colon data | ✓ (4) | ✗ (2) | ✓ (4) |
| Typical performance after fine-tune | ✓ (3) | ✓ (3) | ✓ (5) |
| Export inference speed | ✓ (5) | ✓ (4) | ✓ (3) |
| Homographic adaptation compatibility | ✓ (5) | ✓ (4) | ✓ (4) |
| Ease of NMS/threshold tuning | ✓ (3) | ✓ (4) | ✓ (5) |

(✓ with 1–5 score; higher is better.)

## Recommendation
Use model (C) `SuperPointNet_gauss2` for exporting supervision prior to fine-tuning. The U-Net encoder produces cleaner colon-specific heatmaps while preserving descriptor quality and giving direct control over NMS/thresholds.

## Config Knobs (default values)
- `model.name = "SuperPointNet_gauss2"` in export + training configs.
- Descriptor dimension: 256 (implicit in the head).
- Detector loss: softmax; `lambda_loss = 1`.
- Homographic adaptation count `K = 60` (aggregation `sum`), balancing diversity and GPU memory.
- NMS radius at 240×320: 4 pixels (≈32 px at 960×1344 after scaling).
- Detection threshold range: 0.010–0.020 (start at 0.015; lower for sparse coverage, higher for dense clusters).

## Micro Ablation Plan (export-only, ≤6 runs)
1. Baseline export with `K=60`, threshold `0.015`, NMS `4`; record per-frame keypoint count, convex-hull area coverage, and radial density (polar histogram around the scope centre).
2. Threshold sweep `{0.012, 0.010, 0.0075}` with identical `K`/NMS; for each threshold log (a) mean keypoints/frame, (b) percentage on masked specular pixels, (c) nearest-neighbour match precision on 20 stereo/temporal pairs.
3. Very-low-threshold stress test at `0.00075`; measure false-positive rate via descriptors matched to neighbouring frames (expect precision drop) and visualise heatmaps to identify noise patterns.
4. NMS radius study (`nms ∈ {3,4,5}`) at the threshold chosen from Step 2; evaluate spatial dispersion metrics (Ripley’s K or simple cell occupancy) to avoid clumping.
5. Homographic adaptation stability: vary `K ∈ {40,60,80}` at the candidate threshold; track export runtime, keypoint variance across homography samples, and repeatability between raw image vs HA-aggregated result.
6. Temporal robustness check: run consecutive frames from a procedure (no fine-tune) and monitor keypoint count variance and match persistence (>50% retained across 5 steps); select the configuration that maximises persistence while staying within the 300–400 keypoint band.

## Confidence Threshold Basics
- SuperPoint predicts, for each 8×8 cell, a 65-way softmax heatmap (`semi`) whose values sum to 1 per cell. After homography aggregation, the logits are roughly in `[0,1]` with a heavy tail; meaningful keypoints typically occupy the top 10–20% of scores.
- Practical thresholds published for SuperPoint variants on natural images sit between `0.010` and `0.030` (after normalising the heatmaps). Colon imagery is noisier, so `0.010–0.020` keeps enough peaks while filtering pure noise.
- A threshold like `0.00075` is close to machine precision relative to the softmax output. At that level, almost every cell exceeds the threshold, so you end up labelling low-confidence responses dominated by sensor noise, specular glints, and blur. Training on those targets encourages the detector to accept spurious peaks, harming stability and descriptor quality.
- To sanity-check a candidate threshold, inspect the score histogram: pick a value around the elbow where the tail begins (often ≈85th percentile). Ensure that positive samples retain a signal-to-noise margin of ≥3 standard deviations over the background.

## Homography Aggregation Modes
Let `h_i` be the heatmap from the `i`-th homography sample and `K` the total samples. Aggregation combines them before thresholding.
- **Sum** (`sum`): `H = Σ_i h_i`. Emphasises consistent peaks—locations that survive multiple warps accumulate higher scores. Good default for colon data because it rewards persistence but still distinguishes strong vs weak evidence.
- **Mean** (`mean`): `H = (1/K) Σ_i h_i`. Equivalent to `sum` up to scaling; threshold must be adjusted accordingly (mean values are smaller when K is large). Offers similar behaviour but easier to reason about probabilistically.
- **Max** (`max`): `H = max_i h_i`. Captures the single strongest response per location; useful when the detector is reliable but risks promoting transient artefacts (e.g., a specular highlight in one homography) because it ignores how often the peak recurs.
- **Softmax / Log-Sum-Exp** (`softmax`): `H = log Σ_i exp(h_i / τ)` with temperature τ. Bridges between `max` (small τ) and `sum` (large τ). It boosts strong responses while still accounting for multiplicity. In practice, use when you expect a few homographies to dominate but still want some smoothing.
Example: suppose a point scores `{0.8, 0.7, 0.1}` across three homographies.
- Sum → `1.6` (high confidence).
- Mean → `0.533` (similar ranking; adjust threshold down accordingly).
- Max → `0.8` (still high; ignores the weak support).
- LSE with τ=0.1 → close to `0.8` (near-max), with τ=1.0 → ≈`0.90` (between sum and max).
Noisy point `{0.05, 0.01, 0.80}`: Sum=0.86 (moderate but will pass threshold), Mean=0.287, Max=0.80 (overly optimistic), LSE τ=1 ≈0.57 (still flags). Hence, sum/mean prefer points with repeated support, whereas max (and low-τ LSE) amplifies one-off spikes.

## Homography Adaptation Count (K) vs Keypoint Quality
- Each homography sample exposes different local maxima; aggregating over more samples lifts the long-tail activations that are otherwise suppressed by perspective or occlusions.
- **Low K (≤40):** Fewer aggregated points; high precision but gaps near the circular FOV edge and behind tissue folds because rare views are under-sampled.
- **Moderate K (40–80):** Best balance—coverage improves by 10–20% while duplicate detections remain manageable; descriptor precision typically stays ≥0.9. Runtime grows roughly linearly with K.
- **High K (≥100):** Diminishing returns; aggregation begins to elevate transient specular peaks and motion-induced artefacts, lowering precision and introducing more clustered detections after NMS.
- Keep K coupled with threshold: when K increases, the aggregated logits rise globally, so you may need to tighten the threshold slightly (e.g., from 0.012 to 0.014) to hold the keypoint count in the target band.
- Diagnostic: plot keypoint precision/recall against K while holding the threshold; if recall plateaus but precision drops, you have exceeded the useful K. Measure per-pixel variance across the K homography heatmaps—high variance at specular regions signals noise accumulation.

## Label Quantity vs Confidence Threshold
- Raising the threshold (e.g., 0.020) yields fewer, high-confidence peaks that are usually repeatable; lowering it increases count but eventually admits semi logits that are barely above noise. When threshold drops below the shoulder of the heatmap distribution (typically <0.010 for gauss2), you start exporting points dominated by specular reflections and textureless mucosa, which injects label noise during training.
- Compare the heatmap score percentiles to your chosen threshold: stay above the 85th percentile of activations so that positive labels significantly exceed the background mean by ≥3σ. The `0.00075` figure is several orders of magnitude below the noise floor and will flood the dataset with low signal-to-noise points.
- Quality check: for each threshold, compute NN match precision/recall on held-out frame pairs. If precision collapses (<0.85) while recall rises, you are past the optimal balance—the extra labels are mostly unreliable.
- Practical rule: target a threshold that delivers 250–450 keypoints at full resolution with ≥0.9 precision on pairwise matching and low specular occupancy (<5%). Adjust NMS in tandem to keep the field-of-view coverage uniform instead of lowering the threshold indefinitely.

## Choosing Between Few High-Quality vs Many Lower-Confidence Keypoints
- The detector loss expects a balanced set of positives; too few keypoints (e.g., 15) cause severe class imbalance and poor coverage—fine-tuning will overfit to those locations and degrade generalisation.
- Having many low-confidence points (e.g., >600 from a very low threshold) introduces noisy labels; the network learns to fire on random texture and loses precision, hurting downstream matching.
- Aim for a middle ground: enough positives (~300–400 at 960×1344) to cover the field and provide diverse supervision, but filtered to maintain ≥0.9 precision. This balance keeps the detector responsive while preventing overfitting to artefacts.
- During export-only evaluation, prefer the configuration that maximises repeatability and spatial spread at your target keypoint budget; quality metrics (precision, temporal persistence) should trump raw count.

