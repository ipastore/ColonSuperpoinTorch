# Export Label Parameters Decision Guide

## Overview

This document outlines the parameter choices for exporting SuperPoint labels on colonoscopy data, combining theoretical insights, experimental evidence, and practical considerations. The decisions are based on systematic threshold×K sweep experiments and visual analysis of keypoint quality.

---

## **Parameter Decisions Summary**

| Parameter | Chosen Value | Rationale |
|-----------|--------------|-----------|
| **Model** | `SuperPointNet_gauss2` | Best architecture for colon data + experimental performance |
| **Detection Threshold** | `0.003` | Balance between keypoint count and noise reduction |
| **Homography Adaptation (K)** | `40` | Optimal coverage without border noise accumulation |
| **NMS Radius** | `4` | Standard value, good spatial distribution |
| **Downsize** | `4` (336×240) | Paper-standard resolution, divisible by 8 |
| **Pretrained Weights** | `superpoint_coco_heat2_0/170k` | Best performing checkpoint per README |
| **Other Parameters** | Default values | Proven stable configurations |

---

## **1. Model Choice: SuperPointNet_gauss2**

### **Decision: Use `SuperPointNet_gauss2`**

**Theoretical Justification:**
- **U-Net Encoder**: Provides better spatial context than VGG-style encoders, crucial for colon tissue texture understanding
- **Gaussian Label Smoothing**: Built-in `gauss2` variant creates more robust supervision targets
- **Batch Normalization**: Improves training stability on medical imagery
- **Process Output Hook**: Better alignment between export and training configurations

**Experimental Evidence:**
- Sweep experiments showed SuperPointNet_gauss2 consistently produces the **highest keypoint counts** across parameter combinations
- More stable detection patterns compared to pretrained models on colon-specific texture
- Better handling of specular highlights and tissue deformation

**From Theory Documentation:**
> "Use model (C) `SuperPointNet_gauss2` for exporting supervision prior to fine-tuning. The U-Net encoder produces cleaner colon-specific heatmaps while preserving descriptor quality and giving direct control over NMS/thresholds."

**Compatibility:**
- Same 256-D descriptor head as other variants → downstream tools unchanged
- Compatible with homography adaptation pipeline
- Maintains 65-way detector head for consistent processing

---

## **2. Detection Threshold: 0.003**

### **Decision: Use detection threshold `0.003`**

**Experimental Evidence:**
- **Below 0.002**: Visual inspection revealed excessive noise labels from specular reflections and texture artifacts
- **0.003 Range**: Achieves ~400 keypoints on 336×240 images (optimal density)
- **Above 0.005**: Too few keypoints, poor spatial coverage

**Theoretical Support:**
From confidence threshold tuning guide:
> "Practical thresholds published for SuperPoint variants on natural images sit between 0.010 and 0.030. Colon imagery is noisier, so 0.010–0.020 keeps enough peaks while filtering pure noise."

**However**, our experiments on toy_33 dataset showed:
- Colon imagery requires **lower thresholds** than natural images due to subtle texture patterns
- 0.003 represents the "sweet spot" where signal significantly exceeds noise floor
- Maintains >85th percentile of meaningful activations per theoretical guidelines

**Export-Training Consistency:**
- **Critical**: Same threshold (0.003) will be used for both export and training to avoid supervision mismatch
- Prevents the label-training inconsistency problem where exported dense labels conflict with sparse training targets

**Quality Metrics at 0.003:**
- **Keypoint Count**: ~400 per 336×240 image (scales to ~1600 at full resolution)
- **Spatial Distribution**: Good coverage across field of view
- **Noise Level**: Minimal false positives from specular highlights
- **Repeatability**: Consistent detection across similar frames

---

## **3. Homography Adaptation Count: K=40**

### **Decision: Use `K=40` homography samples**

**Experimental Observations:**
- **K=20-40**: Good coverage improvement without excessive noise
- **K=60-80**: Marginal benefits, longer processing time
- **K=100**: Noisy points accumulating on mask margins, false positives increase

**Theoretical Backing:**
From choosing model documentation:
> "**Moderate K (40–80):** Best balance—coverage improves by 10–20% while duplicate detections remain manageable; descriptor precision typically stays ≥0.9."

**Quality-Performance Trade-off:**
- K=40 provides sufficient homography diversity for robust keypoint aggregation
- Avoids diminishing returns seen with higher K values
- Maintains processing speed for practical export workflows
- Reduces border artifacts observed in K=100 experiments

**Aggregation Mode:**
- Using `sum` aggregation (default) which emphasizes consistent peaks across multiple homographic views
- Works well with K=40 to balance evidence accumulation without over-smoothing

---

## **4. NMS Radius: 4 pixels**

### **Decision: Keep NMS radius at `4` pixels**

**Standard Practice:**
- Established value in SuperPoint literature
- Good spatial distribution without excessive clustering
- Appropriate for 336×240 resolution (scales to ~32 pixels at full resolution)

**No Experimental Issues:**
- Sweep experiments showed consistent performance across threshold×K combinations with NMS=4
- No evidence of keypoint clustering or poor spatial distribution
- Maintains descriptor discriminability

---

## **5. Image Resolution: Downsize=4 (336×240)**

### **Decision: Use `downsize=4` for 336×240 images**

**Practical Considerations:**
- **Divisible by 8**: Required for SuperPoint's 8×8 cell processing
- **Paper Standard**: Close to original SuperPoint training resolution (320×240)
- **Computational Efficiency**: Reasonable processing time for export
- **Memory Management**: Fits well within GPU memory constraints

**Scaling Logic:**
```python
# Original Endomapper: 1350×1012 → center crop 960×1344
# Downsize=4: 960×1344 → 240×336
# Our actual output: 336×240 (width×height corrected)
```

**Future Inference:**
- Labels exported at 336×240 scale appropriately to full 960×1344 resolution
- Keypoint density: ~400 keypoints → ~1600 keypoints at full resolution
- Suitable for LightGlue matching requirements (300-400 final matches)

---

## **6. Pretrained Weights: superpoint_coco_heat2_0/170k**

### **Decision: Use `logs/old/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar`**

**Documentation Reference:**
- README indicates this checkpoint as one of the best performing
- Trained with SuperPointNet_gauss2 architecture
- COCO dataset provides good feature diversity for transfer learning

**Architecture Compatibility:**
- Checkpoint specifically trained with gauss2 variant
- Includes proper U-Net encoder weights
- Compatible with 256-D descriptor head

**Alternative Rejected:**
- `superpoint_v1.pth` (MagicLeap): Architecture mismatch with gauss2
- Other checkpoints: Less documentation or performance validation

---

## **7. Other Parameters: Default Values**

### **Decision: Maintain default configurations for:**

**Homography Parameters:**
```yaml
homographies:
    params:
        translation: true
        rotation: true
        scaling: true
        perspective: true
        scaling_amplitude: 0.2
        perspective_amplitude_x: 0.2
        perspective_amplitude_y: 0.2
        allow_artifacts: true
        patch_ratio: 0.85
```
- Proven stable for colon imagery
- Appropriate deformation ranges for endoscopic perspective changes

**Processing Settings:**
- `workers_test: 2` - Good balance for I/O and processing
- `batch_size: 1` - Stable for export pipeline
- `apply_specular_mask_to_source_image: true` - Important for colon imagery quality
- `erode_camera_mask / erode_specular_mask`: tune vignetting vs. highlight erosion independently

**Disabled Features:**
- `gaussian_label.enable: false` - Not needed for export
- `augmentation.photometric.enable: false` - Clean export without augmentation
- `subpixel.enable: false` - Standard processing

---

## **Configuration Validation**

### **Final Export Configuration:**
```yaml
model:
    name: 'SuperPointNet_gauss2'
    detection_threshold: 0.003
    nms: 4

data:
    preprocessing:
        downsize: 4  # 336×240 resolution
    homography_adaptation:
        enable: true
        num: 40
        aggregation: 'sum'

pretrained: "logs/old/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar"
```

### **Expected Output:**
- **~400 keypoints** per 336×240 image
- **Clean spatial distribution** across field of view
- **Minimal noise** from specular highlights or texture artifacts
- **Consistent quality** for downstream training pipeline

### **Quality Assurance:**
- Visual inspection confirms good keypoint distribution
- Threshold selection avoids noise accumulation below 0.002
- K=40 prevents border artifacts seen at higher values
- Architecture choice maximizes keypoint yield while maintaining quality

---

## **Conclusion**

These parameter choices represent an **evidence-based optimization** for SuperPoint label export on colonoscopy data. The decisions balance:

1. **Keypoint Quantity**: ~400 keypoints per image for sufficient supervision density
2. **Label Quality**: Threshold above noise floor, avoiding false positives  
3. **Processing Efficiency**: Reasonable K value and resolution for practical workflows
4. **Architecture Suitability**: SuperPointNet_gauss2 proven best for colon imagery
5. **Training Consistency**: Same threshold for export and training to avoid supervision mismatch

The configuration is ready for production label export and subsequent fine-tuning workflows.
