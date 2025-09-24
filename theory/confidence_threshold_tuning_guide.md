# Confidence Threshold Tuning Guide for SuperPoint + LightGlue

## **Intuition**

The confidence threshold acts as a quality gate: **lower threshold** → more keypoints but noisier → more potential matches but lower precision; **higher threshold** → fewer, higher-quality keypoints → fewer but more reliable matches. 

**Key insight**: LightGlue typically achieves ~15-25% match rate on well-distributed SuperPoint keypoints. So targeting 300-400 final matches requires extracting ~1500-2000 keypoints per image at inference resolution.

---

## **Starting Values**

### **For 240×320 label export:**
- **Probability scores**: `0.005 - 0.020` (start at 0.010)
- **Logit scores**: `-5.3 to -3.9` (start at -4.6, since logit(-4.6) ≈ 0.010)
- **Target keypoint count at 240×320**: 400-600 keypoints post-NMS

### **Scaling logic:**
```
# Resolution scaling factor
scale_factor = (960 * 1344) / (240 * 320) = 16.8

# Target keypoints at 960×1344 for matching
target_kp_inference = 1500-2000  # to get 300-400 matches
target_kp_training = target_kp_inference / scale_factor = 90-120
```

**But use 400-600 at training resolution** due to:
- Homographic adaptation creating denser supervision
- Need margin for NMS effects and descriptor quality

---

## **2-Stage Tuning Procedure**

### **Stage A: Threshold Sweep at 240×320 (Label Export)**
1. **Fix** NMS radius = 4 pixels, homographic adaptation K=16
2. **Sweep** confidence threshold: `[0.003, 0.005, 0.008, 0.012, 0.018, 0.025]`
3. **Measure** keypoint count post-NMS on 20-50 diverse images
4. **Target**: Median 400-600 keypoints, std < 150 (good consistency)
5. **Select** threshold yielding closest to 500 keypoints median

### **Stage B: Validation at 960×1344 (Inference)**
1. **Train** model with selected threshold from Stage A
2. **Extract** keypoints at full 960×1344 resolution
3. **Measure** keypoint count and spatial distribution
4. **Target**: 1500-2000 keypoints post-NMS
5. **Adjust** inference threshold if needed:
   ```
   # If getting N keypoints but want M:
   new_threshold = current_threshold * (N / M)^0.7
   # Power 0.7 accounts for threshold's non-linear effect
   ```

### **Scaling Heuristic:**
```python
# From training to inference threshold scaling
def scale_threshold(train_thresh, train_res, infer_res, target_ratio=1.0):
    """
    train_thresh: threshold used during training
    train_res: (H, W) training resolution  
    infer_res: (H, W) inference resolution
    target_ratio: desired keypoint_count_ratio (inference/training)
    """
    area_ratio = (infer_res[0] * infer_res[1]) / (train_res[0] * train_res[1])
    # Typically want ~3-4x more keypoints at inference for good matching
    expected_ratio = area_ratio ** 0.5  # sqrt scaling is common
    adjustment = target_ratio / expected_ratio
    return train_thresh * adjustment ** 0.7
```

---

## **Training vs Inference Effects**

### **Changing Threshold During Label Export (Training):**

**Effects on Model Quality:**
- ✅ **DO**: Lower thresholds create denser supervision → better spatial coverage, improved descriptor learning
- ✅ **DO**: Maintain consistency across training dataset
- ❌ **DON'T**: Use extremely low thresholds (< 0.001) → noisy labels degrade detector precision
- ❌ **DON'T**: Change threshold mid-training → inconsistent supervision

### **Changing Threshold Only at Inference:**

**Effects:**
- ✅ **Safe**: No impact on trained weights
- ✅ **Flexible**: Can optimize for specific downstream tasks
- ❌ **Limited**: Can't improve inherent detector quality
- ❌ **Risk**: Very different inference threshold may expose undertrained regions

### **Recommended Practice:**

1. **Training phase**: Use moderate threshold (0.008-0.012) for balanced, dense supervision
2. **Inference phase**: Fine-tune threshold ±50% of training value for task optimization
3. **If major mismatch needed**: Retrain with adjusted label threshold

---

## **Autotune Snippet**

```python
def autotune_confidence_threshold(images, net, target_median_kp=500, 
                                nms_radius=4, min_spread_ratio=0.3):
    """
    Auto-tune confidence threshold to hit target keypoint count.
    
    Args:
        images: List of test images (240x320)
        net: SuperPoint network
        target_median_kp: Target median keypoint count
        nms_radius: NMS radius in pixels
        min_spread_ratio: Min std/mean ratio for spatial distribution
    
    Returns:
        Optimal threshold and diagnostic info
    """
    
    # Define search range
    threshold_candidates = np.logspace(-3, -1.5, 20)  # 0.001 to ~0.03
    
    best_thresh = None
    best_score = float('inf')
    
    for thresh in threshold_candidates:
        kp_counts = []
        spatial_spreads = []
        border_ratios = []
        
        for img in images[:20]:  # Use subset for speed
            # Extract keypoints
            prob_map = net(img)['prob']
            kp_coords = extract_keypoints_nms(prob_map, thresh, nms_radius)
            
            # Metrics
            kp_counts.append(len(kp_coords))
            
            # Spatial spread: std of distances to centroid
            if len(kp_coords) > 10:
                centroid = kp_coords.mean(axis=0)
                distances = np.linalg.norm(kp_coords - centroid, axis=1)
                spread_ratio = distances.std() / distances.mean()
                spatial_spreads.append(spread_ratio)
                
                # Border proximity (within 10px of edge)
                H, W = img.shape[:2]
                border_mask = ((kp_coords[:, 0] < 10) | (kp_coords[:, 0] > W-10) |
                              (kp_coords[:, 1] < 10) | (kp_coords[:, 1] > H-10))
                border_ratios.append(border_mask.mean())
        
        # Scoring function
        median_kp = np.median(kp_counts)
        median_spread = np.median(spatial_spreads) if spatial_spreads else 0
        median_border = np.median(border_ratios) if border_ratios else 1
        
        # Penalty function - closer to target is better
        count_penalty = abs(median_kp - target_median_kp) / target_median_kp
        spread_penalty = max(0, min_spread_ratio - median_spread) * 5  # Heavy penalty for poor spread
        border_penalty = max(0, median_border - 0.15) * 3  # Penalty if >15% near borders
        
        total_score = count_penalty + spread_penalty + border_penalty
        
        print(f"Thresh: {thresh:.4f}, KP: {median_kp:.0f}, "
              f"Spread: {median_spread:.3f}, Border: {median_border:.3f}, "
              f"Score: {total_score:.3f}")
        
        if total_score < best_score:
            best_score = total_score
            best_thresh = thresh
    
    # Guardrails check
    final_test_counts = []
    for img in images:
        prob_map = net(img)['prob'] 
        kp_coords = extract_keypoints_nms(prob_map, best_thresh, nms_radius)
        final_test_counts.append(len(kp_coords))
    
    median_final = np.median(final_test_counts)
    std_final = np.std(final_test_counts)
    
    # Warning checks
    if std_final / median_final > 0.4:
        print(f"⚠️  High variability: std/mean = {std_final/median_final:.3f}")
    if median_final < target_median_kp * 0.7:
        print(f"⚠️  Undershot target: {median_final} vs {target_median_kp}")
    
    return {
        'optimal_threshold': best_thresh,
        'median_keypoints': median_final,
        'std_keypoints': std_final,
        'cv': std_final / median_final,
        'all_counts': final_test_counts
    }


def extract_keypoints_nms(prob_map, threshold, nms_radius):
    """Extract keypoints with NMS - implement based on your codebase"""
    # Your NMS implementation here
    # Return: np.array of shape (N, 2) with (x, y) coordinates
    pass
```

### **Usage Example:**
```python
# Stage A: Find optimal threshold for training labels
results = autotune_confidence_threshold(
    training_images_240x320, 
    superpoint_net,
    target_median_kp=500,
    nms_radius=4
)

optimal_train_thresh = results['optimal_threshold']

# Stage B: Scale for inference
inference_thresh = scale_threshold(
    optimal_train_thresh, 
    train_res=(240, 320),
    infer_res=(960, 1344),
    target_ratio=3.5  # Want ~3.5x more keypoints at inference
)
```

This gives you a systematic approach to dial in the perfect confidence threshold for your colonoscopy matching pipeline!