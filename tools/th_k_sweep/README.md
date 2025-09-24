# SuperPoint Threshold × K Parameter Sweep

## Experiment Overview

This framework systematically tests **238 parameter combinations** to optimize SuperPoint keypoint detection for colonoscopy images by varying:

- **Detection threshold** (17 values: 0.000075 → 0.025)
- **Homography adaptation K** (14 values: 2 → 100)

## Quick Start

```bash
# Run complete 238-experiment sweep
./tools/th_k_sweep/run_threshold_k_sweep.sh configs/superpoint_colon_export.yaml

# Results saved to: logs/export/toy_33/threshold_k_sweep_TIMESTAMP/
```

## What It Tests

**Detection Threshold**: Controls keypoint sensitivity
- Low (0.000075-0.001): Detects weak features, high keypoint count
- Medium (0.005-0.01): Balanced detection
- High (0.015-0.025): Only strong features, low keypoint count

**Homography Adaptation K**: Controls label quality through transformations
- Low K (2-8): Fast processing, basic quality
- Medium K (12-25): Good balance
- High K (32-100): Best quality, slower processing

## Generated Results

### Metrics (for each combination)
- **Keypoint count**: Average detected keypoints per image
- **Spatial spread**: How well-distributed keypoints are
- **Border ratio**: Proportion near image edges

## Metric Details

### Spatial Spread
**Purpose**: Measures how uniformly keypoints are distributed across the image  
**Range**: 0.0 to ~2.0 (higher = more spread out)  
**Formula**: `std(distances_to_centroid) / mean(distances_to_centroid)`

```python
def _compute_spatial_spread(self, keypoints):
    """Compute spatial spread metric: std(distances_to_centroid) / mean(distances_to_centroid)"""
    if len(keypoints) < 2:
        return 0.0
        
    # Get coordinates (assuming keypoints are [x, y] or [x, y, score])
    coords = keypoints[:, :2] if keypoints.shape[1] >= 2 else keypoints
    
    # Compute centroid
    centroid = coords.mean(axis=0)
    
    # Compute distances to centroid
    distances = np.linalg.norm(coords - centroid, axis=1)
    
    # Return normalized spread
    if distances.mean() > 0:
        return distances.std() / distances.mean()
    else:
        return 0.0
```

**Interpretation**:
- **Low values (0.0-0.3)**: Keypoints clustered together (bad for coverage)
- **Medium values (0.3-0.7)**: Good distribution balance  
- **High values (0.7+)**: Very spread out, may indicate good coverage

### Border Ratio
**Purpose**: Measures fraction of keypoints detected near image edges  
**Range**: 0.0 to 1.0 (0 = no border keypoints, 1 = all on borders)  
**Threshold**: 10 pixels from image edge

```python
def _compute_border_ratio(self, keypoints, image_shape):
    """Compute fraction of keypoints near image borders"""
    if len(keypoints) == 0:
        return 0.0
        
    coords = keypoints[:, :2] if keypoints.shape[1] >= 2 else keypoints
    H, W = image_shape
    border_threshold = 10  # pixels
    
    # Check proximity to borders
    near_border = ((coords[:, 0] < border_threshold) | 
                  (coords[:, 0] > W - border_threshold) |
                  (coords[:, 1] < border_threshold) | 
                  (coords[:, 1] > H - border_threshold))
    
    return near_border.mean()
```

**Interpretation**:
- **Low values (0.0-0.2)**: Good - keypoints focus on image content
- **Medium values (0.2-0.5)**: Acceptable - mixed distribution
- **High values (0.5+)**: Poor - too many edge artifacts, may indicate noise

### Visualizations
- `keypoints_heatmap.png`: Keypoint counts across parameter space
- `spatial_spread_heatmap.png`: Distribution quality heatmap  
- `border_ratio_heatmap.png`: Edge proximity heatmap

### Data Files
- `threshold_k_sweep_results.csv`: Tabular results
- `experiment_summary.txt`: Best configuration summary


## Key Findings Format

The experiment identifies optimal parameters for:
1. **Maximum keypoints**: Best threshold×K for feature-rich detection
2. **Best spatial spread**: Most uniform keypoint distribution
3. **Balanced performance**: Optimal quality vs speed trade-off

Use results to configure your production SuperPoint settings based on application needs (speed vs quality vs keypoint density).