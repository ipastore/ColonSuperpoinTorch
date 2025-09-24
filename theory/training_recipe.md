# Training commands

## Export

### Default
```bash
python export.py <export task>  <config file>  <export folder> [--outputImg | output images for visualization (space inefficient)]
```

### For colon
```bash
python export.py export_detector_homoAdapt configs/superpoint_colon_export.yaml ds4_specular_camera_mask_th005_k50_vbm6_nms4_topk600 --outputImg
```

## Training

### Default
```bash
python train4.py <train task> <config file> <export folder> --eval
```

### For Colon
```bash
python train4.py train_joint configs/superpoint_colon_train_heatmap.yaml ds4_specular_camera_mask_th005_k50_vbm6_nms4_topk600 --eval --debug
```