import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import yaml

from utils.loader import dataLoader
from utils.draw import plot_imgs, draw_keypoints

def draw_overlay(img, mask, color=(0, 0, 255), alpha=0.5, s=3):
    """
    Overlay binary mask on top of image.
    Assumes both img and mask are scaled by factor s.
    """
    # import pdb; pdb.set_trace()
    mask = 1 - mask
    mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    idx = np.where(mask)
    overlay[idx] = overlay[idx] * (1 - alpha) + np.array(color) * alpha
    return overlay.astype(np.uint8)


# === Output Directory ===
output_dir = 'colon_visualization'
os.makedirs(output_dir, exist_ok=True)

# === Load Config ===
with open('/home/student/ColonSuperpoinTorch/configs/superpoint_coco_train_heatmap.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Optional: enforce warping
config['data']['warped_pair'] = {
    'enable': True,
    'params': {
        'translation': True,
        'rotation': True,
        'scaling': True,
        'perspective': True,
        'scaling_amplitude': 0.2,
        'perspective_amplitude_x': 0.2,
        'perspective_amplitude_y': 0.2,
        'patch_ratio': 0.85,
        'max_angle': 1.57,
        'allow_artifacts': True,
    },
    'valid_border_margin': 3,
}
config['data']['truncate'] = 5

# === Load Dataset ===
task = config['data']['dataset']
data = dataLoader(config, dataset=task, warp_input=True)
train_loader = data['train_loader']

# === Visualization Parameters ===
scale = 3

for i, sample in enumerate(train_loader):
    d = {k: v[0].numpy() if hasattr(v, 'numpy') else v for k, v in sample.items()}

    img = d['image'].squeeze(0)

    # plt.figure(figsize=(10, 5))
    # plt.imshow(img, cmap='gray')
    # plt.title(f"Raw Image Visualization - Shape: {img.shape}")
    # plt.axis('off')
    # plt.show()

    # import pdb; pdb.set_trace()
    warped = d['warped_img'].squeeze(0)

    # plt.figure(figsize=(10, 5))
    # plt.imshow(warped, cmap='gray')
    # plt.title(f"Raw Image Visualization - Shape: {warped.shape}")
    # plt.axis('off')
    # plt.show()

    # import pdb; pdb.set_trace()


    name = d.get('name', f'sample_{i}')

    # Choose label source
    label_map = d['labels_2D'].squeeze(0)
    warped_label_map = d['warped_labels'].squeeze(0)
    
    # import pdb; pdb.set_trace()

    # Visualize labels as keypoints
    kpts = np.column_stack(np.where(label_map > 0))[:, ::-1]
    warped_kpts = np.column_stack(np.where(warped_label_map > 0))[:, ::-1]


    
    img_draw = draw_keypoints(img, kpts.transpose(), color=(0, 255, 0), radius=5, s=2)
    warped_draw = draw_keypoints(warped, warped_kpts.transpose(), color=(0, 255, 0), radius=5, s=2)

    valid_mask = d['valid_mask'].squeeze(0)
    

    # plt.figure(figsize=(10, 5))
    # plt.imshow(valid_mask, cmap='gray', vmin=0, vmax=1)
    # plt.title(f"Raw Image Visualization - Shape: {valid_mask.shape}")
    # plt.axis('off')
    # plt.show()


    warped_valid_mask = d['warped_valid_mask'].squeeze(0)

    # plt.figure(figsize=(10, 5))
    # plt.imshow(warped_valid_mask, cmap='gray', vmin=0, vmax=1)
    # plt.title(f"Raw Image Visualization - Shape: {warped_valid_mask.shape}")
    # plt.axis('off')
    # plt.show()

    # import pdb; pdb.set_trace()


    # # Overlay invalid (inverted mask) in red
    # img_draw_overlay = draw_overlay(img_draw, valid_mask, color=[0, 0, 255], alpha=0.5, s=scale)
    # warped_draw_overlay = draw_overlay(warped_draw, warped_valid_mask, color=[0, 0, 255], alpha=0.5, s=scale)

    # plt.figure(figsize=(10, 5))
    # plt.imshow(img_draw, cmap='gray', vmin=0, vmax=1)
    # plt.title(f"Raw Image Visualization - Shape: {img_draw.shape}")
    # plt.axis('off')
    # plt.show()

    # import pdb; pdb.set_trace()


    plot_imgs([img_draw, warped_draw],
              ylabel=name,
              titles=['original + labels', 'warped + labels'],
              cmap='gray',
              dpi=200)

    save_path = f'{output_dir}/colon_pair_{i:02d}.png'
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f'Pair {i} saved at {save_path}')

    if i >= 4:
        break
