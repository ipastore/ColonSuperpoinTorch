import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.Colon import Colon



def visualize_mask_for_image(config, image_path, camera_mask_path, output_path=None):

    print("Creating Colon instance...")
    colon = Colon(**config)
    print("Reading image and camera mask...")
    img_np = colon._read_image(image_path)
    if camera_mask_path is None:
        camera_np = np.ones_like(img_np)
    else:
        camera_np = colon._read_image(camera_mask_path)
        camera_np = 1 - camera_np
    if config.get('apply_specular_mask_to_source_image', True):
        specular_np = colon._compute_specular_mask(img_np)
    else:
        specular_np = np.ones_like(img_np)
    combined_np = camera_np * specular_np
    
    print("Generating visualization plot...")
    # Visualize with matplotlib (outside the TensorFlow session)
    plt.figure(figsize=(16, 12))
    
    plt.subplot(221)
    # Explicit vmin/vmax so constant images (all ones) don't render black
    plt.imshow(img_np.squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Original Image')
    
    plt.subplot(222)
    plt.imshow(camera_np.squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Camera Mask (Black = Invalid)')
    
    plt.subplot(223)
    plt.imshow(specular_np.squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Specular Mask (Black = Invalid)')
    
    plt.subplot(224)
    plt.imshow(combined_np.squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
    plt.title('Combined Mask (Black = Invalid)')
    
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Now save the figure
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Replace these with your actual paths
    IMAGE_PATH = "/home/student/ColonSuperpoinTorch/datasets/endomapper/toy_33/train/out04717.png"  
    # CAMERA_MASK_PATH = "/home/student/ColonSuperpoinTorch/datasets/endomapper/camera_mask.png"
    CAMERA_MASK_PATH = None
    OUTPUT_PATH = "/home/student/ColonSuperpoinTorch/notebooks/outputs/example_mask_visualization.png"
    # USE_SPECULAR_MASK = True  # Set to False to disable specular mask
    USE_SPECULAR_MASK = False  # Set to False to disable specular mask

    config = {
        'dataset': 'Colon',
        'export_folder': 'train',
        'camera_mask_path': CAMERA_MASK_PATH,
        'apply_specular_mask_to_source_image': USE_SPECULAR_MASK,
        'erode_camera_mask': 0,
        'erode_specular_mask': 0,
        'images_path': '/home/student/ColonSuperpoinTorch/datasets/endomapper/toy_33/',
        'preprocessing': {
            'downsize': 1,
        }
    }

    visualize_mask_for_image(config, IMAGE_PATH, CAMERA_MASK_PATH, OUTPUT_PATH)
