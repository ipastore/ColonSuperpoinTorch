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
    camera_np = colon._read_image(camera_mask_path)
    camera_np = 1 - camera_np 
    specular_np = colon._compute_specular_mask(img_np)
    combined_np = camera_np * specular_np
    
    print("Generating visualization plot...")
    # Visualize with matplotlib (outside the TensorFlow session)
    plt.figure(figsize=(16, 12))
    
    plt.subplot(221)
    plt.imshow(img_np.squeeze(), cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(222)
    plt.imshow(camera_np.squeeze(), cmap='gray')
    plt.title('Camera Mask (Black = Invalid)')
    
    plt.subplot(223)
    plt.imshow(specular_np.squeeze(), cmap='gray')
    plt.title('Specular Mask (Black = Invalid)')
    
    plt.subplot(224)
    plt.imshow(combined_np.squeeze(), cmap='gray')
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
    CAMERA_MASK_PATH = "/home/student/ColonSuperpoinTorch/datasets/endomapper/camera_mask.png"
    OUTPUT_PATH = "/home/student/ColonSuperpoinTorch/notebooks/outputs/example_mask_visualization.png"

    config = {
        'dataset': 'Colon',
        'export_folder': 'train',
        'camera_mask_path': CAMERA_MASK_PATH,
        'images_path': '/home/student/ColonSuperpoinTorch/datasets/endomapper/toy_33/train',
        'preprocessing': {
            'downsize': 1,
        }
    }

    visualize_mask_for_image(config, IMAGE_PATH, CAMERA_MASK_PATH, OUTPUT_PATH)