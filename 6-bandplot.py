import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

BASE_DIR = r"C:\eysip 2026"
CUBE_PATH = os.path.join(BASE_DIR, "calibrated_datacube.npy")

if os.path.exists(CUBE_PATH):
    data_cube = np.load(CUBE_PATH)
    
    # --- EXTRACT 840nm (NIR) CHANNEL FOR TIGHT LEAF DETECTION ---
    # NIR channel is highly sensitive to leaf tissue and provides excellent discrimination
    nir_band = data_cube[:, :, 4].astype(np.uint8)  # 840nm NIR channel
    
    # --- REMOVE SALT AND PEPPER NOISE ---
    # Kernel size 3 or 5 works best without losing leaf edge crispness
    nir_band_filtered = cv2.medianBlur(nir_band, 3)
    
    # Compute dynamic Otsu tissue separation using the filtered 840nm (NIR) channel
    _, leaf_mask = cv2.threshold(nir_band_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leaf_mask = leaf_mask.astype(bool)
    
    # Morphological cleaning to remove tiny noise bits
    kernel = np.ones((3,3), np.uint8)
    leaf_mask_clean = cv2.morphologyEx(leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    leaf_mask_clean = cv2.morphologyEx(leaf_mask_clean, cv2.MORPH_OPEN, kernel).astype(bool)
    
    # --- TIGHT RECTANGLE EXTRACTION FROM 840nm CHANNEL ---
    # Find contours on the cleaned NIR mask and extract the tightest bounding rectangle
    contours, _ = cv2.findContours(leaf_mask_clean.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (the leaf)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the crop coordinates
        crop_y1, crop_y2 = y, y + h
        crop_x1, crop_x2 = x, x + w
    else:
        # Fallback if no contours found
        crop_y1, crop_y2 = 0, data_cube.shape[0]
        crop_x1, crop_x2 = 0, data_cube.shape[1]
    
    bands = ["Blue (450nm)", "Green (525nm)", "Yellow (590nm)", "Red (650nm)", "NIR (840nm)", "Deep-NIR (940nm)"]
    
    # Create a side-by-side grid plot for all 6 bands (cropped)
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle("CropDrop 6-Band Calibrated Spectral Footprint Map (Cropped to Leaf)", fontsize=16, fontweight='bold')
    
    for i in range(6):
        # Crop each band to the leaf region
        cropped_band = data_cube[crop_y1:crop_y2, crop_x1:crop_x2, i]
        axes[i].imshow(cropped_band, cmap='gray')
        axes[i].set_title(bands[i], fontsize=10)
        axes[i].axis('off')
        
    plt.tight_layout()
    
    # 1. Define the target directory and the full file path
    output_dir = os.path.join(BASE_DIR, "calibrated_spectral_data")
    output_plot = os.path.join(output_dir, "spectral_footprint_grid.png")
    
    # 2. Automatically create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Save the plot securely
    plt.savefig(output_plot, dpi=300)
    plt.show()
    print(f"📊 Summary grid saved to: {output_plot}")