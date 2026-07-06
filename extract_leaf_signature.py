import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  # For plotting

# 1. PATH AND FILE INITIALIZATION
BASE_DIR = r"C:\eysip 2026"
INPUT_IMAGE = os.path.join(BASE_DIR, "calibrated_datacube.npy")

print(f"🔄 Loading target matrix from: {INPUT_IMAGE}")

if not os.path.exists(INPUT_IMAGE):
    print(f"❌ Error: Datacube not found at: {INPUT_IMAGE}")
    print("💡 Available files in your directory are:", os.listdir(BASE_DIR))
else:
    print("✅ Datacube matrix loaded successfully! Extracting profiles...")
    data_cube = np.load(INPUT_IMAGE)
    
    # --- 2. EXTRACT 840nm (NIR) CHANNEL FOR TIGHT LEAF DETECTION ---
    nir_band = data_cube[:, :, 4].astype(np.uint8)  # 840nm NIR channel
    
    print("🤖 Computing dynamic Otsu tissue separation using 840nm (NIR) channel...")
    _, leaf_mask = cv2.threshold(nir_band, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    leaf_mask = leaf_mask.astype(bool)
    
    # Morphological cleaning to remove tiny noise bits
    kernel = np.ones((3,3), np.uint8)
    leaf_mask_clean = cv2.morphologyEx(leaf_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    leaf_mask_clean = cv2.morphologyEx(leaf_mask_clean, cv2.MORPH_OPEN, kernel).astype(bool)
    
    # --- 3. TIGHT RECTANGLE EXTRACTION FROM 840nm CHANNEL ---
    contours, _ = cv2.findContours(leaf_mask_clean.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        tight_rect_mask = np.zeros_like(leaf_mask_clean)
        tight_rect_mask[y:y+h, x:x+w] = True
        
        final_target_mask = leaf_mask_clean & tight_rect_mask
    else:
        final_target_mask = leaf_mask_clean

    # 5. COMPUTE QUANTITATIVE SPECTRAL DATA LAYER STATISTICS
    bands = ["Blue", "Green", "Yellow", "Red", "NIR", "Deep-NIR"]
    wavelengths = [450, 525, 590, 650, 840, 940]
    
    # Lists to store values for plotting
    plot_wavelengths = []
    plot_means = []
    
    print("\n=======================================================")
    print("🛰️  VERIFIED LEAF SPECTRAL PROFILE REPORT")
    print("=======================================================")
    
    total_mask_pixels = np.sum(final_target_mask)
    print(f"🎯 Target Tissue Region Confirmed: {total_mask_pixels} pixels inside sampling envelope.\n")
    
    for i, name in enumerate(bands):
        channel_layer = data_cube[:, :, i]
        leaf_pixels = channel_layer[final_target_mask]
        
        if len(leaf_pixels) > 0:
            mean_reflectance = np.mean(leaf_pixels)
            std_deviation = np.std(leaf_pixels)
            print(f"📊 {name} Channel ({wavelengths[i]}nm) -> Mean Reflectance: {mean_reflectance:.2f} / 255 (± {std_deviation:.2f} variance)")
            
            # Append mean data for plotting
            plot_wavelengths.append(wavelengths[i])
            plot_means.append(mean_reflectance)
        else:
            print(f"⚠️ {name} Channel ({wavelengths[i]}nm) -> No valid tissue pixels found within mask boundaries.")
            
    print("=======================================================")
    
    # --- 6. PLOTTING THE SPECTRAL PROFILE ---
    if plot_means:
        print("\n📈 Generating Spectral Profile Plot...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot only the mean trendline
        plt.plot(plot_wavelengths, plot_means, marker='o', linestyle='-', color='forestgreen', linewidth=2.5, label='Mean Reflectance')
        
        # Formatting titles and axes
        plt.title('Leaf Spectral Profile across Wavelengths', fontsize=14, fontweight='bold')
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Reflectance Intensity (0 - 255)', fontsize=12)
        plt.xticks(wavelengths, [f"{w}nm\n({b})" for w, b in zip(wavelengths, bands)]) # Custom clear X ticks
        plt.ylim(0, 260)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()
    else:
        print("❌ Could not generate plot: No valid pixel data extracted.")