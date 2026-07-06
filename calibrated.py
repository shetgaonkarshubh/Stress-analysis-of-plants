import RPi.GPIO as GPIO
import subprocess
import cv2
import numpy as np
import time
import os

# 1. OPTIMIZED HIGH-SPEED HARDWARE MAPPING
# Shutter speeds dropped significantly to prevent saturation from the new, bright resistors
LED_CONFIG = {
    "Blue":      {"pins": [(27, True)],  "shutter": "8000"},   # Dropped to 8ms
    "Green":     {"pins": [(17, True)],  "shutter": "10000"},  # Dropped to 10ms
    # Yellow combines three high-power streams; dropping down to 4ms to protect the matrix
    "Yellow":    {"pins": [(23, False), (17, True), (22, True)], "shutter": "4000"}, 
    "Red":       {"pins": [(22, True)],  "shutter": "8000"},   # Dropped to 8ms
    "NIR":       {"pins": [(25, True)],  "shutter": "1500"},   # Kept fast (high-power 3W star)
    "Deep-NIR":  {"pins": [(16, False)], "shutter": "10000"}    # Dropped to 6ms
}

OUTPUT_DIR = "calibrated_spectral_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

GPIO.setmode(GPIO.BCM)

print("⚙️ Initializing GPIO channels to default safe OFF states...")
for name, info in LED_CONFIG.items():
    for pin, inverted in info["pins"]:
        init_val = GPIO.HIGH if inverted else GPIO.LOW
        GPIO.setup(pin, GPIO.OUT, initial=init_val)

def set_channel_state(pins_list, turn_on=True):
    """Iterates through a list of mapped pins and sets their state based on their inversion logic."""
    for pin, inverted in pins_list:
        if turn_on:
            val = GPIO.LOW if inverted else GPIO.HIGH
        else:
            val = GPIO.HIGH if inverted else GPIO.LOW
        GPIO.output(pin, val)

def capture_differential_sweep():
    """Cycles through all LEDs, flashing them strictly during the image capture sequence."""
    isolated_bands = {}
    
    for name, info in LED_CONFIG.items():
        print(f"   Flashing Channel: {name} (Shutter: {int(info['shutter'])/1000}ms)")
        
        ambient_path = f"{OUTPUT_DIR}/temp_amb.jpg"
        led_on_path = f"{OUTPUT_DIR}/temp_led.jpg"
        
        # --- CAPTURE AMBIENT BASELINE ---
        cmd_amb = ["rpicam-still", "-o", ambient_path, "--immediate", "--nopreview", "-v", "0",  
                   "--shutter", info["shutter"], "--gain", "1.0"] # Dropped analog gain to 1.0 for noise hygiene
        subprocess.run(cmd_amb, check=True)
        
        # --- TURN TARGET LED(S) ON ---
        set_channel_state(info["pins"], turn_on=True)
        time.sleep(0.05)  # Shorter settling delay since hardware switching is warm
        
        # --- CAPTURE ILLUMINATED FRAME ---
        cmd_led = ["rpicam-still", "-o", led_on_path, "--immediate", "--nopreview", "-v", "0", 
                   "--shutter", info["shutter"], "--gain", "1.0"]
        subprocess.run(cmd_led, check=True)
        
        # --- TURN TARGET LED(S) OFF ---
        set_channel_state(info["pins"], turn_on=False)
            
        # Matrix Math processing (Differential Background Subtraction)
        img_amb = cv2.imread(ambient_path, cv2.IMREAD_GRAYSCALE)
        img_led = cv2.imread(led_on_path, cv2.IMREAD_GRAYSCALE)
        
        if img_amb is not None and img_led is not None:
            diff = cv2.subtract(img_led, img_amb)
            isolated_bands[name] = diff.astype(np.float32)
        else:
            print(f"   ❌ Error processing matrix math for {name}")
            
        # Clean up temporary storage files
        if os.path.exists(ambient_path): os.remove(ambient_path)
        if os.path.exists(led_on_path): os.remove(led_on_path)
        time.sleep(0.05)
        
    return isolated_bands

print("\n=== Hybrid Logic 6-Band Calibration Pipeline Active ===")

try:
    # STEP 1: CAPTURE WHITE CALIBRATION MAP
    print("\n[STEP 1] Place a clean white piece of paper on your black background covering the target zone.")
    input("Press ENTER when the white reference standard is ready...")
    print("Scanning White Reference...")
    white_profiles = capture_differential_sweep()
    
    # STEP 2: CAPTURE LEAF TARGET
    print("\n[STEP 2] Place the leaf TOP-SIDE UP in the exact same spot on the black background.")
    input("Press ENTER to execute the calibrated target scan...")
    print("Scanning Leaf Matrix...")
    target_profiles = capture_differential_sweep()
    
    # STEP 3: MAPPING PURE REFLECTANCE
    calibrated_layers = []
    order_of_bands = ["Blue", "Green", "Yellow", "Red", "NIR", "Deep-NIR"]
    
    for band in order_of_bands:
        if band in target_profiles and band in white_profiles:
            denom = white_profiles[band]
            denom[denom == 0] = 1.0  
            
            reflectance_matrix = target_profiles[band] / denom
            output_frame = (np.clip(reflectance_matrix, 0.0, 1.0) * 255).astype(np.uint8)
            calibrated_layers.append(output_frame)
            
            cv2.imwrite(f"{OUTPUT_DIR}/calibrated_{band}.png", output_frame)
    
    # STEP 4: COMPILE 6-BAND 3D DATA CUBE
    if len(calibrated_layers) == 6:
        data_cube = np.dstack(calibrated_layers)
        cube_filename = f"{OUTPUT_DIR}/calibrated_datacube.npy"
        np.save(cube_filename, data_cube)
        print(f"\n🎉 SUCCESS! 6-BAND CALIBRATED DATACUBE SAVED TO: {cube_filename}")
        print(f"Cube Structure Matrix Shape: {data_cube.shape} -> [Blue, Green, Yellow, Red, NIR, Deep-NIR]")
    else:
        print(f"\n⚠️ Missing spectral layers. Cube skipped. Layer count: {len(calibrated_layers)}/6")

except KeyboardInterrupt:
    print("\n🛑 Execution interrupted. Exiting system safely.")
finally:
    print("🧹 Cleaning up GPIO states and turning off all arrays...")
    for name, info in LED_CONFIG.items():
        set_channel_state(info["pins"], turn_on=False)
    GPIO.cleanup()
    print("👍 System clear.")