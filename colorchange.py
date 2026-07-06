import RPi.GPIO as GPIO
import time
import sys

# HARDWARE MAPPING WITH CORRECT CONFIGURATION LOGIC
# "rgb-yellow" is configured with a tuple containing both the green and red pins
LED_CONFIG = {
    "blue":       {"pin": 27, "inverted": True},
    "green":      {"pin": 17, "inverted": True},
    "yellow":     {"pin": 23, "inverted": False},
    "red":        {"pin": 22, "inverted": True},
    "rgb-yellow": {"pin": (17, 22), "inverted": True}, # Coupled array channel
    "nir":        {"pin": 25, "inverted": True},  # Try switching to Active-High
    "deep-nir":   {"pin": 16, "inverted": False} # 940nm Share-Board Channel
}

# Configure Pi to use BCM GPIO pin numbering
GPIO.setmode(GPIO.BCM)

def set_led_state(target_name, state="OFF"):
    """Safely transitions a specific channel without floating the other pins."""
    info = LED_CONFIG[target_name]
    
    # Handle single pins or a tuple of multiple pins uniformly
    pins = info["pin"] if isinstance(info["pin"], tuple) else (info["pin"],)
    
    if state == "ON":
        val = GPIO.LOW if info["inverted"] else GPIO.HIGH
    else:
        val = GPIO.HIGH if info["inverted"] else GPIO.LOW
        
    for pin in pins:
        GPIO.output(pin, val)

def kill_all_arrays():
    """Forces every single pin into an absolute safe dead state."""
    for name, info in LED_CONFIG.items():
        pins = info["pin"] if isinstance(info["pin"], tuple) else (info["pin"],)
        val = GPIO.HIGH if info["inverted"] else GPIO.LOW
        for pin in pins:
            GPIO.output(pin, val)

# Initialize all channels instantly to safe OFF baselines
print("⚙️ Securing GPIO boundaries...")
for name, info in LED_CONFIG.items():
    pins = info["pin"] if isinstance(info["pin"], tuple) else (info["pin"],)
    init_val = GPIO.HIGH if info["inverted"] else GPIO.LOW
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT, initial=init_val)

print("\n=== Multi-Spectral Manual Debugger Engaged ===")
print("Available Channels: blue | green | yellow | red | rgb-yellow | nir | deep-nir")
print("Control Commands: Type 'off' to kill light, or 'exit' to safe-shutdown.")

active_channel = None

try:
    while True:
        command = input("\nEnter channel name or command -> ").strip().lower()
        
        if command == "exit":
            break
            
        elif command == "off":
            if active_channel:
                set_led_state(active_channel, "OFF")
                print(f"🛑 {active_channel.upper()} matrix deactivated.")
                active_channel = None
            else:
                print("💡 All channels are already dark.")
                
        elif command in LED_CONFIG:
            # Kill any previous active light before shifting channels
            if active_channel:
                set_led_state(active_channel, "OFF")
            
            active_channel = command
            set_led_state(active_channel, "ON")
            print(f"⚡ {active_channel.upper()} loop fired high. Inspecting camera feed live...")
            if active_channel in ["nir", "deep-nir"]:
                print("👀 Remember: Infrared bands are invisible to human eyes! Check the camera sensor.")
                
        else:
            print("❌ Invalid command. Use: blue, green, yellow, red, rgb-yellow, nir, deep-nir, off, or exit.")

except KeyboardInterrupt:
    print("\n🛑 Execution interrupted via terminal command.")
finally:
    print("\n🧹 Initializing safe shutdown routine...")
    kill_all_arrays()
    GPIO.cleanup()
    print("👍 Hardware rails discharged. Grid safe.")