import sys
import numpy as np
import sounddevice as sd

print("=" * 60)
print("AUDIO DIAGNOSTIC TEST")
print("=" * 60)

devices = sd.query_devices()
blackhole_idx = None
for idx, dev in enumerate(devices):
    if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
        blackhole_idx = idx
        break

if blackhole_idx is None:
    print("[ERROR] BlackHole device not found in sounddevice list.")
    sys.exit(1)

dev_info = sd.query_devices(blackhole_idx)
native_sr = int(dev_info["default_samplerate"])
print(f"[*] Found: [{blackhole_idx}] {dev_info['name']}")
print(f"[*] Native Sample Rate: {native_sr} Hz")
print("[*] Recording 2 seconds of audio from BlackHole (Make sure audio is playing in browser!)...")

rec = sd.rec(int(native_sr * 2), samplerate=native_sr, channels=2, device=blackhole_idx, dtype="float32")
sd.wait()

rms = np.sqrt(np.mean(rec**2))
max_amp = np.max(np.abs(rec))

print("-" * 60)
print(f"Result -> RMS: {rms:.6f} | Peak Amplitude: {max_amp:.6f}")

if max_amp == 0.0:
    print("\n[!] WARNING: Audio received is EXACTLY ZERO (pure silence).")
    print("    Because you see the level moving in Mac's Sound Settings, this means")
    print("    macOS Privacy & Security is blocking your Terminal from accessing audio inputs.")
    print("\n[ACTION REQUIRED]:")
    print("    1. Open System Settings -> Privacy & Security -> Microphone")
    print("    2. Turn the toggle ON for your Terminal app (or iTerm / VS Code / Cursor)")
    print("    3. Restart your Terminal and re-run this script.")
else:
    print("\n[SUCCESS] Audio is being received by Python!")
    print("You can now run 'python live_translate.py'")
print("=" * 60)
