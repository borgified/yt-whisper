import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
import whisper

# Audio settings
TARGET_SAMPLE_RATE = 16000  # Whisper expects 16 kHz
CHUNK_DURATION = 4.0        # Seconds per chunk
OVERLAP_DURATION = 1.0      # Overlap duration
RMS_THRESHOLD = 0.001       # Silence cutoff

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Callback collecting raw audio from sounddevice."""
    if status:
        print(f"\n[Audio Warning]: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def find_blackhole_device():
    """Find the BlackHole audio device index."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        if "blackhole" in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx, dev
    return None, None


def draw_meter(rms):
    """Draw a logarithmic (dB) audio VU meter."""
    db = 20 * np.log10(max(rms, 1e-6))
    meter_len = 20
    level = max(0, min(meter_len, int((db + 60) / 3.0)))
    bar = "█" * level + "░" * (meter_len - level)
    sys.stdout.write(f"\r[Listening] Level: [{bar}] ({db:5.1f} dB | RMS: {rms:.4f})  ")
    sys.stdout.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Real-time speech-to-English translation.")
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Input language code (e.g. 'de', 'es', 'fr', 'ja', 'zh'). Default: auto-detect.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small"],
        help="Model size: 'tiny', 'base', or 'small' (default: base).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    device_idx, dev_info = find_blackhole_device()
    if device_idx is None:
        print("[!] BlackHole input not found. Using default microphone.")
        native_sr = 16000
    else:
        native_sr = int(dev_info["default_samplerate"])
        print(f"[*] Audio Input Device: [{device_idx}] {dev_info['name']} ({native_sr} Hz)")

    # Language configuration
    if args.language:
        target_lang = args.language.lower()
        print(f"[*] Input Language: {target_lang.upper()} (specified)")
        initial_prompt = f"Standard {target_lang} speech translated into natural, accurate, and fluent English."
    else:
        target_lang = None
        print("[*] Input Language: AUTO-DETECT (Whisper will detect language dynamically)")
        initial_prompt = "Foreign speech translated into natural, accurate, and fluent English."

    print(f"[*] Loading Whisper model ('{args.model}')...")
    model = whisper.load_model(args.model)
    print("[*] Model loaded successfully.")
    print("[*] Ready! Play audio in your browser (Press Ctrl+C to stop).")
    print("=" * 65)

    downsample_factor = native_sr // TARGET_SAMPLE_RATE if native_sr >= TARGET_SAMPLE_RATE else 1
    buffer = np.array([], dtype=np.float32)
    block_duration = 0.25
    block_size = int(native_sr * block_duration)

    with sd.InputStream(
        device=device_idx,
        channels=2,
        samplerate=native_sr,
        blocksize=block_size,
        callback=audio_callback,
    ):
        try:
            while True:
                block = audio_queue.get()

                if block.ndim > 1 and block.shape[1] > 1:
                    mono_block = np.mean(block, axis=1)
                else:
                    mono_block = block.flatten()

                current_rms = np.sqrt(np.mean(mono_block**2))
                draw_meter(current_rms)

                if downsample_factor > 1:
                    trim_len = len(mono_block) - (len(mono_block) % downsample_factor)
                    resampled = mono_block[:trim_len].reshape(-1, downsample_factor).mean(axis=1)
                else:
                    resampled = mono_block

                buffer = np.concatenate((buffer, resampled))

                if len(buffer) >= TARGET_SAMPLE_RATE * CHUNK_DURATION:
                    chunk_to_process = buffer[: int(TARGET_SAMPLE_RATE * CHUNK_DURATION)]
                    overlap_samples = int(TARGET_SAMPLE_RATE * OVERLAP_DURATION)
                    buffer = buffer[int(TARGET_SAMPLE_RATE * CHUNK_DURATION) - overlap_samples :]

                    total_rms = np.sqrt(np.mean(chunk_to_process**2))

                    if total_rms > RMS_THRESHOLD:
                        sys.stdout.write(f"\r[Transcribing chunk... RMS: {total_rms:.4f}]" + " " * 30 + "\r")
                        sys.stdout.flush()

                        result = model.transcribe(
                            chunk_to_process,
                            language=target_lang,
                            task="translate",
                            initial_prompt=initial_prompt,
                            fp16=False,
                            condition_on_previous_text=False,
                            without_timestamps=True,
                        )
                        text = result.get("text", "").strip()
                        detected_lang = result.get("language", "").upper()

                        if text:
                            if target_lang is None and detected_lang:
                                print(f"\n[EN] ({detected_lang}): {text}")
                            else:
                                print(f"\n[EN]: {text}")

        except KeyboardInterrupt:
            print("\n\n[*] Stopped listening.")


if __name__ == "__main__":
    main()
