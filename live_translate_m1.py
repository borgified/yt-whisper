import argparse
import queue
import sys
import numpy as np
import sounddevice as sd
import mlx_whisper

# Audio settings
TARGET_SAMPLE_RATE = 16000  # 16 kHz audio for Whisper
MIN_SPEECH_DURATION = 1.8   # Minimum seconds of speech before allowing a sentence split
MAX_SPEECH_DURATION = 8.0   # Maximum seconds of speech before forcing a transcription
PAUSE_THRESHOLD = 0.5       # Seconds of silence required to detect end of sentence

# Energy gates to prevent Whisper from transcribing silence / room noise
RMS_SPEECH_LIMIT = 0.003    # RMS threshold to consider a block as active speech
PEAK_SPEECH_LIMIT = 0.02    # Peak amplitude required in chunk (real voice easily hits 0.03 - 0.5)

# Known YouTube artifact hallucinations that Whisper generates on silence/music
GHOST_PHRASES = {
    "thank you", "thank you.", "thank you very much.", "thanks for watching.",
    "thanks for watching", "thank you for watching.", "thank you for watching",
    "please subscribe", "please subscribe.", "subscribe to my channel",
    "subscribe to my channel.", "like and subscribe", "like and subscribe.",
    "subtitles by", "subtitles by the amara.org community", "bye", "bye.",
    "bye bye", "bye-bye", "see you next time", "see you next time.",
    "you", "you.", "so", "so.", "watching", "the end", "the end.",
    "[music]", "[applause]", "...",
}

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


def draw_meter(rms, status_text=""):
    """Draw a logarithmic (dB) audio VU meter."""
    db = 20 * np.log10(max(rms, 1e-6))
    meter_len = 16
    level = max(0, min(meter_len, int((db + 60) / 3.75)))
    bar = "█" * level + "░" * (meter_len - level)
    extra = f" [{status_text}]" if status_text else ""
    sys.stdout.write(f"\r[Listening] [{bar}] ({db:5.1f} dB){extra}   ")
    sys.stdout.flush()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-any-language translation with M1 GPU acceleration."
    )
    parser.add_argument(
        "-i",
        "-l",
        "--input-language",
        type=str,
        default="auto",
        help="Input spoken language code (e.g. 'en', 'de', 'es', 'fr', 'auto'). Default: auto.",
    )
    parser.add_argument(
        "-o",
        "--output-language",
        type=str,
        default="en",
        help="Target translation language code (e.g. 'zh', 'en', 'de', 'es', 'fr', 'orig'). Default: en.",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Show both original spoken text and target translation side-by-side.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="turbo",
        choices=["turbo", "small", "base", "tiny"],
        help="Model: 'turbo' (best accuracy, large-v3-turbo), 'small', 'base', 'tiny'. Default: turbo.",
    )
    return parser.parse_args()


def is_hallucination(text):
    """Check if generated text matches known silence artifacts."""
    clean = text.lower().strip().strip(".!?,")
    if clean in GHOST_PHRASES:
        return True
    for phrase in GHOST_PHRASES:
        if clean == phrase.strip(".!?"):
            return True
    return False


def get_text_translator(from_lang, to_lang):
    """Get or auto-install Argos Translate language package for from_lang -> to_lang."""
    if from_lang == to_lang:
        return lambda x: x

    try:
        import argostranslate.package
        import argostranslate.translate

        installed = argostranslate.translate.get_installed_languages()
        from_obj = next((l for l in installed if l.code == from_lang), None)
        to_obj = next((l for l in installed if l.code == to_lang), None)

        if from_obj and to_obj:
            t = from_obj.get_translation(to_obj)
            if t:
                return t.translate

        # If not installed, download from package index
        sys.stdout.write(f"\n[*] Downloading translation package ({from_lang} -> {to_lang})...\n")
        sys.stdout.flush()
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()

        # Direct pair
        pkg = next((p for p in available if p.from_code == from_lang and p.to_code == to_lang), None)
        if pkg:
            argostranslate.package.install_from_path(pkg.download())
            installed = argostranslate.translate.get_installed_languages()
            from_obj = next((l for l in installed if l.code == from_lang), None)
            to_obj = next((l for l in installed if l.code == to_lang), None)
            return from_obj.get_translation(to_obj).translate

        # Pivot pair through English (e.g. from_lang -> en -> to_lang)
        if from_lang != "en" and to_lang != "en":
            pkg1 = next((p for p in available if p.from_code == from_lang and p.to_code == "en"), None)
            pkg2 = next((p for p in available if p.from_code == "en" and p.to_code == to_lang), None)
            if pkg1 and pkg2:
                argostranslate.package.install_from_path(pkg1.download())
                argostranslate.package.install_from_path(pkg2.download())
                installed = argostranslate.translate.get_installed_languages()
                f_obj = next((l for l in installed if l.code == from_lang), None)
                e_obj = next((l for l in installed if l.code == "en"), None)
                t_obj = next((l for l in installed if l.code == to_lang), None)
                t1 = f_obj.get_translation(e_obj)
                t2 = e_obj.get_translation(t_obj)
                return lambda x: t2.translate(t1.translate(x))

    except Exception as e:
        print(f"\n[Warning]: Offline translator error ({e}).")

    return None


def main():
    args = parse_arguments()

    in_lang_code = None if args.input_language.lower() in ["auto", "none"] else args.input_language.lower()
    out_lang_code = args.output_language.lower()
    show_dual = args.dual

    model_map = {
        "turbo": "mlx-community/whisper-large-v3-turbo",
        "small": "mlx-community/whisper-small-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "tiny": "mlx-community/whisper-tiny",
    }
    model_repo = model_map[args.model]

    device_idx, dev_info = find_blackhole_device()
    if device_idx is None:
        print("[!] BlackHole input not found. Using default microphone.")
        native_sr = 16000
    else:
        native_sr = int(dev_info["default_samplerate"])
        print(f"[*] Audio Input:      [{device_idx}] {dev_info['name']} ({native_sr} Hz)")

    in_label = in_lang_code.upper() if in_lang_code else "AUTO-DETECT"
    out_label = "ORIGINAL" if out_lang_code in ["orig", "same"] else out_lang_code.upper()
    print(f"[*] Input Language:   {in_label}")
    print(f"[*] Output Language:  {out_label} {'(+ Dual View)' if show_dual else ''}")
    print(f"[*] Model:            {args.model.upper()} ({model_repo})")

    # Pre-load text translator if translating to non-English (e.g. en -> zh)
    text_translator = None
    if out_lang_code not in ["en", "orig", "same"]:
        # If in_lang_code is given, prepare from that language, else pivot from en
        source_for_mt = in_lang_code if in_lang_code and in_lang_code != "auto" else "en"
        text_translator = get_text_translator(source_for_mt, out_lang_code)
        if text_translator:
            print(f"[*] Neural MT Engine: Ready ({source_for_mt} -> {out_lang_code})")

    print("[*] Initializing M1 GPU...")
    # Warmup Whisper
    dummy = np.zeros(TARGET_SAMPLE_RATE * 1, dtype=np.float32)
    mlx_whisper.transcribe(dummy, path_or_hf_repo=model_repo, language=in_lang_code, task="transcribe")
    print("[*] Ready! Play audio in your browser (Press Ctrl+C to stop).")
    print("=" * 68)

    downsample_factor = native_sr // TARGET_SAMPLE_RATE if native_sr >= TARGET_SAMPLE_RATE else 1
    buffer = np.array([], dtype=np.float32)
    block_duration = 0.2
    block_size = int(native_sr * block_duration)
    silence_duration = 0.0

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

                # Downmix stereo to mono
                if block.ndim > 1 and block.shape[1] > 1:
                    mono_block = np.mean(block, axis=1)
                else:
                    mono_block = block.flatten()

                # Energy detection
                current_rms = np.sqrt(np.mean(mono_block**2))

                if current_rms < RMS_SPEECH_LIMIT:
                    silence_duration += block_duration
                else:
                    silence_duration = 0.0

                # Resample to 16 kHz
                if downsample_factor > 1:
                    trim_len = len(mono_block) - (len(mono_block) % downsample_factor)
                    resampled = mono_block[:trim_len].reshape(-1, downsample_factor).mean(axis=1)
                else:
                    resampled = mono_block

                if current_rms >= RMS_SPEECH_LIMIT or len(buffer) > 0:
                    buffer = np.concatenate((buffer, resampled))

                current_sec = len(buffer) / TARGET_SAMPLE_RATE
                status = f"Buf: {current_sec:.1f}s" if current_sec > 0 else "Idle"
                draw_meter(current_rms, status)

                # Trigger when sentence pause detected
                should_transcribe = False
                if current_sec >= MIN_SPEECH_DURATION and silence_duration >= PAUSE_THRESHOLD:
                    should_transcribe = True
                elif current_sec >= MAX_SPEECH_DURATION:
                    should_transcribe = True

                if should_transcribe and len(buffer) > 0:
                    audio_to_process = buffer.copy()
                    buffer = np.array([], dtype=np.float32)
                    silence_duration = 0.0

                    total_rms = np.sqrt(np.mean(audio_to_process**2))
                    peak_amp = np.max(np.abs(audio_to_process))

                    if total_rms >= RMS_SPEECH_LIMIT and peak_amp >= PEAK_SPEECH_LIMIT:
                        sys.stdout.write(f"\r[Transcribing sentence ({current_sec:.1f}s)...]" + " " * 25 + "\r")
                        sys.stdout.flush()

                        # Case A: Output is original transcript only
                        if out_lang_code in ["orig", "same"]:
                            res = mlx_whisper.transcribe(
                                audio_to_process,
                                path_or_hf_repo=model_repo,
                                language=in_lang_code,
                                task="transcribe",
                                temperature=0.0,
                                condition_on_previous_text=False,
                                without_timestamps=True,
                            )
                            text = res.get("text", "").strip()
                            det_lang = res.get("language", in_lang_code or "??").upper()
                            if text and not is_hallucination(text):
                                print(f"\n[{det_lang}]: {text}")

                        # Case B: Output is English (Whisper native translation)
                        elif out_lang_code == "en":
                            if show_dual:
                                res_orig = mlx_whisper.transcribe(
                                    audio_to_process,
                                    path_or_hf_repo=model_repo,
                                    language=in_lang_code,
                                    task="transcribe",
                                    temperature=0.0,
                                    condition_on_previous_text=False,
                                    without_timestamps=True,
                                )
                                orig_text = res_orig.get("text", "").strip()
                                det_lang = res_orig.get("language", in_lang_code or "??").upper()

                                if det_lang != "EN":
                                    res_en = mlx_whisper.transcribe(
                                        audio_to_process,
                                        path_or_hf_repo=model_repo,
                                        language=in_lang_code,
                                        task="translate",
                                        temperature=0.0,
                                        condition_on_previous_text=False,
                                        without_timestamps=True,
                                    )
                                    en_text = res_en.get("text", "").strip()
                                else:
                                    en_text = orig_text

                                if orig_text and not is_hallucination(orig_text):
                                    print(f"\n[{det_lang} Spoken]:     {orig_text}")
                                    if det_lang != "EN" and en_text and not is_hallucination(en_text):
                                        print(f"[EN Translated]: {en_text}")
                            else:
                                res_en = mlx_whisper.transcribe(
                                    audio_to_process,
                                    path_or_hf_repo=model_repo,
                                    language=in_lang_code,
                                    task="translate",
                                    temperature=0.0,
                                    condition_on_previous_text=False,
                                    without_timestamps=True,
                                )
                                en_text = res_en.get("text", "").strip()
                                det_lang = res_en.get("language", in_lang_code or "??").upper()
                                if en_text and not is_hallucination(en_text):
                                    print(f"\n[{det_lang}->EN]: {en_text}")

                        # Case C: Output is another language (e.g. ZH, DE, ES, FR, JA)
                        else:
                            # 1. Transcribe speech in source language
                            res_orig = mlx_whisper.transcribe(
                                audio_to_process,
                                path_or_hf_repo=model_repo,
                                language=in_lang_code,
                                task="transcribe",
                                temperature=0.0,
                                condition_on_previous_text=False,
                                without_timestamps=True,
                            )
                            orig_text = res_orig.get("text", "").strip()
                            det_lang = res_orig.get("language", in_lang_code or "??").lower()

                            if orig_text and not is_hallucination(orig_text):
                                # 2. Translate text to target language via Neural MT
                                mt_fn = text_translator
                                if mt_fn is None or (in_lang_code is None and det_lang != "en"):
                                    mt_fn = get_text_translator(det_lang, out_lang_code)

                                translated_text = mt_fn(orig_text) if mt_fn else "[Translation unavailable]"

                                if show_dual:
                                    print(f"\n[{det_lang.upper()} Spoken]:     {orig_text}")
                                    print(f"[{out_lang_code.upper()} Translated]: {translated_text}")
                                else:
                                    print(f"\n[{det_lang.upper()}->{out_lang_code.upper()}]: {translated_text}")

        except KeyboardInterrupt:
            print("\n\n[*] Stopped listening.")


if __name__ == "__main__":
    main()
