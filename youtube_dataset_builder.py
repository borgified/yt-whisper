import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_timestamp(ts_str):
    """Convert HH:MM:SS.mmm or MM:SS.mmm to seconds."""
    parts = ts_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return float(parts[0])


def clean_vtt_text(raw_text):
    """Remove VTT markup tags like <c>, <v Speaker>, <00:01:02.000>, and HTML entities."""
    # Remove timestamp tags like <00:00:01.200>
    text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", raw_text)
    # Remove HTML tags like <c>, <b>, </i>, <v ...>
    text = re.sub(r"<[^>]+>", "", text)
    # Clean whitespace and HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_vtt_file(vtt_path):
    """Parse a WebVTT file into a list of (start_sec, end_sec, text) tuples."""
    with open(vtt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cues = []
    current_start = None
    current_end = None
    current_text_lines = []

    time_pattern = re.compile(
        r"(\d{2}:)?\d{2}:\d{2}\.\d{3}\s+-->\s+(\d{2}:)?\d{2}:\d{2}\.\d{3}"
    )

    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("WEBVTT") or line_clean.startswith("NOTE"):
            continue

        if "-->" in line_clean and time_pattern.search(line_clean):
            # Save previous cue if exists
            if current_start is not None and current_text_lines:
                raw = " ".join(current_text_lines)
                clean = clean_vtt_text(raw)
                if clean:
                    cues.append((current_start, current_end, clean))

            # Parse new timestamps
            times = line_clean.split("-->")
            start_str = times[0].strip().split()[0]
            end_str = times[1].strip().split()[0]
            current_start = parse_timestamp(start_str)
            current_end = parse_timestamp(end_str)
            current_text_lines = []
        else:
            if current_start is not None:
                current_text_lines.append(line_clean)

    # Save last cue
    if current_start is not None and current_text_lines:
        raw = " ".join(current_text_lines)
        clean = clean_vtt_text(raw)
        if clean:
            cues.append((current_start, current_end, clean))

    # Deduplicate consecutive identical lines (YouTube VTT rolling captions)
    deduped_cues = []
    for start, end, text in cues:
        if not deduped_cues:
            deduped_cues.append((start, end, text))
        else:
            prev_start, prev_end, prev_text = deduped_cues[-1]
            if text == prev_text:
                # Extend end time
                deduped_cues[-1] = (prev_start, max(prev_end, end), prev_text)
            elif text.startswith(prev_text) and len(text) > len(prev_text):
                # Rolling update
                deduped_cues[-1] = (prev_start, max(prev_end, end), text)
            else:
                deduped_cues.append((start, end, text))

    return deduped_cues


def group_cues(cues, min_duration=5.0, max_duration=18.0, max_pause=1.0):
    """Group short subtitle cues into cohesive segments between min_duration and max_duration seconds."""
    segments = []
    if not cues:
        return segments

    curr_start = cues[0][0]
    curr_end = cues[0][1]
    curr_texts = [cues[0][2]]

    for i in range(1, len(cues)):
        start, end, text = cues[i]
        pause = start - curr_end
        combined_duration = end - curr_start

        # Break segment if pause is too long or max duration exceeded
        if pause > max_pause or combined_duration > max_duration:
            if (curr_end - curr_start) >= min_duration:
                segments.append((curr_start, curr_end, " ".join(curr_texts)))
            curr_start = start
            curr_end = end
            curr_texts = [text]
        else:
            curr_end = end
            curr_texts.append(text)

    # Last segment
    if (curr_end - curr_start) >= min_duration:
        segments.append((curr_start, curr_end, " ".join(curr_texts)))

    return segments


def build_dataset_from_youtube(
    url,
    lang="en",
    output_dir="dataset",
    allow_auto=False,
    cookies=None,
    proxy=None,
    user_agent=None,
    target_lang=None,
):
    """Download audio & subtitles from YouTube and prepare aligned training dataset.

    If `target_lang` is provided, a second subtitle track is downloaded (if available)
    and a verification file `train_target.jsonl` is written containing the target language
    sentences aligned with the audio clips.
    """
    out_path = Path(output_dir)
    audio_dir = out_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = out_path / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("YOUTUBE TO WHISPER DATASET BUILDER")
    print("=" * 65)
    print(f"[*] Target URL:      {url}")
    print(f"[*] Language:        {lang}")
    if target_lang:
        print(f"[*] Target Language: {target_lang}")
    print(f"[*] Output Dir:      {out_path.resolve()}")

    # 1. Fetch metadata to inspect subtitles
    print("\n[*] Fetching video info...")
    meta_cmd = ["yt-dlp", "--dump-json", url]
    if cookies:
        meta_cmd.extend(["--cookies", cookies])
    if proxy:
        meta_cmd.extend(["--proxy", proxy])
    if user_agent:
        meta_cmd.extend(["--user-agent", user_agent])
    res = subprocess.run(meta_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[Error]: yt-dlp failed to fetch video info: {res.stderr.strip()}", file=sys.stderr)
        return False

    meta = json.loads(res.stdout)
    title = meta.get("title", "Unknown Title")
    duration = meta.get("duration", 0)
    print(f"[*] Video Title:     {title}")
    print(f"[*] Duration:        {duration // 60}m {duration % 60}s")

    subtitles = meta.get("subtitles", {})
    auto_subtitles = meta.get("automatic_captions", {})

    has_manual = lang in subtitles
    has_auto = lang in auto_subtitles

    # Target language availability (optional)
    if target_lang:
        target_manual = target_lang in subtitles
        target_auto = target_lang in auto_subtitles
    else:
        target_manual = target_auto = False

    # Source language subtitle check
    if not has_manual:
        print(f"\n[Notice]: No creator-uploaded (manual) subtitles found for '{lang}'.")
        if has_auto:
            if not allow_auto:
                print(f"[!] Video has AUTO-GENERATED subtitles for '{lang}'.")
                print("    Auto-generated subtitles may contain mistakes from YouTube's speech recognition.")
                print("    To proceed with auto-generated subtitles, add the '--allow-auto' flag.")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return False
            else:
                print(f"[*] Using AUTO-GENERATED subtitles as requested (--allow-auto).")
        else:
            print(f"[Error]: No subtitles found at all for language '{lang}'.", file=sys.stderr)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False
    else:
        print(f"[*] Found CREATOR-UPLOADED human subtitles for '{lang}'! (High quality)")

    # Target language subtitle check
    if target_lang:
        if target_manual:
            print(f"[*] Found CREATOR-UPLOADED subtitles for target language '{target_lang}'! (High quality)")
        elif target_auto:
            if allow_auto:
                print(f"[*] Using AUTO-GENERATED subtitles for target language '{target_lang}'.")
            else:
                print(f"[Notice]: AUTO-GENERATED subtitles available for '{target_lang}' (add --allow-auto to use them).")
        else:
            print(f"[Warning]: No subtitles (manual or auto) found for target language '{target_lang}'.")
            print("           Proceeding without verification file.")

    # 2. Download audio and subtitles
    print("\n[*] Downloading audio and subtitles...")
    raw_audio_template = str(tmp_dir / "raw_audio.%(ext)s")
    dl_cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "-o", raw_audio_template,
    ]
    # Construct the subtitle language list for yt-dlp
    lang_list = [lang]
    if target_lang and (target_manual or target_auto):
        lang_list.append(target_lang)
    dl_cmd.extend(["--sub-lang", ",".join(lang_list)])
    dl_cmd.extend(["--sub-format", "vtt"])

    # Authentication / network options
    if cookies:
        dl_cmd.extend(["--cookies", cookies])
    if proxy:
        dl_cmd.extend(["--proxy", proxy])
    if user_agent:
        dl_cmd.extend(["--user-agent", user_agent])

    # Always request both manual and auto flags so yt-dlp picks whatever exists per language
    dl_cmd.append("--write-subs")
    if not has_manual or (target_lang and not target_manual and target_auto):
        dl_cmd.append("--write-auto-subs")

    dl_cmd.append(url)

    res = subprocess.run(dl_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[Error]: Download failed: {res.stderr.strip()}", file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # Locate downloaded WAV and VTT files
    wav_files = list(tmp_dir.glob("raw_audio*.wav"))
    vtt_files = list(tmp_dir.glob("raw_audio*.vtt"))
    if not wav_files or not vtt_files:
        print(f"[Error]: Could not locate downloaded audio or subtitle files in {tmp_dir}", file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False
    raw_wav = wav_files[0]

    # Identify primary and secondary VTT files by language code in filename
    primary_vtt = None
    secondary_vtt = None
    for vf in vtt_files:
        name = vf.name
        if f".{lang}." in name:
            primary_vtt = vf
        elif target_lang and f".{target_lang}." in name:
            secondary_vtt = vf
    if not primary_vtt:
        print(f"[Error]: Primary subtitle file for '{lang}' not found.", file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # 3. Parse subtitles
    print(f"\n[*] Parsing primary subtitles: {primary_vtt.name}")
    raw_cues = parse_vtt_file(primary_vtt)
    print(f"[*] Parsed {len(raw_cues)} primary subtitle cues.")

    secondary_cues = []
    if target_lang and secondary_vtt:
        print(f"[*] Parsing secondary subtitles: {secondary_vtt.name}")
        secondary_cues = parse_vtt_file(secondary_vtt)
        print(f"[*] Parsed {len(secondary_cues)} secondary subtitle cues.")
    elif target_lang:
        print(f"[Warning]: No secondary VTT file found for '{target_lang}'; skipping verification file.")

    # Group cues into segments (same parameters for both languages)
    segments_src = group_cues(raw_cues, min_duration=4.0, max_duration=18.0, max_pause=0.8)
    segments_tgt = []
    if secondary_cues:
        segments_tgt = group_cues(secondary_cues, min_duration=4.0, max_duration=18.0, max_pause=0.8)

    # Align lengths – truncate to the shorter list to keep pairing valid
    if segments_tgt:
        min_len = min(len(segments_src), len(segments_tgt))
        segments_src = segments_src[:min_len]
        segments_tgt = segments_tgt[:min_len]

    print(f"[*] Grouped into {len(segments_src)} training segments (4s-18s each).")
    if segments_tgt:
        print(f"[*] Also prepared {len(segments_tgt)} verification segments.")

    if not segments_src:
        print("[Error]: No usable speech segments found after grouping.", file=sys.stderr)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # 4. Slice audio and build JSONL dataset
    print(f"\n[*] Slicing audio clips into '{audio_dir}/'...")
    dataset_records = []
    verification_records = []
    total_seconds = 0.0

    tgt_iter = segments_tgt if segments_tgt else [None] * len(segments_src)
    for idx, (src_seg, tgt_seg) in enumerate(zip(segments_src, tgt_iter)):
        start_sec, end_sec, src_text = src_seg
        tgt_text = tgt_seg[2] if tgt_seg else ""
        clip_name = f"clip_{idx+1:04d}.wav"
        clip_path = audio_dir / clip_name
        duration_sec = end_sec - start_sec
        total_seconds += duration_sec

        # ffmpeg slice
        slice_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-to", f"{end_sec:.3f}",
            "-i", str(raw_wav),
            "-ar", "16000",
            "-ac", "1",
            str(clip_path),
        ]
        subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        dataset_records.append({
            "id": f"clip_{idx+1:04d}",
            "audio": f"audio/{clip_name}",
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "duration": round(duration_sec, 2),
            "sentence": src_text,
        })
        if tgt_text:
            verification_records.append({
                "id": f"clip_{idx+1:04d}",
                "sentence_target": tgt_text,
            })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(segments_src):
            sys.stdout.write(f"\r[*] Processed {idx+1}/{len(segments_src)} audio clips...")
            sys.stdout.flush()

    print()  # newline after progress line

    # 5. Write train.jsonl (source) and optional verification file (target)
    jsonl_path = out_path / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in dataset_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = out_path / "train.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("audio,sentence\n")
        for rec in dataset_records:
            clean_s = rec["sentence"].replace('"', '""')
            f.write(f'"{rec["audio"]}","{clean_s}"\n')

    verif_path = None
    if verification_records:
        verif_path = out_path / "train_target.jsonl"
        with open(verif_path, "w", encoding="utf-8") as f:
            for rec in verification_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Cleanup temp directory
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 65)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 65)
    print(f"[*] Total Clips Prepared:  {len(dataset_records)}")
    print(f"[*] Total Paired Audio:    {total_seconds // 60:.0f}m {total_seconds % 60:.1f}s")
    print(f"[*] Dataset Metadata:      {jsonl_path}")
    print(f"[*] CSV Metadata:          {csv_path}")
    print(f"[*] Audio Directory:       {audio_dir}")
    if verif_path:
        print(f"[*] Verification File:     {verif_path}")
    print("-" * 65)
    print("[*] Sample Pair 1:")
    print(f"    Audio:    {dataset_records[0]['audio']} ({dataset_records[0]['duration']}s)")
    print(f"    Sentence: \"{dataset_records[0]['sentence']}\"")
    if verification_records:
        print(f"    Target:   \"{verification_records[0]['sentence_target']}\"")
    if len(dataset_records) > 1:
        print("[*] Sample Pair 2:")
        print(f"    Audio:    {dataset_records[1]['audio']} ({dataset_records[1]['duration']}s)")
        print(f"    Sentence: \"{dataset_records[1]['sentence']}\"")
        if len(verification_records) > 1:
            print(f"    Target:   \"{verification_records[1]['sentence_target']}\"")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download YouTube video and build paired (audio, transcript) dataset for Whisper fine-tuning."
    )
    parser.add_argument("url", type=str, help="YouTube video URL.")
    parser.add_argument(
        "-l", "--lang",
        type=str,
        default="en",
        help="Source subtitle language code (e.g. 'en', 'de'). Default: en.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="dataset",
        help="Output directory for dataset. Default: dataset.",
    )
    parser.add_argument(
        "--allow-auto",
        action="store_true",
        help="Allow using YouTube auto-generated captions if manual creator subtitles are not available.",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to cookies file for authenticated YouTube access (optional).",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Proxy URL for yt-dlp (optional).",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Custom user-agent string for yt-dlp (optional).",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default=None,
        help=(
            "Secondary subtitle language code (e.g. 'es') to download for verification. "
            "When provided, a train_target.jsonl file is written with paired target-language sentences."
        ),
    )
    args = parser.parse_args()

    build_dataset_from_youtube(
        args.url,
        lang=args.lang,
        output_dir=args.output_dir,
        allow_auto=args.allow_auto,
        cookies=args.cookies,
        proxy=args.proxy,
        user_agent=args.user_agent,
        target_lang=args.target_lang,
    )
