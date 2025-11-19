import os
import json
import uuid
import math
from typing import List, Dict, Any

from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
from moviepy import AudioFileClip, TextClip, CompositeVideoClip, ColorClip


# ---------------------------------------------------------------------------
# Configuration / Setup
# ---------------------------------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set or empty")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are a scriptwriter for TikTok and YouTube Shorts. 
You take rough story ideas and turn them into short, punchy scripts for 30–60 second vertical videos.

Your style:
- Strong hook in the first line
- Short, emotionally charged sentences
- Very visual and easy to imagine
- Written in casual, modern language

You ALWAYS respond with valid JSON only. No explanations, no markdown, no extra text.
""".strip()


# ---------------------------------------------------------------------------
# Script generation (LLM)
# ---------------------------------------------------------------------------

def build_user_prompt(idea: str) -> str:
    return f"""
Turn the following story idea into a short script for a 30–60 second vertical video.

Requirements:
- The script must be in ENGLISH.
- The tone should match the idea (if the idea feels sad, keep it sad; if it feels light, keep it light).
- The first part is a HOOK: 1–2 sentences that grab attention and make people want to keep watching.
- Then write 5–10 short LINES that tell the story in order.
- Finally write 1 CLOSER sentence that ends on a strong emotional beat or cliffhanger.

Return your answer as valid JSON exactly in this format:

{{
  "hook": "string",
  "lines": ["string", "string", "..."],
  "closer": "string"
}}

Story idea:
---
{idea}
---
""".strip()


def generate_script(idea: str) -> Dict[str, Any]:
    user_prompt = build_user_prompt(idea)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
    )

    raw_text = response.choices[0].message.content

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from model:", e)
        print("Raw text:", raw_text)
        raise

    return data


# ---------------------------------------------------------------------------
# Text-to-speech helpers
# ---------------------------------------------------------------------------

def tts_to_file(text: str, out_path: str) -> str:
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        response_format="mp3",
    ) as resp:
        resp.stream_to_file(out_path)

    return out_path


# ---------------------------------------------------------------------------
# Timeline / caption utilities
# ---------------------------------------------------------------------------

def chunk_timeline_entry(entry: Dict[str, Any], words_per_chunk: int = 3) -> List[Dict[str, Any]]:
    """
    Takes a timeline entry like:
      {"text": "...", "start_ms": 0, "end_ms": 3000, "type": "line"}
    and splits it into smaller chunks of ~words_per_chunk each,
    distributing the time evenly.
    """
    text = entry["text"]
    words = text.split()
    if not words:
        return [entry]

    num_chunks = max(1, math.ceil(len(words) / words_per_chunk))
    total_ms = entry["end_ms"] - entry["start_ms"]
    chunk_ms = total_ms / num_chunks

    chunks: List[Dict[str, Any]] = []
    for i in range(num_chunks):
        start_idx = i * words_per_chunk
        end_idx = min((i + 1) * words_per_chunk, len(words))
        chunk_words = words[start_idx:end_idx]
        if not chunk_words:
            continue

        chunk_text = " ".join(chunk_words)
        start_ms = entry["start_ms"] + int(i * chunk_ms)
        end_ms = entry["start_ms"] + int((i + 1) * chunk_ms)

        chunk_entry = {
            "type": entry.get("type", "line"),
            "text": chunk_text,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        chunks.append(chunk_entry)

    return chunks


def build_chunked_timeline(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunked: List[Dict[str, Any]] = []
    for entry in timeline:
        if entry["type"] in ("hook", "closer"):
            chunked.extend(chunk_timeline_entry(entry, words_per_chunk=4))
        else:
            chunked.extend(chunk_timeline_entry(entry, words_per_chunk=3))
    return chunked


# ---------------------------------------------------------------------------
# Audio synthesis + timeline building
# ---------------------------------------------------------------------------

def synthesize_script_audio(script: Dict[str, Any], out_dir: str = "audio_out") -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    # Collect segments as (type, text, duration_ms)
    segments_meta: List[Dict[str, Any]] = []

    # Hook
    hook_text = script["hook"]
    hook_path = os.path.join(out_dir, f"hook_{uuid.uuid4().hex}.mp3")
    tts_to_file(hook_text, hook_path)
    hook_audio = AudioSegment.from_file(hook_path)
    segments_meta.append({"type": "hook", "text": hook_text, "duration_ms": len(hook_audio)})

    # Lines
    for i, line in enumerate(script["lines"]):
        line_path = os.path.join(out_dir, f"line_{i}_{uuid.uuid4().hex}.mp3")
        tts_to_file(line, line_path)
        line_audio = AudioSegment.from_file(line_path)
        segments_meta.append({"type": "line", "text": line, "duration_ms": len(line_audio)})

    # Closer
    closer_text = script["closer"]
    closer_path = os.path.join(out_dir, f"closer_{uuid.uuid4().hex}.mp3")
    tts_to_file(closer_text, closer_path)
    closer_audio = AudioSegment.from_file(closer_path)
    segments_meta.append({"type": "closer", "text": closer_text, "duration_ms": len(closer_audio)})

    # Merge into one file
    full = AudioSegment.silent(duration=0)
    for seg in [hook_audio] + \
               [AudioSegment.from_file(os.path.join(out_dir, f)) for f in []] + \
               []:
        # This dummy comprehension is removed below; we'll just rebuild full explicitly.
        pass

    # Rebuild full from meta by reloading files in order
    full = AudioSegment.silent(duration=0)
    cursor = 0
    timeline: List[Dict[str, Any]] = []

    # We need to know which file corresponds to which segment. To keep it simple,
    # reload in the same order we created them.
    audio_paths: List[str] = [hook_path]
    audio_paths.extend(
        [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("line_")]
    )
    audio_paths = sorted(audio_paths)  # ensure deterministic order

    # But we already have durations in segments_meta, so just loop that instead.
    # We'll rebuild full using the same TTS outputs in the same order:
    # hook -> all lines (in order) -> closer.
    full = AudioSegment.silent(duration=0)
    cursor = 0
    timeline = []

    # Rebuild in the same order we appended: hook, lines, closer
    for seg in segments_meta:
        # Reload by type/order: simple but explicit
        if seg["type"] == "hook":
            audio = hook_audio
        elif seg["type"] == "closer":
            audio = closer_audio
        else:
            # For lines, re-synthesize from text to avoid tracking each path.
            # If you prefer to avoid extra calls, you can track paths in segments_meta.
            line_path = os.path.join(out_dir, f"line_rebuild_{uuid.uuid4().hex}.mp3")
            tts_to_file(seg["text"], line_path)
            audio = AudioSegment.from_file(line_path)

        full += audio
        start_ms = cursor
        end_ms = cursor + len(audio)
        cursor = end_ms

        timeline.append(
            {
                "type": seg["type"],
                "text": seg["text"],
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )

    full_path = os.path.join(out_dir, "full_story.mp3")
    full.export(full_path, format="mp3")

    chunked_timeline = build_chunked_timeline(timeline)

    return {
        "timeline": chunked_timeline,
        "full_audio_path": full_path,
    }


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def create_video(
    script: Dict[str, Any],
    audio_info: Dict[str, Any],
    output_path: str = "storyshort_test.mp4",
) -> str:
    WIDTH, HEIGHT = 1080, 1920
    BG_COLOR = (10, 10, 15)

    audio_clip = AudioFileClip(audio_info["full_audio_path"])
    total_duration = audio_clip.duration

    background = (
        ColorClip(size=(WIDTH, HEIGHT), color=BG_COLOR)
        .with_duration(total_duration)
        .with_audio(audio_clip)
    )

    text_clips = []
    CAPTION_BOX_HEIGHT = HEIGHT // 3

    for t in audio_info["timeline"]:
        start_s = t["start_ms"] / 1000.0
        end_s = t["end_ms"] / 1000.0
        dur_s = end_s - start_s
        txt = t["text"]

        txt_clip = (
            TextClip(
                text=txt,
                font_size=60,
                font=r"C:\Windows\Fonts\arial.ttf",  # consider making this a constant
                color="white",
                method="caption",
                size=(WIDTH - 200, CAPTION_BOX_HEIGHT),
            )
            .with_start(start_s)
            .with_duration(dur_s)
            .with_position(("center", HEIGHT - CAPTION_BOX_HEIGHT - 150))
        )

        text_clips.append(txt_clip)

    video = CompositeVideoClip([background, *text_clips])

    video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        audio=True,
        ffmpeg_params=["-preset", "ultrafast"],
    )

    return output_path


# ---------------------------------------------------------------------------
# CLI / entrypoint
# ---------------------------------------------------------------------------

def main():
    idea = "Gojo loses his powers and has to live like a normal teacher in Tokyo."
    script = generate_script(idea)
    audio_info = synthesize_script_audio(script)

    print("Full audio saved to:", audio_info["full_audio_path"])
    for t in audio_info["timeline"]:
        print(t)

    video_path = create_video(script, audio_info, output_path="storyshort_test.mp4")
    print("Video saved to:", video_path)


if __name__ == "__main__":
    main()
