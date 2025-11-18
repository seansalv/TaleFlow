import os
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
import json
import uuid


load_dotenv()  # load from .env

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


def generate_script(idea: str) -> dict:
    user_prompt = build_user_prompt(idea)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or whatever model you want
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

def tts_to_file(text: str, out_path: str):
    # Use streaming helper and save directly to file
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",  # or "tts-1" if you prefer
        voice="alloy",
        input=text,
        response_format="mp3",    # optional, mp3 is default, but explicit is nice
    ) as resp:
        resp.stream_to_file(out_path)

    return out_path

def synthesize_script_audio(script: dict, out_dir: str = "audio_out"):
    os.makedirs(out_dir, exist_ok=True)

    segments = []

    # 1) Hook
    hook_text = script["hook"]
    hook_path = os.path.join(out_dir, f"hook_{uuid.uuid4().hex}.mp3")
    tts_to_file(hook_text, hook_path)
    hook_audio = AudioSegment.from_file(hook_path)
    segments.append({"type": "hook", "text": hook_text, "path": hook_path, "audio": hook_audio})

    # 2) Lines
    for i, line in enumerate(script["lines"]):
        line_path = os.path.join(out_dir, f"line_{i}_{uuid.uuid4().hex}.mp3")
        tts_to_file(line, line_path)
        line_audio = AudioSegment.from_file(line_path)
        segments.append({"type": "line", "index": i, "text": line, "path": line_path, "audio": line_audio})

    # 3) Closer
    closer_text = script["closer"]
    closer_path = os.path.join(out_dir, f"closer_{uuid.uuid4().hex}.mp3")
    tts_to_file(closer_text, closer_path)
    closer_audio = AudioSegment.from_file(closer_path)
    segments.append({"type": "closer", "text": closer_text, "path": closer_path, "audio": closer_audio})

    # Optional: merge into one file so you can play it easily
    full = AudioSegment.silent(duration=0)
    for seg in segments:
        full += seg["audio"]

    full_path = os.path.join(out_dir, "full_story.mp3")
    full.export(full_path, format="mp3")

    # Also return timing info (durations in ms)
    timeline = []
    cursor = 0
    for seg in segments:
        dur = len(seg["audio"])
        timeline.append({
            "type": seg["type"],
            "text": seg["text"],
            "start_ms": cursor,
            "end_ms": cursor + dur,
        })
        cursor += dur

    return {
        "segments": segments,
        "timeline": timeline,
        "full_audio_path": full_path,
    }

if __name__ == "__main__":
    idea = "Gojo loses his powers and has to live like a normal teacher in Tokyo."
    script = generate_script(idea)
    audio_info = synthesize_script_audio(script)

    print("Full audio saved to:", audio_info["full_audio_path"])
    print("Timeline:")
    for t in audio_info["timeline"]:
        print(t)

