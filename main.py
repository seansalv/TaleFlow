import os
from dotenv import load_dotenv
from openai import OpenAI
import json

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


if __name__ == "__main__":
    idea = "Gojo loses his powers and has to live like a normal teacher in Tokyo."
    script = generate_script(idea)
    print(json.dumps(script, indent=2, ensure_ascii=False))
