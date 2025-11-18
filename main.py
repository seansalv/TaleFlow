import json
# from openai import OpenAI  # or whatever client you use

# client = OpenAI()

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

{
  "hook": "string",
  "lines": ["string", "string", "..."],
  "closer": "string"
}

Story idea:
---
{idea}
---
""".strip()


def generate_script(idea: str) -> dict:
    user_prompt = build_user_prompt(idea)


    raw_text = """{
      "hook": "He used to be the strongest sorcerer alive. Now he can’t even open a cursed door.",
      "lines": [
        "Gojo wakes up to an alarm instead of a mission call.",
        "No blindfold, no uniform. Just a wrinkled dress shirt and a stack of graded quizzes.",
        "The students only know him as the weird new teacher who stares out the window too long.",
        "He reaches for cursed energy on instinct… and feels nothing.",
        "The city feels louder without the hum of spirits in the background.",
        "For the first time, he has to rely on train schedules, paychecks, and coffee to survive."
      ],
      "closer": "And on an ordinary Tuesday, something in the crowd looks back at him with eyes that remember his power."
    }"""

    # Parse JSON safely
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from model:", e)
        print("Raw text:", raw_text)
        raise

    # Basic sanity checks
    if not isinstance(data.get("hook"), str):
        raise ValueError("Missing or invalid 'hook'")
    if not isinstance(data.get("lines"), list):
        raise ValueError("Missing or invalid 'lines'")
    if not isinstance(data.get("closer"), str):
        raise ValueError("Missing or invalid 'closer'")

    return data


if __name__ == "__main__":
    idea = "Gojo loses his powers and has to live like a normal teacher in Tokyo."
    script = generate_script(idea)
    print(json.dumps(script, indent=2, ensure_ascii=False))
