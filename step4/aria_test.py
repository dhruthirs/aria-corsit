"""
ARIA -- basic test version, running on Google's Gemini API (free tier).

Deliberately simple: no server, no database, no retrieval logic yet.
Just proof that the personality file + club facts + member dossiers actually
work together and sound right.

Setup:
    pip install google-genai
    Get a free key at https://aistudio.google.com/app/apikey (no credit card needed)
    Windows (same terminal, before running):  set GEMINI_API_KEY=your-key-here
    Mac/Linux:                                 export GEMINI_API_KEY=your-key-here

Run:
    python aria_test.py
"""

import json
import os
from google import genai
from google.genai import types

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")


def load_knowledge():
    with open(os.path.join(KNOWLEDGE_DIR, "aria_personality.md"), "r") as f:
        personality = f.read()
    with open(os.path.join(KNOWLEDGE_DIR, "club_facts.json"), "r") as f:
        club_facts = json.load(f)
    with open(os.path.join(KNOWLEDGE_DIR, "member_dossiers.json"), "r") as f:
        member_dossiers = json.load(f)
    return personality, club_facts, member_dossiers


def build_system_prompt(personality, club_facts, member_dossiers):
    facts_text = "\n".join(f"- {f['topic']}: {f['content']}" for f in club_facts)

    members_text = "\n\n".join(
        f"- {m['name']} ({m.get('year_branch', 'unknown year')}): "
        + json.dumps({k: v for k, v in m.items() if k not in ("name", "year_branch")})
        for m in member_dossiers
    )

    return f"""{personality}

CLUB FACTS:
{facts_text}

MEMBER DOSSIERS:
{members_text}
"""


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set your GEMINI_API_KEY environment variable first, in THIS terminal window, before running the script.")
        print("Windows:    set GEMINI_API_KEY=your-key-here")
        print("Mac/Linux:  export GEMINI_API_KEY=your-key-here")
        print("Get a free key at https://aistudio.google.com/app/apikey")
        return

    client = genai.Client(api_key=api_key)

    personality, club_facts, member_dossiers = load_knowledge()
    system_prompt = build_system_prompt(personality, club_facts, member_dossiers)

    chat = client.chats.create(
        model="gemini-3.6-flash",
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    print("ARIA is online. Type a question (or 'quit' to exit).\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        response = chat.send_message(user_input)
        print(f"ARIA: {response.text}\n")


if __name__ == "__main__":
    main()
