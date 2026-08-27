"""
ARIA backend -- FastAPI server.

Run locally:
    pip install -r requirements.txt
    Set your key: copy .env.example to .env and fill in GEMINI_API_KEY
    uvicorn main:app --reload --port 8000

This single /chat endpoint serves BOTH chat and voice frontends -- voice mode
just means the frontend converts speech to text before sending it here, and
reads the response out loud after. The backend doesn't need to know or care
which mode sent the message, except for logging purposes (input_mode field).
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag import knowledge_base
from database import init_db, log_conversation, get_analytics

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ARIA - CorSIT Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend URL before going live
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSONALITY = knowledge_base.load_personality()


class ChatRequest(BaseModel):
    message: str
    input_mode: str = "text"  # "text" or "voice"


class ChatResponse(BaseModel):
    response: str
    retrieved_topics: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not set. Add it to backend/.env (see .env.example).",
        )

    # 1. Retrieve only what's relevant to this specific question
    retrieved = knowledge_base.retrieve(req.message, top_k=5)
    context_block = "\n".join(f"- [{r['type']}] {r['text']}" for r in retrieved)

    system_prompt = f"{PERSONALITY}\n\nRELEVANT INFO FOR THIS QUESTION:\n{context_block if context_block else 'Nothing specific matched -- answer from general CorSIT knowledge or admit you are not sure.'}"

    # 2. Generate the response
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=req.message,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    reply_text = response.text

    # 3. Log it
    topics = [r["label"] for r in retrieved]
    log_conversation(req.message, reply_text, topics, input_mode=req.input_mode)

    return ChatResponse(response=reply_text, retrieved_topics=topics)


@app.get("/analytics")
def analytics():
    return get_analytics()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "facts_loaded": len(knowledge_base.club_facts),
        "members_loaded": len(knowledge_base.member_dossiers),
    }
