"""
Logs every ARIA conversation to SQLite -- this is what lets you show real
numbers afterward ("we logged and analyzed N visitor interactions") and lets
you build a live analytics screen for the booth if you want one later.
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "aria.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_mode TEXT NOT NULL,      -- 'text' or 'voice'
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            retrieved_topics TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_conversation(user_message: str, bot_response: str, retrieved_topics: list[str], input_mode: str = "text"):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO conversations (timestamp, input_mode, user_message, bot_response, retrieved_topics)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            input_mode,
            user_message,
            bot_response,
            ",".join(retrieved_topics),
        ),
    )
    conn.commit()
    conn.close()


def get_analytics():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) AS c FROM conversations").fetchone()["c"]
    voice_count = conn.execute("SELECT COUNT(*) AS c FROM conversations WHERE input_mode = 'voice'").fetchone()["c"]

    recent = conn.execute(
        "SELECT timestamp, input_mode, user_message FROM conversations ORDER BY id DESC LIMIT 10"
    ).fetchall()

    topic_rows = conn.execute("SELECT retrieved_topics FROM conversations").fetchall()
    topic_counts = {}
    for row in topic_rows:
        for topic in (row["retrieved_topics"] or "").split(","):
            topic = topic.strip()
            if topic:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    conn.close()
    return {
        "total_conversations": total,
        "voice_conversations": voice_count,
        "text_conversations": total - voice_count,
        "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
        "recent_conversations": [dict(r) for r in recent],
    }
