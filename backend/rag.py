"""
Retrieval-Augmented Generation (RAG) for ARIA.

Instead of stuffing every club fact and every member dossier into every single
prompt (which is what aria_test.py did, fine for testing, wasteful for real use),
we build a searchable index once at startup. At question time, we retrieve only
the handful of facts/members most relevant to what was actually asked.

Uses TF-IDF + cosine similarity -- lightweight, no external embedding API calls
needed, runs instantly even on a laptop.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")


class KnowledgeBase:
    def __init__(self):
        self.club_facts = self._load_json("club_facts.json")
        self.member_dossiers = self._load_json("member_dossiers.json")

        # Build one combined searchable corpus: club facts + member dossiers
        self.entries = []  # each item: {"type": "fact"/"member", "label": str, "text": str, "raw": dict}

        for fact in self.club_facts:
            self.entries.append({
                "type": "fact",
                "label": fact["topic"],
                "text": f"{fact['topic']}: {fact['content']}",
                "raw": fact,
            })

        for member in self.member_dossiers:
            searchable_text = f"{member['name']} {member.get('year_branch', '')} " + \
                " ".join(str(v) for k, v in member.items() if k not in ("name", "year_branch"))
            self.entries.append({
                "type": "member",
                "label": member["name"],
                "text": searchable_text,
                "raw": member,
            })

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([e["text"] for e in self.entries])

    def _load_json(self, filename):
        with open(os.path.join(KNOWLEDGE_DIR, filename), "r") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = 5):
        """Return the top_k most relevant entries (facts and/or members) for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                continue  # skip completely irrelevant entries
            results.append(self.entries[i])
        return results

    def get_member_by_name(self, name: str):
        """Exact/partial name lookup -- useful fallback alongside semantic search."""
        name_lower = name.lower()
        for member in self.member_dossiers:
            if name_lower in member["name"].lower():
                return member
        return None

    def load_personality(self):
        with open(os.path.join(KNOWLEDGE_DIR, "aria_personality.md"), "r") as f:
            return f.read()


knowledge_base = KnowledgeBase()
