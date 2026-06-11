"""
rag_tool.py — Pinecone-backed structured RAG tool

Intent-aware retrieval across 3 document layers:
  law_text   → exact §§ paragraphs ("what the rule IS")
  reasoning  → Skälen för ("WHY the rule exists")
  committee  → Kommitténs förslag / Remissinstanserna ("historical debate")
"""

import os
from google.adk.tools import FunctionTool

INDEX_NAME = "swedish-law-structured"

_embed_model = None
_pinecone_index = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embed_model


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        existing = [idx.name for idx in pc.list_indexes()]
        if INDEX_NAME not in existing:
            return None
        _pinecone_index = pc.Index(INDEX_NAME)
    return _pinecone_index


def _classify_intent(query: str, search_mode: str) -> str:
    if search_mode in ("law", "reasoning", "committee", "all"):
        return {"law": "law_text", "reasoning": "reasoning", "committee": "committee"}.get(search_mode, "all")

    q = query.lower()
    law_score = sum(1 for s in [
        "requirement", "krav", "skall", "ska", "must", "§", "paragraph",
        "section", "rule", "artikel", "innehåll", "framgå", "föreskriva",
        "specify", "contains", "include", "hembudsförbehåll innehåll",
        "förköpsförbehåll krav", "samtyckesförbehåll"
    ] if s in q)
    reasoning_score = sum(1 for s in [
        "why", "varför", "purpose", "syfte", "intent", "skäl", "explain",
        "background", "bakgrund", "meaning", "innebär", "interpretation",
        "tolkning", "significance", "motivering"
    ] if s in q)
    committee_score = sum(1 for s in [
        "history", "debate", "originally", "proposed", "committee",
        "kommitté", "sou", "remiss", "alternative", "rejected",
        "changed from", "old law", "previous", "formerly", "reform"
    ] if s in q)

    if committee_score > law_score and committee_score > reasoning_score:
        return "committee"
    if reasoning_score > law_score:
        return "reasoning"
    return "law_text"


def search_swedish_law_docs(query: str, search_mode: str = "auto") -> dict:
    """
    Search the structured Swedish legal document database (Pinecone).

    Use for questions about:
    - Hembudsförbehåll (right of first refusal) in Swedish AB — ABL 4 kap §27-§36
    - Förköpsförbehåll (pre-emption rights)
    - Samtyckesförbehåll (consent clauses)
    - Aktieägaravtal (shareholders agreement) requirements
    - Aktiebolagslagen (ABL) — any chapter or paragraph
    - SFS 2026:495 amendment to ABL Chapter 4 §41
    - Legislative reasoning (Skälen för) and committee debate history

    The search is TYPED — retrieves the right document layer:
    - Exact legal requirements → law_text nodes (§ paragraphs)
    - Why a rule exists / intent → reasoning nodes (Skälen för)
    - Historical debate → committee nodes (Kommitténs förslag)

    For complex questions call TWICE:
      1. search_mode="law"       → exact rule text
      2. search_mode="reasoning" → legislative intent

    Args:
        query: Question in English or Swedish. Include legal terms.
               e.g. "hembudsförbehåll innehåll krav §28 bolagsordning"
        search_mode: "auto" | "law" | "reasoning" | "committee" | "all"

    Returns:
        dict with keys: query, intent, nodes, answer_hint, sources_found
    """
    index = _get_index()
    if index is None:
        return {
            "query": query,
            "intent": "unknown",
            "nodes": [],
            "answer_hint": "RAG database not found. Run: python build_rag.py",
            "sources_found": [],
        }

    model = _get_embed_model()
    query_vector = model.encode(query).tolist()
    intent = _classify_intent(query, search_mode)

    nodes_by_layer = {}
    try:
        if intent == "all":
            raw = index.query(vector=query_vector, top_k=8, include_metadata=True)
            nodes_by_layer["mixed"] = _parse_matches(raw.matches)
        else:
            raw_primary = index.query(
                vector=query_vector, top_k=5,
                filter={"type": {"$eq": intent}},
                include_metadata=True,
            )
            nodes_by_layer[intent] = _parse_matches(raw_primary.matches)

            if intent != "law_text":
                raw_law = index.query(
                    vector=query_vector, top_k=2,
                    filter={"type": {"$eq": "law_text"}},
                    include_metadata=True,
                )
                nodes_by_layer["law_text"] = _parse_matches(raw_law.matches)

    except Exception as e:
        return {"query": query, "intent": intent, "nodes": [],
                "answer_hint": f"Pinecone error: {e}", "sources_found": []}

    seen, all_nodes = set(), []
    for nodes in nodes_by_layer.values():
        for n in nodes:
            if n["text"] not in seen:
                seen.add(n["text"])
                all_nodes.append(n)

    if not all_nodes:
        return {"query": query, "intent": intent, "nodes": [],
                "answer_hint": f"No results for '{query}'. Try search_mode='all'.",
                "sources_found": []}

    law_nodes = [n for n in all_nodes if n["type"] == "law_text"]
    reason_nodes = [n for n in all_nodes if n["type"] == "reasoning"]
    committee_nodes = [n for n in all_nodes if n["type"] == "committee"]

    hint_parts = []
    if law_nodes:
        hint_parts.append(f"RULE TEXT ({len(law_nodes)} passages):")
        for n in law_nodes[:3]:
            hint_parts.append(f"  [{n['header']}] {n['text'][:300]}...")
    if reason_nodes:
        hint_parts.append(f"\nLEGISLATIVE INTENT ({len(reason_nodes)} passages):")
        for n in reason_nodes[:2]:
            hint_parts.append(f"  [{n['header']}] {n['text'][:300]}...")
    if committee_nodes:
        hint_parts.append(f"\nCOMMITTEE DEBATE ({len(committee_nodes)} passages):")
        for n in committee_nodes[:2]:
            hint_parts.append(f"  [{n['header']}] {n['text'][:200]}...")

    return {
        "query": query,
        "intent": intent,
        "nodes": all_nodes,
        "answer_hint": "\n".join(hint_parts),
        "sources_found": list({n["source"] for n in all_nodes}),
    }


def _parse_matches(matches) -> list[dict]:
    return [{
        "text": (m.metadata or {}).get("text", ""),
        "type": (m.metadata or {}).get("type", "unknown"),
        "chapter": (m.metadata or {}).get("chapter", ""),
        "header": (m.metadata or {}).get("header", ""),
        "source": (m.metadata or {}).get("source", ""),
        "relevance_pct": round(m.score * 100, 1),
    } for m in matches]


search_swedish_law_docs_tool = FunctionTool(func=search_swedish_law_docs)
