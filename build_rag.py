"""
build_rag.py — Build structured Pinecone index from Swedish legal PDFs.

Run once from ADK_Legal/:
    python build_rag.py

PDFs must be in:  ADK_Legal/docs/   (sweden.pdf, download.pdf)
Index builds to:  Pinecone cloud            (swedish-law-structured)

Structural RAG — 3 document layers with type metadata:
  law_text   — §§ paragraphs (what the rule IS)
  reasoning  — Skälen för (WHY it exists)
  committee  — Kommitténs förslag / Remissinstanserna (historical debate)
"""

import os
import re
import sys
import json

_ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(_ROOT, "docs")
METADATA_FILE = os.path.join(_ROOT, "node_index.json")
INDEX_NAME = "swedish-law-structured"
EMBED_DIM = 384
BATCH_SIZE = 100


# ── PDF text extraction ────────────────────────────────────────────────────────

def extract_text(pdf_path: str) -> str:
    import PyPDF2
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    pages.append(t)
            except Exception:
                pages.append("")
    return "\n\f\n".join(pages)


# ── Structural node parser ─────────────────────────────────────────────────────

def parse_into_nodes(text: str, source: str) -> list[dict]:
    nodes = []
    node_id = 0
    current_chapter = ""
    current_section = ""
    current_type = "law_text"
    buffer, buffer_type, buffer_header = [], "law_text", ""

    chapter_pattern = re.compile(r'^(\d+)\s+kap\.\s+(.+)$')
    section_header_pattern = re.compile(r'^(\d+\.\d+(?:\.\d+)?)\s+(.{5,80})$')
    paragraph_start = re.compile(r'^(\d+)\s+§\s*(.*)$')
    reasoning_start = re.compile(r'^Skälen för', re.IGNORECASE)
    committee_start = re.compile(r'^Kommitténs förslag|^Remissinstanserna', re.IGNORECASE)
    proposal_start = re.compile(r'^Regeringens förslag:', re.IGNORECASE)

    def flush_buffer():
        nonlocal buffer, buffer_type, buffer_header, node_id
        if not buffer:
            return
        text_block = re.sub(r'\s+', ' ', ' '.join(buffer).strip())
        if len(text_block) > 60:
            nodes.append({
                "id": f"{source}_{node_id}",
                "type": buffer_type,
                "chapter": current_chapter,
                "section": current_section,
                "header": buffer_header,
                "text": text_block,
                "source": source,
            })
            node_id += 1
        buffer.clear()
        buffer_header = ""

    for line in text.split('\n'):
        line = line.strip()

        if re.match(r'^\f?\d+$', line) or line == '\f':
            continue

        m = chapter_pattern.match(line)
        if m:
            flush_buffer()
            current_chapter = f"Kap {m.group(1)}: {m.group(2)}"
            current_type = "law_text"
            continue

        m = section_header_pattern.match(line)
        if m and len(line) < 100:
            flush_buffer()
            current_section = line
            buffer_header = line
            continue

        if reasoning_start.match(line):
            flush_buffer(); buffer_type = "reasoning"; buffer_header = line[:80]; buffer = [line]; continue
        if committee_start.match(line):
            flush_buffer(); buffer_type = "committee"; buffer_header = line[:80]; buffer = [line]; continue
        if proposal_start.match(line):
            flush_buffer(); buffer_type = "proposal_summary"; buffer_header = line[:80]; buffer = [line]; continue

        m = paragraph_start.match(line)
        if m:
            flush_buffer()
            buffer_type = "law_text"
            buffer_header = f"§{m.group(1)} {current_chapter}"
            buffer = [line]
            continue

        if len(line) < 80 and re.match(r'^[A-ZÅÄÖ][a-zåäöA-ZÅÄÖ\s]+$', line) and not line.endswith('.') and len(line) > 10:
            flush_buffer(); buffer_type = current_type; buffer_header = line; continue

        if line:
            buffer.append(line)
        if sum(len(b) for b in buffer) > 1200:
            flush_buffer(); buffer_type = current_type

    flush_buffer()
    return nodes


def parse_sfs_amendment(text: str) -> list[dict]:
    return [{
        "id": "SFS2026_495_0",
        "type": "law_text",
        "chapter": "4 kap. §41 (2026 amendment)",
        "section": "SFS 2026:495",
        "header": "Amendment to ABL Chapter 4 §41 — wrong recipient of securities",
        "text": text[:3000],
        "source": "SFS2026-495",
    }]


# ── Main build ─────────────────────────────────────────────────────────────────

def build():
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone, ServerlessSpec
    from dotenv import load_dotenv
    import time

    load_dotenv(os.path.join(_ROOT, ".env"))

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not in .env"); sys.exit(1)

    if not os.path.isdir(DOCS_DIR):
        print(f"ERROR: {DOCS_DIR} not found"); sys.exit(1)

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"ERROR: No PDFs in {DOCS_DIR}"); sys.exit(1)

    print(f"PDFs found: {pdf_files}")
    print(f"Loading embedding model (paraphrase-multilingual-MiniLM-L12-v2)...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Parse
    all_nodes = []
    for pdf_file in sorted(pdf_files):
        path = os.path.join(DOCS_DIR, pdf_file)
        source = pdf_file.replace(".pdf", "")
        print(f"\nParsing {pdf_file}...")
        text = extract_text(path)
        print(f"  {len(text):,} chars extracted")
        nodes = parse_sfs_amendment(text) if "SFS2026" in pdf_file else parse_into_nodes(text, source)
        type_counts = {}
        for n in nodes:
            type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1
        print(f"  {len(nodes)} nodes → {type_counts}")
        all_nodes.extend(nodes)

    print(f"\nTotal: {len(all_nodes)} nodes")

    # Pinecone setup
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"Deleting existing index '{INDEX_NAME}'...")
        pc.delete_index(INDEX_NAME)

    print(f"Creating index '{INDEX_NAME}' (dim={EMBED_DIM}, cosine, serverless)...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index to be ready...")
    while not pc.describe_index(INDEX_NAME).status.get("ready", False):
        time.sleep(2)

    index = pc.Index(INDEX_NAME)

    # Embed + upsert
    print(f"Embedding and upserting to Pinecone...")
    for i in range(0, len(all_nodes), BATCH_SIZE):
        batch = all_nodes[i:i + BATCH_SIZE]
        vectors = model.encode([n["text"] for n in batch], show_progress_bar=False).tolist()
        index.upsert(vectors=[{
            "id": n["id"],
            "values": v,
            "metadata": {
                "type": n["type"],
                "chapter": n["chapter"][:500],
                "header": n["header"][:500],
                "source": n["source"],
                "text": n["text"][:2000],
            },
        } for n, v in zip(batch, vectors)])
        print(f"  {min(i + BATCH_SIZE, len(all_nodes))}/{len(all_nodes)} done")

    # Save local index for inspection
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump([{"id": n["id"], "type": n["type"], "chapter": n["chapter"],
                    "header": n["header"], "preview": n["text"][:100]}
                   for n in all_nodes], f, ensure_ascii=False, indent=2)

    stats = index.describe_index_stats()
    print(f"\n✅ Done — {stats.total_vector_count} vectors in Pinecone index '{INDEX_NAME}'")
    type_dist = {}
    for n in all_nodes:
        type_dist[n["type"]] = type_dist.get(n["type"], 0) + 1
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"   {t:25s}: {c} nodes")


if __name__ == "__main__":
    build()
