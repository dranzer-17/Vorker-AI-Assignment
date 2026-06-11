# SwedLex — Swedish Legal Compliance AI Agent

> An AI assistant that answers Swedish corporate law, tax, and labor questions with cited sources — built on Google ADK, Pinecone, and live government data.

---

## What is this?

Running a business in Sweden means navigating a maze of laws — the Companies Act (ABL), Skatteverket tax rules, LAS labor regulations, Bolagsverket registration requirements. Getting it wrong is expensive.

SwedLex is an AI agent that reads the actual legal texts, searches live government sources, and gives you grounded, cited answers — not hallucinated guesses.

---

## How it works (the pipeline)

```mermaid
graph TD
    Q([User Query]) --> SWEDLEX

    SWEDLEX["✦ swed_lex
    SequentialAgent"]

    SWEDLEX --> LR["✦ language_router_agent
    Detects language, classifies domain
    Extracts user-provided URLs"]

    SWEDLEX --> PR["✦ parallel_research
    ParallelAgent"]

    SWEDLEX --> SC["✦ scraper_agent
    Fetches full HTML pages
    or user-provided PDF documents"]

    SWEDLEX --> SY["✦ synthesis_agent
    SwedLex — reads ALL upstream
    context, writes grounded answer"]

    PR --> RAG["✦ rag_corporate_law_agent
    Searches 5982 Pinecone chunks
    law · reasoning · committee layers"]

    PR --> SE["✦ search_agent
    Live Tavily search
    Skatteverket · Riksdagen · Bolagsverket"]

    RAG -. search_swedish_law_docs .-> T1[("Pinecone
    swedish-law-structured
    5982 vectors")]

    SE -. tavily_search_swedish .-> T2[("Tavily
    Swedish gov sources")]

    SC -. scrape_url .-> T3[("httpx + PyPDF2
    HTML and PDF")]

    LR -->|"LANGUAGE, DOMAIN, URL"| SY
    RAG -->|"law nodes + references"| SY
    SE -->|"search results + URLs"| SY
    SC -->|"full page or PDF text"| SY

    SY --> ANS(["Structured Answer
    paragraph references, sources,
    legal basis, follow-up questions"])

    style SWEDLEX fill:#1a2744,stroke:#4a9eff,color:#fff
    style PR fill:#1a2744,stroke:#4a9eff,color:#fff
    style LR fill:#1e2d1e,stroke:#4aaa4a,color:#fff
    style RAG fill:#1e2d1e,stroke:#4aaa4a,color:#fff
    style SE fill:#1e2d1e,stroke:#4aaa4a,color:#fff
    style SC fill:#1e2d1e,stroke:#4aaa4a,color:#fff
    style SY fill:#2d1a2d,stroke:#aa4aff,color:#fff
    style T1 fill:#0d1117,stroke:#555,color:#aaa
    style T2 fill:#0d1117,stroke:#555,color:#aaa
    style T3 fill:#0d1117,stroke:#555,color:#aaa
```

Every answer goes through four stages:

1. **Router** — figures out what language you're asking in and what domain the question belongs to (tax, corporate law, labor, registration)
2. **Parallel research** — while the RAG agent queries 5,982 pre-indexed chunks from Swedish legal PDFs, the search agent simultaneously fires off a live Tavily search against Skatteverket, Riksdagen, Bolagsverket — both finish before either waits for the other
3. **Scraper** — grabs the full text of the top-ranked pages (or your uploaded PDF document)
4. **Synthesis** — combines everything and writes a structured answer with specific §§ citations, source URLs, and a disclaimer if you should really be talking to a lawyer

---

## What can you ask it?

**Corporate law**
- "What are the requirements for a hembudsförbehåll in a Swedish AB?"
- "Explain förköpsförbehåll vs samtyckesförbehåll"
- "What does ABL 4 kap §27 say about share transfer restrictions?"

**Tax**
- "What VAT rate applies to SaaS sold to a Norwegian B2B customer?"
- "How does Swedish reverse-charge VAT work for digital services to Germany?"
- "When do I need to register for F-skatt?"

**Labor law**
- "How do I calculate karensavdrag for a part-time employee?"
- "What are the minimum notice periods under LAS?"
- "Can I include a non-compete clause in a Swedish employment contract?"

**Website / document compliance**
- Paste any URL: "Does this page comply with Swedish privacy rules? https://example.se"
- Paste a PDF link: "Review this shareholders agreement for compliance issues: https://example.com/avtal.pdf"

---

## What's under the hood

| Component | What it does |
|-----------|-------------|
| **Google ADK 2.2.0** | Agent orchestration framework — SequentialAgent, ParallelAgent, LlmAgent |
| **Gemini 2.5 Flash** | The language model powering all four agents |
| **Pinecone** | Cloud vector database holding 5,982 structured legal document chunks |
| **Sentence Transformers** | `paraphrase-multilingual-MiniLM-L12-v2` — embeds both Swedish and English queries into the same vector space |
| **Tavily** | Search API that returns content from Skatteverket, Riksdagen, Bolagsverket, and other Swedish government sources |
| **Structural RAG** | Documents are split into three typed layers: `law_text` (the actual §§), `reasoning` (why the rule exists), `committee` (legislative debate history) — queries route to the right layer |

---

## The RAG setup

The indexed documents cover:
- `docs/sweden.pdf` — Main Swedish Companies Act (ABL, 893 pages)
- `docs/download.pdf` — Additional Swedish corporate law documents
- **SFS 2026:495** — Latest amendment to ABL Chapter 4 §41

When you ask about hembudsförbehåll, the RAG agent doesn't just do a keyword search. It detects your intent:

- Asking *"what are the requirements?"* → searches `law_text` nodes (exact §§ paragraphs)
- Asking *"why does this rule exist?"* → searches `reasoning` nodes (legislative intent, Skälen för)
- Asking *"how was this debated?"* → searches `committee` nodes (Kommitténs förslag, Remissinstanserna)

This means you get the actual statutory text, not just summaries.

---

## Project structure

```
ADK_Legal/
├── __init__.py          ← ADK entry point (exports root_agent)
├── agent.py             ← Root pipeline (SequentialAgent + all sub-agents)
├── tools.py             ← Tavily search tool + HTML/PDF scraper
├── build_rag.py         ← One-time script to index PDFs into Pinecone
├── rag_agent/
│   ├── __init__.py      ← Sub-package entry point
│   ├── agent.py         ← RAG LlmAgent with Pinecone tool
│   └── rag_tool.py      ← Intent-aware 3-layer Pinecone retrieval
├── docs/                ← Source PDFs for indexing (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running it

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and fill in your API keys

# 4. Add source PDFs to docs/
#    docs/sweden.pdf   — Swedish Companies Act (ABL)
#    docs/download.pdf — Additional legal documents

# 5. Build the Pinecone RAG index (run once, ~5 minutes)
python build_rag.py

# 6. Launch
adk web .
# Open http://localhost:8000 → select ADK_Legal
```

---

## Design decisions worth knowing

**Why parallel research?** The RAG query and web search are completely independent — running them sequentially would add 15-20 seconds of dead wait time per query. Running them together means the total time is `max(rag_time, search_time)` instead of `rag_time + search_time`.

**Why Pinecone over ChromaDB?** Production-ready cloud storage, persistent across sessions, no local disk management. The index survives restarts.

**Why structural RAG instead of naive chunking?** Swedish legal PDFs have a clear three-layer structure — statutory text, legislative reasoning, and committee debate. Chunking naively loses that signal. By tagging each chunk with its document layer type, the retriever can answer "what does the law say" and "why was it written that way" with different source material.

**Why Tavily instead of Google Search?** Google's grounding tool modifies the LLM request internally and can't return structured URLs for pipeline passing. Tavily returns clean JSON with scores, URLs, and page content that downstream agents can act on.

---

## Limitations

- Answers are grounded in indexed documents and live search, but **this is not legal advice**. Always consult a licensed Swedish lawyer (advokat) or certified accountant before making legal or financial decisions.
- The RAG index covers the documents listed above. For very recent SFS amendments (post-2025), the live web search path is more reliable.
- PDF analysis works on direct PDF URLs. Native file upload requires additional frontend integration.
