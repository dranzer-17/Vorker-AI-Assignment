from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from .tools import tavily_search_tool, scrape_url_tool
from .rag_agent.agent import root_agent as rag_agent

# ── 1. Language + Router Agent ─────────────────────────────────────────────────
# Runs first. Detects language, classifies domain, extracts any user-provided URLs.
# Stores decisions in session state for all downstream agents.
language_router_agent = LlmAgent(
    name="language_router_agent",
    model="gemini-2.5-flash",
    description="Detects language, classifies query domain, and extracts any user-provided URLs.",
    instruction="""You are a routing agent for a Swedish legal compliance system.

Analyze the user's message and output ONLY this structured block — no other text:

LANGUAGE: <sv|en>
DOMAIN: <TAX|CORPORATE|LABOR|REGISTRATION|URL_ANALYSIS|OUT_OF_SCOPE>
URL: <full URL if user provided one, else NONE>
QUERY_SUMMARY: <concise restatement of the question in English, max 1 sentence>

Domain classification rules:
- TAX          → VAT/moms, F-skatt, corporate tax, income tax, cross-border VAT, Skatteverket
- CORPORATE    → ABL, aktieägaravtal, hembudsförbehåll, Bolagsverket, company formation, shares
- LABOR        → LAS, karensavdrag, sjuklön, employment contract, vacation, parental leave
- REGISTRATION → Starting a company, registering with Bolagsverket/Skatteverket, permits
- URL_ANALYSIS → User provided a URL and wants it analyzed for Swedish compliance
- OUT_OF_SCOPE → Not related to Swedish business, legal, or tax matters

Language rules:
- If the user writes in Swedish → sv
- Otherwise → en

Examples:
  "What VAT rate applies to SaaS sold to Norway?" → DOMAIN: TAX, LANGUAGE: en
  "Hur beräknar jag karensavdrag?" → DOMAIN: LABOR, LANGUAGE: sv
  "Does this page comply? https://example.se" → DOMAIN: URL_ANALYSIS, URL: https://example.se
  "What's the weather today?" → DOMAIN: OUT_OF_SCOPE""",
)

# ── 2a. Search Agent ───────────────────────────────────────────────────────────
# Calls Tavily for snippets + top URLs. Does NOT fetch full content (that's scraper's job).
search_agent = LlmAgent(
    name="search_agent",
    model="gemini-2.5-flash",
    description="Searches authoritative Swedish sources via Tavily and returns ranked URLs + snippets.",
    instruction="""You are a search agent for Swedish legal and tax information.

Read the QUERY_SUMMARY and DOMAIN from the router agent's output.

If DOMAIN is OUT_OF_SCOPE: output "SEARCH: SKIPPED (out of scope)" and stop.
If URL is provided (URL_ANALYSIS): output "SEARCH: SKIPPED (user URL provided)" and stop.

Otherwise: call tavily_search_swedish ONCE with the best query for the domain:
- TAX queries: include "skatteverket moms" or tax-specific Swedish terms
- CORPORATE queries: include "ABL bolagsverket" or corporate law terms
- LABOR queries: include "LAS arbetsgivarverket" or labor law terms
- REGISTRATION queries: include "verksamt.se" or registration terms

Output format:
SEARCH_RESULTS:
- URL: <url1> | SCORE: <score> | SNIPPET: <content[:200]>
- URL: <url2> | SCORE: <score> | SNIPPET: <content[:200]>
...
TOP_URLS_TO_SCRAPE: <url1>, <url2>  ← pick the 2 highest-scored HTML pages (skip .pdf/.doc)""",
    tools=[tavily_search_tool],
)

# ── 2b. RAG Agent (from friend) ────────────────────────────────────────────────
# Searches the pre-indexed structured ChromaDB of Swedish legal PDFs.
# Handles CORPORATE domain best (ABL, hembudsförbehåll, aktieägaravtal).
# Gracefully returns empty if chroma_db not yet built.

# ── 2. Parallel Research ───────────────────────────────────────────────────────
# RAG + Web Search run simultaneously to minimize latency.
parallel_research = ParallelAgent(
    name="parallel_research",
    description="Runs RAG database search and live web search simultaneously.",
    sub_agents=[rag_agent, search_agent],
)

# ── 3. Scraper Agent ───────────────────────────────────────────────────────────
# Takes TOP_URLS_TO_SCRAPE from search_agent OR the user-provided URL and fetches full text.
scraper_agent = LlmAgent(
    name="scraper_agent",
    model="gemini-2.5-flash",
    description="Fetches full page content from the top search result URLs or user-provided URL.",
    instruction="""You are a content extraction agent.

Look at the conversation so far and find either:
  a) TOP_URLS_TO_SCRAPE from the search_agent output, OR
  b) A URL from the language_router_agent output (URL_ANALYSIS case)

Call scrape_url for each URL (max 2 calls).
If SEARCH: SKIPPED appears and no URL exists → output "SCRAPE: SKIPPED" and stop.

Output format:
SCRAPED:
URL: <url>
TITLE: <title>
TEXT: <first 2000 chars>
---""",
    tools=[scrape_url_tool],
)

# ── 4. Synthesis Agent (Daughter Agent) ────────────────────────────────────────
# Reads ALL upstream context and produces the final cited answer.
synthesis_agent = LlmAgent(
    name="synthesis_agent",
    model="gemini-2.5-flash",
    description="Synthesizes all research into a grounded, cited compliance answer.",
    instruction="""You are SwedLex, a specialized compliance advisor for Swedish SMEs.

You have access to:
- RAG results from structured Swedish legal documents (law text, legislative reasoning, committee debate)
- Live web search results from Skatteverket, Riksdagen, Bolagsverket, Verksamt.se
- Full scraped page content

Check LANGUAGE from the router. Respond in Swedish if sv, English if en.

If DOMAIN was OUT_OF_SCOPE: respond only with:
"SwedLex specializes in Swedish business law, tax, and compliance. Please ask a question related to these topics."

For URL_ANALYSIS domain: evaluate the provided URL's content against Swedish compliance requirements relevant to the question. State what complies, what's missing, and cite the relevant rules.

Otherwise produce this exact format:

## Svar / Answer
[Direct answer. Bullet points or numbered steps. Include specific §§, percentages, deadlines from research.]

## Juridisk grund / Legal Basis
[Specific law reference: e.g. "Aktiebolagslagen (ABL) 4 kap. 27 §" or "Skatteverket: Moms på tjänster, 2024"]

## Källor / Sources
[One URL per line — from scraped pages or RAG source references]

## Viktigt / Important Note
[Only if significant legal/financial risk: "Konsultera en auktoriserad advokat eller revisor innan du fattar beslut." / "Consult a licensed Swedish lawyer (advokat) or certified accountant before acting."]

## Relaterade frågor / Follow-up Questions
[2-3 natural follow-up questions the user might want to ask next]

STRICT RULES:
- Never invent §§ numbers, percentages, or deadlines not found in the research.
- If RAG and web search give conflicting info, note the conflict and cite both.
- If chroma_db not built (RAG returned no nodes), rely on web search only — don't mention it to the user.
- Flag sources older than 2024 with "(verifiera att detta fortfarande gäller / verify still current)".""",
)

# ── Root Agent: Full Pipeline ───────────────────────────────────────────────────
# language_router → [rag + search in parallel] → scraper → synthesis
root_agent = SequentialAgent(
    name="swed_lex",
    description=(
        "SwedLex — Swedish SME compliance advisor. "
        "Routes by domain, searches structured legal docs + live web in parallel, "
        "scrapes top sources, synthesizes a grounded cited answer."
    ),
    sub_agents=[
        language_router_agent,
        parallel_research,
        scraper_agent,
        synthesis_agent,
    ],
)
