import os
import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient
from google.adk.tools import FunctionTool

AUTHORITATIVE_DOMAINS = [
    "skatteverket.se",
    "bolagsverket.se",
    "verksamt.se",
    "riksdagen.se",
    "arbetsgivarverket.se",
    "arbetsmiljoverket.se",
]

# ── Tavily: search + content in one call ──────────────────────────────────────

def tavily_search_swedish(query: str) -> dict:
    """
    Searches authoritative Swedish legal and government sources using Tavily.
    Returns structured results with URLs, snippets, and full page content.
    Use this for any question about Swedish tax, corporate law, labor law, or business registration.

    Args:
        query: The search query. Include Swedish legal terms for best results.

    Returns:
        A dict with keys:
          - results: list of {url, title, content, raw_content, score}
          - answer: Tavily's AI-generated summary (if available)
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    response = client.search(
        query=query,
        search_depth="advanced",
        include_raw_content=True,
        max_results=5,
        include_domains=AUTHORITATIVE_DOMAINS,
    )

    BINARY_EXTENSIONS = (".doc", ".docx", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx", ".zip")

    results = []
    for r in response.get("results", []):
        url = r.get("url", "")
        # Skip binary file URLs — they return garbled content and waste tokens
        if any(url.lower().endswith(ext) for ext in BINARY_EXTENSIONS):
            continue
        raw = (r.get("raw_content") or "")
        # Drop raw_content that looks like binary garbage (high ratio of non-ASCII)
        non_ascii = sum(1 for c in raw[:500] if ord(c) > 127)
        if len(raw) > 100 and non_ascii / min(len(raw), 500) > 0.3:
            raw = ""
        results.append({
            "url": url,
            "title": r.get("title", ""),
            "content": r.get("content", ""),
            "raw_content": raw[:2000],  # tighter cap to reduce token bloat
            "score": round(r.get("score", 0), 3),
        })

    return {
        "query": query,
        "results": results,
        "answer": response.get("answer", ""),
    }


# ── Fallback scraper: direct HTTP fetch ───────────────────────────────────────

async def scrape_url(url: str) -> dict:
    """
    Fallback: fetches and extracts clean text directly from a Swedish government URL.
    Only works on authoritative domains. Use when Tavily did not return full content for a specific URL.

    Args:
        url: Full URL to scrape (must be from an authoritative Swedish domain).

    Returns:
        A dict with keys: url, title, text, error.
    """
    if not any(domain in url for domain in AUTHORITATIVE_DOMAINS):
        return {
            "url": url,
            "title": "",
            "text": "",
            "error": f"Not an authoritative domain. Allowed: {', '.join(AUTHORITATIVE_DOMAINS)}",
        }

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            response = await client.get(url, headers={"Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8"})
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else ""
        main = soup.find("main") or soup.find("article") or soup.find("div", {"id": "content"}) or soup.body
        text = " ".join(main.get_text(separator=" ", strip=True).split()) if main else ""

        return {"url": url, "title": title, "text": text[:4000], "error": ""}

    except Exception as e:
        return {"url": url, "title": "", "text": "", "error": str(e)}


tavily_search_tool = FunctionTool(func=tavily_search_swedish)
scrape_url_tool = FunctionTool(func=scrape_url)
