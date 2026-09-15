import json
import os
import re
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from urllib.parse import urlparse

from google import genai
from google.genai import types


MODEL = "gemini-3.1-flash-lite"
MAX_RESEARCH_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_QUESTION_LENGTH = 1_500
MAX_CLAIMS = 5
MAX_SOURCES = 6
request_times = defaultdict(deque)


def _allow_request(request_key="local"):
    """Limit expensive grounded searches within one running process."""
    now = time.monotonic()
    timestamps = request_times[request_key]
    while timestamps and now - timestamps[0] >= RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()
    if len(timestamps) >= MAX_RESEARCH_REQUESTS:
        return False
    timestamps.append(now)
    return True


def _extract_json(text):
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate:
        raise ValueError("The research assistant did not return JSON.")
    return json.loads(candidate)


def _is_http_url(value):
    parsed = urlparse(str(value))
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _validate_report(report):
    required = {"question", "as_of", "answer", "claims", "sources", "limitations"}
    missing = required.difference(report)
    if missing:
        raise ValueError(f"Research report is missing: {', '.join(sorted(missing))}.")

    sources = report["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("The report returned no verifiable sources.")

    source_ids = set()
    for source in sources:
        source_id = source.get("id")
        if not source_id or source_id in source_ids or not _is_http_url(source.get("url")):
            raise ValueError("The report contains an invalid or duplicate source.")
        source_ids.add(source_id)
        if not source.get("title") or not source.get("publisher"):
            raise ValueError("Every source needs a title and publisher.")

    claims = report["claims"]
    if not isinstance(claims, list) or not claims:
        raise ValueError("The report returned no source-backed claims.")
    if len(claims) > MAX_CLAIMS:
        raise ValueError("The report returned more claims than requested.")
    if len(sources) > MAX_SOURCES:
        raise ValueError("The report returned more sources than requested.")
    for claim in claims:
        citations = claim.get("source_ids")
        if not claim.get("text") or not citations or not set(citations).issubset(source_ids):
            raise ValueError("Every claim must include valid source citations.")
        if claim.get("confidence") not in {"high", "medium", "low"}:
            raise ValueError("Every claim needs a high, medium, or low confidence level.")

    report["verified_at"] = datetime.now(timezone.utc).isoformat()
    return report


def research_verified(question, date_range="latest available", source_preference="credible primary sources"):
    """Research a question with live search grounding and claim-level citations."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a research question.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Research question must be {MAX_QUESTION_LENGTH} characters or fewer.")
    if not _allow_request():
        raise RuntimeError("Research rate limit reached. Wait a minute before searching again.")

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in .env before using Verified Research.")

    prompt = f"""You are a rigorous research verification assistant.

Research question: {question}
Date requirement: {date_range}
Source preference: {source_preference}

Use live web search grounding. Do not answer from memory when the question asks for current, recent, updated, or time-sensitive information.
Prefer primary sources: government agencies, official statistics, universities, standards bodies, company filings, and original research. Use reputable secondary sources only when primary sources are unavailable.

Rules:
- Answer only the exact question. Do not add background, tangents, recommendations, or extra statistics that were not requested.
- Return at most 5 claims and 6 sources. Include only sources cited by a claim.
- Keep the answer under 120 words unless the question explicitly requires a comparison or explanation.
- Never invent facts, dates, numbers, quotations, URLs, or sources.
- Every factual claim must cite one or more source IDs.
- If sources disagree, report the disagreement instead of choosing silently.
- Distinguish observed facts from estimates, forecasts, interpretation, and opinion.
- If evidence is insufficient, say exactly what could not be verified.
- Use the source publication date when available. Do not call an undated source current.
- Return only JSON. No markdown fences or extra commentary.

Return this exact shape:
{{
  "question": "...",
  "as_of": "YYYY-MM-DD or unknown",
  "answer": "A concise answer based only on the cited claims. Say insufficient evidence when necessary.",
  "claims": [
    {{
      "text": "One independently checkable factual claim.",
      "source_ids": ["s1"],
      "confidence": "high|medium|low",
      "kind": "fact|estimate|forecast|interpretation",
      "published_date": "YYYY-MM-DD or unknown"
    }}
  ],
  "sources": [
    {{
      "id": "s1",
      "title": "Exact source title",
      "publisher": "Organization or publication",
      "url": "https://example.com/source",
      "published_date": "YYYY-MM-DD or unknown",
      "accessed_date": "YYYY-MM-DD"
    }}
  ],
  "limitations": ["What remains uncertain or could not be verified"]
}}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=1200,
            temperature=0.1,
        ),
    )
    return _validate_report(_extract_json(response.text))
