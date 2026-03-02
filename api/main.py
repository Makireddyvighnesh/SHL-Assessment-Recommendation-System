"""
SHL Assessment Recommendation API
FastAPI backend with /health and /recommend endpoints.

Endpoints:
    GET  /health     → {"status": "healthy"}
    POST /recommend  → {"query": "..."} → {"recommended_assessments": [...]}
"""

import os
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from bs4 import BeautifulSoup

from engine.recommender import get_recommender

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends relevant SHL assessments given a job description or query.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        description="Job description text, natural language query, or a URL to a JD page",
        example="I am hiring for Java developers who can also collaborate effectively with my business teams.",
    )


class AssessmentItem(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[AssessmentItem]


async def fetch_url_text(url: str) -> str:
    """Fetch text content from a URL (for JD URLs)."""
    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        response = await client.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SHL-Recommender/1.0)"},
        )
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Remove scripts and styles
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    # Trim to avoid token limits
    return text[:4000]


def is_url(text: str) -> bool:
    return text.strip().startswith(("http://", "https://"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_duration(value) -> int:
    """Safely parse duration to int — handles None, strings, ints."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    # It's a string — try to extract first number from it
    import re
    m = re.search(r'(\d+)', str(value))
    return int(m.group(1)) if m else 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Recommend SHL assessments for a job description or query.
    Accepts plain text or a URL pointing to a job description.
    """
    query = request.query.strip()

    # If a URL is provided, fetch the page text
    if is_url(query):
        try:
            query = await fetch_url_text(query)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Could not fetch URL content: {e}",
            )

    if not query:
        raise HTTPException(status_code=400, detail="Query text is empty")

    try:
        recommender = get_recommender()
        results = recommender.recommend(query)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant assessments found for this query.",
        )

    # Coerce types to match schema
    assessments = [
        AssessmentItem(
            url=r.get("url", ""),
            name=r.get("name", ""),
            adaptive_support=r.get("adaptive_support", "No"),
            description=r.get("description", ""),
            duration=_parse_duration(r.get("duration")),
            remote_support=r.get("remote_support", "No"),
            test_type=r.get("test_type", []),
        )
        for r in results
    ]

    return RecommendResponse(recommended_assessments=assessments)


# ── Dev Server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
