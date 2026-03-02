"""
SHL Assessment Recommendation Engine
Pipeline:
1. Query preprocessing — LLM extracts role, skills, constraints from raw JD
2. Vector search — retrieves top-K candidates using clean query
3. LLM re-ranking — selects 5-10 balanced assessments
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from indexer.build_index import load_index, EMBED_MODEL_NAME

load_dotenv()
# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL     = "openai/gpt-oss-20b" #"llama-3.1-8b-instant" #"llama-3.3-70b-versatile"
RETRIEVAL_TOP_K = 40
FINAL_MIN      = 5
FINAL_MAX      = 10

# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """You are an expert HR analyst. Extract the key hiring requirements from the job description below.

Return ONLY a JSON object with these fields:
{{
  "role": "Job title/role name",
  "technical_skills": ["skill1", "skill2"],
  "soft_skills": ["skill1", "skill2"],
  "experience_level": "entry/mid/senior/manager/executive",
  "duration_preference": null or integer minutes if mentioned,
  "other_requirements": ["any other important requirements"]
}}

STRICT Rules:
- List ONLY skills explicitly mentioned in the job description — do not infer or add related skills
- If any abbreviation or short form is explicitly mentioned in the job description, expand it to its full form in the output.
  Example:
    SDLC → Software Development Lifecycle
    JS → JavaScript
    ML → Machine Learning
- Do NOT expand terms that are already written in full.
- Do NOT invent expansions for abbreviations not present in the job description.
- Java and JavaScript are DIFFERENT — list them separately and exactly as mentioned
- Keep each skill short 
- duration_preference: if "1 hour" → 60, "30 minutes" → 30, "max X minutes" → X. Otherwise null.
- Return ONLY valid JSON, no extra text, no markdown

Job Description:
{query}
"""

RERANK_PROMPT = """You are an expert HR assessment consultant for SHL.

Hiring requirements:
{extracted}

Candidate SHL assessments:
{candidates}

TASK: Select the most relevant assessments for the hiring requirements above.

RULES:
1. Only include assessments that directly match what is asked for in the requirements. Do not include loosely related ones just to fill the count.
2. Match precisely — similar sounding skills are not the same skill.
3. If the requirements mention both technical and behavioral needs, include a balanced mix of Knowledge & Skills (K) and Personality & Behavior (P) type assessments.
4. If a maximum duration is specified, only include assessments that fit within it.
5. Select between {min_n} and {max_n} assessments. Prioritize quality and relevance over hitting the maximum count.
6. Return ONLY a valid JSON array with no extra text, no markdown:

[
  {{"name": "Assessment Name", "url": "https://...", "reason": "one sentence explanation"}}
]
"""


class SHLRecommender:
    def __init__(self):
        self._index: Optional[VectorStoreIndex] = None
        self._llm: Optional[Groq] = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        print("Initializing SHL Recommender...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        self._llm = Groq(model=GROQ_MODEL, api_key=groq_api_key)
        Settings.llm = self._llm
        self._index = load_index()
        self._initialized = True
        print("✅ SHL Recommender ready")

    # ── Step 1: Extract key requirements from raw query ───────────────────────

    def _extract_requirements(self, query: str) -> dict:
        """
        Use LLM to extract structured requirements from raw JD or query.
        Returns dict with role, skills, experience, duration_preference etc.
        """
        # Skip extraction for short clean queries (not a full JD)
        # if len(query.split()) < 30:
        #     return {
        #         "role": query,
        #         "technical_skills": [],
        #         "soft_skills": [],
        #         "experience_level": "",
        #         "duration_preference": None,
        #         "other_requirements": []
        #     }

        prompt = EXTRACT_PROMPT.format(query=query)

        try:
            response = self._llm.complete(prompt)
            content = response.text.strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                print(f"  Extracted requirements: {extracted}")
                return extracted
        except Exception as e:
            print(f"  Warning: extraction failed ({e}), using raw query")

        # Fallback: return raw query as role
        return {
            "role": query[:200],
            "technical_skills": [],
            "soft_skills": [],
            "experience_level": "",
            "duration_preference": None,
            "other_requirements": []
        }

    def _build_search_query(self, extracted: dict) -> str:
        """
        Build a clean, focused search query from extracted requirements.
        This is what gets embedded for vector search.
        """
        parts = []

        if extracted.get("role"):
            parts.append(extracted["role"])

        if extracted.get("technical_skills"):
            parts.append("Technical skills: " + ", ".join(extracted["technical_skills"]))

        if extracted.get("soft_skills"):
            parts.append("Soft skills: " + ", ".join(extracted["soft_skills"]))

        if extracted.get("experience_level"):
            parts.append(f"Experience level: {extracted['experience_level']}")

        if extracted.get("other_requirements"):
            parts.append(", ".join(extracted["other_requirements"]))

        clean_query = ". ".join(parts)
        print(f"  Search query: {clean_query}")
        return clean_query

    # ── Step 2: Vector retrieval ──────────────────────────────────────────────

    def _retrieve_candidates(self, search_query: str) -> list[dict]:
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=RETRIEVAL_TOP_K,
        )
        nodes = retriever.retrieve(search_query)

        candidates = []
        seen_urls = set()
        for node in nodes:
            meta = node.metadata
            test_types = json.loads(meta.get("test_type", "[]"))
            c = {
                "name":             meta.get("name", ""),
                "url":              meta.get("url", ""),
                "description":      meta.get("description", ""),
                "test_type":        test_types,
                "duration":         meta.get("duration", 0),
                "remote_support":   meta.get("remote_support", "No"),
                "adaptive_support": meta.get("adaptive_support", "No"),
                "score":            node.score,
            }
            candidates.append(c)
            seen_urls.add(c["url"])

        return candidates, seen_urls

    def _keyword_fetch(self, extracted: dict, seen_urls: set) -> list[dict]:
        """
        Force-fetch assessments for skills explicitly mentioned
        that vector search may have missed.
        """
        from llama_index.core.retrievers import VectorIndexRetriever

        tech_skills = extracted.get("technical_skills", [])
        injected = []

        for skill in tech_skills:
            # Search specifically for this skill
            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=5,
            )
            nodes = retriever.retrieve(skill)
            for node in nodes[:2]:  # top 2 per skill
                meta = node.metadata
                url  = meta.get("url", "")
                if url in seen_urls:
                    continue
                # Only inject if skill name appears in assessment name
                name = meta.get("name", "").lower()
                skill_words = [w for w in skill.lower().split() if len(w) > 2]
                if any(w in name for w in skill_words):
                    test_types = json.loads(meta.get("test_type", "[]"))
                    injected.append({
                        "name":             meta.get("name", ""),
                        "url":              url,
                        "description":      meta.get("description", ""),
                        "test_type":        test_types,
                        "duration":         meta.get("duration", 0),
                        "remote_support":   meta.get("remote_support", "No"),
                        "adaptive_support": meta.get("adaptive_support", "No"),
                        "score":            node.score,
                    })
                    seen_urls.add(url)
                    print(f"  [Keyword inject] {meta.get('name')} ← skill: {skill}")

        return injected

    # ── Step 3: LLM re-ranking ────────────────────────────────────────────────

    def _format_candidates(self, candidates: list[dict]) -> str:
        lines = []
        for i, c in enumerate(candidates, 1):
            types    = ", ".join(c["test_type"]) if c["test_type"] else "Unknown"
            duration = f"{c['duration']} min" if c["duration"] else "N/A"
            # Clean description — remove \r\n and extra whitespace
            desc = c['description'].replace('\r', ' ').replace('\n', ' ').strip()
            lines.append(
                f"{i}. [{types}] {c['name']} (Duration: {duration})\n"
                f"   URL: {c['url']}\n"
                f"   {desc[:120]}"
            )
        return "\n\n".join(lines)


    def _rerank_with_llm(self, extracted: dict, candidates: list[dict]) -> list[dict]:
        prompt = RERANK_PROMPT.format(
            extracted=json.dumps(extracted, indent=2),
            candidates=self._format_candidates(candidates),
            min_n=FINAL_MIN,
            max_n=FINAL_MAX,
        )

        response = self._llm.complete(prompt)
        llm_output = response.text.strip()

        print("LLMs response: ", llm_output)

        # Try to extract JSON array — handle markdown fences and trailing text
        # First try strict match
        json_match = re.search(r'\[.*?\]', llm_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Second try: find [ and ] manually and parse between them
        start = llm_output.find('[')
        end   = llm_output.rfind(']')
        if start != -1 and end != -1:
            try:
                return json.loads(llm_output[start:end+1])
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM did not return valid JSON: {e} | Output: {llm_output[:300]}")

        raise ValueError(f"No JSON array found in LLM output: {llm_output[:200]}")
    # ── Step 4: Build final response ──────────────────────────────────────────

    def _build_response(self, selected: list[dict], candidates: list[dict]) -> list[dict]:
        meta_by_url  = {c["url"]: c  for c in candidates}
        meta_by_name = {c["name"].lower(): c for c in candidates}

        results = []
        for item in selected:
            url  = item.get("url", "")
            name = item.get("name", "")
            meta = meta_by_url.get(url) or meta_by_name.get(name.lower()) or {}

            results.append({
                "name":             name or meta.get("name", ""),
                "url":              url  or meta.get("url", ""),
                "description":      meta.get("description", ""),
                "duration":         meta.get("duration") or 0,
                "remote_support":   meta.get("remote_support", "No"),
                "adaptive_support": meta.get("adaptive_support", "No"),
                "test_type":        meta.get("test_type", []),
            })

        # Ensure minimum 5 results
        results = results[:FINAL_MAX]
        if len(results) < FINAL_MIN:
            existing = {r["url"] for r in results}
            for c in candidates:
                if len(results) >= FINAL_MIN:
                    break
                if c["url"] not in existing:
                    results.append(c)
                    existing.add(c["url"])

        return results

    # ── Main entry point ──────────────────────────────────────────────────────

    def recommend(self, query: str) -> list[dict]:
        """
        Full pipeline:
        1. Extract structured requirements from raw query/JD
        2. Build clean search query
        3. Vector retrieval
        4. LLM re-ranking with awareness of duration/balance requirements
        5. Return 5-10 assessments
        """
        self.initialize()

        print(f"\n--- Recommendation Pipeline ---")
        print(f"Raw query length: {len(query.split())} words")

        # Step 1: Extract requirements
        extracted = self._extract_requirements(query)

        # Step 2: Build focused search query
        search_query = self._build_search_query(extracted)
        if not search_query.strip():
            search_query = query[:500]

        # Step 3: Vector retrieval
        # candidates = self._retrieve_candidates(search_query)
        candidates, seen_urls = self._retrieve_candidates(search_query)
        print(f"  Retrieved {len(candidates)} candidates")

        # Step 3.5: Keyword inject — force include explicitly mentioned skills
        injected = self._keyword_fetch(extracted, seen_urls)
        if injected:
            print(f"  Injected {len(injected)} keyword-matched candidates")
            # Put injected at front so LLM sees them first
            candidates = injected + candidates
        if not candidates:
            return []

        print(f"  Retrieved {len(candidates)} candidates")
        print(f"  Retrieved candidates: {candidates}")


        # Step 4: LLM re-ranking
        # try:
        #     selected = self._rerank_with_llm(extracted, candidates)
        # except Exception as e:
        #     print(f"  Warning: LLM re-ranking failed ({e}), using vector results")
        #     selected = [{"name": c["name"], "url": c["url"]} for c in candidates[:FINAL_MAX]]
        # Step 4: LLM re-ranking
        try:
            selected = self._rerank_with_llm(extracted, candidates)
        except Exception as e:
            print(f"  Warning: LLM re-ranking failed ({e}), using filtered vector results")
            selected = []  # ← initialize FIRST
            tech_skills = [s.lower() for s in extracted.get("technical_skills", [])]
            if tech_skills:
                skill_keywords = set()
                for skill in tech_skills:
                    for word in skill.split():
                        if len(word) > 2:
                            skill_keywords.add(word)
                selected = [
                    {"name": c["name"], "url": c["url"]}
                    for c in candidates
                    if any(kw in c["name"].lower() for kw in skill_keywords)
                ][:FINAL_MAX]
            if not selected:
                selected = [{"name": c["name"], "url": c["url"]} for c in candidates[:FINAL_MAX]]

        # Step 5: Build final response
        print(selected)
        return self._build_response(selected, candidates)


# Singleton
_recommender = SHLRecommender()

def get_recommender() -> SHLRecommender:
    return _recommender
