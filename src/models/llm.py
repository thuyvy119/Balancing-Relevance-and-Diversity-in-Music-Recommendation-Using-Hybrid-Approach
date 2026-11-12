import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class LLMReranker:
    def __init__(self, model="llama3", temperature=0.0, top_k=10):
        self.top_k = top_k
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.n_calls = 0
        self.n_fallback = 0          # calls where the LLM output could not be used
        self.last_fallback = False

    def build_prompt(self, query, candidates):
        formatted = []
        for idx, item in enumerate(candidates):
            formatted.append({
                "rank": idx + 1,
                "iid": item.get("iid"),
                "title": item.get("title"),
                "categories": item.get("categories"),
                "description": item.get("description"),
                "cf_score": round(float(item.get("cf_score", 0)), 4),
                "semantic_score": round(float(item.get("semantic_score", 0)), 4),
                "hybrid_score": round(float(item.get("hybrid_score", 0)), 4),
                "diversity_score": round(float(item.get("diversity_score", 0)), 4),
            })
        return f"""You are a recommendation reranking system.
            User Query:
            {query}

            Candidate Items:
            {json.dumps(formatted, indent=2)}

            Task:
            Rerank the items to maximize:
            1. relevance to the query
            2. diversity across recommendations
            3. overall recommendation quality

            Rules:
            - prioritize semantically relevant items
            - avoid redundant items
            - preserve diverse categories/styles
            - use all provided scores jointly
            - return ONLY valid JSON
            - do not add explanations

            Return format:
            [
            {{
                "iid": "..."
            }}
            ]
            """

    @staticmethod
    def _parse(content):
        m = re.search(r"\[.*\]", content, re.S)       # tolerate text around the JSON array
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return None
        return [str(o.get("iid")) if isinstance(o, dict) else str(o) for o in parsed]

    def rerank(self, query, candidates):
        if not candidates:
            return []
        self.n_calls += 1
        self.last_fallback = False
        try:
            content = self.llm.invoke([HumanMessage(content=self.build_prompt(query, candidates))]).content.strip()
            ids = self._parse(content)
        except Exception:
            ids = None

        cmap = {str(c["iid"]): c for c in candidates}
        reranked, seen = [], set()
        for i in (ids or []):
            if i in cmap and i not in seen:
                reranked.append(cmap[i]); seen.add(i)

        if not reranked:                       # unusable output -> keep MMR order, record it
            self.n_fallback += 1
            self.last_fallback = True
            return candidates[: self.top_k]

        for c in candidates:                   # pad so every list has top_k items
            if len(reranked) >= self.top_k:
                break
            if str(c["iid"]) not in seen:
                reranked.append(c)
        return reranked[: self.top_k]