import json
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage

class LLMReranker:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        temperature=0.0,
        top_k=10,
    ):

        self.top_k = top_k

        self.llm = ChatOllama(
        model="llama3",
        temperature=0
    )

    def build_prompt(self, query, candidates):

        formatted_candidates = []

        for idx, item in enumerate(candidates):
            formatted_candidates.append(
                {
                    "rank": idx + 1,
                    "iid": item.get("iid"),
                    "title": item.get("title"),
                    "categories": item.get("categories"),
                    "description": item.get("description"),
                    "cf_score": round(float(item.get("cf_score", 0)), 4),
                    "semantic_score": round(float(item.get("semantic_score", 0)), 4),
                    "hybrid_score": round(float(item.get("hybrid_score", 0)), 4),
                    "diversity_score": round(float(item.get("diversity_score", 0)), 4),
                }
            )

        prompt = f"""You are a recommendation reranking system.
            User Query:
            {query}

            Candidate Items:
            {json.dumps(formatted_candidates, indent=2)}

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

        return prompt

    def rerank(self, query, candidates):

        if not candidates:
            return []

        prompt = self.build_prompt(query, candidates)

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        content = response.content.strip()

        try:
            parsed = json.loads(content)
        except Exception:
            return candidates[: self.top_k]

        candidate_map = {
            item["iid"]: item
            for item in candidates
        }

        reranked = []

        for obj in parsed:
            iid = obj.get("iid")

            if iid in candidate_map:
                reranked.append(candidate_map[iid])

        if len(reranked) == 0:
            return candidates[: self.top_k]

        return reranked[: self.top_k]