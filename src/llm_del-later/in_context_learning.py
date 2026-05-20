from typing import List, Dict, Any
from .llm_interface import LLMInterface

class InContextLearning:
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.examples = self._load_examples()

    def _load_examples(self) -> List[Dict[str, Any]]:
        """
        Load few-shot examples for in-context learning
        
        Returns:
            List of example queries and their recommendations
        """
        return [
            {
                "query": "I'm looking for action movies with strong female leads",
                "recommendations": [
                    "Mad Max: Fury Road",
                    "Kill Bill",
                    "Atomic Blonde"
                ],
                "explanation": "These movies feature strong female protagonists in action-packed scenarios."
            },
            {
                "query": "Recommend me some classic rock albums from the 70s",
                "recommendations": [
                    "Led Zeppelin IV",
                    "Dark Side of the Moon",
                    "Rumours"
                ],
                "explanation": "These are iconic rock albums from the 1970s that defined the era."
            }
        ]

    def create_prompt(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        num_examples: int = 2
    ) -> str:
        """
        Create a prompt with few-shot examples and current query
        
        args:
            query: current user query
            candidates: list of candidate items
            num_examples: Number of few-shot examples to include
        
        returns:
            Formatted prompt
        """
        prompt = "Here are some examples of how to make recommendations:\n\n"
        
        # Add few-shot examples
        for example in self.examples[:num_examples]:
            prompt += f"Query: {example['query']}\n"
            prompt += "Recommendations:\n"
            for rec in example['recommendations']:
                prompt += f"- {rec}\n"
            prompt += f"Explanation: {example['explanation']}\n\n"
        
        # Add current query and candidates
        prompt += "Now, based on the following query and candidates:\n\n"
        prompt += f"Query: {query}\n"
        prompt += "Candidates:\n"
        for candidate in candidates:
            prompt += f"- {candidate['title']}\n"
            if 'description' in candidate:
                prompt += f"  Description: {candidate['description']}\n"
            prompt += f"  Relevance Score: {candidate['score']}\n"
        
        prompt += "\nPlease provide a personalized recommendation list with explanations."
        return prompt

    def generate_recommendations(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate final recommendations using in-context learning
        
        Args:
            query: User query
            candidates: List of candidate items
        
        Returns:
            Dictionary containing recommendations and explanations
        """
        # Create prompt with examples and current query
        prompt = self.create_prompt(query, candidates)
        
        # Generate response from LLM
        response = self.llm.generate_response(prompt)
        
        # Parse and structure the response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured format
        
        args:
            response: raw LLM response
        
        returns:
            structured recommendations with explanations
        """
        # Implementation depends on your specific needs
        # This is a simple example
        lines = response.split('\n')
        recommendations = []
        explanation = ""
        
        for line in lines:
            if line.startswith('- '):
                recommendations.append(line[2:])
            elif line.startswith('Explanation:'):
                explanation = line[12:].strip()
        
        return {
            'recommendations': recommendations,
            'explanation': explanation
        } 