from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMInterface:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        """Load the LLM model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> str:
        """
        Generate response from the LLM
        
        args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Controls randomness (higher = more random)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to return
        
        returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def format_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Format the prompt with query and context
        
        args:
            query: User query
            context: List of relevant items with their metadata
        
        returns:
            Formatted prompt
        """
        prompt = f"User Query: {query}\n\n"
        prompt += "Relevant Items:\n"
        
        for item in context:
            prompt += f"- {item['title']}\n"
            if 'description' in item:
                prompt += f"  Description: {item['description']}\n"
            prompt += f"  Relevance Score: {item['score']}\n\n"
        
        prompt += "Based on the user's query and the relevant items above, "
        prompt += "provide a personalized recommendation list with explanations."
        
        return prompt 