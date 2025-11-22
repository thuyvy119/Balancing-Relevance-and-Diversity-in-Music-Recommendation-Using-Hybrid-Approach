from transformers import pipeline
from huggingface_hub import login
import os
import re
import torch

class LLMReranker:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        self.model_name = model_name
        self.llm_pipeline = None
        
    def initialize_pipeline(self):
        try:
            if "HUGGINGFACE_TOKEN" in os.environ:
                login(token=os.environ["HUGGINGFACE_TOKEN"])
            else:
                raise ValueError("HUGGINGFACE_TOKEN environment variable must be set")
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                token=True
            )
        except Exception as e:
            print(f"LLM initialization failed: {str(e)}")
            self.llm_pipeline = None
            
    
    def generate_prompt(user_id, query, items, user_type):
    
        # Build items list with all relevant metadata
        items_str = "\n".join(
            f"{i}. {item['title']} (ID: {item['iid']}) | "
            f"Categories: {', '.join(item.get('categories', []))} | "
            f"Description: {item.get('description', 'No description')[:100]}... | "
            f"Scores: Hybrid={item.get('hybrid_score', 0):.2f}, MMR={item.get('mmr_score', 0):.2f}"
            for i, item in enumerate(items, 1)
        )
        
        prompt = f"""
        **Music Recommendation Expert System**
        
        === User Context ===
        - User ID: {user_id}
        - Type: {user_type} user
        - Query: "{query}"
        {'- History: Available (prioritize hybrid scores and MMR)' if user_type == 'existing' else '- History: New user (focus on query relevance)'}
        
        === Candidate Items ===
        {items_str}
        
        === Task Instructions ===
        Provide 10 final recommendations with:
        1. Clear ranking from 1-10
        2. Complete item information
        3. Justification for each position
        
        === Required Output Format ===
        === IMPORTANT OUTPUT FORMATTING ===
        Please STRICTLY follow these rules:
        1. Each recommendation MUST start with "[Rank]. [Title] (ID: [exact_item_id])"
        2. MUST include ALL metadata and generate reasoning components for each item:
            • Relevance: Explicitly explain how this matches the query "{query}"
            • Personalization/Popularity: [explanation] 
            • Diversity: How this complements other recommendations
        3. Maintain EXACTLY the specified indentation
        
        Results example:
        ---- Top 10 Recommendations ----
        Example:
        1. Dark Side of the Moon (ID: B000002UAZ)
        - Categories: Rock, Progressive Rock
        - Description: The Dark Side of the Moon is the eighth studio album...
        - Scores: Hybrid=0.92 | MMR=0.88
        
        ... (continue numbering to 10)
        
        === Critical Constraints ===
        • MUST include exactly 10 items
        • MUST maintain category diversity (min. 3 distinct categories)
        • For existing users: Hybrid score and MMR are primary factors
        • For new users: Pure query relevance is primary factor
        """
        
        return prompt

    def parse_llm_output(output, items, query, user_type):
        results = []
        current_item = None
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # detect new item entry 
            item_match = re.match(r'^(\d+)\.\s+(.+?)\s*\(ID:\s*([^)]+)\)', line)
            if item_match:
                rank = int(item_match.group(1))
                title = item_match.group(2).strip()
                item_id = item_match.group(3).strip()
                
                matching_items = [
                    item for item in items 
                    if str(item['iid']).lower() == item_id.lower()
                    and title.lower() in item['title'].lower()
                ]
                
                if matching_items:
                    current_item = {
                        'item': matching_items[0],
                        'reason': []
                    }
                    results.append(current_item)
            
            elif current_item and (line.startswith(('•', '-', '*', 'Reason', 'Relevance:', 'Diversity:'))):
                current_item['reason'].append(line)

        formatted_results = []
        for result in results[:10]:
            formatted_item = result['item'].copy()
            
            # process reasoning lines
            if result['reason']:
                # Group related reasoning points
                formatted_reason = []
                current_category = None
                
                for line in result['reason']:
                    if 'Relevance:' in line:
                        current_category = 'Relevance'
                        formatted_reason.append(f"• {line}")
                    elif 'Diversity:' in line:
                        current_category = 'Diversity'
                        formatted_reason.append(f"• {line}")
                    elif 'Personalization:' in line or 'Popularity:' in line:
                        current_category = 'Personalization' if user_type == 'existing' else 'Popularity'
                        formatted_reason.append(f"• {line}")
                    elif current_category:
                        formatted_reason.append(f"   {line.strip()}")
                
                formatted_item['llm_reason'] = '\n'.join(formatted_reason)
            else:
                formatted_item['llm_reason'] = "• No reasoning provided by LLM"
            
            formatted_results.append(formatted_item)
        
        return formatted_results if formatted_results else [
            {**item, 'llm_reason': "• Fallback ranking - no LLM reasoning"} 
            for item in sorted(
                items,
                key=lambda x: x.get('hybrid_score', x.get('semantic_score', 0)),
                reverse=True
            )[:10]
        ]
