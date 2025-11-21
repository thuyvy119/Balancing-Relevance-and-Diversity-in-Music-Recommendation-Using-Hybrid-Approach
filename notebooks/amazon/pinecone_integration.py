import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import json

class PineconeRecommendationSystem:
    def __init__(self, api_key, environment, index_name="recommendations-index"):
        """
        Initialize Pinecone recommendation system
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        
        # Create or connect to index
        self._setup_index()
        
    def _setup_index(self):
        """Create index if it doesn't exist, otherwise connect to it"""
        if self.index_name not in pinecone.list_indexes():
            print(f"Creating new index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name,
                dimension=768,  # all-mpnet-base-v2 dimension
                metric="cosine"
            )
        
        self.index = pinecone.Index(self.index_name)
        print(f"Connected to index: {self.index_name}")
    
    def prepare_recommendations_for_pinecone(self, user_recs):
        """
        Prepare user recommendations for Pinecone upload
        
        Args:
            user_recs: Dictionary {user_id: [list of recommendation dicts]}
            
        Returns:
            List of tuples: (id, embedding, metadata)
        """
        vectors = []
        
        print("Preparing recommendations for Pinecone...")
        
        for user_id, recommendations in tqdm(user_recs.items(), desc="Processing users"):
            for i, rec in enumerate(recommendations):
                # Create combined text for embedding
                combined_text = f"Title: {rec['title']}. Categories: {rec['categories']}"
                
                # Generate embedding
                embedding = self.embedding_model.encode(combined_text)
                
                # Create unique ID
                vector_id = f"user_{user_id}_item_{rec['iid']}"
                
                # Create metadata
                metadata = {
                    "user_id": int(user_id),
                    "item_id": int(rec['iid']),
                    "title": str(rec['title']),
                    "categories": str(rec['categories']),
                    "fm_score": float(rec['score']),
                    "combined_text": combined_text,
                    "recommendation_rank": i + 1
                }
                
                vectors.append((vector_id, embedding.tolist(), metadata))
        
        print(f"Prepared {len(vectors)} vectors for Pinecone")
        return vectors
    
    def upload_recommendations_to_pinecone(self, user_recs, batch_size=100):
        """
        Upload user recommendations to Pinecone
        
        Args:
            user_recs: Dictionary {user_id: [list of recommendation dicts]}
            batch_size: Number of vectors to upload per batch
        """
        vectors = self.prepare_recommendations_for_pinecone(user_recs)
        
        print(f"Uploading {len(vectors)} vectors to Pinecone in batches of {batch_size}")
        
        for i in tqdm(range(0, len(vectors), batch_size), desc="Uploading to Pinecone"):
            batch = vectors[i:i + batch_size]
            
            # Upsert batch to Pinecone
            self.index.upsert(vectors=batch)
        
        print("Upload completed!")
    
    def search_recommendations_for_user(self, user_id, query_text=None, top_k=10, filter_user_id=None):
        """
        Search for recommendations for a specific user
        
        Args:
            user_id: User ID to search for
            query_text: Optional query text (if None, uses user's preferences)
            top_k: Number of top results to return
            filter_user_id: If provided, only return recommendations from this user
            
        Returns:
            List of recommendation dictionaries
        """
        if query_text is None:
            # Use a generic query for the user
            query_text = f"recommendations for user {user_id}"
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text)
        
        # Prepare filter if specified
        filter_dict = None
        if filter_user_id is not None:
            filter_dict = {"user_id": {"$eq": int(filter_user_id)}}
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Process results
        recommendations = []
        for match in results['matches']:
            metadata = match['metadata']
            recommendations.append({
                "item_id": metadata['item_id'],
                "title": metadata['title'],
                "categories": metadata['categories'],
                "fm_score": metadata['fm_score'],
                "similarity_score": match['score'],
                "recommendation_rank": metadata['recommendation_rank'],
                "source_user_id": metadata['user_id']
            })
        
        return recommendations
    
    def get_diversified_recommendations(self, user_id, top_k=10, diversity_threshold=0.3):
        """
        Get diversified recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations to return
            diversity_threshold: Minimum similarity threshold for diversity
            
        Returns:
            List of diversified recommendations
        """
        # Get initial recommendations
        initial_recs = self.search_recommendations_for_user(
            user_id, 
            top_k=top_k * 2  # Get more candidates for diversification
        )
        
        if not initial_recs:
            return []
        
        # Simple diversification: select items with different categories
        diversified = []
        used_categories = set()
        
        for rec in initial_recs:
            categories = set(rec['categories'].split(', '))
            
            # Check if this item adds diversity
            if not categories.intersection(used_categories):
                diversified.append(rec)
                used_categories.update(categories)
                
                if len(diversified) >= top_k:
                    break
        
        # If we don't have enough diverse items, add remaining ones
        if len(diversified) < top_k:
            for rec in initial_recs:
                if rec not in diversified:
                    diversified.append(rec)
                    if len(diversified) >= top_k:
                        break
        
        return diversified[:top_k]
    
    def get_user_preference_summary(self, user_id, top_k=5):
        """
        Get a summary of user preferences based on their recommendations
        
        Args:
            user_id: User ID
            top_k: Number of top recommendations to analyze
            
        Returns:
            Dictionary with preference summary
        """
        recommendations = self.search_recommendations_for_user(
            user_id, 
            top_k=top_k
        )
        
        if not recommendations:
            return {"error": "No recommendations found"}
        
        # Analyze categories
        all_categories = []
        for rec in recommendations:
            categories = rec['categories'].split(', ')
            all_categories.extend(categories)
        
        # Count category frequencies
        from collections import Counter
        category_counts = Counter(all_categories)
        
        # Get top categories
        top_categories = category_counts.most_common(5)
        
        # Calculate average scores
        avg_fm_score = np.mean([rec['fm_score'] for rec in recommendations])
        avg_similarity = np.mean([rec['similarity_score'] for rec in recommendations])
        
        return {
            "user_id": user_id,
            "top_categories": top_categories,
            "average_fm_score": avg_fm_score,
            "average_similarity_score": avg_similarity,
            "recommendation_count": len(recommendations)
        }

# Example usage functions
def save_user_recs_to_json(user_recs, filename):
    """Save user recommendations to JSON file"""
    # Convert numpy types to native Python types
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert the recommendations
    json_safe_recs = {}
    for user_id, recs in user_recs.items():
        json_safe_recs[str(user_id)] = [
            {k: convert_for_json(v) for k, v in rec.items()}
            for rec in recs
        ]
    
    with open(filename, 'w') as f:
        json.dump(json_safe_recs, f, indent=2)
    
    print(f"Saved recommendations for {len(user_recs)} users to {filename}")

def load_user_recs_from_json(filename):
    """Load user recommendations from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert string keys back to integers
    user_recommendations = {int(k): v for k, v in data.items()}
    
    return user_recommendations

# Example usage:
"""
# Initialize the system
pinecone_system = PineconeRecommendationSystem(
    api_key="your_pinecone_api_key",
    environment="your_environment"
)

# Upload recommendations
pinecone_system.upload_recommendations_to_pinecone(user_recs)

# Search for recommendations
recommendations = pinecone_system.search_recommendations_for_user(user_id=123)

# Get diversified recommendations
diversified = pinecone_system.get_diversified_recommendations(user_id=123)

# Get user preference summary
summary = pinecone_system.get_user_preference_summary(user_id=123)
""" 