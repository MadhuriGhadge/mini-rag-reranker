import numpy as np
from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re
from database import ChunkDatabase

class HybridReranker:
    def __init__(self, alpha: float = 0.6):
        """
        Hybrid reranker combining vector and keyword scores
        Args:
            alpha: Weight for vector score (1-alpha for keyword score)
        """
        self.alpha = alpha
        self.db = ChunkDatabase()
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def rerank(self, query: str, vector_results: List[Tuple[int, float]], 
               fts_results: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        """
        Rerank results by combining vector and keyword scores
        """
        # Get all unique chunk IDs
        all_chunk_ids = set()
        vector_dict = {}
        fts_dict = {}
        
        for chunk_id, score in vector_results:
            all_chunk_ids.add(chunk_id)
            vector_dict[chunk_id] = score
        
        for chunk_id, score in fts_results:
            all_chunk_ids.add(chunk_id)
            fts_dict[chunk_id] = score
        
        if not all_chunk_ids:
            return []
        
        # Normalize scores
        vector_scores = [vector_dict.get(cid, 0.0) for cid in all_chunk_ids]
        fts_scores = [fts_dict.get(cid, 0.0) for cid in all_chunk_ids]
        
        norm_vector_scores = self.normalize_scores(vector_scores)
        norm_fts_scores = self.normalize_scores(fts_scores)
        
        # Combine scores
        final_scores = []
        chunk_ids_list = list(all_chunk_ids)
        
        for i, chunk_id in enumerate(chunk_ids_list):
            vector_score = norm_vector_scores[i]
            fts_score = norm_fts_scores[i]
            final_score = self.alpha * vector_score + (1 - self.alpha) * fts_score
            final_scores.append((chunk_id, final_score))
        
        # Sort by final score and return top k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:k]

class LearnedReranker:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.db = ChunkDatabase()
        self.model_path = "data/learned_reranker.pkl"
        self.scaler_path = "data/learned_scaler.pkl"
    
    def extract_features(self, query: str, chunk_id: int, vector_score: float, 
                        fts_score: float) -> List[float]:
        """Extract features for a query-chunk pair"""
        chunk = self.db.get_chunk_by_id(chunk_id)
        if not chunk:
            return [0.0] * 7
        
        features = []
        
        # 1. Vector score
        features.append(vector_score)
        
        # 2. FTS score
        features.append(fts_score)
        
        # 3. Title hit (1 if query term appears in title)
        query_terms = set(re.findall(r'\w+', query.lower()))
        title_terms = set(re.findall(r'\w+', chunk['source_title'].lower()))
        title_hit = 1.0 if query_terms & title_terms else 0.0
        features.append(title_hit)
        
        # 4. Query length
        features.append(len(query.split()))
        
        # 5. Chunk length (normalized)
        features.append(chunk['chunk_length'] / 1000.0)  # Normalize by 1000
        
        # 6. Is first paragraph
        features.append(1.0 if chunk['is_first_paragraph'] else 0.0)
        
        # 7. Exact match bonus (if query terms appear exactly in chunk)
        chunk_text_lower = chunk['text'].lower()
        exact_matches = sum(1 for term in query_terms if term in chunk_text_lower)
        exact_match_ratio = exact_matches / len(query_terms) if query_terms else 0.0
        features.append(exact_match_ratio)
        
        return features
    
    def create_training_data(self, training_questions: List[Dict]) -> Tuple[List[List[float]], List[int]]:
        """Create training data from questions with expected keywords"""
        X = []
        y = []
        
        from embeddings import EmbeddingIndex
        embedding_index = EmbeddingIndex()
        embedding_index.load_index()
        
        for question_data in training_questions:
            query = question_data['question']
            expected_keywords = set(kw.lower() for kw in question_data.get('expected_keywords', []))
            
            # Get candidate chunks from both vector and FTS search
            vector_results = embedding_index.search(query, k=20)
            fts_results = self.db.fts_search(query, limit=20)
            
            # Create feature vectors and labels
            all_chunk_ids = set()
            vector_dict = {}
            fts_dict = {}
            
            for chunk_id, score in vector_results:
                all_chunk_ids.add(chunk_id)
                vector_dict[chunk_id] = score
            
            for chunk_id, score in fts_results:
                all_chunk_ids.add(chunk_id)
                fts_dict[chunk_id] = score
            
            for chunk_id in all_chunk_ids:
                vector_score = vector_dict.get(chunk_id, 0.0)
                fts_score = fts_dict.get(chunk_id, 0.0)
                
                features = self.extract_features(query, chunk_id, vector_score, fts_score)
                
                # Create label based on whether chunk contains expected keywords
                chunk = self.db.get_chunk_by_id(chunk_id)
                if chunk:
                    chunk_text_lower = chunk['text'].lower()
                    has_keywords = any(keyword in chunk_text_lower for keyword in expected_keywords)
                    label = 1 if has_keywords else 0
                    
                    X.append(features)
                    y.append(label)
        
        return X, y
    
    def train(self, training_questions: List[Dict]):
        """Train the learned reranker"""
        print("Creating training data...")
        X, y = self.create_training_data(training_questions)
        
        if not X:
            raise ValueError("No training data created")
        
        print(f"Training with {len(X)} samples, {sum(y)} positive")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train logistic regression
        self.model = LogisticRegression(
            random_state=42, 
            class_weight='balanced',  # Handle imbalanced classes
            max_iter=1000
        )
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Learned reranker trained and saved")
    
    def load_model(self):
        """Load trained model and scaler"""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Trained model not found. Please train first.")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def rerank(self, query: str, vector_results: List[Tuple[int, float]], 
               fts_results: List[Tuple[int, float]], k: int) -> List[Tuple[int, float]]:
        """Rerank results using the learned model"""
        if self.model is None:
            self.load_model()
        
        # Get all unique chunk IDs
        all_chunk_ids = set()
        vector_dict = {}
        fts_dict = {}
        
        for chunk_id, score in vector_results:
            all_chunk_ids.add(chunk_id)
            vector_dict[chunk_id] = score
        
        for chunk_id, score in fts_results:
            all_chunk_ids.add(chunk_id)
            fts_dict[chunk_id] = score
        
        if not all_chunk_ids:
            return []
        
        # Extract features for all chunks
        features_list = []
        chunk_ids_list = list(all_chunk_ids)
        
        for chunk_id in chunk_ids_list:
            vector_score = vector_dict.get(chunk_id, 0.0)
            fts_score = fts_dict.get(chunk_id, 0.0)
            features = self.extract_features(query, chunk_id, vector_score, fts_score)
            features_list.append(features)
        
        # Scale features
        X_scaled = self.scaler.transform(features_list)
        
        # Get probabilities from model
        probabilities = self.model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
        
        # Create results with probabilities as scores
        results = [(chunk_ids_list[i], float(prob)) for i, prob in enumerate(probabilities)]
        
        # Sort by probability and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

# Training questions with expected keywords for the learned reranker
TRAINING_QUESTIONS = [
    {
        "question": "What is ISO 13849?",
        "expected_keywords": ["iso 13849", "safety", "performance level", "control system"]
    },
    {
        "question": "PLC safety requirements",
        "expected_keywords": ["plc", "programmable logic controller", "safety", "requirements"]
    },
    {
        "question": "How to calculate Performance Level d?",
        "expected_keywords": ["performance level", "pld", "calculation", "category"]
    },
    {
        "question": "Emergency stop standards",
        "expected_keywords": ["emergency stop", "e-stop", "standard", "safety"]
    },
    {
        "question": "Risk assessment methodology",
        "expected_keywords": ["risk assessment", "methodology", "hazard", "analysis"]
    },
    {
        "question": "Safety integrity levels",
        "expected_keywords": ["safety integrity level", "sil", "functional safety"]
    },
    {
        "question": "Machine guarding requirements",
        "expected_keywords": ["machine guarding", "guard", "barrier", "protection"]
    },
    {
        "question": "Lockout tagout procedures",
        "expected_keywords": ["lockout", "tagout", "loto", "procedure", "energy isolation"]
    }
]

def train_learned_reranker():
    """Train the learned reranker with predefined questions"""
    reranker = LearnedReranker()
    reranker.train(TRAINING_QUESTIONS)

if __name__ == "__main__":
    print("Training learned reranker...")
    train_learned_reranker()
    print("Training complete!")