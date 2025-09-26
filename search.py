from typing import List, Tuple, Dict, Optional
import re
from database import ChunkDatabase
from embeddings import EmbeddingIndex
from reranker import HybridReranker, LearnedReranker

class SearchEngine:
    def __init__(self):
        self.db = ChunkDatabase()
        self.embedding_index = None
        self.hybrid_reranker = HybridReranker()
        self.learned_reranker = LearnedReranker()
        
        # Abstention thresholds
        self.abstention_thresholds = {
            'baseline': 0.6,
            'hybrid': 0.7,
            'learned': 0.65
        }
    
    def _load_embedding_index(self):
        """Lazy load embedding index"""
        if self.embedding_index is None:
            self.embedding_index = EmbeddingIndex()
            self.embedding_index.load_index()
    
    def baseline_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Baseline cosine similarity search"""
        self._load_embedding_index()
        return self.embedding_index.search(query, k)
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Hybrid search combining vector and keyword similarity"""
        self._load_embedding_index()
        
        # Get candidates from both vector and FTS search
        vector_results = self.embedding_index.search(query, k=min(30, k*3))
        fts_results = self.db.fts_search(query, limit=min(30, k*3))
        
        # Rerank using hybrid approach
        return self.hybrid_reranker.rerank(query, vector_results, fts_results, k)
    
    def learned_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Learned reranking search"""
        self._load_embedding_index()
        
        try:
            # Get candidates from both vector and FTS search
            vector_results = self.embedding_index.search(query, k=min(30, k*3))
            fts_results = self.db.fts_search(query, limit=min(30, k*3))
            
            # Rerank using learned approach
            return self.learned_reranker.rerank(query, vector_results, fts_results, k)
        except FileNotFoundError:
            # Fall back to hybrid if learned model not available
            print("Learned model not found, falling back to hybrid search")
            return self.hybrid_search(query, k)
    
    def search(self, query: str, k: int = 10, mode: str = "baseline") -> List[Tuple[int, float]]:
        """Main search interface"""
        if mode == "baseline":
            return self.baseline_search(query, k)
        elif mode == "hybrid":
            return self.hybrid_search(query, k)
        elif mode == "learned":
            return self.learned_search(query, k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
    
    def generate_answer(self, query: str, contexts: List[Dict], mode: str) -> Tuple[Optional[str], bool, Optional[str]]:
        """
        Generate answer from retrieved contexts
        Returns: (answer, abstained, abstain_reason)
        """
        if not contexts:
            return None, True, "No relevant contexts found"
        
        # Check confidence threshold
        top_score = contexts[0]['score'] if contexts else 0.0
        threshold = self.abstention_thresholds[mode]
        
        if top_score < threshold:
            return None, True, f"Top result confidence ({top_score:.3f}) below threshold ({threshold:.3f})"
        
        # Find the most relevant context with sufficient detail
        best_context = None
        for context in contexts[:3]:  # Check top 3 contexts
            text = context['text']
            # Look for substantial content (not just headers or short fragments)
            if len(text.split()) >= 20 and any(keyword in text.lower() for keyword in query.lower().split()):
                best_context = context
                break
        
        if not best_context:
            best_context = contexts[0]  # Fall back to top result
        
        # Extract answer from the best context
        answer_text = best_context['text']
        
        # Try to extract the most relevant sentence(s)
        sentences = re.split(r'[.!?]+', answer_text)
        relevant_sentences = []
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            sentence_words = set(sentence.lower().split())
            # Check if sentence contains query terms
            if query_words & sentence_words:
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            # Take first 2 relevant sentences or up to 200 words
            answer = ""
            word_count = 0
            for sentence in relevant_sentences[:2]:
                if word_count + len(sentence.split()) <= 200:
                    answer += sentence + ". "
                    word_count += len(sentence.split())
                else:
                    break
            
            if answer.strip():
                return answer.strip(), False, None
        
        # Fall back to first part of the context if no good sentences found
        words = answer_text.split()
        if len(words) > 50:
            fallback_answer = ' '.join(words[:50]) + "..."
        else:
            fallback_answer = answer_text
        
        return fallback_answer, False, None
    
    def search_and_answer(self, query: str, k: int = 5, mode: str = "baseline") -> Dict:
        """
        Complete search and answer pipeline
        Returns full response with answer, contexts, and metadata
        """
        # Perform search
        search_results = self.search(query, k, mode)
        
        # Get chunk details
        chunk_ids = [chunk_id for chunk_id, _ in search_results]
        chunks = self.db.get_chunks_by_ids(chunk_ids)
        
        # Create context objects with scores
        contexts = []
        score_dict = {chunk_id: score for chunk_id, score in search_results}
        
        for chunk in chunks:
            contexts.append({
                'chunk_id': chunk['id'],
                'text': chunk['text'],
                'score': score_dict[chunk['id']],
                'source_title': chunk['source_title'],
                'source_url': chunk['source_url']
            })
        
        # Generate answer
        answer, abstained, abstain_reason = self.generate_answer(query, contexts, mode)
        
        return {
            'answer': answer,
            'contexts': contexts,
            'reranker_used': mode,
            'abstained': abstained,
            'abstain_reason': abstain_reason
        }

def test_search():
    """Test the search functionality"""
    engine = SearchEngine()
    
    # Test queries
    queries = [
        "What is ISO 13849?",
        "PLC safety requirements",
        "How to calculate Performance Level d?",
        "Emergency stop standards"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Test all modes
        for mode in ['baseline', 'hybrid', 'learned']:
            try:
                result = engine.search_and_answer(query, k=3, mode=mode)
                print(f"\n{mode.upper()} MODE:")
                print(f"Abstained: {result['abstained']}")
                if result['abstained']:
                    print(f"Reason: {result['abstain_reason']}")
                else:
                    print(f"Answer: {result['answer'][:100]}...")
                print(f"Top score: {result['contexts'][0]['score']:.3f}")
            except Exception as e:
                print(f"{mode.upper()} MODE: Error - {e}")
        
        print("-" * 80)

if __name__ == "__main__":
    test_search()