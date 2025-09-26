import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import pickle
import os
from database import ChunkDatabase

class EmbeddingIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunk_id_mapping = None
        self.db = ChunkDatabase()
        
        # Paths for saving/loading
        self.index_path = "data/faiss_index.bin"
        self.mapping_path = "data/chunk_mapping.pkl"
        self.model_path = "data/model_name.txt"
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Set random seed for reproducibility
            np.random.seed(42)
    
    def build_index(self, batch_size: int = 32):
        """Build FAISS index from all chunks in database"""
        self.load_model()
        
        # Get all chunk IDs
        chunk_ids = self.db.get_all_chunk_ids()
        if not chunk_ids:
            raise ValueError("No chunks found in database. Run ingest.py first.")
        
        print(f"Building index for {len(chunk_ids)} chunks...")
        
        # Get chunk texts
        chunks = self.db.get_chunks_by_ids(chunk_ids)
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunk ID mapping
        self.chunk_id_mapping = chunk_ids
        
        print(f"Index built with {self.index.ntotal} vectors of dimension {dimension}")
        self.save_index()
    
    def save_index(self):
        """Save the FAISS index and mapping to disk"""
        faiss.write_index(self.index, self.index_path)
        
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.chunk_id_mapping, f)
        
        with open(self.model_path, 'w') as f:
            f.write(self.model_name)
        
        print(f"Index saved to {self.index_path}")
    
    def load_index(self):
        """Load the FAISS index and mapping from disk"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        self.index = faiss.read_index(self.index_path)
        
        with open(self.mapping_path, 'rb') as f:
            self.chunk_id_mapping = pickle.load(f)
        
        with open(self.model_path, 'r') as f:
            saved_model_name = f.read().strip()
            if saved_model_name != self.model_name:
                print(f"Warning: Model name mismatch. Expected {self.model_name}, got {saved_model_name}")
        
        self.load_model()
        print(f"Index loaded with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Search for similar chunks"""
        if self.index is None:
            self.load_index()
        
        # Encode query
        query_embedding = self.model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert to chunk IDs and scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                chunk_id = self.chunk_id_mapping[idx]
                results.append((chunk_id, float(score)))
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.index is not None:
            return self.index.d
        elif self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        else:
            self.load_model()
            return self.model.get_sentence_embedding_dimension()

def main():
    """Build the embedding index"""
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Check if database exists
    db = ChunkDatabase()
    chunk_count = db.get_chunk_count()
    
    if chunk_count == 0:
        print("No chunks found in database. Please run ingest.py first.")
        return
    
    print(f"Found {chunk_count} chunks in database")
    
    # Build index
    embedding_index = EmbeddingIndex()
    embedding_index.build_index()
    
    # Test the index
    print("\nTesting index...")
    results = embedding_index.search("safety requirements", k=3)
    print(f"Test search returned {len(results)} results")
    for chunk_id, score in results:
        chunk = db.get_chunk_by_id(chunk_id)
        print(f"  Chunk {chunk_id} (score: {score:.3f}): {chunk['text'][:100]}...")

if __name__ == "__main__":
    main()