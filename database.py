import sqlite3
import json
from typing import List, Dict, Optional, Tuple
import os

class ChunkDatabase:
    def __init__(self, db_path: str = "data/chunks.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables and FTS index"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                source_title TEXT NOT NULL,
                source_url TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                chunk_length INTEGER NOT NULL,
                is_first_paragraph BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # FTS5 table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                source_title,
                content_rowid=id
            )
        """)
        
        conn.commit()
        conn.close()
    
    def insert_chunk(self, source_file: str, source_title: str, source_url: str, 
                    chunk_index: int, text: str, is_first_paragraph: bool = False) -> int:
        """Insert a chunk and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chunks (source_file, source_title, source_url, chunk_index, 
                              text, chunk_length, is_first_paragraph)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source_file, source_title, source_url, chunk_index, 
              text, len(text), is_first_paragraph))
        
        chunk_id = cursor.lastrowid
        
        # Insert into FTS index
        cursor.execute("""
            INSERT INTO chunks_fts (rowid, text, source_title)
            VALUES (?, ?, ?)
        """, (chunk_id, text, source_title))
        
        conn.commit()
        conn.close()
        return chunk_id
    
    def get_chunk_by_id(self, chunk_id: int) -> Optional[Dict]:
        """Get a chunk by its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, source_file, source_title, source_url, chunk_index, 
                   text, chunk_length, is_first_paragraph
            FROM chunks WHERE id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'source_file': row[1],
                'source_title': row[2],
                'source_url': row[3],
                'chunk_index': row[4],
                'text': row[5],
                'chunk_length': row[6],
                'is_first_paragraph': row[7]
            }
        return None
    
    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Dict]:
        """Get multiple chunks by their IDs"""
        if not chunk_ids:
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(chunk_ids))
        cursor.execute(f"""
            SELECT id, source_file, source_title, source_url, chunk_index, 
                   text, chunk_length, is_first_paragraph
            FROM chunks WHERE id IN ({placeholders})
        """, chunk_ids)
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'id': row[0],
                'source_file': row[1],
                'source_title': row[2],
                'source_url': row[3],
                'chunk_index': row[4],
                'text': row[5],
                'chunk_length': row[6],
                'is_first_paragraph': row[7]
            })
        return result
    
    def fts_search(self, query: str, limit: int = 10) -> List[Tuple[int, float]]:
        """Full-text search returning (chunk_id, bm25_score) pairs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Clean query for FTS5 - remove special characters that cause issues
            # FTS5 doesn't like certain characters like "?"
            clean_query = query.replace('?', '').replace('"', '').replace("'", '').strip()
            
            if not clean_query:
                conn.close()
                return []
            
            # Split into words and create OR query
            words = clean_query.split()
            # Limit to first 5 words to avoid complexity
            words = words[:5]
            fts_query = ' OR '.join(words)
            
            # FTS5 with BM25 ranking
            cursor.execute("""
                SELECT rowid, bm25(chunks_fts) as score
                FROM chunks_fts 
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (fts_query, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            # Convert BM25 scores to positive values (FTS5 returns negative scores)
            return [(row[0], abs(row[1])) for row in results]
            
        except sqlite3.OperationalError as e:
            print(f"FTS search error for query '{query}': {e}")
            conn.close()
            # Return empty results on error
            return []
        
    def get_all_chunk_ids(self) -> List[int]:
        """Get all chunk IDs for embedding indexing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM chunks ORDER BY id")
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        count = cursor.fetchone()[0]
        conn.close()
        
        return count