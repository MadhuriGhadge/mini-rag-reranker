from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from search import SearchEngine

# Initialize FastAPI app
app = FastAPI(
    title="Mini RAG + Reranker API",
    description="Question answering service over industrial safety documents",
    version="1.0.0"
)

# Initialize search engine
search_engine = SearchEngine()

# Request models
class QuestionRequest(BaseModel):
    q: str  # Question
    k: int = 5  # Number of contexts to return
    mode: str = "baseline"  # Search mode: baseline, hybrid, or learned

class ContextResponse(BaseModel):
    chunk_id: int
    text: str
    score: float
    source_title: str
    source_url: str

class AnswerResponse(BaseModel):
    answer: Optional[str]
    contexts: List[ContextResponse]
    reranker_used: str
    abstained: bool
    abstain_reason: Optional[str]

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions
    """
    try:
        # Validate mode
        if request.mode not in ["baseline", "hybrid", "learned"]:
            raise HTTPException(
                status_code=400, 
                detail="Mode must be 'baseline', 'hybrid', or 'learned'"
            )
        
        # Validate k
        if request.k < 1 or request.k > 20:
            raise HTTPException(
                status_code=400,
                detail="k must be between 1 and 20"
            )
        
        # Validate query
        if not request.q.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Perform search and generate answer
        result = search_engine.search_and_answer(
            query=request.q.strip(),
            k=request.k,
            mode=request.mode
        )
        
        # Convert contexts to response format
        contexts = [
            ContextResponse(
                chunk_id=ctx['chunk_id'],
                text=ctx['text'],
                score=ctx['score'],
                source_title=ctx['source_title'],
                source_url=ctx['source_url']
            )
            for ctx in result['contexts']
        ]
        
        return AnswerResponse(
            answer=result['answer'],
            contexts=contexts,
            reranker_used=result['reranker_used'],
            abstained=result['abstained'],
            abstain_reason=result['abstain_reason']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test that the search engine can load
        search_engine._load_embedding_index()
        return {"status": "healthy", "message": "API is running and index is loaded"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Mini RAG + Reranker API",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask - Ask a question about industrial safety",
            "health": "GET /health - Check API health",
        },
        "example_request": {
            "url": "/ask",
            "method": "POST",
            "body": {
                "q": "What is ISO 13849?",
                "k": 5,
                "mode": "hybrid"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )