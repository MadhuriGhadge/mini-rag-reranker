# Mini RAG + Reranker System for Industrial Safety Documents

# A complete question-answering service built over 20 industrial safety PDF documents

```bash
# 1. Create environment & install dependencies
python -m venv rag_env
rag_env\Scripts\activate
pip install -r requirements.txt

# 2. Run setup (extract PDFs, process docs, build index, train reranker, evaluate)
python setup.py

# 3. Start API
python app.py

# 4. Example Queries
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"q": "What is ISO 13849?", "k": 3, "mode": "learned"}'

curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"q": "What is the torque for a 3/8-inch socket on machine X model 12A?", "k": 3, "mode": "hybrid"}'


