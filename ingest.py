import os
import json
import re
from typing import List, Dict
from pathlib import Path
import PyPDF2
from database import ChunkDatabase

class DocumentProcessor:
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db = ChunkDatabase()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (simple heuristics)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely headers/footers (short lines with numbers or common words)
            if len(line) > 10 and not re.match(r'^\d+$', line) and not re.match(r'^page \d+', line.lower()):
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines)
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        # First try to split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph is too long, split by sentences
            if len(para) > self.chunk_size:
                sentences = re.split(r'[.!?]+', para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            # Overlap: keep last few words
                            words = current_chunk.split()
                            overlap_words = words[-10:] if len(words) > 10 else words
                            current_chunk = ' '.join(overlap_words) + ' ' + sentence
                        else:
                            current_chunk = sentence
                    else:
                        current_chunk += ' ' + sentence
            else:
                # Normal paragraph handling
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        # Overlap: keep last few words
                        words = current_chunk.split()
                        overlap_words = words[-10:] if len(words) > 10 else words
                        current_chunk = ' '.join(overlap_words) + ' ' + para
                    else:
                        current_chunk = para
                else:
                    current_chunk += ' ' + para if current_chunk else para
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        return [chunk for chunk in chunks if len(chunk.split()) >= 10]
    
    def process_document(self, pdf_path: str, source_info: Dict) -> int:
        """Process a single document and store chunks"""
        print(f"Processing {pdf_path}...")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print(f"Warning: No text extracted from {pdf_path}")
            return 0
        
        # Clean text
        clean_text = self.clean_text(raw_text)
        
        # Chunk text
        chunks = self.chunk_text(clean_text)
        
        if not chunks:
            print(f"Warning: No chunks created from {pdf_path}")
            return 0
        
        # Store chunks in database
        chunk_count = 0
        for i, chunk in enumerate(chunks):
            self.db.insert_chunk(
                source_file=pdf_path,
                source_title=source_info['title'],
                source_url=source_info['url'],
                chunk_index=i,
                text=chunk,
                is_first_paragraph=(i == 0)
            )
            chunk_count += 1
        
        print(f"Created {chunk_count} chunks from {source_info['title']}")
        return chunk_count
    
    def process_all_documents(self, pdfs_dir: str, sources_file: str):
        """Process all documents in the directory"""
        # Load source information
        with open(sources_file, 'r') as f:
            sources = json.load(f)
        
        total_chunks = 0
        processed_files = 0
        
        # Get all PDF files
        pdf_files = [f for f in os.listdir(pdfs_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"ERROR: No PDF files found in {pdfs_dir}/")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process\n")
        
        # Process each PDF
        for filename in pdf_files:
            pdf_path = os.path.join(pdfs_dir, filename)
            
            # Try to match PDF filename with sources by looking for similar titles
            source_info = None
            
            # Create a searchable version of the filename (without .pdf and normalized)
            search_name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').lower()
            
            # Try to find matching source
            for source in sources:
                source_title_lower = source['title'].lower()
                # Check if filename is contained in title or vice versa
                if search_name in source_title_lower or any(word in source_title_lower for word in search_name.split() if len(word) > 3):
                    source_info = source
                    break
            
            # If no match found, create default source info
            if not source_info:
                print(f"Warning: No source match found for {filename}, using default info")
                source_info = {
                    'title': filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title(),
                    'url': f'local://pdfs/{filename}'
                }
            
            chunk_count = self.process_document(pdf_path, source_info)
            total_chunks += chunk_count
            processed_files += 1
        
        print(f"\nProcessing complete:")
        print(f"Files processed: {processed_files}")
        print(f"Total chunks created: {total_chunks}")
        if processed_files > 0:
            print(f"Average chunks per file: {total_chunks/processed_files:.1f}")


def main():
    """Main processing function"""
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process all documents
    processor.process_all_documents("pdfs", "sources.json")

if __name__ == "__main__":
    main()