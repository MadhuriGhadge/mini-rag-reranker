
import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {description} completed!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    if os.path.exists(filepath):
        print(f"✓ Found: {filepath}")
        return True
    else:
        print(f"✗ Missing: {filepath} - {description}")
        return False

def extract_pdfs(zip_path="industrial-safety-pdfs.zip", extract_dir="pdfs"):
    """Extract PDFs from zip file"""
    if not os.path.exists(zip_path):
        print(f"ERROR: {zip_path} not found!")
        print("Please ensure the industrial-safety-pdfs.zip file is in the current directory")
        return False
    
    # Create extraction directory
    os.makedirs(extract_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only PDF files
            pdf_files = [f for f in zip_ref.namelist() if f.endswith('.pdf')]
            for pdf_file in pdf_files:
                # Extract to pdfs/ directory
                zip_ref.extract(pdf_file, extract_dir)
                # Move to root of pdfs/ directory if in subdirectory
                src = os.path.join(extract_dir, pdf_file)
                dst = os.path.join(extract_dir, os.path.basename(pdf_file))
                if src != dst:
                    shutil.move(src, dst)
        
        # Clean up empty directories
        for root, dirs, files in os.walk(extract_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)  # Only removes empty directories
                except OSError:
                    pass  # Directory not empty, ignore
        
        pdf_count = len([f for f in os.listdir(extract_dir) if f.endswith('.pdf')])
        print(f"✓ Extracted {pdf_count} PDF files to {extract_dir}/")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to extract PDFs: {e}")
        return False

def setup_environment():
    """Set up the development environment"""
    print("Mini RAG + Reranker Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Check required files
    required_files = [
        ("sources.json", "Document source information"),
        ("industrial-safety-pdfs.zip", "PDF documents archive")
    ]
    
    all_files_present = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_files_present = False
    
    if not all_files_present:
        print("\nERROR: Missing required files. Please ensure all files are present.")
        return False
    
    return True

def main():
    """Main setup function"""
    if not setup_environment():
        sys.exit(1)
    
    # Extract PDFs
    if not extract_pdfs():
        sys.exit(1)
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        sys.exit(1)
    
    # Process documents
    if not run_command("python ingest.py", "Processing documents and creating chunks"):
        sys.exit(1)
    
    # Build embeddings index
    if not run_command("python embeddings.py", "Building embeddings index"):
        sys.exit(1)
    
    # Train learned reranker
    if not run_command("python reranker.py", "Training learned reranker"):
        sys.exit(1)
    
    # Run evaluation
    if not run_command("python evaluation.py", "Running evaluation"):
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("🎉 SETUP COMPLETE!")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python app.py")
    print("\n2. Test the API:")
    print('   curl -X POST http://localhost:8000/ask \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"q": "What is ISO 13849?", "k": 3, "mode": "hybrid"}\'')
    print("\n3. Check evaluation results:")
    print("   cat evaluation_report.md")
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
