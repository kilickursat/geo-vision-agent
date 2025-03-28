import runpod
import torch
import base64
import io
import tempfile
import traceback
from typing import Dict, Any, List, Tuple
from PIL import Image
import PyPDF2
import pdf2image

# Import ColPali model components
from colpali_engine.models import ColPali, ColPaliProcessor


# Add at beginning of handler.py
import os
import shutil

def check_and_clear_space():
    """Check available space and clean up if needed"""
    try:
        stats = shutil.disk_usage("/")
        free_mb = stats.free / (1024 * 1024)
        print(f"Available disk space: {free_mb:.2f} MB")
        
        if free_mb < 1000:  # If less than 1GB free
            print("Low disk space detected. Cleaning caches...")
            os.system("rm -rf /root/.cache/huggingface/hub/*")
            os.system("rm -rf /tmp/*")
            os.system("pip cache purge")
            
            # Check space after cleanup
            stats = shutil.disk_usage("/")
            free_mb = stats.free / (1024 * 1024)
            print(f"Available disk space after cleanup: {free_mb:.2f} MB")
            
        return free_mb
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return 0

# Call this function before model loading
check_and_clear_space()

# --- Global model initialization (executed once per worker startup) ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vidore/colpali-v1.3"

# Initialize the model globally to avoid reloading
try:
    print(f"Loading ColPali model ({MODEL_NAME}) on {DEVICE}...")
    colpali_model = ColPali.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE=="cuda:0" else torch.float32,
        device_map=DEVICE,
    ).eval()
    colpali_processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
    print("ColPali model loaded successfully.")
except Exception as e:
    print(f"FATAL: Failed to load ColPali model: {e}")
    print(traceback.format_exc())
    colpali_model = None
    colpali_processor = None

def pdf_to_images_and_text(pdf_bytes: bytes) -> Tuple[List[Image.Image], List[str]]:
    """Convert PDF bytes to a list of PIL images and extract text."""
    images = []
    texts = []
    
    # Read PDF with PyPDF2
    with io.BytesIO(pdf_bytes) as data:
        reader = PyPDF2.PdfReader(data)
        num_pages = len(reader.pages)
        
        # Extract text from each page
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            texts.append(page.extract_text())
        
        # Convert PDF to images using pdf2image
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            
            # Adjust DPI as needed for quality vs performance
            pdf_images = pdf2image.convert_from_path(
                temp_pdf.name, 
                dpi=150,
                fmt="jpeg"
            )
            
            images.extend(pdf_images)
    
    return images, texts

def process_pdf_with_colpali(pdf_bytes: bytes, query: str) -> Dict[str, Any]:
    """Process PDF using ColPali model."""
    if not colpali_model or not colpali_processor:
        return {"error": "ColPali model not available on worker."}

    try:
        # Convert PDF to images and extract text
        pdf_images, pdf_texts = pdf_to_images_and_text(pdf_bytes)
        
        # Process images with ColPali
        batch_images = colpali_processor.process_images(pdf_images).to(colpali_model.device)
        
        # Get image embeddings
        with torch.no_grad():
            image_embeddings = colpali_model(**batch_images)
        
        # Process query if provided
        query_scores = {}
        if query:
            batch_query = colpali_processor.process_queries([query]).to(colpali_model.device)
            with torch.no_grad():
                query_embedding = colpali_model(**batch_query)
            
            # Calculate similarity scores
            scores = colpali_processor.score_multi_vector(query_embedding, image_embeddings)
            query_scores = {
                "query": query,
                "page_scores": [float(score) for score in scores[0]]
            }
        
        # Find relevant sections based on scores
        relevant_sections = []
        if query and len(pdf_texts) > 0:
            # Sort pages by score
            page_scores = query_scores["page_scores"]
            scored_pages = sorted(
                [(i, page_scores[i], pdf_texts[i]) for i in range(len(page_scores))],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top 3 most relevant pages
            top_pages = scored_pages[:min(3, len(scored_pages))]
            
            # Extract relevant snippets
            for page_idx, score, text in top_pages:
                # Split text into paragraphs
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                
                # Filter short fragments
                filtered_paragraphs = []
                for para in paragraphs:
                    if len(para) > 30:  # Avoid tiny fragments
                        filtered_paragraphs.append(para)
                
                relevant_sections.append({
                    "page_number": page_idx + 1,
                    "similarity_score": round(float(score), 4),
                    "content": text,
                    "snippets": filtered_paragraphs[:3]  # Take up to 3 paragraphs
                })
        
        # Return the results
        return {
            "status": "success",
            "num_pages": len(pdf_images),
            "page_text": pdf_texts,
            "embedding_dimensions": image_embeddings.embeddings.shape[1],
            "tokens_per_page": image_embeddings.embeddings.shape[2],
            "query_scores": query_scores,
            "relevant_sections": relevant_sections,
            "total_pages": len(pdf_images)
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing PDF: {e}")
        print(error_trace)
        return {"status": "error", "message": str(e), "trace": error_trace}

def handler(job):
    """
    Runpod Serverless handler function.
    Expects job['input'] to be a dict like:
    {
        "pdf_base64": "BASE64_ENCODED_PDF_STRING",
        "query": "Search query text"
    }
    """
    job_input = job.get('input', {})
    pdf_base64 = job_input.get('pdf_base64')
    query = job_input.get('query', '')  # Make query optional
    
    if not pdf_base64:
        return {"status": "error", "message": "Missing 'pdf_base64' in input"}
    
    try:
        # Decode the PDF data
        pdf_bytes = base64.b64decode(pdf_base64)
    except Exception as e:
        return {"status": "error", "message": f"Failed to decode base64 PDF data: {e}"}
    
    # Process the PDF
    analysis_result = process_pdf_with_colpali(pdf_bytes, query)
    
    # Return the result
    return analysis_result

# Start the Runpod serverless worker
runpod.serverless.start({"handler": handler})
