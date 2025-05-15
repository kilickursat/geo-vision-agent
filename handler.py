import runpod
import torch
import base64
import io
import tempfile
import traceback
import os
import shutil
import re
from typing import Dict, Any, List, Tuple
from PIL import Image
import PyPDF2
import pdf2image

# Try to import NLTK for better sentence tokenization
try:
    import nltk
    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
    SENTENCE_TOKENIZER = sent_tokenize
    print("Using NLTK sentence tokenizer.")
except ImportError:
    def regex_sentence_tokenizer(text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
        return [s.strip() for s in sentences if s.strip()]
    SENTENCE_TOKENIZER = regex_sentence_tokenizer
    print("NLTK not found. Using regex sentence tokenizer.")

# Import ColPali model components
from colpali_engine.models import ColPali, ColPaliProcessor

# Function to check and clear disk space
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

# Check disk space before loading model
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
    """Process PDF using ColPali model with enhanced snippet extraction."""
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
        
        # Get embedding dimensions based on actual shape
        embedding_dim = 0
        tokens_per_page = 0
        if isinstance(image_embeddings, torch.Tensor):
            # Check the actual shape structure of ColPali output
            if len(image_embeddings.shape) == 3:  # (batch, num_tokens, dim)
                embedding_dim = image_embeddings.shape[2]
                tokens_per_page = image_embeddings.shape[1]
            elif len(image_embeddings.shape) == 2:  # (batch, dim)
                embedding_dim = image_embeddings.shape[1]
        
        # Process query if provided
        query_scores = {}
        query_text_embedding = None
        if query:
            # Get query embedding for image similarity
            batch_query = colpali_processor.process_queries([query]).to(colpali_model.device)
            with torch.no_grad():
                query_embedding = colpali_model(**batch_query)
                # Store this for text-level similarity later
                query_text_embedding = query_embedding
            
            # Calculate page-level similarity scores
            scores = colpali_processor.score_multi_vector(query_embedding, image_embeddings)
            query_scores = {
                "query": query,
                "page_scores": [float(score) for score in scores[0]]
            }
        
        # Find relevant sections based on scores
        refined_relevant_sections = []
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
            
            # Use global sentence tokenizer
            sentence_tokenizer = SENTENCE_TOKENIZER
            
            # Process each top page for better snippet extraction
            for page_idx, score, text in top_pages:
                # Skip empty pages
                if not text.strip():
                    continue
                    
                # Split text into sentences for more granular analysis
                try:
                    # Normalize newlines first to better handle paragraphs
                    normalized_text = re.sub(r'\n+', '\n', text)
                    paragraphs = [p.strip() for p in normalized_text.split('\n') if p.strip()]
                    
                    # Extract sentences from each paragraph
                    sentences = []
                    for para in paragraphs:
                        para_sentences = sentence_tokenizer(para)
                        sentences.extend(para_sentences)
                    
                    # Filter out too short sentences
                    sentences = [s for s in sentences if len(s) > 15]
                    
                    # If no sentences were found, fall back to paragraphs
                    if not sentences:
                        if len(paragraphs) > 0:
                            refined_relevant_sections.append({
                                "page_number": page_idx + 1,
                                "similarity_score": round(float(score), 4),
                                "content": text,
                                "snippets": paragraphs[:3]  # Original fallback behavior
                            })
                        continue
                    
                    # Verify we have the query embedding already
                    if query_text_embedding is None:
                        batch_query = colpali_processor.process_queries([query]).to(colpali_model.device)
                        with torch.no_grad():
                            query_text_embedding = colpali_model(**batch_query)
                    
                    # Process sentences in batches to avoid OOM
                    batch_size = 16
                    all_sentence_scores = []
                    
                    for i in range(0, len(sentences), batch_size):
                        batch_sentences = sentences[i:i+batch_size]
                        batch_sentences_processed = colpali_processor.process_queries(batch_sentences).to(colpali_model.device)
                        
                        with torch.no_grad():
                            sentence_embeddings = colpali_model(**batch_sentences_processed)
                        
                        # Calculate similarity scores between query and sentences
                        batch_similarities = colpali_processor.score_multi_vector(
                            query_text_embedding, 
                            sentence_embeddings
                        )[0]
                        
                        # Store scores with sentence indices
                        for j, sim in enumerate(batch_similarities):
                            all_sentence_scores.append((i+j, float(sim)))
                    
                    # Sort sentences by similarity score (descending)
                    sorted_sentence_scores = sorted(all_sentence_scores, key=lambda x: x[1], reverse=True)
                    
                    # Take top 5 sentences (or fewer if there aren't that many)
                    top_k = min(5, len(sorted_sentence_scores))
                    top_sentence_indices = [idx for idx, _ in sorted_sentence_scores[:top_k]]
                    
                    # Sort indices to maintain original text flow
                    top_sentence_indices.sort()
                    
                    # Group adjacent sentences for context
                    grouped_snippets = []
                    current_group = []
                    prev_idx = -2  # Start with a non-adjacent value
                    
                    for idx in top_sentence_indices:
                        if idx == prev_idx + 1:  # Adjacent to previous sentence
                            current_group.append(sentences[idx])
                        else:
                            if current_group:  # Save previous group if exists
                                grouped_snippets.append(" ".join(current_group))
                            current_group = [sentences[idx]]  # Start new group
                        prev_idx = idx
                    
                    # Don't forget the last group
                    if current_group:
                        grouped_snippets.append(" ".join(current_group))
                    
                    # Ensure we have context for single-sentence snippets
                    enhanced_snippets = []
                    for snippet_idx in top_sentence_indices:
                        # If this snippet isn't already part of a group with context
                        if len(sentences[snippet_idx].split()) > 3:  # Only for substantive sentences
                            context = []
                            # Add preceding sentence if available and not already in a top snippet
                            if snippet_idx > 0 and snippet_idx-1 not in top_sentence_indices:
                                context.append(sentences[snippet_idx-1])
                            
                            # Add the main sentence
                            context.append(sentences[snippet_idx])
                            
                            # Add following sentence if available and not already in a top snippet
                            if snippet_idx < len(sentences)-1 and snippet_idx+1 not in top_sentence_indices:
                                context.append(sentences[snippet_idx+1])
                                
                            enhanced_snippets.append(" ".join(context))
                    
                    # Combine grouped snippets with enhanced snippets, avoiding duplicates
                    all_snippets = grouped_snippets + [s for s in enhanced_snippets if s not in grouped_snippets]
                    
                    # Take up to 3 best snippets
                    final_snippets = all_snippets[:3]
                    
                    # Add to refined sections
                    refined_relevant_sections.append({
                        "page_number": page_idx + 1,
                        "similarity_score": round(float(score), 4),
                        "content": text,
                        "snippets": final_snippets
                    })
                    
                except Exception as snippet_error:
                    print(f"Error processing snippets for page {page_idx+1}: {snippet_error}")
                    print(traceback.format_exc())
                    
                    # Fallback to standard paragraph extraction
                    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                    refined_relevant_sections.append({
                        "page_number": page_idx + 1,
                        "similarity_score": round(float(score), 4),
                        "content": text,
                        "snippets": paragraphs[:3]  # Original fallback behavior
                    })
        
        # Return the results
        return {
            "status": "success",
            "num_pages": len(pdf_images),
            "page_text": pdf_texts,
            "embedding_dimensions": embedding_dim,
            "tokens_per_page": tokens_per_page,
            "query_scores": query_scores,
            "relevant_sections": refined_relevant_sections,
            "total_pages": len(pdf_images)
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing PDF: {e}")
        print(error_trace)
        return {"status": "error", "message": str(e), "trace": error_trace}

def handler(job):
    """
    RunPod Serverless handler function.
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

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})
