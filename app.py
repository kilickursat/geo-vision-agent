import streamlit as st
import torch
from PIL import Image
import io
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument
from duckduckgo_search import DDGS
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
import PyPDF2
import json
import time
import markdown
from bs4 import BeautifulSoup
import base64
from io import BytesIO

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.info("Downloading language model for the first time...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set page configuration
st.set_page_config(
    page_title="Enhanced Document OCR Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 16px;
            height: auto;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4682B4 !important;
            color: white !important;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            border-radius: 0.5rem;
            background-color: white;
            box-shadow: 0 0.15rem 1.75rem 0 rgba(33, 40, 50, 0.15);
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        .chart-container {
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 1.75rem 0 rgba(33, 40, 50, 0.15);
            padding: 1rem;
            margin-top: 1rem;
        }
        .search-result {
            background-color: white;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 1.75rem 0 rgba(33, 40, 50, 0.15);
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .stProgress .st-bo {
            background-color: #4682B4;
        }
        .stSelectbox label, .stSlider label {
            font-weight: 500;
        }
        h1, h2, h3 {
            color: #1E3A8A;
        }
    </style>
    """, unsafe_allow_html=True)

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache model loading to improve performance
@st.cache_resource
def load_model(model_name="ds4sd/SmolDocling-256M-preview"):
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Define OCR pipeline function
def ocr_pipeline(image, processor, model):
    try:
        # Prepare input message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."}
                ]
            },
        ]
        
        # Process input
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
        
        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        trimmed_generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        
        # Decode output
        doctags = processor.batch_decode(trimmed_generated_ids, skip_special_tokens=False)[0].lstrip()
        
        # Create Docling document
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        doc = DoclingDocument(name="Document")
        doc.load_from_doctags(doctags_doc)
        
        return doc.export_to_markdown()
    except Exception as e:
        return f"Error in OCR processing: {str(e)}"

# Process PDF function (handles multi-page PDFs)
def process_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        
        all_pages = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            
            # Save PDF page as image
            pdf_bytes = BytesIO()
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(page)
            pdf_writer.write(pdf_bytes)
            pdf_bytes.seek(0)
            
            # Convert PDF page to image using pdf2image
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes.read())
            if images:
                all_pages.append(images[0])
        
        return all_pages, num_pages
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return [], 0

# DuckDuckGo search function
def search_duckduckgo(query, max_results=5):
    try:
        search_results = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                search_results.append({
                    'title': r['title'],
                    'href': r['href'],
                    'body': r['body']
                })
        return search_results
    except Exception as e:
        st.error(f"Error in DuckDuckGo search: {str(e)}")
        return []

# Extract entities and important terms from text
def extract_entities(text):
    try:
        doc = nlp(text)
        
        # Extract named entities
        entities = {ent.text: ent.label_ for ent in doc.ents}
        
        # Extract keywords (excluding stopwords)
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        # Count word frequencies and get top keywords
        word_freq = Counter(filtered_tokens)
        common_words = word_freq.most_common(10)
        
        return entities, common_words
    except Exception as e:
        st.error(f"Error extracting entities: {str(e)}")
        return {}, []

# Extract text features for visualization
def extract_text_features(text):
    try:
        # Basic text statistics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
        
        # Word frequencies (excluding stopwords)
        word_freq = Counter(filtered_words)
        common_words = word_freq.most_common(10)
        
        # Average word length
        avg_word_length = sum(len(word) for word in filtered_words) / len(filtered_words) if filtered_words else 0
        
        # Sentence length distribution
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        
        # Entity analysis
        doc = nlp(text)
        entity_types = Counter([ent.label_ for ent in doc.ents])
        
        # Readability (simple metric)
        total_syllables = sum(count_syllables(word) for word in filtered_words)
        if len(sentences) > 0 and len(filtered_words) > 0:
            flesch_kincaid = 206.835 - 1.015 * (len(filtered_words) / len(sentences)) - 84.6 * (total_syllables / len(filtered_words))
        else:
            flesch_kincaid = 0
            
        features = {
            "total_sentences": len(sentences),
            "total_words": len(words),
            "total_unique_words": len(set(word.lower() for word in filtered_words)),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": sum(sentence_lengths) / len(sentences) if sentences else 0,
            "readability_score": flesch_kincaid,
            "sentence_lengths": sentence_lengths,
            "common_words": common_words,
            "entity_types": dict(entity_types)
        }
        
        return features
    except Exception as e:
        st.error(f"Error extracting text features: {str(e)}")
        return {}

# Helper function for syllable counting (for readability metrics)
def count_syllables(word):
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    # Remove ending silent e
    if word.endswith('e'):
        word = word[:-1]
    
    # Count vowel groups
    count = 0
    vowels = "aeiouy"
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    return max(1, count)  # At least 1 syllable per word

# Create visualizations based on extracted features
def create_visualizations(features):
    visualizations = {}
    
    # Word frequency chart
    if "common_words" in features and features["common_words"]:
        words, counts = zip(*features["common_words"])
        word_freq_fig = px.bar(
            x=words, y=counts,
            labels={"x": "Words", "y": "Frequency"},
            title="Most Common Words",
            color=counts,
            color_continuous_scale="Blues"
        )
        word_freq_fig.update_layout(
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            coloraxis_showscale=False
        )
        visualizations["word_frequency"] = word_freq_fig
    
    # Entity types chart
    if "entity_types" in features and features["entity_types"]:
        entity_labels = list(features["entity_types"].keys())
        entity_counts = list(features["entity_types"].values())
        
        entity_fig = px.pie(
            values=entity_counts,
            names=entity_labels,
            title="Named Entity Types",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        entity_fig.update_layout(
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='white'
        )
        visualizations["entity_types"] = entity_fig
    
    # Text statistics chart
    if "total_sentences" in features:
        stats_labels = ["Sentences", "Words", "Unique Words"]
        stats_values = [
            features.get("total_sentences", 0),
            features.get("total_words", 0),
            features.get("total_unique_words", 0)
        ]
        
        stats_fig = px.bar(
            x=stats_labels,
            y=stats_values,
            title="Document Statistics",
            labels={"x": "Metric", "y": "Count"},
            color=stats_values,
            color_continuous_scale="Blues"
        )
        stats_fig.update_layout(
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white',
            coloraxis_showscale=False
        )
        visualizations["text_stats"] = stats_fig
    
    # Sentence length distribution
    if "sentence_lengths" in features and features["sentence_lengths"]:
        sent_lengths = features["sentence_lengths"]
        sent_fig = px.histogram(
            x=sent_lengths,
            nbins=20,
            title="Sentence Length Distribution",
            labels={"x": "Words per Sentence", "y": "Frequency"},
            color_discrete_sequence=["#4682B4"]
        )
        sent_fig.update_layout(
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        visualizations["sentence_lengths"] = sent_fig
    
    # Readability gauge chart
    if "readability_score" in features:
        readability = min(max(features["readability_score"], 0), 100)
        
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=readability,
            title={"text": "Readability Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#4682B4"},
                "steps": [
                    {"range": [0, 30], "color": "#FF9AA2"},
                    {"range": [30, 70], "color": "#FFDAC1"},
                    {"range": [70, 100], "color": "#B5EAD7"}
                ]
            }
        ))
        gauge_fig.update_layout(
            font=dict(size=12),
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='white'
        )
        visualizations["readability"] = gauge_fig
    
    return visualizations

# Function to convert markdown to HTML safely
def markdown_to_html(markdown_text):
    html = markdown.markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove potentially unsafe tags/attributes
    for tag in soup.find_all():
        if tag.name in ["script", "iframe", "form", "object", "embed"]:
            tag.decompose()
        
        # Remove event handler attributes
        for attr in list(tag.attrs.keys()):
            if attr.startswith("on"):
                del tag.attrs[attr]
    
    return str(soup)

# Function to display search results
def display_search_results(results):
    if not results:
        st.info("No search results found.")
        return
    
    for i, result in enumerate(results):
        with st.container():
            st.markdown(f"""
            <div class="search-result">
                <h4><a href="{result['href']}" target="_blank">{result['title']}</a></h4>
                <p>{result['body']}</p>
                <p><small>{result['href']}</small></p>
            </div>
            """, unsafe_allow_html=True)

# Main application
def main():
    apply_custom_css()
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/text-recognition.png", width=80)
    st.sidebar.title("Document OCR")
    
    # Model selection
    model_option = st.sidebar.selectbox(
        "Select OCR Model",
        ["ds4sd/SmolDocling-256M-preview"],
        help="Select the model for OCR processing"
    )
    
    # Cache the selected model
    processor, model = load_model(model_option)
    
    # Settings
    st.sidebar.subheader("Settings")
    search_enabled = st.sidebar.checkbox("Enable DuckDuckGo Search", value=True)
    search_depth = st.sidebar.slider("Search Result Depth", min_value=1, max_value=10, value=5)
    
    # Display settings
    st.sidebar.subheader("Display Settings")
    max_search_queries = st.sidebar.slider("Max Search Queries", min_value=1, max_value=5, value=3)
    show_raw_text = st.sidebar.checkbox("Show Raw OCR Text", value=False)
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This application extracts text from documents using OCR, "
        "performs analysis, and provides relevant search results. "
        "Upload any document image or PDF to begin."
    )
    
    # Main content
    st.title("Enhanced Document OCR Text Extractor")
    st.markdown(
        "Upload an image or PDF containing text to extract its content, analyze features, "
        "and find relevant information through DuckDuckGo search."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=["jpg", "jpeg", "png", "tiff", "pdf"],
        help="Upload a document image or PDF file to extract text"
    )
    
    # Process the uploaded file
    if uploaded_file is not None:
        # Check if processor and model loaded successfully
        if processor is None or model is None:
            st.error("Failed to load OCR model. Please check your internet connection and try again.")
            return
        
        # Determine file type
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        # Initialize variables
        images = []
        current_page = 0
        num_pages = 1
        extracted_text = ""
        
        # Process based on file type
        if file_extension == "pdf":
            # Process PDF (multi-page)
            images, num_pages = process_pdf(uploaded_file)
            if num_pages > 0:
                st.success(f"PDF loaded successfully with {num_pages} pages.")
                # Add page selector if more than one page
                if num_pages > 1:
                    current_page = st.slider("Select Page", min_value=0, max_value=num_pages-1, value=0)
                
                # Process selected page
                if 0 <= current_page < len(images):
                    current_image = images[current_page]
                    
                    # Display a spinner during processing
                    with st.spinner(f"Processing page {current_page+1} of {num_pages}..."):
                        # Extract text using OCR pipeline
                        extracted_text = ocr_pipeline(current_image, processor, model)
                else:
                    st.error("Selected page is out of range.")
            else:
                st.error("Could not extract pages from the PDF.")
        else:
            # Process single image
            try:
                # Read the image
                image_bytes = uploaded_file.read()
                current_image = Image.open(io.BytesIO(image_bytes))
                images = [current_image]
                
                # Display a spinner during processing
                with st.spinner("Processing the image..."):
                    # Extract text using OCR pipeline
                    extracted_text = ocr_pipeline(current_image, processor, model)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                return
        
        # Create tabs for different sections
        tabs = st.tabs(["OCR Results", "Search Results", "Analysis & Visualization"])
        
        # OCR Results Tab
        with tabs[0]:
            if images:
                # Create two columns for image and text
                col1, col2 = st.columns(2)
                
                # Display the current image
                with col1:
                    st.subheader("Document Image")
                    st.image(images[current_page], use_column_width=True)
                
                # Display the extracted text
                with col2:
                    st.subheader("Extracted Text")
                    
                    # Option to show raw text or formatted
                    if show_raw_text:
                        st.text_area("Raw Text", extracted_text, height=400)
                    else:
                        # Convert markdown to safe HTML
                        safe_html = markdown_to_html(extracted_text)
                        st.markdown(safe_html, unsafe_allow_html=True)
                
                # Provide download button for the text
                st.download_button(
                    label="Download Extracted Text",
                    data=extracted_text,
                    file_name=f"extracted_text_page_{current_page+1}.md",
                    mime="text/markdown"
                )
        
        # Process text for search and analysis
        if extracted_text:
            # Extract entities and important terms
            entities, common_words = extract_entities(extracted_text)
            
            # Extract text features for visualization
            text_features = extract_text_features(extracted_text)
            
            # Generate search queries based on entities and common words
            search_queries = []
            
            # Add entity-based queries
            for entity, entity_type in list(entities.items())[:max_search_queries]:
                search_queries.append(f"{entity} {entity_type.lower()}")
            
            # Add common word-based queries if needed
            if len(search_queries) < max_search_queries and common_words:
                for word, _ in common_words[:max_search_queries - len(search_queries)]:
                    if len(word) > 3:  # Only use meaningful words
                        search_queries.append(word)
            
            # Search Results Tab
            with tabs[1]:
                if search_enabled and search_queries:
                    st.subheader("Relevant Information")
                    
                    # Display entity information
                    if entities:
                        st.markdown("### Key Entities Detected")
                        entity_df = pd.DataFrame(list(entities.items()), columns=["Entity", "Type"])
                        st.dataframe(entity_df, hide_index=True)
                    
                    # Display search results for each query
                    for i, query in enumerate(search_queries[:max_search_queries]):
                        st.markdown(f"### Search Results for \"{query}\"")
                        
                        with st.spinner(f"Searching for information about '{query}'..."):
                            results = search_duckduckgo(query, max_results=search_depth)
                            display_search_results(results)
                else:
                    st.info("Search is disabled or no relevant search queries could be generated.")
            
            # Analysis & Visualization Tab
            with tabs[2]:
                st.subheader("Document Analysis")
                
                # Create visualizations
                visualizations = create_visualizations(text_features)
                
                # Display document statistics
                st.markdown("### Document Statistics")
                
                # Create metrics in a row
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("Sentences", text_features.get("total_sentences", 0))
                
                with metrics_col2:
                    st.metric("Words", text_features.get("total_words", 0))
                
                with metrics_col3:
                    st.metric("Unique Words", text_features.get("total_unique_words", 0))
                
                with metrics_col4:
                    avg_sentence = round(text_features.get("avg_sentence_length", 0), 1)
                    st.metric("Avg. Sentence Length", f"{avg_sentence} words")
                
                # Display visualizations
                if visualizations:
                    # Arrange charts in a grid
                    viz_cols = st.columns(2)
                    
                    # Word frequency chart
                    if "word_frequency" in visualizations:
                        with viz_cols[0]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(visualizations["word_frequency"], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Entity types chart
                    if "entity_types" in visualizations:
                        with viz_cols[1]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(visualizations["entity_types"], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Sentence length distribution
                    if "sentence_lengths" in visualizations:
                        with viz_cols[0]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(visualizations["sentence_lengths"], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Readability gauge
                    if "readability" in visualizations:
                        with viz_cols[1]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(visualizations["readability"], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Not enough data to generate visualizations.")

if __name__ == "__main__":
    main()
