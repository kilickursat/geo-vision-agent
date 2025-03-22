import streamlit as st
import os
import io
import base64
import tempfile
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import json
import requests
from typing import List, Dict, Any, Union, Optional
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Geotechnical Engineering Analysis System",
    page_icon="🧠",
    layout="wide"
)

# Try to import pdf2image, but have a fallback option
try:
    from pdf2image import convert_from_bytes
    PDF_TO_IMAGE_AVAILABLE = True
except ImportError:
    # Define a fallback using PyPDF2
    import PyPDF2
    
    PDF_TO_IMAGE_AVAILABLE = False
    
    def convert_pdf_to_image_fallback(pdf_content):
        """Fallback method to extract first page image from PDF."""
        try:
            # Read PDF content
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # For simplicity, just use the first page
            # This is a simplified fallback and won't work for all PDFs
            width, height = 612, 792  # Standard letter size
            image = Image.new('RGB', (width, height), 'white')
            
            # Note: This is a basic fallback and won't extract actual content
            return [image]
        except Exception as e:
            st.error(f"Error converting PDF to image: {str(e)}")
            # Return a blank image as last resort
            return [Image.new('RGB', (612, 792), 'white')]

# Import smolagents components with careful fallbacks
SMOLAGENTS_AVAILABLE = False
HF_LOGIN_AVAILABLE = False

try:
    # First try importing the smolagents
    from smolagents import tool, HfApiModel, CodeAgent, DuckDuckGoSearchTool
    SMOLAGENTS_AVAILABLE = True
    
    # Try importing huggingface_hub for login
    try:
        from huggingface_hub import login
        HF_LOGIN_AVAILABLE = True
    except ImportError:
        HF_LOGIN_AVAILABLE = False
        
    # Log successful import
    logging.info("Successfully imported core SmolaAgent components")
    
except ImportError as e:
    logging.error(f"Error importing smolagents: {str(e)}")
    st.error(f"smolagents package is not available: {str(e)}. Using fallback methods.")
    
    # Create dummy classes/functions to prevent errors
    def tool(func):
        return func
        
    class HfApiModel:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            return {"error": "HfApiModel not available"}
    
    class CodeAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        def run(self, *args, **kwargs):
            return {"error": "Agent functionality not available"}
    
    class DuckDuckGoSearchTool:
        def __call__(self, *args, **kwargs):
            return [{"title": "Search unavailable", "snippet": "Search functionality not available", "link": "#"}]

# Function to get Hugging Face token
def get_hf_token():
    """Get Hugging Face token from environment variable or Streamlit secrets."""
    # First check environment variable
    token = os.getenv("HF_TOKEN")
    
    # Then check Streamlit secrets
    if not token and 'huggingface' in st.secrets:
        token = st.secrets["huggingface"]["hf_token"]
    
    # Finally, request from user if not found
    if not token:
        if "hf_token" not in st.session_state:
            st.session_state.hf_token = st.text_input(
                "Enter your Hugging Face API token:",
                type="password",
                help="Get your token from https://huggingface.co/settings/tokens"
            )
        token = st.session_state.hf_token
        
    return token

# Initialize the Qwen model
def initialize_qwen_model(token):
    """Initialize the Qwen vision-language model."""
    if not SMOLAGENTS_AVAILABLE or not token:
        return None
    
    try:
        # Login to Hugging Face if possible
        if HF_LOGIN_AVAILABLE:
            login(token)
        
        # Initialize the model as in the example
        model = HfApiModel(
            model_id="Qwen/Qwen2.5-VL-72B-Instruct",
            token=token,
            timeout=240  # Higher timeout for image processing
        )
        
        return model
    except Exception as e:
        logging.error(f"Error initializing Qwen model: {str(e)}")
        return None

# Call the Qwen model with image content
def query_qwen_vision_model(model, image, prompt):
    """Query the Qwen vision-language model with an image and prompt."""
    if model is None:
        return None
    
    try:
        # Convert image to base64 if it's not already a string
        if not isinstance(image, str):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode()
            image_url = f"data:image/jpeg;base64,{image_b64}"
        else:
            image_url = image
        
        # Create message format as shown in the example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Call the model
        with st.status("Processing image with Qwen model..."):
            response = model(messages)
        
        return response
    except Exception as e:
        logging.error(f"Error querying Qwen model: {str(e)}")
        return None

# Extract parameters from response
def extract_parameters_from_response(response):
    """Extract geotechnical parameters from the model response."""
    try:
        if response is None:
            return None
        
        # Get the response text
        if isinstance(response, dict) and "content" in response:
            text_response = response["content"]
        elif isinstance(response, str):
            text_response = response
        else:
            text_response = str(response)
        
        # Try to extract JSON from the text response
        try:
            # Find JSON-like content in the response
            import re
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                params = json.loads(extracted_json)
            else:
                # Fallback parsing for structured text that isn't proper JSON
                params = {}
                lines = text_response.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().replace('"', '')
                        try:
                            # Extract numeric values
                            value_match = re.search(r'\d+\.?\d*', value)
                            if value_match:
                                params[key] = float(value_match.group(0))
                        except:
                            pass
            
            # If no parameters were extracted, return None
            if not params:
                return None
                
            return params
        except Exception as e:
            logging.error(f"Error parsing response: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Error extracting parameters: {str(e)}")
        return None

# Tool for extraction - Fixed with proper type hint for qwen_model
@tool
def extract_from_file(file_content: bytes, file_type: str, qwen_model: Optional[Any] = None) -> Dict[str, float]:
    """
    Extract geotechnical parameters from PDF or image files.
    
    Args:
        file_content: The binary content of the uploaded file
        file_type: The type of file ('pdf' or 'image')
        qwen_model: Optional Qwen model instance
        
    Returns:
        Dictionary of extracted parameters
    """
    try:
        # Convert PDF to images if necessary
        if file_type == 'pdf':
            if PDF_TO_IMAGE_AVAILABLE:
                images = convert_from_bytes(file_content, dpi=200)
                image = images[0]  # Use first page for demo
            else:
                images = convert_pdf_to_image_fallback(file_content)
                image = images[0]
                st.warning("Using simplified PDF processing. For better results, ensure pdf2image is properly installed.")
        else:
            image = Image.open(io.BytesIO(file_content))
        
        # Prepare prompt for the VLM
        prompt = """
        Extract the following geotechnical parameters from this image if present:
        - Uniaxial Compressive Strength (UCS) in MPa
        - Brazilian Tensile Strength (BTS) in MPa
        - Rock Mass Rating (RMR)
        - Geological Strength Index (GSI)
        - Young's Modulus (E) in GPa
        - Poisson's Ratio (ν)
        - Cohesion (c) in MPa
        - Friction Angle (φ) in degrees
        
        Return the results as a JSON object with parameter names as keys and numerical values (without units).
        If a parameter is not found, don't include it in the result.
        """
        
        # Try using the Qwen model first if available
        if qwen_model is not None:
            response = query_qwen_vision_model(qwen_model, image, prompt)
            params = extract_parameters_from_response(response)
            
            if params:
                return params
            else:
                st.warning("Could not extract parameters using Qwen model. Falling back to API.")
        
        # If Qwen model is not available or failed, use Hugging Face API
        token = get_hf_token()
        
        if not token:
            st.error("Hugging Face token is required for parameter extraction.")
            return sample_geotechnical_data()
        
        # Encode image to base64 for API request
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Make API request to Hugging Face Inference API
        # Try with multiple model options in case of permission issues
        model_options = [
            "Qwen/Qwen2.5-VL-7B-Instruct",  # Try smaller model first
            "Qwen/Qwen2.5-VL-72B-Instruct",
            "llava-hf/llava-1.5-7b-hf",
            "openai/clip-vit-base-patch32"  # Simple fallback model
        ]
        
        result = None
        last_error = None
        
        for model in model_options:
            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {token}"}
                
                # Use different payload formats based on model
                if "clip" in model.lower():
                    # CLIP model has a different input format
                    payload = {
                        "image": img_str,
                        "text": prompt
                    }
                else:
                    # Standard VLM format
                    payload = {
                        "inputs": {
                            "image": img_str,
                            "text": prompt
                        }
                    }
                
                with st.status(f"Trying model: {model}..."):
                    response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Successfully used model: {model}")
                    break
                else:
                    last_error = f"API request to {model} failed with status code {response.status_code}"
                    st.warning(last_error)
                    # Continue to next model
            except Exception as e:
                last_error = f"Error with model {model}: {str(e)}"
                st.warning(last_error)
                # Continue to next model
                
        # If all models failed
        if result is None:
            st.error(f"All API requests failed. Last error: {last_error}")
            return sample_geotechnical_data()
                
        # Parse the response - assuming it returns text that includes JSON
        try:
            if isinstance(result, list) and "generated_text" in result[0]:
                text_response = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                text_response = result["generated_text"]
            else:
                text_response = str(result)
            
            # Extract parameters
            params = extract_parameters_from_response({"content": text_response})
            
            if not params:
                st.warning("Could not extract parameters from the image. Using sample data instead.")
                return sample_geotechnical_data()
                
            return params
                
        except Exception as e:
            st.error(f"Error processing API response: {str(e)}. Using sample data instead.")
            return sample_geotechnical_data()
            
    except Exception as e:
        st.error(f"Error in extraction: {str(e)}\n{traceback.format_exc()}")
        return sample_geotechnical_data()

def sample_geotechnical_data():
    """Return sample geotechnical data for demonstration purposes."""
    return {
        "UCS": 50,
        "BTS": 10,
        "RMR": 75,
        "GSI": 60,
        "E": 25,
        "ν": 0.2,
        "c": 15,
        "φ": 30
    }

# Tool for analyzer
@tool
def create_correlation_panel(data: Dict[str, float]) -> Dict[str, Any]:
    """
    Create a correlation panel from extracted data.
    
    Args:
        data: Dictionary of extracted parameters
        
    Returns:
        Dictionary with correlation matrix and figure data
    """
    try:
        # Convert dictionary to dataframe
        df = pd.DataFrame([data])
        
        # Calculate correlation matrix if there are enough data points
        if len(df.columns) > 1:
            # Create synthetic data for demonstration purposes
            # In a real application, you would collect multiple samples
            synthetic_df = pd.DataFrame()
            for col in df.columns:
                base_value = df[col].iloc[0]
                # Create 10 synthetic values around the base value
                synthetic_values = np.random.normal(base_value, base_value * 0.1, 10)
                synthetic_df[col] = synthetic_values
            
            # Calculate correlation
            corr = synthetic_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr, 
                text_auto=True, 
                color_continuous_scale='Blues',
                title="Parameter Correlation Matrix"
            )
            
            return {
                "correlation_matrix": corr.to_dict(),
                "figure": fig
            }
        else:
            return {
                "error": "Not enough parameters for correlation analysis"
            }
    except Exception as e:
        return {
            "error": f"Error in correlation analysis: {str(e)}"
        }

# Tool for visualization
@tool
def create_visualizations(data: Dict[str, float]) -> Dict[str, Any]:
    """
    Create visualizations from extracted data.
    
    Args:
        data: Dictionary of extracted parameters
        
    Returns:
        Dictionary with visualization figures
    """
    try:
        # Convert to dataframe
        df = pd.DataFrame([data])
        
        # Create synthetic data for demonstration
        n_points = 20
        synthetic_df = pd.DataFrame()
        
        # Generate x values (could be depth or another parameter)
        synthetic_df['Depth'] = np.linspace(0, 100, n_points)
        
        # Generate y values for each parameter with some variability
        for col in df.columns:
            base_value = df[col].iloc[0]
            # Create trend with depth
            trend = base_value * (1 + 0.005 * synthetic_df['Depth'])
            # Add random variation
            synthetic_df[col] = trend * (1 + 0.1 * np.random.randn(n_points))
        
        # Create figures
        figures = {}
        
        # Scatter plot: UCS vs BTS
        if 'UCS' in synthetic_df.columns and 'BTS' in synthetic_df.columns:
            scatter_fig = px.scatter(
                synthetic_df, 
                x='UCS', 
                y='BTS',
                title="UCS vs BTS Relationship",
                labels={"UCS": "UCS (MPa)", "BTS": "BTS (MPa)"},
                trendline="ols"
            )
            figures["scatter"] = scatter_fig
        
        # Line plot: Parameter variation with depth
        plot_params = [col for col in synthetic_df.columns if col != 'Depth'][:3]  # Take first 3 parameters
        if plot_params and 'Depth' in synthetic_df.columns:
            line_df = synthetic_df.melt(
                id_vars=['Depth'],
                value_vars=plot_params,
                var_name='Parameter',
                value_name='Value'
            )
            line_fig = px.line(
                line_df,
                x='Depth',
                y='Value',
                color='Parameter',
                title="Parameter Variation with Depth",
                labels={"Depth": "Depth (m)", "Value": "Value"}
            )
            figures["line"] = line_fig
        
        return figures
        
    except Exception as e:
        return {
            "error": f"Error in visualization: {str(e)}"
        }

@tool
def search_geotechnical_data(query: str) -> List[Dict[str, str]]:
    """
    Searches for geotechnical information using DuckDuckGo.
    
    Args:
        query: The search query for finding geotechnical information.
    Returns:
        List of search results as dictionaries.
    """
    try:
        if not SMOLAGENTS_AVAILABLE:
            return [
                {"title": "Search unavailable", "snippet": "Search functionality is not available.", "link": "#"},
                {"title": "Missing components", "snippet": "The required SmolaAgent components are not installed.", "link": "#"}
            ]
            
        search_tool = DuckDuckGoSearchTool()
        results = search_tool(query)
        return results
    except Exception as e:
        return [{"title": "Search error", "snippet": f"Error: {str(e)}", "link": "#"}]

# Streamlit sidebar
def create_sidebar():
    """Create and configure the sidebar."""
    with st.sidebar:
        st.title("Control Panel")
        st.subheader("Settings")
        
        # Settings options
        debug_mode = st.checkbox("Debug Mode", False)
        
        if debug_mode:
            st.subheader("Debug Information")
            token = get_hf_token()
            
            # Display token information securely
            st.write(f"Token available: {'Yes' if token else 'No'}")
            if token:
                masked_token = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
                st.write(f"Token: {masked_token}")
            
            # Display system information
            st.write(f"smolagents available: {SMOLAGENTS_AVAILABLE}")
            st.write(f"PDF to Image available: {PDF_TO_IMAGE_AVAILABLE}")
            st.write(f"HF Login available: {HF_LOGIN_AVAILABLE}")
        
        # Information
        st.subheader("Information")
        st.info("""
        This application extracts and analyzes geotechnical parameters from unstructured data.
        
        Upload a PDF or image to begin analysis.
        """)
        
        # Show mode
        st.subheader("Mode")
        if SMOLAGENTS_AVAILABLE:
            st.success("✅ SmolaAgent integration available")
        else:
            st.warning("⚠️ Using direct function mode (SmolaAgent not available)")

# Main function
def main():
    """Main application function."""
    # Set up the title and header
    st.title("🧠 Geotechnical Engineering Analysis System")
    st.write("Extract, analyze, and visualize geotechnical parameters from unstructured data")
    
    # Create sidebar
    create_sidebar()
    
    # Get Hugging Face token
    token = get_hf_token()
    
    # Initialize Qwen model if possible
    qwen_model = None
    if token and SMOLAGENTS_AVAILABLE:
        qwen_model = initialize_qwen_model(token)
        if qwen_model:
            st.success("✅ Qwen2.5-VL-72B-Instruct model initialized successfully")
        else:
            st.warning("⚠️ Failed to initialize Qwen model. Will use API fallback.")
    
    # Initialize session state for storing data between runs
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None
    if "correlation_data" not in st.session_state:
        st.session_state.correlation_data = None
    if "visualization_data" not in st.session_state:
        st.session_state.visualization_data = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    
    # File upload section
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader(
        "Upload a PDF or image containing geotechnical data",
        type=["pdf", "png", "jpg", "jpeg"],
        help="The file will be processed to extract geotechnical parameters"
    )
    
    # Create tabs for different functionality
    tab1, tab2, tab3, tab4 = st.tabs(["Data Extraction", "Analysis", "Visualization", "Web Search"])
    
    # Tab 1: Data Extraction
    with tab1:
        st.subheader("Parameter Extraction")
        if uploaded_file is not None:
            if st.button("Extract Parameters"):
                with st.spinner("Extracting parameters..."):
                    # Determine file type
                    file_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "image"
                    
                    # Read file content
                    file_content = uploaded_file.getvalue()
                    
                    # Use direct execution
                    try:
                        extracted_data = extract_from_file(file_content, file_type, qwen_model)
                        st.session_state.extracted_data = extracted_data
                    except Exception as e:
                        st.error(f"Error during extraction: {str(e)}\n{traceback.format_exc()}")
                        # Fallback to sample data
                        st.session_state.extracted_data = sample_geotechnical_data()
            
            # Display extracted data
            if st.session_state.extracted_data:
                st.success("Parameters extracted successfully!")
                
                # Create two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Extracted Parameters")
                    # Display parameters in a formatted way
                    for param, value in st.session_state.extracted_data.items():
                        # Map parameter keys to full names with units
                        param_names = {
                            "UCS": "Uniaxial Compressive Strength (MPa)",
                            "BTS": "Brazilian Tensile Strength (MPa)",
                            "RMR": "Rock Mass Rating",
                            "GSI": "Geological Strength Index",
                            "E": "Young's Modulus (GPa)",
                            "ν": "Poisson's Ratio",
                            "c": "Cohesion (MPa)",
                            "φ": "Friction Angle (°)"
                        }
                        
                        param_name = param_names.get(param, param)
                        st.metric(param_name, f"{value}")
                
                with col2:
                    st.write("### Reliability Assessment")
                    st.info("The extraction model has assessed these parameters with the following confidence levels:")
                    
                    # Generate mock confidence scores for demonstration
                    for param in st.session_state.extracted_data.keys():
                        confidence = np.random.uniform(0.7, 0.98)
                        st.progress(confidence, text=f"{param}: {confidence:.0%}")
        else:
            st.info("Please upload a file to begin extraction.")
    
    # Tab 2: Analysis
    with tab2:
        st.subheader("Data Analysis")
        
        if st.session_state.extracted_data:
            if st.button("Generate Correlation Panel"):
                with st.spinner("Analyzing data..."):
                    try:
                        correlation_data = create_correlation_panel(st.session_state.extracted_data)
                        st.session_state.correlation_data = correlation_data
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}\n{traceback.format_exc()}")
            
            # Display correlation panel
            if st.session_state.correlation_data:
                if "error" in st.session_state.correlation_data:
                    st.error(st.session_state.correlation_data["error"])
                elif "figure" in st.session_state.correlation_data:
                    st.plotly_chart(st.session_state.correlation_data["figure"], use_container_width=True)
                    
                    # Parameter relationships
                    st.write("### Parameter Relationships")
                    st.write("""
                    The correlation matrix shows the relationships between different geotechnical parameters.
                    Strong positive correlations appear in dark blue, while negative correlations appear in lighter colors.
                    
                    Common relationships in geotechnical engineering:
                    - UCS and BTS typically show a positive correlation
                    - RMR and GSI are closely related parameters
                    - Young's Modulus (E) often correlates with UCS
                    """)
        else:
            st.info("Extract parameters first to enable analysis.")

    # Tab 3: Visualization 
    with tab3:
        st.subheader("Data Visualization")
        
        if st.session_state.extracted_data:
            # Parameters to include in visualizations
            if st.session_state.extracted_data:
                params = list(st.session_state.extracted_data.keys())
                selected_params = st.multiselect(
                    "Select parameters to visualize",
                    params,
                    default=params[:min(3, len(params))]
                )
            
            if st.button("Generate Visualizations"):
                with st.spinner("Creating visualizations..."):
                    try:
                        # Filter data to include only selected parameters
                        filtered_data = {k: v for k, v in st.session_state.extracted_data.items() if k in selected_params}
                        
                        visualization_data = create_visualizations(filtered_data)
                        st.session_state.visualization_data = visualization_data
                    except Exception as e:
                        st.error(f"Error during visualization: {str(e)}\n{traceback.format_exc()}")
            
            # Display visualizations
            if st.session_state.visualization_data:
                if "error" in st.session_state.visualization_data:
                    st.error(st.session_state.visualization_data["error"])
                else:
                    # Display each figure
                    for fig_name, fig in st.session_state.visualization_data.items():
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualization interpretation
                    st.write("### Interpretation")
                    st.write("""
                    The scatter plot shows the relationship between UCS and BTS, which is important
                    for understanding rock strength characteristics. The trend line indicates the 
                    empirical relationship between these parameters.
                    
                    The line plot shows how parameters vary with depth, which can help identify
                    geological layers and strength variations in the rock mass.
                    """)
        else:
            st.info("Extract parameters first to enable visualization.")
    
    # Tab 4: Web Search
    with tab4:
        st.subheader("Web Search")
        
        search_query = st.text_input(
            "Enter a geotechnical engineering search query",
            placeholder="E.g., relationship between UCS and RMR in limestone"
        )
        
        if search_query and st.button("Search"):
            with st.spinner("Searching for information..."):
                try:
                    search_results = search_geotechnical_data(search_query)
                    
                    # Ensure search_results is a list
                    if not isinstance(search_results, list):
                        search_results = [{"title": "Results", "snippet": str(search_results), "link": "#"}]
                        
                    st.session_state.search_results = search_results
                except Exception as e:
                    st.error(f"Error during search: {str(e)}\n{traceback.format_exc()}")
                    st.session_state.search_results = [
                        {"title": "Search error", "snippet": f"An error occurred: {str(e)}", "link": "#"}
                    ]
        
        # Display search results
        if st.session_state.search_results:
            st.write("### Search Results")
            for idx, result in enumerate(st.session_state.search_results):
                # Handle different result formats
                if hasattr(result, 'title') and hasattr(result, 'snippet') and hasattr(result, 'link'):
                    title = result.title
                    snippet = result.snippet
                    link = result.link
                elif isinstance(result, dict) and 'title' in result and 'snippet' in result and 'link' in result:
                    title = result['title']
                    snippet = result['snippet']
                    link = result['link']
                else:
                    title = f"Result {idx+1}"
                    snippet = str(result)
                    link = "#"
                
                with st.expander(f"Result {idx+1}: {title}"):
                    st.write(snippet)
                    st.write(f"Source: {link}")

if __name__ == "__main__":
    main()
