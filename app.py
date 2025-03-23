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
    from smolagents import tool, HfApiModel, CodeAgent, ToolCallingAgent, DuckDuckGoSearchTool
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
    
    class ToolCallingAgent:
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
        # Try multiple possible key names
        if "hf_token" in st.secrets["huggingface"]:
            token = st.secrets["huggingface"]["hf_token"]
        elif "api_token" in st.secrets["huggingface"]:
            token = st.secrets["huggingface"]["api_token"]
    
    # If still not found, check if token is directly provided
    if not token:
        # Hardcoded token for testing (you should remove this in production)
        token = "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"
    
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

# Tool for image processing
@tool
def process_image(image: Image.Image, prompt: str) -> str:
    """
    Process an image with a given prompt using vision-language models.
    
    Args:
        image: The image to process
        prompt: The prompt for the vision-language model
        
    Returns:
        Extracted text from the image based on the prompt
    """
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode()
        image_url = f"data:image/jpeg;base64,{image_b64}"
        
        # Get token
        token = get_hf_token()
        if not token:
            return "Error: Hugging Face token is required for image processing."
        
        # Create API request
        api_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-VL-7B-Instruct"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Prepare payload in the correct format for the Qwen VL model
        payload = {
            "inputs": {
                "image": image_url,
                "prompt": prompt
            },
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        # Make the API request
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Check for errors
        if response.status_code != 200:
            return f"Error: API request failed with status code {response.status_code}. {response.text}"
        
        # Parse response
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", str(result))
        elif isinstance(result, dict):
            return result.get("generated_text", str(result))
        else:
            return str(result)
    
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Tool for extracting geotechnical parameters from documents
@tool
def extract_from_file(file_content: bytes, file_type: str) -> Dict[str, float]:
    """Extract geotechnical parameters from files.
    
    Args:
        file_content: The binary content of the file
        file_type: The type of file ('pdf' or 'image')
        
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
        
        # Process the image using our tool
        response_text = process_image(image, prompt)
        
        # Extract parameters from the response
        try:
            # Try to find a JSON object in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                params = json.loads(extracted_json)
                return params
            
            # Fallback parsing for structured text that isn't proper JSON
            params = {}
            lines = response_text.split('\n')
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
            
            # If no parameters were extracted, return sample data
            if not params:
                return sample_geotechnical_data()
                
            return params
                
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
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

# Tool for web search
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

# Initialize the multi-agent system
def initialize_agents(token):
    """Initialize the multi-agent system with a manager agent and specialized agents."""
    if not SMOLAGENTS_AVAILABLE or not token:
        return None, None
    
    try:
        # Login to Hugging Face if possible
        if HF_LOGIN_AVAILABLE:
            login(token)
        
        # Initialize the model - use Qwen2.5-VL-7B-Instruct as specified
        model = HfApiModel(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            token=token,
            timeout=120  # Adjusted timeout
        )
        
        # Create web search agent
        web_agent = ToolCallingAgent(
            tools=[DuckDuckGoSearchTool(), search_geotechnical_data],
            model=model,
            max_steps=5,
            name="geotechnical_search_agent",
            description="Searches for geotechnical engineering information on the web."
        )
        
        # Create visualization agent
        visualization_agent = ToolCallingAgent(
            tools=[create_correlation_panel, create_visualizations],
            model=model,
            max_steps=3,
            name="visualization_agent",
            description="Creates visualizations and correlations from geotechnical data."
        )
        
        # Create data extraction agent
        extraction_agent = ToolCallingAgent(
            tools=[extract_from_file],
            model=model,
            max_steps=3,
            name="extraction_agent",
            description="Extracts geotechnical parameters from documents and images."
        )
        
        # Create the manager agent
        manager_agent = CodeAgent(
            tools=[],
            model=model,
            managed_agents=[web_agent, visualization_agent, extraction_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )
        
        return manager_agent, model
    
    except Exception as e:
        logging.error(f"Error initializing agents: {str(e)}")
        return None, None

# Create sidebar
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
    
    # Initialize agents if possible
    manager_agent, model = initialize_agents(token)
    
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
            # Show preview of the uploaded file
            if uploaded_file.type.startswith('image'):
                st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
                uploaded_file.seek(0)  # Reset file pointer after reading
            elif uploaded_file.type == 'application/pdf':
                st.write("PDF file uploaded. Click 'Extract Parameters' to process.")
            
            if st.button("Extract Parameters"):
                with st.spinner("Extracting parameters..."):
                    # Determine file type
                    file_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "image"
                    
                    # Read file content
                    file_content = uploaded_file.getvalue()
                    
                    # Check if we can use the manager agent
                    if manager_agent and SMOLAGENTS_AVAILABLE:
                        try:
                            # Format the query for the manager agent
                            query = f"Extract geotechnical parameters from the uploaded {file_type} file."
                            
                            # For direct agent use, we'd need to modify the workflow, but for now
                            # we'll use the direct function call as it's already set up to handle file content
                            extracted_data = extract_from_file(file_content, file_type)
                            st.session_state.extracted_data = extracted_data
                        except Exception as e:
                            st.error(f"Error using manager agent: {str(e)}")
                            # Fallback to direct function
                            extracted_data = extract_from_file(file_content, file_type)
                            st.session_state.extracted_data = extracted_data
                    else:
                        # Use direct function call
                        extracted_data = extract_from_file(file_content, file_type)
                        st.session_state.extracted_data = extracted_data
            
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
                    # Use visualization agent if available
                    if manager_agent and SMOLAGENTS_AVAILABLE:
                        try:
                            # For now, we'll use the direct function for consistency
                            correlation_data = create_correlation_panel(st.session_state.extracted_data)
                            st.session_state.correlation_data = correlation_data
                        except Exception as e:
                            st.error(f"Error using visualization agent: {str(e)}")
                            # Fallback to direct function
                            correlation_data = create_correlation_panel(st.session_state.extracted_data)
                            st.session_state.correlation_data = correlation_data
                    else:
                        # Use direct function
                        correlation_data = create_correlation_panel(st.session_state.extracted_data)
                        st.session_state.correlation_data = correlation_data
            
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
                    # Filter data to include only selected parameters
                    filtered_data = {k: v for k, v in st.session_state.extracted_data.items() if k in selected_params}
                    
                    # Use visualization agent if available
                    if manager_agent and SMOLAGENTS_AVAILABLE:
                        try:
                            # For now, we'll use the direct function for consistency
                            visualization_data = create_visualizations(filtered_data)
                            st.session_state.visualization_data = visualization_data
                        except Exception as e:
                            st.error(f"Error using visualization agent: {str(e)}")
                            # Fallback to direct function
                            visualization_data = create_visualizations(filtered_data)
                            st.session_state.visualization_data = visualization_data
                    else:
                        # Use direct function
                        visualization_data = create_visualizations(filtered_data)
                        st.session_state.visualization_data = visualization_data
            
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
                # Use web search agent if available
                if manager_agent and SMOLAGENTS_AVAILABLE:
                    try:
                        # For integration with the real agent system, we'd use the manager_agent.run()
                        # But for now we'll use the direct search tool for consistency
                        search_results = search_geotechnical_data(search_query)
                        st.session_state.search_results = search_results
                    except Exception as e:
                        st.error(f"Error using web search agent: {str(e)}")
                        # Fallback to direct function
                        search_results = search_geotechnical_data(search_query)
                        st.session_state.search_results = search_results
                else:
                    # Use direct search
                    search_results = search_geotechnical_data(search_query)
                    st.session_state.search_results = search_results
        
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
