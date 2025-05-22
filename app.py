import streamlit as st
import math
import logging
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import requests
import base64
import io
import tempfile
import time
import hashlib
from datetime import datetime
from huggingface_hub import login
from typing import Dict, List, Any, Tuple
from markdownify import markdownify
from requests.exceptions import RequestException
import traceback
import sys
import os
from PIL import Image
import PyPDF2
import pdf2image

# CRITICAL FIX: PyTorch/Streamlit compatibility fixes
# Fix 1: Disable Streamlit file watcher for problematic modules
if hasattr(st, '_config'):
    st._config.set_option('global.disableWatchdogWarning', True)

# Fix 2: Set environment variables to prevent conflicts
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Fix 3: Import torch with compatibility patches
try:
    # Patch torch.classes before importing smolagents
    import torch
    if hasattr(torch, 'classes'):
        # Create a dummy __path__ attribute if it doesn't exist
        if not hasattr(torch.classes, '__path__'):
            class DummyPath:
                def __init__(self):
                    self._path = []
                
                def __iter__(self):
                    return iter(self._path)
                
                def __getattr__(self, name):
                    if name == '_path':
                        return []
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
            
            torch.classes.__path__ = DummyPath()
    
    logger_init = True
except ImportError:
    # If torch is not available, continue without it
    logger_init = True
except Exception as e:
    print(f"Warning: PyTorch compatibility patch failed: {e}")
    logger_init = True

# Fix 4: Monkey patch Streamlit's asyncio handling
import asyncio
original_get_running_loop = asyncio.get_running_loop

def patched_get_running_loop():
    try:
        return original_get_running_loop()
    except RuntimeError:
        # If no event loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Apply the patch
asyncio.get_running_loop = patched_get_running_loop

# Now safely import smolagents after applying patches
try:
    from smolagents import tool, CodeAgent, InferenceClientModel, ToolCallingAgent, DuckDuckGoSearchTool
    smolagents_available = True
except ImportError as e:
    print(f"Warning: smolagents not available: {e}")
    smolagents_available = False
    # Create dummy classes for graceful degradation
    class DummyAgent:
        def __init__(self, *args, **kwargs):
            pass
        def run(self, query):
            return "Agents not available - please install smolagents"
    
    CodeAgent = DummyAgent
    ToolCallingAgent = DummyAgent
    InferenceClientModel = lambda x: None
    DuckDuckGoSearchTool = lambda: None
    
    def tool(func):
        return func

# Enhanced logging configuration
def setup_enhanced_logging():
    """Setup enhanced logging for better debugging."""
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # File handler for detailed logs
    file_handler = logging.FileHandler("geotechnical_app_detailed.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize enhanced logging
logger = setup_enhanced_logging()

# Page configuration with additional fixes
st.set_page_config(
    page_title="Advanced Geotechnical AI Agent by Qwen2.5-Coder-32B-Instruct",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix 5: Add Streamlit configuration overrides
if 'configured' not in st.session_state:
    # Override problematic configuration settings
    try:
        import streamlit.web.bootstrap as bootstrap
        if hasattr(bootstrap, '_fix_matplotlib_crash'):
            bootstrap._fix_matplotlib_crash()
    except:
        pass
    
    st.session_state.configured = True

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = {}
if 'pdf_analysis' not in st.session_state:
    st.session_state.pdf_analysis = None
if 'uploaded_pdf' not in st.session_state:
    st.session_state.uploaded_pdf = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'pdf_page_images' not in st.session_state:
    st.session_state.pdf_page_images = []
if 'current_pdf_hash' not in st.session_state:
    st.session_state.current_pdf_hash = None

# Response validation function
def validate_response(response, context="general"):
    """Validate and clean up response before displaying."""
    logger.debug(f"Validating response in context '{context}': type={type(response)}, length={len(str(response)) if response else 0}")
    
    # Handle None responses
    if response is None:
        logger.warning(f"Received None response in context '{context}'")
        return "I couldn't generate a response. Please try again with a different question."
    
    # Convert to string if needed
    if not isinstance(response, str):
        logger.info(f"Converting response from {type(response)} to string")
        response = str(response)
    
    # Clean up the response
    response = response.strip()
    
    # Handle empty responses
    if not response:
        logger.warning(f"Received empty response in context '{context}'")
        return "I generated an empty response. Please try rephrasing your question."
    
    # Remove any problematic characters that might interfere with Streamlit
    response = response.replace('\x00', '').replace('\r\n', '\n').replace('\r', '\n')
    
    # Limit response length to prevent UI issues
    if len(response) > 10000:
        logger.warning(f"Response too long ({len(response)} chars), truncating")
        response = response[:9950] + "\n\n[Response truncated due to length]"
    
    logger.debug(f"Response validation complete: length={len(response)}")
    return response

# Enhanced error handling wrapper with asyncio fixes
def safe_agent_call(agent, query, context="unknown"):
    """Safely call an agent with comprehensive error handling and asyncio fixes."""
    try:
        logger.info(f"Making agent call in context '{context}': {query[:100]}...")
        
        # Fix 6: Ensure we're in the right asyncio context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Call the agent using .run() method
        if smolagents_available and agent:
            result = agent.run(query)
        else:
            result = "Agent service is not available. Please check your configuration."
        
        # Validate the result
        validated_result = validate_response(result, context)
        
        logger.info(f"Agent call successful in context '{context}': response length={len(validated_result)}")
        return validated_result
        
    except Exception as e:
        error_msg = f"Agent call failed in context '{context}': {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return f"I encountered an error while processing your request: {str(e)}"

# Runpod API integration functions
def call_runpod_endpoint(pdf_bytes, query=None):
    """Call the Runpod serverless endpoint for PDF analysis."""
    try:
        runpod_endpoint_url = st.secrets["runpod"]["endpoint_url"]
        runpod_api_key = st.secrets["runpod"]["api_key"]
    except Exception as e:
        logger.error(f"Failed to access Runpod secrets: {str(e)}")
        return {"error": "Runpod configuration missing. Please check secrets.toml"}
    
    try:
        # Encode PDF bytes to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Prepare the payload
        payload = {
            "input": {
                "pdf_base64": pdf_base64,
                "query": query if query else ""
            }
        }
        
        # Set up headers with API key
        headers = {
            "Authorization": f"Bearer {runpod_api_key}",
            "Content-Type": "application/json"
        }
        
        # Send synchronous request to Runpod
        with st.spinner("Connecting to analysis service..."):
            response = requests.post(
                runpod_endpoint_url, 
                json=payload, 
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
        
        # Check for status endpoint in the response (async API)
        result = response.json()
        
        if "id" in result:
            # This is an async job, poll for results
            status_url = f"{runpod_endpoint_url.split('/run')[0]}/status/{result['id']}"
            st.session_state.processing_status = "Processing PDF with ColPali model..."
            
            with st.spinner("Processing PDF (this may take a minute)..."):
                max_attempts = 30
                for attempt in range(max_attempts):
                    status_response = requests.get(status_url, headers=headers)
                    status_data = status_response.json()
                    
                    if status_data.get("status") == "COMPLETED":
                        st.session_state.processing_status = "Analysis complete"
                        return status_data.get("output", {})
                    elif status_data.get("status") in ["FAILED", "CANCELLED"]:
                        st.session_state.processing_status = "Analysis failed"
                        return {"error": f"Processing failed: {status_data.get('error', 'Unknown error')}"}
                    
                    progress_msg = f"Processing PDF (attempt {attempt+1}/{max_attempts})..."
                    st.session_state.processing_status = progress_msg
                    
                    wait_time = min(2 * (1.5 ** attempt), 15)
                    time.sleep(wait_time)
                
                st.session_state.processing_status = "Analysis timed out"
                return {"error": "PDF analysis timed out. Please try again or with a smaller document."}
        else:
            # Direct result (sync API)
            st.session_state.processing_status = "Analysis complete"
            return result
            
    except requests.exceptions.RequestException as e:
        st.session_state.processing_status = "Connection error"
        logger.error(f"Failed to connect to PDF analysis service: {str(e)}")
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        st.session_state.processing_status = "Error occurred"
        logger.error(f"Error processing PDF analysis: {str(e)}\n{traceback.format_exc()}")
        return {"error": f"Error: {str(e)}"}

# Tools
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.
    Args:
        url: The URL of the webpage to visit and retrieve content from.
    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

@tool
def search_geotechnical_data(query: str) -> str:
    """Searches for geotechnical information using DuckDuckGo.
    Args:
        query: The search query for finding geotechnical information.
    Returns:
        Search results as formatted text.
    """
    if smolagents_available:
        search_tool = DuckDuckGoSearchTool()
        try:
            results = search_tool(query)
            return str(results)
        except Exception as e:
            return f"Search error: {str(e)}"
    else:
        return "Search functionality not available - smolagents not installed"

@tool
def classify_soil(soil_type: str, plasticity_index: float, liquid_limit: float) -> Dict:
    """Classify soil using USCS classification system.
    Args:
        soil_type: Type of soil (clay, sand, silt)
        plasticity_index: Plasticity index value
        liquid_limit: Liquid limit value
    Returns:
        Dictionary containing soil classification and description
    """
    if soil_type.lower() == 'clay':
        if plasticity_index > 50:
            return {"classification": "CH", "description": "High plasticity clay"}
        elif plasticity_index > 30:
            return {"classification": "CI", "description": "Medium plasticity clay"}
        else:
            return {"classification": "CL", "description": "Low plasticity clay"}
    elif soil_type.lower() == 'sand':
        if plasticity_index < 4 and liquid_limit < 50:
            return {"classification": "SP", "description": "Poorly graded sand"}
        elif plasticity_index < 7 and liquid_limit < 50:
            return {"classification": "SW", "description": "Well-graded sand"}
        else:
            return {"classification": "SC", "description": "Clayey sand"}
    elif soil_type.lower() == 'silt':
        if liquid_limit < 50:
            return {"classification": "ML", "description": "Low plasticity silt"}
        else:
            return {"classification": "MH", "description": "High plasticity silt"}
    return {"classification": "Unknown", "description": "Unknown soil type"}

@tool
def calculate_tunnel_support(depth: float, soil_density: float, k0: float, tunnel_diameter: float) -> Dict:
    """Calculate tunnel support pressure and related parameters.
    Args:
        depth: Tunnel depth from surface in meters
        soil_density: Soil density in kg/m³
        k0: At-rest earth pressure coefficient
        tunnel_diameter: Tunnel diameter in meters
    Returns:
        Dictionary containing support pressures, stresses and safety factors
    """
    g = 9.81
    vertical_stress = depth * soil_density * g / 1000
    horizontal_stress = k0 * vertical_stress
    support_pressure = (vertical_stress + horizontal_stress) / 2
    safety_factor = 1.5 if depth < 30 else 2.0
    return {
        "support_pressure": support_pressure,
        "design_pressure": support_pressure * safety_factor,
        "safety_factor": safety_factor,
        "vertical_stress": vertical_stress,
        "horizontal_stress": horizontal_stress
    }

def get_support_recommendations(rmr: int) -> Dict:
    """Get support recommendations based on RMR value.
    Args:
        rmr: Rock Mass Rating value
    Returns:
        Dictionary containing support recommendations
    """
    if rmr > 80:
        return {
            "excavation": "Full face, 3m advance",
            "support": "Generally no support required",
            "bolting": "Spot bolting if needed",
            "shotcrete": "None required",
            "steel_sets": "None required"
        }
    elif rmr > 60:
        return {
            "excavation": "Full face, 1.0-1.5m advance",
            "support": "Complete within 20m of face",
            "bolting": "Systematic bolting, 4m length, spaced 1.5-2m",
            "shotcrete": "50mm in crown where required",
            "steel_sets": "None required"
        }
    elif rmr > 40:
        return {
            "excavation": "Top heading and bench, 1.5-3m advance",
            "support": "Complete within 10m of face",
            "bolting": "Systematic bolting, 4-5m length, spaced 1-1.5m",
            "shotcrete": "50-100mm in crown and 30mm in sides",
            "steel_sets": "Light to medium ribs spaced 1.5m where required"
        }
    else:
        return {
            "excavation": "Multiple drifts, 0.5-1.5m advance",
            "support": "Install support concurrent with excavation",
            "bolting": "Systematic bolting with shotcrete and steel sets",
            "shotcrete": "100-150mm in crown and sides",
            "steel_sets": "Medium to heavy ribs spaced 0.75m"
        }

def get_q_support_category(q: float) -> Dict:
    """Get Q-system support recommendations.
    Args:
        q: Q-system value
    Returns:
        Dictionary containing support recommendations
    """
    if q > 40:
        return {
            "support_type": "No support required",
            "bolting": "None or occasional spot bolting",
            "shotcrete": "None required"
        }
    elif q > 10:
        return {
            "support_type": "Spot bolting",
            "bolting": "Spot bolts in crown, spaced 2.5m",
            "shotcrete": "None required"
        }
    elif q > 4:
        return {
            "support_type": "Systematic bolting",
            "bolting": "Systematic bolts in crown spaced 2m, occasional wire mesh",
            "shotcrete": "40-100mm where needed"
        }
    elif q > 1:
        return {
            "support_type": "Systematic bolting with shotcrete",
            "bolting": "Systematic bolts spaced 1-1.5m with wire mesh in crown and sides",
            "shotcrete": "50-90mm in crown and 30mm on sides"
        }
    else:
        return {
            "support_type": "Heavy support",
            "bolting": "Systematic bolts spaced 1m with wire mesh",
            "shotcrete": "90-120mm in crown and 100mm on sides",
            "additional": "Consider steel ribs, forepoling, or face support"
        }

@tool
def calculate_rmr(ucs: float, rqd: float, spacing: float, condition: int, groundwater: int, orientation: int) -> Dict:
    """Calculate Rock Mass Rating (RMR) classification.
    Args:
        ucs: Uniaxial compressive strength in MPa
        rqd: Rock Quality Designation as percentage
        spacing: Joint spacing in meters
        condition: Joint condition rating (0-30)
        groundwater: Groundwater condition rating (0-15)
        orientation: Joint orientation rating (-12-0)
    Returns:
        Dictionary containing RMR value, rock class, and component ratings
    """
    if ucs > 250: ucs_rating = 15
    elif ucs > 100: ucs_rating = 12
    elif ucs > 50: ucs_rating = 7
    elif ucs > 25: ucs_rating = 4
    else: ucs_rating = 2
    if rqd > 90: rqd_rating = 20
    elif rqd > 75: rqd_rating = 17
    elif rqd > 50: rqd_rating = 13
    elif rqd > 25: rqd_rating = 8
    else: rqd_rating = 3
    if spacing > 2: spacing_rating = 20
    elif spacing > 0.6: spacing_rating = 15
    elif spacing > 0.2: spacing_rating = 10
    elif spacing > 0.06: spacing_rating = 8
    else: spacing_rating = 5
    total_rmr = ucs_rating + rqd_rating + spacing_rating + condition + groundwater + orientation
    if total_rmr > 80: rock_class = "I - Very good rock"
    elif total_rmr > 60: rock_class = "II - Good rock"
    elif total_rmr > 40: rock_class = "III - Fair rock"
    elif total_rmr > 20: rock_class = "IV - Poor rock"
    else: rock_class = "V - Very poor rock"
    return {
        "rmr_value": total_rmr,
        "rock_class": rock_class,
        "support_recommendations": get_support_recommendations(total_rmr),
        "component_ratings": {
            "ucs_rating": ucs_rating,
            "rqd_rating": rqd_rating,
            "spacing_rating": spacing_rating,
            "condition_rating": condition,
            "groundwater_rating": groundwater,
            "orientation_rating": orientation
        }
    }

@tool
def calculate_q_system(rqd: float, jn: float, jr: float, ja: float, jw: float, srf: float) -> Dict:
    """Calculate Q-system rating and support requirements.
    Args:
        rqd: Rock Quality Designation as percentage
        jn: Joint set number
        jr: Joint roughness number
        ja: Joint alteration number
        jw: Joint water reduction factor
        srf: Stress Reduction Factor
    Returns:
        Dictionary containing Q-value and support recommendations
    """
    q_value = (rqd/jn) * (jr/ja) * (jw/srf)
    if q_value > 40: quality = "Exceptionally Good"
    elif q_value > 10: quality = "Very Good"
    elif q_value > 4: quality = "Good"
    elif q_value > 1: quality = "Fair"
    elif q_value > 0.1: quality = "Poor"
    else: quality = "Extremely Poor"
    return {
        "q_value": round(q_value, 2),
        "rock_quality": quality,
        "support_category": get_q_support_category(q_value),
        "parameters": {
            "RQD/Jn": round(rqd/jn, 2),
            "Jr/Ja": round(jr/ja, 2),
            "Jw/SRF": round(jw/srf, 2)
        }
    }

@tool
def estimate_tbm_performance(ucs: float, rqd: float, joint_spacing: float,
                             abrasivity: float, diameter: float) -> Dict:
    """Estimate TBM performance parameters.
    Args:
        ucs: Uniaxial compressive strength in MPa
        rqd: Rock Quality Designation as percentage
        joint_spacing: Average joint spacing in meters
        abrasivity: Cerchar abrasivity index
        diameter: TBM diameter in meters
    Returns:
        Dictionary containing TBM performance estimates
    """
    pr = 20 * (1/ucs) * (rqd/100) * (1/abrasivity)
    utilization = 0.85 - (0.01 * (abrasivity/2))
    advance_rate = pr * utilization * 24
    cutter_life = 100 * (250/ucs) * (2/abrasivity)
    return {
        "penetration_rate": round(pr, 2),
        "daily_advance": round(advance_rate, 2),
        "utilization": round(utilization * 100, 1),
        "cutter_life_hours": round(cutter_life, 0),
        "estimated_completion_days": round(1000/advance_rate, 0)
    }

@tool
def analyze_face_stability(depth: float, diameter: float, soil_density: float,
                           cohesion: float, friction_angle: float, water_table: float) -> str:
    """Analyze tunnel face stability.
    Args:
        depth: Tunnel depth in meters
        diameter: Tunnel diameter in meters
        soil_density: Soil density in kg/m³
        cohesion: Soil cohesion in kPa
        friction_angle: Soil friction angle in degrees
        water_table: Water table depth from surface in meters
    Returns:
        Formatted string containing stability analysis results
    """
    g = 9.81
    sigma_v = depth * soil_density * g / 1000
    water_pressure = (depth - water_table) * 9.81 if water_table < depth else 0
    N = (sigma_v - water_pressure) * math.tan(math.radians(friction_angle)) + cohesion
    fs = N / (0.5 * soil_density * g * diameter / 1000)
    return json.dumps({
        "stability_ratio": round(N, 2),
        "factor_of_safety": round(fs, 2), 
        "water_pressure": round(water_pressure, 2),
        "support_pressure_required": round(sigma_v/fs, 2) if fs < 1.5 else 0
    })

@tool
def import_borehole_data(file_path: str) -> Dict:
    """Import and process borehole data.
    Args:
        file_path: Path to borehole data CSV file
    Returns:
        Dictionary containing processed borehole data
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = ['depth', 'soil_type', 'N_value', 'moisture']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in borehole data")
        return {
            "total_depth": df['depth'].max(),
            "soil_layers": df['soil_type'].nunique(),
            "ground_water_depth": df[df['moisture'] > 50]['depth'].min(),
            "average_N_value": df['N_value'].mean(),
            "soil_profile": df.groupby('soil_type')['depth'].agg(['min', 'max']).to_dict()
        }
    except Exception as e:
        logger.error(f"Error processing borehole data: {e}")
        raise

@tool
def visualize_3d_results(coordinates: str, geology_data: str, analysis_data: str) -> Dict:
    """Create 3D visualization of tunnel and analysis results.
    Args:
        coordinates: JSON string of tunnel coordinates in format [[x1,y1,z1], [x2,y2,z2], ...] 
        geology_data: JSON string of geological layers containing type, color and bounds
        analysis_data: JSON string with stability analysis results including factor of safety
    Returns:
        Dict containing plot data (Plotly figure) and statistics (length, depth, critical sections)
    """
    tunnel_path = json.loads(coordinates)
    geology = json.loads(geology_data)
    analysis_results = json.loads(analysis_data)
    fig = go.Figure()
    x, y, z = zip(*tunnel_path)
    # Tunnel alignment and stability analysis
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Tunnel Alignment'))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=[r['factor_of_safety'] for r in analysis_results['stability']],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Factor of Safety')
        ),
        name='Stability Analysis'
    ))
    # Add geology layers
    for layer in geology:
        fig.add_trace(go.Surface(
            x=layer['bounds']['x'],
            y=layer['bounds']['y'],
            z=layer['bounds']['z'],
            colorscale=[[0, layer['color']], [1, layer['color']]],
            showscale=False,
            name=layer['type'],
            opacity=0.6
        ))
    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    stats = {
        "tunnel_length": sum(math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2)
                             for i in range(1, len(x))),
        "depth_range": [min(z), max(z)],
        "critical_sections": [i for i, r in enumerate(analysis_results['stability'])
                              if r['factor_of_safety'] < 1.5]
    }
    return {"plot": fig.to_dict(), "statistics": stats}

@tool
def calculate_tbm_penetration(alpha: float, fracture_spacing: float, peak_slope: float, csm_rop: float) -> Dict:
    """Calculate TBM Rate of Penetration using advanced formula.
    Args:
        alpha: Angle between tunnel axis and weakness plane
        fracture_spacing: Fracture spacing
        peak_slope: Peak slope from punch tests
        csm_rop: CSM model basic ROP
    """
    rfi = 1.44 * math.log(alpha) - 0.0187 * fracture_spacing
    bi = 0.0157 * peak_slope
    rop = 0.859 - rfi + bi + 0.0969 * csm_rop
    return {"penetration_rate": rop}

@tool
def calculate_cutter_specs(max_speed: float, cutter_diameter: float) -> Dict:
    """Calculate cutter head specs including RPM and power requirements.
    Args:
        max_speed: Maximum cutting speed
        cutter_diameter: Diameter of cutter
    """
    rpm = max_speed / (math.pi * cutter_diameter)
    return {
        "rpm": rpm,
        "max_speed": max_speed,
        "diameter": cutter_diameter
    }

@tool
def calculate_specific_energy(normal_force: float, spacing: float, penetration: float, 
                              rolling_force: float, tip_angle: float) -> Dict:
    """Calculate specific energy for disc cutters.
    Args:
        normal_force: Normal force on cutter
        spacing: Spacing between cutters
        penetration: Penetration per revolution
        rolling_force: Rolling force
        tip_angle: Angle of cutter tip in radians
    """
    se = (normal_force / (spacing * penetration)) * (1 + (rolling_force/normal_force) * math.tan(tip_angle))
    return {"specific_energy": se}

@tool
def predict_cutter_life(ucs: float, penetration: float, rpm: float, diameter: float, 
                        cai: float, constants: Dict[str, float]) -> Dict:
    """Predict cutter life using empirical relationship.
    Args:
        ucs: Uniaxial compressive strength
        penetration: Penetration rate
        rpm: Cutterhead revolution speed
        diameter: Tunnel diameter
        cai: Cerchar abrasivity index
        constants: Dictionary of C1-C6 constants
    """
    cl = (constants['C1'] * (ucs ** constants['C2'])) / \
         ((penetration ** constants['C3']) * (rpm ** constants['C4']) * \
          (diameter ** constants['C5']) * (cai ** constants['C6']))
    return {"cutter_life_m3": cl}

@tool
def find_relevant_pdf_sections(pdf_bytes: bytes, query: str) -> Dict[str, Any]:
    """Find and extract sections from a PDF that are most relevant to a query using Runpod API.
    
    Args:
        pdf_bytes: The binary content of the PDF file
        query: The search query
        
    Returns:
        Dictionary containing relevant sections, similarity scores, and snippets
    """
    # Call the Runpod API
    result = call_runpod_endpoint(pdf_bytes, query)
    
    # Check for errors
    if "error" in result or result.get("status") == "error":
        error_message = result.get("error", result.get("message", "Unknown error"))
        return {"error": error_message}
    
    # Return the processed results
    return {
        "query": query,
        "relevant_sections": result.get("relevant_sections", []),
        "total_pages": result.get("num_pages", result.get("total_pages", 0))
    }

@tool
def extract_pdf_features(pdf_bytes: bytes, query: str = None) -> Dict[str, Any]:
    """Extract visual and textual features from PDF using Runpod-hosted ColPali VLM model.
    
    Args:
        pdf_bytes: The binary content of the PDF file
        query: Optional query to compare PDF content against
        
    Returns:
        Dictionary containing extracted features, page content, and similarity scores
    """
    # Call the Runpod API
    result = call_runpod_endpoint(pdf_bytes, query)
    
    # Check for errors
    if "error" in result or result.get("status") == "error":
        error_message = result.get("error", result.get("message", "Unknown error"))
        return {"error": error_message}
    
    # Return the processed results
    return {
        "num_pages": result.get("num_pages", 0),
        "page_text": result.get("page_text", []),
        "embedding_dimensions": result.get("embedding_dimensions", 0),
        "tokens_per_page": result.get("tokens_per_page", 0),
        "query_scores": result.get("query_scores", {})
    }

def pdf_to_images_and_text(pdf_bytes):
    """Convert PDF bytes to images and text locally (for display purposes only).
    This doesn't use the ColPali model - only for previewing pages."""
    try:
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
    except Exception as e:
        logger.error(f"Error processing PDF: {e}\n{traceback.format_exc()}")
        return [], []

def initialize_agents():
    """Initialize the multi-agent system with enhanced error handling and asyncio fixes."""
    try:
        if not smolagents_available:
            st.warning("⚠️ SmolaAgents not available. Some features will be limited.")
            return None, None, None
        
        # Fix 7: Handle HuggingFace authentication with asyncio fixes
        hf_key = None
        try:
            hf_key = st.secrets["huggingface"]["HUGGINGFACE_API_KEY"]
        except Exception as e:
            logger.warning(f"Couldn't access Hugging Face secrets: {str(e)}")
            hf_key = os.environ.get("HUGGINGFACE_API_KEY")

        if not hf_key:
            st.warning("""
                **Hugging Face API key not found.**  
                Please add your key to either:
                - A `secrets.toml` file (for local development)
                - Environment variables (for deployment)
                
                The app will continue with limited functionality.
            """)
            return None, None, None

        # Enhanced authentication with asyncio fixes
        max_retries = 3
        retry_delay = 2
        success = False
        
        for attempt in range(max_retries):
            try:
                # Ensure we have proper asyncio context
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                login(hf_key)
                success = True
                break
            except requests.exceptions.HTTPError as http_err:
                if "429" in str(http_err):
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit. Retrying in {retry_delay} seconds (attempt {attempt+1}/{max_retries})...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"Failed to authenticate after {max_retries} attempts: Rate limit exceeded")
                        st.error("**Hugging Face API rate limit exceeded.** Please try again in a few minutes.")
                else:
                    logger.error(f"Hugging Face API authentication error: {str(http_err)}")
                    st.error(f"**Hugging Face API authentication failed.** Please check your API key.")
                    break
            except Exception as e:
                logger.error(f"Unexpected error during Hugging Face authentication: {str(e)}")
                st.error(f"**Error connecting to Hugging Face.** {str(e)}")
                break
        
        if not success:
            return None, None, None
            
        # Initialize model with error handling
        try:
            model = InferenceClientModel("Qwen/Qwen2.5-Coder-32B-Instruct")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            st.error(f"Failed to initialize AI model: {str(e)}")
            return None, None, None
        
        # Check Runpod configuration
        try:
            runpod_endpoint_url = st.secrets["runpod"]["endpoint_url"]
            runpod_api_key = st.secrets["runpod"]["api_key"]
            st.success("✅ Runpod API configuration found. PDF analysis available.")
        except Exception as e:
            logger.warning(f"Runpod configuration not found: {str(e)}")
            st.warning("⚠️ PDF analysis features will be limited without Runpod configuration.")
        
        # Initialize agents with error handling
        try:
            web_agent = ToolCallingAgent(
                tools=[search_geotechnical_data, visit_webpage],
                model=model,
                max_steps=10
            )
            
            geotech_agent = ToolCallingAgent(
                tools=[
                    classify_soil,
                    calculate_tunnel_support,
                    calculate_rmr,
                    calculate_q_system,
                    estimate_tbm_performance,
                    analyze_face_stability,
                    import_borehole_data,
                    visualize_3d_results,
                    calculate_tbm_penetration,
                    calculate_cutter_specs,
                    calculate_specific_energy,
                    predict_cutter_life,
                    extract_pdf_features,
                    find_relevant_pdf_sections
                ],
                model=model,
                max_steps=10
            )
            
            manager_agent = CodeAgent(
                tools=[search_geotechnical_data],
                model=model,
                additional_authorized_imports=["time", "numpy", "pandas"]
            )
            
            logger.info("All agents initialized successfully")
            st.success("✅ AI Agents initialized successfully")
            return web_agent, geotech_agent, manager_agent
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            st.error(f"Failed to initialize agents: {str(e)}")
            return None, None, None
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}\n{traceback.format_exc()}")
        st.error(f"Failed to initialize agents: {str(e)}")
        return None, None, None

def process_request(user_input: str):
    """Process user requests with enhanced error handling and asyncio fixes."""
    
    # Check if agents are initialized
    if not smolagents_available or manager_agent is None or web_agent is None or geotech_agent is None:
        logger.error("Agents are not initialized. Cannot process request.")
        return (
            "I'm unable to process your request at this time because the AI agents could not be initialized. "
            "This is likely due to missing API keys or compatibility issues. "
            "Please check your configuration and try again."
        )
    
    try:
        # Fix 8: Ensure proper asyncio context for all operations
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Handle PDF queries
        if st.session_state.get("uploaded_pdf"):
            logger.info(f"Processing PDF query: {user_input}")
            st.session_state.processing_status = "Analyzing PDF for your query..."
            
            pdf_analysis_result = find_relevant_pdf_sections(
                pdf_bytes=st.session_state.uploaded_pdf,
                query=user_input
            )
            
            st.session_state.pdf_analysis = pdf_analysis_result
            
            # Cache PDF page images
            if "pdf_page_images" not in st.session_state or st.session_state.get("current_pdf_hash") != hash(str(st.session_state.get("uploaded_pdf"))):
                with st.spinner("Preparing PDF page previews..."):
                    try:
                        images, _ = pdf_to_images_and_text(st.session_state.uploaded_pdf)
                        st.session_state.pdf_page_images = images
                        st.session_state.current_pdf_hash = hash(str(st.session_state.uploaded_pdf))
                    except Exception as img_error:
                        logger.error(f"Error preparing PDF previews: {str(img_error)}")
                        st.session_state.pdf_page_images = []
            
            st.session_state.processing_status = None
            
            # Process PDF results
            if pdf_analysis_result and not pdf_analysis_result.get("error"):
                extracted_snippets = []
                for section in pdf_analysis_result.get("relevant_sections", []):
                    for snippet in section.get("snippets", []):
                        if snippet and len(snippet.strip()) > 20:
                            extracted_snippets.append(snippet.strip())
                
                if extracted_snippets:
                    snippet_context = "\n\n".join([f"• {s}" for s in extracted_snippets[:3]])
                    
                    final_response = f"""Based on the PDF analysis, here's what I found regarding "{user_input}":

{snippet_context}

This information was extracted from {len(pdf_analysis_result.get("relevant_sections", []))} relevant page(s) in the document."""
                    
                    logger.info(f"Direct PDF response created, length: {len(final_response)}")
                    return validate_response(final_response, "pdf_analysis")
                else:
                    return validate_response("I analyzed the PDF but couldn't find information specifically relevant to your query. Try asking about a different topic covered in the document.", "pdf_no_results")
            elif pdf_analysis_result and pdf_analysis_result.get("error"):
                error_message = pdf_analysis_result.get("error")
                logger.error(f"PDF analysis error: {error_message}")
                return validate_response(f"I encountered an issue when analyzing the PDF: {error_message}", "pdf_error")
            else:
                return validate_response("I couldn't properly analyze the PDF document. Please try again or with a different query.", "pdf_general_error")
        
        # Handle general queries with asyncio fixes
        logger.info(f"Processing general query: {user_input}")
        st.session_state.processing_status = "Processing your request..."
        
        try:
            # Get web search results if appropriate
            web_result_str = ""
            if any(term in user_input.lower() for term in ["search", "find", "what is", "who is", "when", "where", "how"]):
                try:
                    web_result = safe_agent_call(web_agent, user_input, "web_search")
                    web_result_str = str(web_result) if web_result else ""
                    logger.info("Retrieved web search results")
                except Exception as e:
                    logger.warning(f"Web search failed: {str(e)}")
                    web_result_str = ""
            
            # Check for geotechnical calculations
            is_calculation_query = any(term in user_input.lower() for term in [
                "calculate", "computation", "analysis", "soil", "rock", "tunnel", 
                "pressure", "support", "stability", "tbm", "boring", "classification"
            ])
            
            geotech_result_str = ""
            if is_calculation_query:
                try:
                    geotech_result = safe_agent_call(geotech_agent, user_input, "geotechnical")
                    geotech_result_str = str(geotech_result) if geotech_result else ""
                    logger.info("Retrieved geotechnical analysis results")
                except Exception as e:
                    logger.warning(f"Geotechnical analysis failed: {str(e)}")
                    geotech_result_str = ""
            
            # Response synthesis
            if geotech_result_str:
                response = geotech_result_str
            elif web_result_str:
                response = web_result_str
            else:
                try:
                    simple_prompt = f"Please answer this question directly: {user_input}"
                    manager_result = safe_agent_call(manager_agent, simple_prompt, "manager_fallback")
                    response = str(manager_result) if manager_result else "I couldn't generate a response to your query."
                except Exception as e:
                    logger.error(f"Manager agent failed: {str(e)}")
                    response = "I encountered an error while processing your request. Please try rephrasing your question."
            
            st.session_state.processing_status = None
            logger.info(f"Final response created, length: {len(response)}")
            return validate_response(response, "general_query")
            
        except Exception as e:
            logger.error(f"Error in agent processing: {str(e)}\n{traceback.format_exc()}")
            st.session_state.processing_status = None
            return validate_response(f"I encountered an error while processing your request: {str(e)}", "agent_error")
            
    except Exception as e:
        logger.error(f"Error in process_request: {str(e)}\n{traceback.format_exc()}")
        st.session_state.processing_status = None
        return validate_response(f"I encountered an unexpected error. Please try again or rephrase your query.", "general_error")

# Initialize agents with all fixes applied
web_agent, geotech_agent, manager_agent = initialize_agents()

# Add debugging sidebar
def add_debug_sidebar():
    """Add debugging information to sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("🔍 Debug Info")
        
        # Debug mode toggle
        debug_mode = st.checkbox("Enable Debug Mode", False)
        
        if debug_mode:
            # Session state info
            with st.expander("Session State", expanded=False):
                st.write("Chat History Length:", len(st.session_state.get('chat_history', [])))
                st.write("Processing Status:", st.session_state.get('processing_status', 'None'))
                st.write("Current Analysis:", st.session_state.get('current_analysis', 'None'))
                st.write("PDF Uploaded:", bool(st.session_state.get('uploaded_pdf')))
            
            # Agent status
            with st.expander("Agent Status", expanded=False):
                st.write("Web Agent:", "✅ Initialized" if web_agent else "❌ Not initialized")
                st.write("Geotech Agent:", "✅ Initialized" if geotech_agent else "❌ Not initialized")
                st.write("Manager Agent:", "✅ Initialized" if manager_agent else "❌ Not initialized")
            
            # Recent logs
            if st.button("Show Recent Logs"):
                try:
                    with open("geotechnical_app_detailed.log", "r") as f:
                        lines = f.readlines()
                        recent_lines = lines[-20:]  # Last 20 lines
                        st.text_area("Recent Log Lines", "\n".join(recent_lines), height=200)
                except Exception as e:
                    st.error(f"Could not read log file: {str(e)}")
        
        # Reset button
        if st.button("🔄 Reset Everything"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# Test function for debugging
def test_agent_responses():
    """Test function to verify agent responses work correctly."""
    st.subheader("🧪 Agent Response Test")
    
    if st.button("Test Web Agent"):
        try:
            test_query = "What is Python programming?"
            with st.spinner("Testing web agent..."):
                if web_agent:
                    result = safe_agent_call(web_agent, test_query, "test_web")
                    st.success("Web Agent Test Result:")
                    st.write(result)
                else:
                    st.error("Web agent not initialized")
        except Exception as e:
            st.error(f"Web agent test failed: {str(e)}")
    
    if st.button("Test Geotech Agent"):
        try:
            test_query = "Calculate tunnel support for 10m depth"
            with st.spinner("Testing geotech agent..."):
                if geotech_agent:
                    result = safe_agent_call(geotech_agent, test_query, "test_geotech")
                    st.success("Geotech Agent Test Result:")
                    st.write(result)
                else:
                    st.error("Geotech agent not initialized")
        except Exception as e:
            st.error(f"Geotech agent test failed: {str(e)}")
    
    if st.button("Test Manager Agent"):
        try:
            test_query = "What is 2 + 2?"
            with st.spinner("Testing manager agent..."):
                if manager_agent:
                    result = safe_agent_call(manager_agent, test_query, "test_manager")
                    st.success("Manager Agent Test Result:")
                    st.write(result)
                else:
                    st.error("Manager agent not initialized")
        except Exception as e:
            st.error(f"Manager agent test failed: {str(e)}")

# Main UI with status indicators
st.title("🏗️ Geotechnical AI Agent by Qwen2.5-Coder-32B-Instruct")

# Show system status
col1, col2, col3 = st.columns(3)
with col1:
    if smolagents_available:
        st.success("✅ SmolaAgents Available")
    else:
        st.error("❌ SmolaAgents Not Available")

with col2:
    if web_agent and geotech_agent and manager_agent:
        st.success("✅ AI Agents Ready")
    else:
        st.error("❌ AI Agents Not Ready")

with col3:
    try:
        runpod_status = st.secrets["runpod"]["endpoint_url"]
        st.success("✅ PDF Analysis Ready")
    except:
        st.warning("⚠️ PDF Analysis Limited")

# Show processing status if available
if st.session_state.processing_status:
    st.info(f"Status: {st.session_state.processing_status}")

# Sidebar
with st.sidebar:
    st.title("🔧 Analysis Tools")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Soil Classification", "Tunnel Support", "RMR Analysis", "TBM Performance"]
    )
    with st.expander("Analysis Parameters", expanded=True):
        if analysis_type == "Soil Classification":
            st.session_state.analysis_params = {
                "soil_type": st.selectbox("Soil Type", ["clay", "sand", "silt"]),
                "plasticity_index": st.number_input("Plasticity Index", 0.0, 100.0, 25.0),
                "liquid_limit": st.number_input("Liquid Limit", 0.0, 100.0, 50.0)
            }
        elif analysis_type == "Tunnel Support":
            st.session_state.analysis_params = {
                "depth": st.number_input("Depth (m)", 0.0, 1000.0, 100.0),
                "soil_density": st.number_input("Soil Density (kg/m³)", 1000.0, 3000.0, 1800.0),
                "k0": st.number_input("K₀ Coefficient", 0.0, 2.0, 0.5),
                "tunnel_diameter": st.number_input("Tunnel Diameter (m)", 1.0, 20.0, 6.0)
            }
        elif analysis_type == "RMR Analysis":
            st.session_state.analysis_params = {
                "ucs": st.number_input("UCS (MPa)", 0.0, 250.0, 100.0),
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 75.0),
                "spacing": st.number_input("Joint Spacing (m)", 0.0, 2.0, 0.6),
                "condition": st.slider("Joint Condition", 0, 30, 15),
                "groundwater": st.slider("Groundwater Condition", 0, 15, 10),
                "orientation": st.slider("Joint Orientation", -12, 0, -5)
            }
        elif analysis_type == "TBM Performance":
            st.session_state.analysis_params = {
                "ucs": st.number_input("UCS (MPa)", 0.0, 250.0, 100.0),
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 75.0),
                "joint_spacing": st.number_input("Joint Spacing (m)", 0.0, 2.0, 0.6),
                "abrasivity": st.number_input("Cerchar Abrasivity Index", 0.0, 6.0, 2.0),
                "diameter": st.number_input("TBM Diameter (m)", 1.0, 15.0, 6.0)
            }
    
    if st.button("Run Analysis"):
        with st.spinner("Processing..."):
            if analysis_type == "Soil Classification":
                st.session_state.current_analysis = classify_soil(
                    st.session_state.analysis_params["soil_type"],
                    st.session_state.analysis_params["plasticity_index"],
                    st.session_state.analysis_params["liquid_limit"]
                )
            elif analysis_type == "Tunnel Support":
                st.session_state.current_analysis = calculate_tunnel_support(
                    st.session_state.analysis_params["depth"],
                    st.session_state.analysis_params["soil_density"],
                    st.session_state.analysis_params["k0"],
                    st.session_state.analysis_params["tunnel_diameter"]
                )
            elif analysis_type == "RMR Analysis":
                st.session_state.current_analysis = calculate_rmr(
                    st.session_state.analysis_params["ucs"],
                    st.session_state.analysis_params["rqd"],
                    st.session_state.analysis_params["spacing"],
                    st.session_state.analysis_params["condition"],
                    st.session_state.analysis_params["groundwater"],
                    st.session_state.analysis_params["orientation"]
                )
            elif analysis_type == "TBM Performance":
                st.session_state.current_analysis = estimate_tbm_performance(
                    st.session_state.analysis_params["ucs"],
                    st.session_state.analysis_params["rqd"],
                    st.session_state.analysis_params["joint_spacing"],
                    st.session_state.analysis_params["abrasivity"],
                    st.session_state.analysis_params["diameter"]
                )
    
    st.markdown("---")
    
    # PDF Analysis section in sidebar
    st.title("📄 PDF Analysis")
    
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file is not None:
        st.session_state.uploaded_pdf = uploaded_file.getvalue()
        
        # PDF info display
        with st.expander("PDF Information", expanded=True):
            with io.BytesIO(st.session_state.uploaded_pdf) as pdf_data:
                reader = PyPDF2.PdfReader(pdf_data)
                st.write(f"Pages: {len(reader.pages)}")
                st.write(f"Size: {len(st.session_state.uploaded_pdf)/1024:.1f} KB")
        
        # Query input for PDF analysis
        pdf_query = st.text_input("Enter query for PDF analysis:")
        
        if st.button("Analyze PDF"):
            if pdf_query:
                st.session_state.processing_status = "Preparing PDF for analysis..."
                with st.spinner("Analyzing PDF with ColPali VLM via Runpod..."):
                    st.session_state.pdf_analysis = find_relevant_pdf_sections(
                        st.session_state.uploaded_pdf, 
                        pdf_query
                    )
            else:
                st.warning("Please enter a query for analysis.")
    
    # Test functionality
    if st.button("🧪 Test System"):
        st.write("Testing system components...")
        
        # Test PyTorch
        try:
            import torch
            st.success(f"✅ PyTorch {torch.__version__}")
        except Exception as e:
            st.error(f"❌ PyTorch: {e}")
        
        # Test Asyncio
        try:
            loop = asyncio.get_running_loop()
            st.success("✅ Asyncio Event Loop")
        except Exception as e:
            st.warning(f"⚠️ Asyncio: Creating new loop")
        
        # Test Agents
        if smolagents_available and manager_agent:
            try:
                test_result = safe_agent_call(manager_agent, "What is 1+1?", "system_test")
                st.success(f"✅ Agent Test: {test_result}")
            except Exception as e:
                st.error(f"❌ Agent Test: {e}")
    
    # Add debug sidebar
    add_debug_sidebar()

# Tabs for different functionalities
chat_tab, analysis_tab, pdf_tab, debug_tab = st.tabs(["Chat", "Analysis Results", "PDF Analysis", "Debug"])

# FIXED CHAT TAB
with chat_tab:
    st.subheader("💬 Chat Interface")
    
    # Display previous chat messages - FIXED VERSION
    for msg in st.session_state.chat_history:
        try:
            with st.chat_message(msg["role"]):
                # Ensure content is properly handled as string
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = str(content)
                
                # Clean and display content
                if content.strip():
                    st.markdown(content.strip())
                else:
                    st.markdown("*No content*")
        except Exception as e:
            logger.error(f"Error displaying chat message: {str(e)}")
            with st.chat_message(msg["role"]):
                st.error("Error displaying message")
    
    # Chat input - IMPROVED VERSION
    if user_prompt := st.chat_input("Ask a question or describe your task:"):
        # Add user message to chat history and display immediately
        user_msg = {"role": "user", "content": user_prompt}
        st.session_state.chat_history.append(user_msg)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Process and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            try:
                # Show processing indicator
                with st.spinner("Processing your request..."):
                    # CRITICAL FIX: Call the corrected process_request function
                    response = process_request(user_prompt)
                
                # CRITICAL FIX: Ensure response is a clean string
                if response is None:
                    response = "I couldn't generate a response to your question."
                elif not isinstance(response, str):
                    response = str(response)
                
                # Clean the response
                response = response.strip()
                if not response:
                    response = "I received an empty response. Please try rephrasing your question."
                
                # Display response immediately - no complex processing
                message_placeholder.markdown(response)
                
                # Add to chat history
                assistant_msg = {"role": "assistant", "content": response}
                st.session_state.chat_history.append(assistant_msg)
                
                # Log success
                logger.info(f"Successfully processed and displayed response (length: {len(response)})")
                
            except Exception as e:
                # Handle any errors gracefully
                error_msg = f"I encountered an error while processing your request: {str(e)}"
                logger.error(f"Chat processing error: {str(e)}\n{traceback.format_exc()}")
                
                # Display error message
                message_placeholder.error(error_msg)
                
                # Add error to history
                error_assistant_msg = {"role": "assistant", "content": error_msg}
                st.session_state.chat_history.append(error_assistant_msg)
            
            finally:
                # Clear processing status
                st.session_state.processing_status = None

with analysis_tab:
    st.subheader("📊 Analysis Results")
    if st.session_state.current_analysis:
        with st.expander("Detailed Results", expanded=True):
            st.json(st.session_state.current_analysis)
        if analysis_type == "Tunnel Support":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, st.session_state.analysis_params["tunnel_diameter"]],
                y=[st.session_state.analysis_params["depth"], 
                   st.session_state.analysis_params["depth"]],
                mode='lines',
                name='Tunnel Level'
            ))
            fig.update_layout(
                title="Tunnel Cross Section",
                xaxis_title="Width (m)",
                yaxis_title="Depth (m)",
                yaxis_autorange="reversed"
            )
            st.plotly_chart(fig)
        elif analysis_type == "RMR Analysis":
            if "component_ratings" in st.session_state.current_analysis:
                ratings = st.session_state.current_analysis["component_ratings"]
                labels = {
                    "ucs_rating": "UCS",
                    "rqd_rating": "RQD",
                    "spacing_rating": "Spacing",
                    "condition_rating": "Condition",
                    "groundwater_rating": "Groundwater",
                    "orientation_rating": "Orientation"
                }
                fig = go.Figure(data=[
                    go.Bar(x=[labels[k] for k in ratings.keys()], 
                           y=list(ratings.values()))
                ])
                fig.update_layout(
                    title="RMR Component Ratings",
                    xaxis_title="Parameters",
                    yaxis_title="Rating"
                )
                st.plotly_chart(fig)

with pdf_tab:
    st.subheader("📊 PDF Analysis Results")
    
    if st.session_state.pdf_analysis:
        if "error" in st.session_state.pdf_analysis:
            st.error(f"Error during PDF analysis: {st.session_state.pdf_analysis['error']}")
        else:
            st.write(f"**Analysis Results for Query:** '{st.session_state.pdf_analysis.get('query', 'Unknown')}'")
            st.write(f"**Total Pages:** {st.session_state.pdf_analysis.get('total_pages', 0)}")
            
            # Display relevant sections with detailed information
            for i, section in enumerate(st.session_state.pdf_analysis.get('relevant_sections', [])):
                with st.expander(f"Page {section.get('page_number', 'Unknown')} (Score: {section.get('similarity_score', 0):.4f})", expanded=i==0):
                    # Display snippets
                    for j, snippet in enumerate(section.get('snippets', [])):
                        st.markdown(f"**Snippet {j+1}:**")
                        st.markdown(f"> {snippet}")
                    
                    # Option to view full page content
                    if st.checkbox(f"Show full page content for Page {section.get('page_number', 'Unknown')}", key=f"full_page_{i}"):
                        st.text_area("Full Page Content:", section.get('content', ''), height=300)
                    
                    # Display page image preview using cached images
                    if st.checkbox(f"Show image preview for Page {section.get('page_number', 'Unknown')}", key=f"img_preview_page_{i}"):
                        page_num = section.get('page_number', 0)
                        if "pdf_page_images" in st.session_state and page_num > 0 and page_num <= len(st.session_state.pdf_page_images):
                            st.image(st.session_state.pdf_page_images[page_num-1], 
                                     caption=f"Page {page_num}", 
                                     use_container_width=True)
                        else:
                            st.warning(f"Preview not available for page {page_num}")
    elif st.session_state.uploaded_pdf:
        st.info("Ask a question about the PDF in the chat interface to see analysis results here.")
    else:
        st.info("Please upload a PDF document using the sidebar to analyze it.")

with debug_tab:
    test_agent_responses()

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Kilic Intelligence")
st.sidebar.markdown("🛠️ **Fixed Version** - PyTorch/Streamlit Compatible")
