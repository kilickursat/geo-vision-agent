import streamlit as st
import math
import logging
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import requests
from datetime import datetime
from huggingface_hub import login
from typing import Dict, List
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool, CodeAgent, HfApiModel, ManagedAgent, ToolCallingAgent, DuckDuckGoSearchTool
import traceback
import sys

# Page configuration
st.set_page_config(
    page_title="Advanced Geotechnical AI Agent by Qwen2.5-Coder-32B-Instruct",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = {}

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
    search_tool = DuckDuckGoSearchTool()
    try:
        results = search_tool(query)  # Changed from .run() to direct call
        return str(results)
    except Exception as e:
        return f"Search error: {str(e)}"

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
        logging.error(f"Error processing borehole data: {e}")
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

@st.cache_resource
def initialize_agents():
    try:
        hf_key = st.secrets["HUGGINGFACE_API_KEY"]
        login(hf_key)
        model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")  # Updated to Qwen model <button class="citation-flag" data-index="4">
        # Web search agent
        web_agent = ToolCallingAgent(
            tools=[search_geotechnical_data, visit_webpage],
            model=model,
            max_steps=10
        )
        managed_web_agent = ManagedAgent(
            agent=web_agent,
            name="geotech_web_search",
            description="Performs web searches for geotechnical data."
        )
        # Geotech calculation agent
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
                predict_cutter_life
            ],
            model=model,
            max_steps=10
        )
        managed_geotech_agent = ManagedAgent(
            agent=geotech_agent,
            name="geotech_analysis",
            description="Performs geotechnical calculations."
        )
        # Manager agent with search_geotechnical_data tool
        manager_agent = CodeAgent(
            tools=[search_geotechnical_data],
            model=model,
            managed_agents=[managed_web_agent, managed_geotech_agent],
            additional_authorized_imports=["time", "numpy", "pandas"]
        )
        return managed_web_agent, managed_geotech_agent, manager_agent
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}\nFull traceback:\n{traceback.format_exc()}")
        return None, None, None

def process_request(request: str):
    try:
        web_result = search_geotechnical_data(request)
        geotech_result = managed_geotech_agent(request=request)
        final_result = list(manager_agent.run(
            request,
            {
                "web_data": web_result,
                "technical_analysis": str(geotech_result)
            }
        ))
        if final_result:
            result = final_result[-1].content if hasattr(final_result[-1], 'content') else final_result[-1]
            return result
        else:
            return "No results generated"
    except Exception as e:
        return f"Error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"

# Initialize agent
managed_web_agent, managed_geotech_agent, manager_agent = initialize_agents()

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

def display_chat_message(msg):
    try:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        content = msg["content"]
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2)
        st.markdown(f"{role_icon} **{msg['role'].title()}:** {content}")
    except Exception as e:
        st.error(f"Error displaying message: {str(e)}")

# Main content
st.title("🏗️ Geotechnical AI Agent by Qwen2.5-Coder-32B-Instruct")
st.subheader("💬 Chat Interface")
user_input = st.text_input("Ask a question:")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Processing..."):
        response = process_request(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Update chat display section
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        display_chat_message(msg)

# Analysis Results Section
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

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Kilic Intelligence")
