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
from typing import Dict, List, Optional
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool, CodeAgent, HfApiModel, ToolCallingAgent, DuckDuckGoSearchTool
import traceback
import sys
import os
from io import BytesIO

# --- New Imports ---
import torch
from PIL import Image
import fitz  # PyMuPDF
from transformers import AutoProcessor, AutoModelForCausalLM
# --- End New Imports ---


# --- Page configuration ---
st.set_page_config(
    page_title="Advanced Geotechnical AI Agent by Qwen2.5 + SmolVLM2", # Updated Title
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    /* Add custom styles if needed */
    .stSpinner > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize session state ---
DEFAULT_CHAT_HISTORY = [{"role": "assistant", "content": "Hello! Ask me a geotechnical question, perform an analysis using the sidebar tools, or upload a document/image for analysis."}]
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = DEFAULT_CHAT_HISTORY
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = {}
# --- New Session State Variables ---
if 'vlm_description' not in st.session_state:
    st.session_state.vlm_description = None
if 'vlm_search_results' not in st.session_state:
    st.session_state.vlm_search_results = None
if 'uploaded_file_info' not in st.session_state:
    st.session_state.uploaded_file_info = None
# --- End New Session State Variables ---


# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Existing Tools (No changes needed here initially) ---
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.
    Args:
        url: The URL of the webpage to visit and retrieve content from.
    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    logging.info(f"Visiting webpage: {url}")
    try:
        response = requests.get(url, timeout=15) # Added timeout
        response.raise_for_status()
        # Basic HTML cleaning before markdownify
        # Consider using BeautifulSoup for more robust cleaning if needed
        text = re.sub(r"<script.*?</script>", "", response.text, flags=re.DOTALL)
        text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<head.*?</head>", "", text, flags=re.DOTALL)
        text = re.sub(r"<nav.*?</nav>", "", text, flags=re.DOTALL) # Remove nav
        text = re.sub(r"<footer.*?</footer>", "", text, flags=re.DOTALL) # Remove footer
        markdown_content = markdownify(text).strip()
        # Clean up excessive newlines more aggressively
        markdown_content = re.sub(r"\n\s*\n", "\n\n", markdown_content)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        # Limit content length to avoid overwhelming the context
        max_length = 10000
        if len(markdown_content) > max_length:
            logging.warning(f"Webpage content truncated ({len(markdown_content)} > {max_length})")
            markdown_content = markdown_content[:max_length] + "... (truncated)"
        return markdown_content
    except RequestException as e:
        logging.error(f"Error fetching webpage {url}: {str(e)}")
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error visiting webpage {url}: {str(e)}", exc_info=True)
        return f"Unexpected error processing webpage: {str(e)}"

@tool
def search_geotechnical_data(query: str) -> str:
    """Searches for geotechnical information using DuckDuckGo.
    Args:
        query: The search query for finding geotechnical information.
    Returns:
        Search results as formatted text, or an error message.
    """
    logging.info(f"Performing web search for: {query}")
    search_tool = DuckDuckGoSearchTool(max_results=5) # Limit results
    try:
        results = search_tool(query)
        if not results:
            return "No search results found."
        # Format results for better readability
        formatted_results = f"Search results for '{query}':\n\n"
        # DuckDuckGoSearchTool now returns a list of dicts
        if isinstance(results, list):
             for i, res in enumerate(results, 1):
                title = res.get('title', 'N/A')
                link = res.get('link', '#')
                snippet = res.get('snippet', 'N/A')
                formatted_results += f"{i}. **{title}**\n   *Link:* {link}\n   *Snippet:* {snippet}\n\n"
        elif isinstance(results, str): # Fallback if format changes
             formatted_results += results
        else:
             formatted_results += str(results)

        return formatted_results

    except Exception as e:
        logging.error(f"Search error for query '{query}': {str(e)}", exc_info=True)
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
    logging.info(f"Classifying soil: type={soil_type}, PI={plasticity_index}, LL={liquid_limit}")
    soil_type = soil_type.strip().lower()
    try:
        # Basic validation
        if not soil_type or plasticity_index < 0 or liquid_limit < 0:
            raise ValueError("Invalid input parameters for soil classification.")

        # Simplified USCS Logic (expand as needed)
        if liquid_limit < 50: # Low plasticity
            if soil_type == 'clay':
                # Need more info (e.g., % passing #200 sieve) for full USCS, this is simplified
                if plasticity_index > 7 and plasticity_index > (0.73 * (liquid_limit - 20)): # Above A-line
                     return {"classification": "CL", "description": "Low plasticity clay"}
                elif plasticity_index < 4 or plasticity_index < (0.73 * (liquid_limit - 20)): # Below A-line
                     return {"classification": "ML", "description": "Low plasticity silt (or organic clay/silt - ML/OL)"}
                else: # On A-line
                     return {"classification": "CL-ML", "description": "Dual symbol: Silty clay"}
            elif soil_type == 'silt':
                 return {"classification": "ML", "description": "Low plasticity silt"}
            # Add logic for Sand (requires grain size distribution)
            elif soil_type == 'sand':
                 return {"classification": "SM/SC (Incomplete)", "description": "Sand classification requires grain size distribution."}
        else: # High plasticity
            if soil_type == 'clay':
                if plasticity_index > (0.73 * (liquid_limit - 20)): # Above A-line
                    return {"classification": "CH", "description": "High plasticity clay"}
                else: # Below A-line
                    return {"classification": "MH", "description": "High plasticity silt (or organic clay/silt - MH/OH)"}
            elif soil_type == 'silt':
                return {"classification": "MH", "description": "High plasticity silt"}
            elif soil_type == 'sand':
                 return {"classification": "SM/SC (Incomplete)", "description": "Sand classification requires grain size distribution."}

        return {"classification": "Unknown", "description": f"Cannot classify based only on type '{soil_type}', PI, and LL. Need grain size distribution for sands/gravels or more detailed plasticity chart analysis."}
    except ValueError as ve:
        logging.warning(f"Soil classification input error: {ve}")
        return {"classification": "Error", "description": str(ve)}
    except Exception as e:
        logging.error("Error during soil classification", exc_info=True)
        return {"classification": "Error", "description": f"An unexpected error occurred: {str(e)}"}

@tool
def calculate_tunnel_support(depth: float, soil_density: float, k0: float, tunnel_diameter: float) -> Dict:
    """Calculate tunnel support pressure and related parameters, assuming simplified elastic conditions.
    Args:
        depth: Tunnel depth from surface in meters (centerline or crown, specify assumption) - Assuming centerline.
        soil_density: Average soil unit weight in kN/m³ (typical range 16-22).
        k0: At-rest earth pressure coefficient (typically 0.5-1.0 for soils).
        tunnel_diameter: Tunnel diameter in meters.
    Returns:
        Dictionary containing estimated support pressures, stresses and suggested safety factors. Note: This is a simplified calculation.
    """
    logging.info(f"Calculating tunnel support: depth={depth}m, density={soil_density}kN/m³, k0={k0}, diameter={tunnel_diameter}m")
    try:
        if not all(p > 0 for p in [depth, soil_density, k0, tunnel_diameter]):
             raise ValueError("Input parameters must be positive.")
        # Note: Using density in kN/m³ directly simplifies calculation. If kg/m³ is input, convert (multiply by 9.81/1000)
        # Assume soil_density is provided in kN/m³ as per typical geotech practice for these calcs.
        # If kg/m3 is expected, uncomment below:
        # unit_weight = soil_density * 9.81 / 1000 # kN/m³

        unit_weight = soil_density # Assuming input is kN/m³
        vertical_stress_total = depth * unit_weight # Total vertical stress at tunnel centerline (kPa)
        horizontal_stress_total = k0 * vertical_stress_total # Total horizontal stress (kPa)

        # Simplified estimation of required support pressure (can vary significantly based on method)
        # Using average stress, common simple approach but may not be conservative
        avg_stress = (vertical_stress_total + horizontal_stress_total) / 2

        # More refined methods exist (e.g., considering plasticity, arching), this is basic.
        # Example using convergence-confinement, very simplified: P_support ~ (1-lambda)*P_initial
        # Where lambda depends on allowed convergence. Let's stick to avg stress for simplicity here.
        support_pressure_estimated = avg_stress # kPa

        # Safety factor depends on many things (uncertainty, consequence of failure, design stage)
        # Providing typical ranges, actual SF should be determined by project standards.
        safety_factor_range = "1.5-2.5+" # Typical range for ultimate limit state

        return {
            "calculation_notes": "Simplified elastic calculation assuming centerline depth and uniform soil.",
            "vertical_stress_total_kPa": round(vertical_stress_total, 2),
            "horizontal_stress_total_kPa": round(horizontal_stress_total, 2),
            "average_in_situ_stress_kPa": round(avg_stress, 2),
            "estimated_support_pressure_kPa": round(support_pressure_estimated, 2),
            "typical_safety_factor_range": safety_factor_range,
            "design_pressure_example_SF2": round(support_pressure_estimated * 2.0, 2) # Example with SF=2
        }
    except ValueError as ve:
        logging.warning(f"Tunnel support calculation input error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error("Error calculating tunnel support", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Helper functions for RMR/Q ---
def get_support_recommendations(rmr: int) -> Dict:
    """Get support recommendations based on Bieniawski's RMR (1989 version).
    Args:        rmr: Rock Mass Rating value
    Returns:     Dictionary containing support recommendations
    """
    # ... (keep existing RMR recommendations) ...
    if rmr > 80:
        return {
            "rock_class": "I - Very good rock",
            "excavation": "Full face, 3m advance",
            "support": "Generally no support required",
            "bolting": "Spot bolting if needed",
            "shotcrete": "None required",
            "steel_sets": "None required"
        }
    elif rmr > 60:
        return {
            "rock_class": "II - Good rock",
            "excavation": "Full face, 1.0-1.5m advance",
            "support": "Install support concurrent with excavation, 10m from face", # Adjusted distance
            "bolting": "Systematic bolts, 2.5m spacing (staggered)", # Updated spacing
            "shotcrete": "Occasional 50mm in crown where required", # Updated description
            "steel_sets": "None required"
        }
    elif rmr > 40:
        return {
            "rock_class": "III - Fair rock",
            "excavation": "Top heading and bench, 1.0-1.5m advance in top heading", # Refined excavation
            "support": "Install support concurrent with excavation, <10m from face",
            "bolting": "Systematic bolts, 1.5-2m spacing in crown and walls", # Updated spacing/location
            "shotcrete": "50-100mm in crown and 30mm+ in walls", # Updated thickness/location
            "steel_sets": "Light ribs spaced 1.5m where required (especially in poor zones)" # Added context
        }
    elif rmr > 20:
         return {
            "rock_class": "IV - Poor rock",
            "excavation": "Top heading and bench, 1.0m advance. Immediate support.", # Refined excavation
            "support": "Install support concurrent with excavation, close to face",
            "bolting": "Systematic bolts, 1-1.5m spacing in crown and walls",
            "shotcrete": "100-150mm in crown, 100mm+ in walls, potentially with mesh", # Added mesh note
            "steel_sets": "Medium to heavy ribs spaced 0.75-1.5m, potential lagging" # Added lagging note
        }
    else: # RMR <= 20
        return {
            "rock_class": "V - Very poor rock",
            "excavation": "Multiple drifts or short rounds (0.5-1m), immediate support", # Refined excavation
            "support": "Install support concurrent with excavation, at the face",
            "bolting": "Systematic bolts, 1m spacing, often with mesh/straps", # Added straps
            "shotcrete": "150mm+ in crown and walls, often reinforced", # Added reinforced note
            "steel_sets": "Heavy ribs spaced 0.75m or less, often with forepoling/invert strut" # Added techniques
        }

def get_q_support_category(q: float, span_or_height: float = 10.0, esr: float = 1.6) -> Dict:
    """Get Q-system support recommendations (Barton et al., 1974, simplified chart lookup).
    Args:
        q: Q-system value
        span_or_height: Tunnel span or height (m). Default 10m.
        esr: Excavation Support Ratio (related to safety level/use). Default 1.6 (standard tunnels).
    Returns:
        Dictionary containing support category and potentially recommendations.
    """
    # Simplified - requires Q-chart lookup for accurate recommendations based on De and ESR.
    # This provides general categories based on Q value only.
    # De = Span / ESR
    equiv_dimension = span_or_height / esr

    # Rough category mapping based on Q alone (very approximate)
    if q > 400: cat = "Exceptionally good" # Extended range
    elif q > 100: cat = "Extremely good"
    elif q > 40: cat = "Very good"
    elif q > 10: cat = "Good"
    elif q > 4: cat = "Fair"
    elif q > 1: cat = "Poor"
    elif q > 0.1: cat = "Very poor"
    elif q > 0.01: cat = "Extremely poor"
    else: cat = "Exceptionally poor" # Extended range

    # Placeholder recommendations - **Actual support depends heavily on De/ESR**
    support_details = "Specific support (bolt length/spacing, shotcrete thickness) requires lookup on Barton's Q-chart using Span/ESR."
    if q > 10:
        recommendation = "Generally unsupported or spot bolting."
    elif q > 1:
        recommendation = "Systematic bolting, possibly with shotcrete in crown."
    elif q > 0.1:
        recommendation = "Systematic bolting with mesh, shotcrete (50-100mm)."
    else:
        recommendation = "Heavy support: Bolting, mesh, reinforced shotcrete (100mm+), potentially steel sets/ribs."


    return {
        "q_value": round(q, 3), # More precision for low Q
        "rock_mass_quality": cat,
        "equiv_dimension_for_chart": round(equiv_dimension, 2),
        "esr_used": esr,
        "support_note": support_details,
        "general_recommendation": recommendation
        # Add detailed chart lookup logic here if possible/needed
    }

# --- RMR Calculation Tool ---
@tool
def calculate_rmr(ucs: float, rqd: float, spacing: float, condition: int, groundwater: int, orientation: int) -> Dict:
    """Calculate Rock Mass Rating (RMR - Bieniawski 1989) classification.
    Args:
        ucs: Uniaxial compressive strength in MPa. Use point load index Is(50) * ~22-25 if UCS not available.
        rqd: Rock Quality Designation as percentage (0-100).
        spacing: Average discontinuity spacing in meters.
        condition: Discontinuity condition rating (0-30). Sum of length, aperture, roughness, infilling, weathering.
        groundwater: Groundwater condition rating (0-15). Based on inflow or water pressure.
        orientation: Discontinuity orientation adjustment rating (-12 to 0). Depends on tunnel drive direction relative to joints. Set to 0 if orientation is favorable or not critical.
    Returns:
        Dictionary containing RMR value, rock class, support recommendations, and component ratings.
    """
    logging.info(f"Calculating RMR: UCS={ucs}, RQD={rqd}, Spacing={spacing}, Cond={condition}, GW={groundwater}, Orient={orientation}")
    try:
        # Parameter validation
        if not (0 <= rqd <= 100): raise ValueError("RQD must be between 0 and 100.")
        if spacing <= 0: raise ValueError("Spacing must be positive.")
        if not (0 <= condition <= 30): raise ValueError("Condition rating must be between 0 and 30.")
        if not (0 <= groundwater <= 15): raise ValueError("Groundwater rating must be between 0 and 15.")
        if not (-12 <= orientation <= 0): raise ValueError("Orientation rating must be between -12 and 0.")
        if ucs < 0: raise ValueError("UCS must be non-negative.")

        # RMR component ratings (Bieniawski 1989)
        # 1. Strength of Intact Rock Material
        if ucs > 250: ucs_rating = 15
        elif ucs >= 100: ucs_rating = 12 # Use >= for boundary cases
        elif ucs >= 50: ucs_rating = 7
        elif ucs >= 25: ucs_rating = 4
        elif ucs >= 5: ucs_rating = 2 # Added lower ranges based on some charts
        elif ucs >= 1: ucs_rating = 1
        else: ucs_rating = 0 # For very weak material

        # 2. RQD Rating
        if rqd >= 90: rqd_rating = 20
        elif rqd >= 75: rqd_rating = 17
        elif rqd >= 50: rqd_rating = 13
        elif rqd >= 25: rqd_rating = 8
        else: rqd_rating = 3

        # 3. Spacing of Discontinuities Rating
        if spacing > 2.0: spacing_rating = 20
        elif spacing > 0.6: spacing_rating = 15
        elif spacing > 0.2: spacing_rating = 10
        elif spacing >= 0.06: spacing_rating = 8 # Changed to >=
        else: spacing_rating = 5 # spacing < 0.06m

        # 4. Condition of Discontinuities Rating (Direct input)
        condition_rating = condition

        # 5. Groundwater Rating (Direct input)
        groundwater_rating = groundwater

        # Basic RMR = Sum of 1-5
        rmr_basic = ucs_rating + rqd_rating + spacing_rating + condition_rating + groundwater_rating

        # 6. Adjustment for Discontinuity Orientation (Direct input)
        orientation_adjustment = orientation
        rmr_adjusted = rmr_basic + orientation_adjustment

        # Ensure RMR is not > 100 or < 0 (though negative adjustment could make it slightly < 0)
        rmr_final = max(0, min(100, rmr_adjusted))

        # Get Rock Class and Support Recommendations
        support_recs = get_support_recommendations(rmr_final)
        rock_class = support_recs.get("rock_class", "N/A") # Get class from recommendations dict

        return {
            "rmr_basic (RMR_b)": rmr_basic,
            "orientation_adjustment": orientation_adjustment,
            "rmr_adjusted (RMR_89)": rmr_final,
            "rock_class": rock_class,
            "support_recommendations": support_recs, # Now includes rock class too
            "component_ratings": {
                "ucs_rating (R1)": ucs_rating,
                "rqd_rating (R2)": rqd_rating,
                "spacing_rating (R3)": spacing_rating,
                "condition_rating (R4)": condition_rating,
                "groundwater_rating (R5)": groundwater_rating,
                "orientation_adjustment (R6)": orientation_adjustment
            }
        }
    except ValueError as ve:
        logging.warning(f"RMR calculation input error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error("Error calculating RMR", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}

# --- Q-System Calculation Tool ---
@tool
def calculate_q_system(rqd: float, jn: float, jr: float, ja: float, jw: float, srf: float, span: Optional[float]=None, esr: Optional[float]=None) -> Dict:
    """Calculate Q-system rating (Barton et al.) and provide general support category.
    Args:
        rqd: Rock Quality Designation as percentage (0-100, use 10 if RQD is measured <10).
        jn: Joint set number (0.5-20).
        jr: Joint roughness number (0.5-4).
        ja: Joint alteration number (0.75-20).
        jw: Joint water reduction factor (0.05-1.0).
        srf: Stress Reduction Factor (1-400, depends on stress conditions, rock strength, depth).
        span: Tunnel span or height (m). Optional, used for support chart context.
        esr: Excavation Support Ratio. Optional, used for support chart context.
    Returns:
        Dictionary containing Q-value, quality description, and context for support lookup.
    """
    logging.info(f"Calculating Q-System: RQD={rqd}, Jn={jn}, Jr={jr}, Ja={ja}, Jw={jw}, SRF={srf}, Span={span}, ESR={esr}")
    try:
        # Input validation
        if not (0 <= rqd <= 100): raise ValueError("RQD must be between 0 and 100.")
        # Use RQD=10 if measured RQD is very low, as per Barton's recommendation
        rqd_calc = max(rqd, 10.0) if rqd < 10 else rqd

        # Check typical ranges for other parameters (warnings, not errors)
        if not (0.5 <= jn <= 20): logging.warning(f"Jn={jn} is outside typical range (0.5-20).")
        if not (0.5 <= jr <= 4): logging.warning(f"Jr={jr} is outside typical range (0.5-4).")
        # Ja can go up to 20 for clay fillings
        if not (0.75 <= ja <= 20): logging.warning(f"Ja={ja} is outside typical range (0.75-20).")
        if not (0.05 <= jw <= 1.0): logging.warning(f"Jw={jw} is outside typical range (0.05-1.0).")
        # SRF can be high in squeezing/swelling ground
        if not (0.5 <= srf <= 400): logging.warning(f"SRF={srf} is outside broad typical range (0.5-400).")
        if span is not None and span <= 0: raise ValueError("Span must be positive if provided.")
        if esr is not None and esr <= 0: raise ValueError("ESR must be positive if provided.")

        # Avoid division by zero or negative inputs that would break the formula
        if jn <= 0 or ja <= 0 or srf <= 0:
            raise ValueError("Jn, Ja, and SRF must be positive.")

        # Calculate Q = (RQD/Jn) * (Jr/Ja) * (Jw/SRF)
        term1 = rqd_calc / jn
        term2 = jr / ja
        term3 = jw / srf
        q_value = term1 * term2 * term3

        # Provide support category context
        support_context = get_q_support_category(q_value, span if span else 10.0, esr if esr else 1.6) # Use defaults if not provided

        return {
            "q_value": q_value, # Keep high precision
            "rock_mass_quality": support_context["rock_mass_quality"],
            "support_category_info": support_context,
            "calculation_components": {
                "RQD_used": rqd_calc,
                "Block Size (RQD/Jn)": round(term1, 3),
                "Inter-block Shear Strength (Jr/Ja)": round(term2, 3),
                "Active Stress (Jw/SRF)": round(term3, 4) # Higher precision for potentially small number
            }
        }
    except ValueError as ve:
        logging.warning(f"Q-System calculation input error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error("Error calculating Q-System", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def estimate_tbm_performance(ucs: float, rqd: float, joint_spacing: float,
                             abrasivity_index: float, # Changed name for clarity (e.g., CAI)
                             tbm_diameter: float,
                             brittleness_index: Optional[float] = None, # Optional advanced input
                             thrust_per_cutter_kN: Optional[float] = None # Optional advanced input
                            ) -> Dict:
    """Estimate TBM performance parameters using simplified empirical models.
    Args:
        ucs: Uniaxial compressive strength in MPa.
        rqd: Rock Quality Designation as percentage.
        joint_spacing: Average discontinuity spacing in meters.
        abrasivity_index: Abrasivity measure (e.g., Cerchar Abrasivity Index - CAI). Specify which index if possible. Assumes CAI if unspecified.
        tbm_diameter: TBM diameter in meters.
        brittleness_index: Optional - Brittleness index (e.g., B_prime). Improves prediction.
        thrust_per_cutter_kN: Optional - Average thrust per cutter (kN). Improves prediction.
    Returns:
        Dictionary containing TBM performance estimates (penetration rate, advance rate, utilization, cutter life). Results are indicative.
    """
    logging.info(f"Estimating TBM Performance: UCS={ucs}MPa, RQD={rqd}%, Spacing={joint_spacing}m, CAI={abrasivity_index}, Dia={tbm_diameter}m")
    try:
        # Input validation
        if not all(p > 0 for p in [ucs, rqd, joint_spacing, abrasivity_index, tbm_diameter]):
             raise ValueError("Core input parameters (UCS, RQD, Spacing, Abrasivity, Diameter) must be positive.")

        # --- Simplified Penetration Rate (PR) Models ---
        # Example: Simple empirical relation (adjust coefficients based on specific models/data)
        # This is a placeholder - REAL models are more complex (e.g., CSM, NTNU)
        # Higher UCS -> lower PR; Higher RQD/Spacing -> sometimes higher PR (more intact blocks) or lower (less free breaks) - complex effect
        # Let's use a very basic conceptual formula for illustration:
        # Assume PR is inversely related to UCS and Abrasivity, positively to spacing (simplified)
        # Coefficients below are purely illustrative and lack empirical basis!
        base_pr_factor = 5.0 # Adjust this based on TBM type/power/rock type calibration
        pr_m_per_hour = base_pr_factor * (joint_spacing + 1)**0.3 / (ucs**0.6 * (abrasivity_index + 1)**0.4)
        pr_m_per_hour = max(0.1, min(pr_m_per_hour, 10.0)) # Apply realistic bounds

        # --- Utilization ---
        # Affected by geology, machine, logistics. Simple estimate based on RMR/Q or just typical values.
        # Lower RQD/Spacing -> more support -> lower utilization. Higher abrasivity -> more cutter changes -> lower util.
        # Very rough estimate:
        utilization_percent = 60.0 - (100 - rqd) * 0.1 - (abrasivity_index * 3)
        utilization_percent = max(20.0, min(utilization_percent, 75.0)) # Realistic bounds
        utilization_factor = utilization_percent / 100.0

        # --- Advance Rate (AR) ---
        advance_rate_m_per_day = pr_m_per_hour * utilization_factor * 24 # Hours per day

        # --- Cutter Life ---
        # Very complex. Depends on rock properties, cutter type, thrust, RPM, etc.
        # Simplified relation: Inversely related to UCS, Abrasivity.
        # Coefficients below are purely illustrative!
        base_cutter_life_factor = 5000 # Adjust based on cutter type/rock
        cutter_life_m3_per_cutter = base_cutter_life_factor / (ucs**0.8 * abrasivity_index**1.2)
        cutter_life_m3_per_cutter = max(10, cutter_life_m3_per_cutter) # Lower bound

        # Estimate cutter consumption (very rough)
        volume_per_meter = math.pi * (tbm_diameter / 2)**2 * 1.0 # m³ per meter advance
        approx_cutters_used_per_meter = volume_per_meter / cutter_life_m3_per_cutter if cutter_life_m3_per_cutter > 0 else float('inf')
        # cutter_life_hours = ?? Needs PR and cutter number - complex. Skip for now.

        # Add notes about limitations
        notes = [
            "Estimates are based on highly simplified empirical relationships.",
            "Actual performance depends heavily on specific TBM design, operation parameters (thrust, RPM), detailed geology, and site logistics.",
            "Penetration and cutter life models used here are illustrative and lack rigorous empirical calibration.",
            "Consult specialized TBM performance models (e.g., CSM, NTNU) and manufacturer data for reliable predictions."
        ]

        return {
            "notes": notes,
            "estimated_penetration_rate_m_hr": round(pr_m_per_hour, 2),
            "estimated_utilization_percent": round(utilization_percent, 1),
            "estimated_advance_rate_m_day": round(advance_rate_m_per_day, 1),
            "estimated_cutter_life_m3_rock_per_cutter": round(cutter_life_m3_per_cutter, 1),
            "estimated_cutters_consumed_per_meter": round(approx_cutters_used_per_meter, 3) if approx_cutters_used_per_meter != float('inf') else "N/A",
            "input_parameters_used": {
                "ucs_mpa": ucs, "rqd_pct": rqd, "joint_spacing_m": joint_spacing,
                "abrasivity_index": abrasivity_index, "tbm_diameter_m": tbm_diameter,
                 "brittleness_index": brittleness_index, "thrust_per_cutter_kN": thrust_per_cutter_kN
            }
        }
    except ValueError as ve:
         logging.warning(f"TBM Performance estimation input error: {ve}")
         return {"error": str(ve)}
    except Exception as e:
        logging.error("Error estimating TBM performance", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def analyze_face_stability(depth: float, diameter: float,
                           unit_weight: float, # Changed from density, common geotech unit kN/m³
                           cohesion: float, friction_angle: float,
                           water_table_depth: Optional[float] = None, # Depth below surface, optional
                           support_pressure: float = 0.0 # Applied support pressure (e.g., from TBM or air)
                          ) -> str: # Returns JSON string
    """Analyze tunnel face stability using simplified limit equilibrium methods.
    Args:
        depth: Tunnel depth to centerline in meters.
        diameter: Tunnel diameter in meters.
        unit_weight: Soil/Rock unit weight in kN/m³.
        cohesion: Soil/Rock cohesion in kPa.
        friction_angle: Soil/Rock friction angle in degrees.
        water_table_depth: Depth of water table from surface in meters. If None, assume dry conditions above tunnel. If 0, water at surface.
        support_pressure: Applied support pressure at the face in kPa (e.g., slurry, EPB, air). Default 0.
    Returns:
        JSON string containing stability analysis results (e.g., stability ratio, Factor of Safety). Note: Simplified analysis.
    """
    logging.info(f"Analyzing face stability: Depth={depth}m, Dia={diameter}m, UnitWeight={unit_weight}kN/m³, c={cohesion}kPa, phi={friction_angle}deg, WT_depth={water_table_depth}m, SupportP={support_pressure}kPa")
    try:
        # Input validation
        if not all(p >= 0 for p in [depth, diameter, unit_weight, cohesion, friction_angle, support_pressure]):
            raise ValueError("Input parameters cannot be negative.")
        if diameter == 0 or unit_weight == 0:
             raise ValueError("Diameter and Unit Weight must be positive.")
        if water_table_depth is not None and water_table_depth < 0:
             raise ValueError("Water table depth cannot be negative.")
        phi_rad = math.radians(friction_angle)

        # --- Calculate Stresses at Tunnel Axis ---
        sigma_v_total = depth * unit_weight # Total vertical stress (kPa)

        # Pore water pressure at axis
        pore_pressure = 0.0
        water_unit_weight = 9.81 # kN/m³
        if water_table_depth is not None:
            depth_below_wt = depth - water_table_depth
            if depth_below_wt > 0:
                pore_pressure = depth_below_wt * water_unit_weight

        sigma_v_effective = sigma_v_total - pore_pressure # Effective vertical stress

        # Estimate horizontal stress (using K0=1-sin(phi) as an example, could be input)
        k0_estimated = 1 - math.sin(phi_rad) if friction_angle < 90 else 0.5 # Handle phi=90 case
        sigma_h_effective = k0_estimated * sigma_v_effective
        sigma_h_total = sigma_h_effective + pore_pressure

        # --- Simplified Stability Analysis (e.g., Limit Equilibrium Wedge or similar) ---
        # Many methods exist (Anagnostou & Kovari, Leca & Dormieux, etc.)
        # Using a very basic concept: Resisting forces vs Driving forces

        # Driving Force Estimation (very simplified, relates to overburden removing support)
        # Often related to sigma_v_total or an active earth pressure concept.
        # Let's use total vertical stress as a simple proxy for driving stress P_driving ~ sigma_v_total
        driving_stress = sigma_v_total # kPa

        # Resisting Force Estimation (from soil strength and applied support)
        # Simplified model based on passive resistance concepts or bearing capacity
        # N_gamma term often related to unit weight, N_c term to cohesion
        # Very simplified form: P_resisting ~ c*Nc + 0.5*gamma*B*N_gamma + applied_support
        # Let's use a simplified form related to cohesion and friction, plus support pressure
        # This needs refinement based on a specific chosen model.
        # Example using a simple formula sometimes seen (needs verification for applicability):
        # Stability ratio N = (sigma_v_total - support_pressure) / (c' * cot(phi') + ...) - Needs better model.

        # Let's try a Factor of Safety approach: FS = Resisting / Driving
        # Resisting: Cohesion contribution + Frictional contribution + Applied Support
        # Driving: Overburden stress (potentially reduced by arching, ignored here)

        # Simplified resisting stress (e.g., based on wedge failure, ignoring 3D effects)
        # This requires a proper limit equilibrium model. Placeholder calculation:
        # Using cohesion and applied support directly as resisting components for simplicity.
        resisting_stress = cohesion * 5.0 + support_pressure # Very rough estimate, 5.0 is placeholder Nc factor

        # Factor of Safety
        fs = resisting_stress / driving_stress if driving_stress > 0 else float('inf')

        # Stability assessment
        status = "Likely Stable" if fs >= 1.5 else ("Marginally Stable" if fs >= 1.0 else "Likely Unstable")

        results = {
            "notes": "Highly simplified stability analysis. Uses placeholder calculations for resisting stress. Consult specialized geotechnical software or methods (e.g., FLAC, Plaxis, limit equilibrium programs) for design.",
            "inputs": {
                "depth_m": depth, "diameter_m": diameter, "unit_weight_kN_m3": unit_weight,
                "cohesion_kPa": cohesion, "friction_angle_deg": friction_angle,
                "water_table_depth_m": water_table_depth, "support_pressure_kPa": support_pressure
            },
            "calculated_stresses_kPa": {
                "total_vertical_stress_axis": round(sigma_v_total, 2),
                "pore_water_pressure_axis": round(pore_pressure, 2),
                "effective_vertical_stress_axis": round(sigma_v_effective, 2),
                "estimated_total_horizontal_stress_axis": round(sigma_h_total, 2),
            },
            "simplified_stability_assessment": {
                 "driving_stress_proxy_kPa": round(driving_stress, 2),
                 "resisting_stress_proxy_kPa": round(resisting_stress, 2),
                 "factor_of_safety_estimated": round(fs, 2) if fs != float('inf') else "Infinite",
                 "status": status
            }
        }
        return json.dumps(results, indent=2)

    except ValueError as ve:
        logging.warning(f"Face stability analysis input error: {ve}")
        return json.dumps({"error": str(ve)})
    except Exception as e:
        logging.error("Error analyzing face stability", exc_info=True)
        return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

@tool
def import_borehole_data(file_path: str) -> Dict:
    """Import and process borehole data from a CSV file.
    Args:
        file_path: Path to borehole data CSV file. Expects columns like 'depth', 'soil_type', 'N_value' (SPT), 'moisture_content'.
    Returns:
        Dictionary containing processed borehole data summary.
    """
    logging.info(f"Importing borehole data from: {file_path}")
    try:
        # Basic check for file existence (optional, pandas handles it too)
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        # --- Data Validation ---
        required_columns = ['depth', 'soil_type', 'N_value', 'moisture_content'] # Adjusted moisture column name
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in borehole data: {', '.join(missing_cols)}")

        # Check for essential data types (basic check)
        if not pd.api.types.is_numeric_dtype(df['depth']):
             raise TypeError("Column 'depth' must be numeric.")
        if not pd.api.types.is_numeric_dtype(df['N_value']):
             raise TypeError("Column 'N_value' must be numeric.")
        if not pd.api.types.is_numeric_dtype(df['moisture_content']):
             raise TypeError("Column 'moisture_content' must be numeric.")
        # Ensure depth is sorted (important for layer analysis)
        if not df['depth'].is_monotonic_increasing:
             logging.warning("Borehole data 'depth' column is not sorted. Sorting...")
             df = df.sort_values('depth').reset_index(drop=True)

        # --- Data Processing & Summary ---
        total_depth = df['depth'].max()
        unique_soil_types = df['soil_type'].unique().tolist()
        num_layers = len(unique_soil_types) # Simplistic, layers might repeat

        # Estimate groundwater depth (e.g., where moisture content significantly increases or noted)
        # This is heuristic. Actual GWL often noted separately in logs.
        # Let's assume high moisture (> saturated percentage, e.g., 30-40% depending on soil) indicates GWL
        potential_gw_df = df[df['moisture_content'] > 35.0] # Threshold is arbitrary
        ground_water_depth_estimated = potential_gw_df['depth'].min() if not potential_gw_df.empty else None

        # Calculate average N value (consider weighted average by layer thickness?)
        average_n_value = df['N_value'].mean() # Simple average

        # Identify soil profile layers (start and end depths)
        soil_profile = {}
        current_layer_type = None
        layer_start_depth = 0.0
        for index, row in df.iterrows():
            depth = row['depth']
            soil_type = row['soil_type']
            if current_layer_type is None: # First row
                current_layer_type = soil_type
                layer_start_depth = 0.0 # Assuming log starts at surface or first depth is start
                if df['depth'].iloc[0] != 0:
                     layer_start_depth = df['depth'].iloc[0] # Start at the first recorded depth if not 0

            elif soil_type != current_layer_type: # Change in soil type
                 # Record previous layer
                 if current_layer_type not in soil_profile:
                      soil_profile[current_layer_type] = []
                 soil_profile[current_layer_type].append({"start": layer_start_depth, "end": depth})
                 # Start new layer
                 current_layer_type = soil_type
                 layer_start_depth = depth

        # Add the last layer
        if current_layer_type is not None:
             if current_layer_type not in soil_profile:
                soil_profile[current_layer_type] = []
             soil_profile[current_layer_type].append({"start": layer_start_depth, "end": total_depth})


        return {
            "file_path": file_path,
            "total_depth_m": total_depth,
            "unique_soil_types_found": unique_soil_types,
            "number_of_distinct_soil_types": num_layers,
            "estimated_ground_water_depth_m": ground_water_depth_estimated if ground_water_depth_estimated is not None else "Not detected (based on moisture > 35%)",
            "average_spt_n_value": round(average_n_value, 1),
            "soil_profile_layers": soil_profile # More detailed profile
        }
    except FileNotFoundError as fnf:
         logging.error(f"Borehole data file not found: {fnf}")
         return {"error": str(fnf)}
    except (ValueError, TypeError) as ve:
        logging.error(f"Error processing borehole data CSV: {ve}")
        return {"error": f"Data validation error: {str(ve)}"}
    except Exception as e:
        logging.error(f"Unexpected error processing borehole data: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during processing: {str(e)}"}


# --- Placeholder for visualize_3d_results ---
# This requires significant data formatting. Let's make it accept simpler data for now or just note its complexity.
@tool
def visualize_3d_results(tunnel_alignment_coords: str, stability_data: str, geology_layers: Optional[str] = None) -> Dict:
    """Create a 3D visualization Plotly figure dictionary for tunnel alignment and stability.
    Args:
        tunnel_alignment_coords: JSON string of tunnel coordinates [[x1,y1,z1], [x2,y2,z2], ...].
        stability_data: JSON string of stability results per segment, e.g., [{'segment': i, 'factor_of_safety': fs}, ...]. Length must match number of segments in coords.
        geology_layers: Optional JSON string describing geological layers, e.g., [{'type': 'Sand', 'color': 'yellow', 'bounds': ...}]. Bounds are complex to define generally.
    Returns:
        Dict containing plot data (Plotly figure as dict) and basic statistics. Visualization of geology is highly simplified/conceptual.
    """
    logging.info("Generating 3D visualization data...")
    try:
        tunnel_path = json.loads(tunnel_alignment_coords)
        stability_results = json.loads(stability_data) # Assumes list of dicts with 'factor_of_safety'

        if not isinstance(tunnel_path, list) or not all(isinstance(p, list) and len(p) == 3 for p in tunnel_path):
            raise ValueError("tunnel_alignment_coords must be a list of [x, y, z] lists.")
        if not isinstance(stability_results, list) or not all('factor_of_safety' in item for item in stability_results):
             raise ValueError("stability_data must be a list of dictionaries containing 'factor_of_safety'.")
        # Check if stability data length matches tunnel segments (len(coords)-1) or points (len(coords))
        num_points = len(tunnel_path)
        num_segments = num_points - 1
        if len(stability_results) != num_points and len(stability_results) != num_segments:
            raise ValueError(f"Length mismatch: Tunnel has {num_points} points ({num_segments} segments), but stability data has {len(stability_results)} entries.")

        fig_dict = {'data': [], 'layout': {}} # Store as dict directly
        x, y, z = zip(*tunnel_path)

        # Tunnel alignment line
        fig_dict['data'].append({
            'type': 'scatter3d', 'x': x, 'y': y, 'z': z,
            'mode': 'lines', 'name': 'Tunnel Alignment',
            'line': {'color': 'blue', 'width': 4}
        })

        # Stability analysis markers (color-coded by FS)
        # Decide if FS applies to points or segments. Let's assume points for simplicity.
        # If it applies to segments, need to plot markers at segment midpoints.
        fs_values = [r['factor_of_safety'] for r in stability_results]
        if len(fs_values) == num_segments:
             # Calculate midpoints if FS applies to segments
             mid_x = [(x[i] + x[i+1])/2 for i in range(num_segments)]
             mid_y = [(y[i] + y[i+1])/2 for i in range(num_segments)]
             mid_z = [(z[i] + z[i+1])/2 for i in range(num_segments)]
             plot_x, plot_y, plot_z = mid_x, mid_y, mid_z
        else: # Assume applies to points
             plot_x, plot_y, plot_z = x, y, z

        fig_dict['data'].append({
            'type': 'scatter3d', 'x': plot_x, 'y': plot_y, 'z': plot_z,
            'mode': 'markers', 'name': 'Stability (FS)',
            'marker': {
                'size': 6, 'color': fs_values, 'colorscale': 'RdYlGn', # Red-Yellow-Green scale typical for FS
                'cmin': 0.5, 'cmax': 2.5, # Set color range bounds
                'showscale': True, 'colorbar': {'title': 'Factor of Safety'}
            }
        })

        # Add simplified geology (if provided) - This is very hard to do generally.
        # Placeholder: Just add text annotations or basic surfaces if format is known.
        if geology_layers:
            try:
                geology = json.loads(geology_layers)
                # Example: Add simple horizontal plane surfaces if bounds define them
                # This part needs significant work based on expected geology_layers format
                logging.warning("Geology layer visualization is conceptual and may not render correctly without specific format.")
                # for layer in geology:
                #    # Add Plotly trace for surface/mesh based on layer['bounds'] - COMPLEX
                #    pass
            except Exception as geo_e:
                logging.error(f"Could not process or visualize geology layers: {geo_e}")


        fig_dict['layout'] = {
            'title': '3D Tunnel Alignment and Stability',
            'scene': {
                'xaxis_title': 'X Coordinate',
                'yaxis_title': 'Y Coordinate',
                'zaxis_title': 'Z Coordinate (Elevation)',
                'aspectmode': 'data' # Ensure correct aspect ratio
            },
             'margin': {'l': 0, 'r': 0, 'b': 0, 't': 40} # Adjust margins
        }

        # Calculate basic stats
        tunnel_length = sum(math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2)
                             for i in range(1, num_points))
        depth_range = [min(z), max(z)] # Assuming Z is elevation/depth
        critical_indices = [i for i, fs in enumerate(fs_values) if isinstance(fs, (int, float)) and fs < 1.2] # Example threshold

        stats = {
            "tunnel_length_m": round(tunnel_length, 2),
            "elevation_range_m": [round(min(z), 2), round(max(z), 2)],
            "number_of_points": num_points,
            "number_of_segments": num_segments,
            "indices_with_low_fs (<1.2)": critical_indices
        }

        # Return the figure dictionary and stats
        # Plotly chart object cannot be directly returned if agent runs separately.
        # Return the dict representation. The app needs to use go.Figure(fig_dict).
        return {"plot_dict": fig_dict, "statistics": stats}

    except json.JSONDecodeError as jde:
        logging.error(f"JSON decoding error in visualize_3d_results: {jde}")
        return {"error": f"Invalid JSON input: {str(jde)}"}
    except ValueError as ve:
        logging.warning(f"Data validation error in visualize_3d_results: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error("Error generating 3D visualization", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- More Advanced/Specific Tools (Keep existing) ---
@tool
def calculate_tbm_penetration(alpha: float, fracture_spacing: float, peak_slope: float, csm_rop: float) -> Dict:
    """Calculate TBM Rate of Penetration using advanced formula (Hassanpour et al.).
    Args:
        alpha: Angle between tunnel axis and main weakness plane (degrees).
        fracture_spacing: Average spacing of main discontinuities (meters).
        peak_slope: Peak slope from punch tests (kN/mm).
        csm_rop: CSM model basic ROP (Rate of Penetration) in m/hr.
    Returns:
        Dictionary with calculated ROP. Note: Check applicability of the model.
    """
    logging.info(f"Calculating TBM Penetration (Hassanpour): alpha={alpha}, spacing={fracture_spacing}, peak_slope={peak_slope}, CSM_ROP={csm_rop}")
    try:
        # Basic validation
        if not (0 < alpha <= 90): raise ValueError("Alpha must be between 0 and 90 degrees.")
        if fracture_spacing <= 0: raise ValueError("Fracture spacing must be positive.")
        if peak_slope <= 0: raise ValueError("Peak slope must be positive.")
        if csm_rop <= 0: raise ValueError("CSM ROP must be positive.")

        # Rock Fabric Index (RFi) - Ensure log input is > 0
        rfi = 1.44 * math.log(alpha) - 0.0187 * fracture_spacing

        # Brittleness Index (BI)
        bi = 0.0157 * peak_slope

        # Penetration Rate (ROP) in m/hr
        rop_m_hr = 0.859 - rfi + bi + 0.0969 * csm_rop
        rop_m_hr = max(0, rop_m_hr) # Ensure non-negative ROP

        return {
            "method": "Hassanpour et al.",
            "calculated_rop_m_hr": round(rop_m_hr, 3),
            "intermediate_indices": {
                "Rock Fabric Index (RFi)": round(rfi, 3),
                "Brittleness Index (BI)": round(bi, 3)
            }
        }
    except ValueError as ve:
         logging.warning(f"TBM Penetration calculation input error: {ve}")
         return {"error": str(ve)}
    except Exception as e:
        logging.error("Error calculating TBM penetration (Hassanpour)", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def calculate_cutter_specs(max_cutting_speed_mps: float, cutter_diameter_m: float) -> Dict:
    """Calculate cutter head RPM based on maximum cutting speed at periphery and cutter diameter.
    Args:
        max_cutting_speed_mps: Maximum desired cutting speed at the cutterhead periphery (m/s).
        cutter_diameter_m: Diameter of the TBM cutterhead (m).
    Returns:
        Dictionary containing calculated RPM and input parameters.
    """
    logging.info(f"Calculating Cutter Specs: MaxSpeed={max_cutting_speed_mps} m/s, CutterheadDia={cutter_diameter_m} m")
    try:
        if max_cutting_speed_mps <= 0 or cutter_diameter_m <= 0:
            raise ValueError("Max cutting speed and cutterhead diameter must be positive.")

        # Circumference = pi * D
        circumference = math.pi * cutter_diameter_m # meters

        # Speed = Circumference * Revolutions per second
        # Revolutions per second = Speed / Circumference
        rps = max_cutting_speed_mps / circumference

        # Revolutions per minute = RPS * 60
        rpm = rps * 60

        return {
            "cutterhead_diameter_m": cutter_diameter_m,
            "max_peripheral_speed_mps": max_cutting_speed_mps,
            "calculated_cutterhead_rpm": round(rpm, 2)
        }
    except ValueError as ve:
         logging.warning(f"Cutter spec calculation input error: {ve}")
         return {"error": str(ve)}
    except Exception as e:
        logging.error("Error calculating cutter specs", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


@tool
def calculate_specific_energy(normal_force_kN: float, spacing_mm: float, penetration_mm_rev: float,
                              rolling_force_kN: float, tip_radius_mm: float, # Changed from angle
                              cutter_diameter_mm: float # Added cutter dia
                             ) -> Dict:
    """Calculate Specific Energy (SE) for disc cutters based on CSM model components.
    Args:
        normal_force_kN: Average normal force per cutter (kN).
        spacing_mm: Spacing between cutter grooves (mm).
        penetration_mm_rev: Penetration depth per revolution (mm).
        rolling_force_kN: Average rolling force per cutter (kN).
        tip_radius_mm: Cutter tip radius (width/2) in mm (e.g., 9.5mm for standard 19mm tip).
        cutter_diameter_mm: Diameter of the disc cutter (mm).
    Returns:
        Dictionary containing calculated Specific Energy (MJ/m³) and related metrics.
    """
    logging.info(f"Calculating Specific Energy: Fn={normal_force_kN}kN, S={spacing_mm}mm, P={penetration_mm_rev}mm/rev, Fr={rolling_force_kN}kN, TipRad={tip_radius_mm}mm, CutterDia={cutter_diameter_mm}mm")
    try:
        # Input validation
        if not all(p > 0 for p in [normal_force_kN, spacing_mm, penetration_mm_rev, tip_radius_mm, cutter_diameter_mm]):
             raise ValueError("Normal force, spacing, penetration, tip radius, and cutter diameter must be positive.")
        # Rolling force can theoretically be zero, but usually positive
        if rolling_force_kN < 0: raise ValueError("Rolling force cannot be negative.")

        # Convert forces to N and dimensions to meters for SE in J/m³
        fn_N = normal_force_kN * 1000
        fr_N = rolling_force_kN * 1000
        s_m = spacing_mm / 1000
        p_m = penetration_mm_rev / 1000
        tip_rad_m = tip_radius_mm / 1000
        cutter_dia_m = cutter_diameter_mm / 1000

        # Energy = Force * Distance. Work done per revolution per cutter.
        # Work primarily by rolling force over contact length. Contact length estimation is complex.
        # Alternative: SE = (Fn * P + Fr * L_contact) / (S * P * W_cutter) - needs L_contact
        # Simpler CSM approach: SE relates to forces and geometry.
        # SE = Energy / Volume = (F_rolling * 2*pi*R_cutterhead) / (Area * PR * Time) -> Complex

        # Specific Energy (empirical definition often used):
        # SE = Thrust * RPM / (Area * Advance Rate) - Needs machine params

        # Let's use the definition: Energy input per unit volume excavated.
        # Energy input per cutter per revolution ~ Rolling_Force * Contact_Arc_Length
        # Volume excavated per cutter per revolution ~ Spacing * Penetration * Cutter_Width (approx)

        # Using a simplified formula relating forces to SE (Check source/applicability):
        # SE (MJ/m³) = ( F_normal [kN] / (Spacing [mm] * Penetration [mm]) ) * ( CONSTANT? )
        # Or relating to cutting coefficient: SE = F_cutting / (Penetration * Width) - needs F_cutting

        # Let's use a formula from Balci et al. 2004 (check reference for context):
        # SE (MJ/m³) = (Fn[kN] * tan(theta/2) + Fr[kN]) / (S[mm] * p[mm]) * 1000
        # where theta is contact angle. Estimate theta/2 ~ penetration / cutter_radius
        cutter_radius_m = cutter_dia_m / 2
        # Ensure argument for asin is valid (-1 to 1)
        contact_angle_rad_approx = math.asin(min(1, max(-1, math.sqrt(p_m * cutter_dia_m - p_m**2) / cutter_radius_m))) if p_m < cutter_dia_m else math.pi / 2 # simplified geometry

        # Formula from original request (needs clarification on tip_angle, assuming it related to contact):
        # se = (normal_force / (spacing * penetration)) * (1 + (rolling_force/normal_force) * math.tan(tip_angle))
        # Let's reinterpret 'tip_angle' as related to the contact mechanics, perhaps half the contact angle.
        # Using the reinterpreted formula with forces in kN, dims in mm:
        # Need consistent units. Let's calculate Volume (m³) and Energy (MJ)
        volume_m3 = s_m * p_m * (tip_rad_m * 2) # Approx volume per rev per cutter (using tip width) - ROUGH
        # Energy per rev = Rolling Force * distance rolled. Distance ~ contact arc * (cutterhead_rpm / cutter_rpm?) - complex.
        # Use work done by forces: Energy ~ Fn*penetration + Fr*contact_length
        # Let's use the formula conceptually: Intensity * Factor
        intensity_kPa = fn_N / (s_m * p_m) # Normal force intensity (Pa)
        force_ratio_factor = (1 + (fr_N / fn_N) * math.tan(contact_angle_rad_approx / 2.0)) if fn_N > 0 else 1.0

        # This formula's units/derivation are unclear from the original code.
        # Let's try the Balci formula structure (MJ/m³)
        se_MJ_m3 = (fn_N * math.tan(contact_angle_rad_approx / 2.0) + fr_N) / (s_m * p_m) / 1e6 if (s_m * p_m) > 0 else float('inf')

        return {
            "calculation_method_note": "Based on Balci et al. (2004) structure, requires verification. Contact angle estimated geometrically.",
            "specific_energy_MJ_m3": round(se_MJ_m3, 3) if se_MJ_m3 != float('inf') else "Infinite (zero area)",
            "inputs": {
                "normal_force_kN": normal_force_kN, "spacing_mm": spacing_mm, "penetration_mm_rev": penetration_mm_rev,
                "rolling_force_kN": rolling_force_kN, "tip_radius_mm": tip_radius_mm, "cutter_diameter_mm": cutter_diameter_mm
            },
            "intermediate_calc": {
                 "estimated_contact_angle_deg": round(math.degrees(contact_angle_rad_approx), 1)
            }
        }
    except ValueError as ve:
         logging.warning(f"Specific Energy calculation input error: {ve}")
         return {"error": str(ve)}
    except Exception as e:
        # Catch potential math errors (e.g., log of non-positive, division by zero)
        logging.error("Error calculating specific energy", exc_info=True)
        return {"error": f"An unexpected error occurred during SE calculation: {str(e)}"}


@tool
def predict_cutter_life(ucs: float, penetration_mm_rev: float, rpm: float,
                        tbm_diameter_m: float, cai: float,
                        constants: Optional[Dict[str, float]] = None) -> Dict:
    """Predict cutter life (m³/cutter) using a generic empirical relationship structure.
    Args:
        ucs: Uniaxial compressive strength (MPa).
        penetration_mm_rev: Penetration per revolution (mm).
        rpm: Cutterhead revolution speed (rev/min).
        tbm_diameter_m: Tunnel Boring Machine diameter (m).
        cai: Cerchar Abrasivity Index.
        constants: Optional Dictionary of constants C1-C6 for the formula: CL = (C1 * UCS^C2) / (Pen^C3 * RPM^C4 * Dia^C5 * CAI^C6). Provide defaults if None.
    Returns:
        Dictionary containing predicted cutter life in m³ of rock excavated per cutter.
    """
    logging.info(f"Predicting Cutter Life: UCS={ucs}, Pen={penetration_mm_rev}, RPM={rpm}, Dia={tbm_diameter_m}, CAI={cai}")
    try:
        # Input validation
        if not all(p > 0 for p in [ucs, penetration_mm_rev, rpm, tbm_diameter_m, cai]):
             raise ValueError("UCS, Penetration, RPM, Diameter, and CAI must be positive.")

        # Default constants if not provided (These are placeholders - MUST be calibrated)
        if constants is None:
            constants = { # EXAMPLE VALUES ONLY - DO NOT USE FOR REAL PREDICTIONS
                'C1': 5000.0,  # Base life factor
                'C2': -0.5,   # UCS exponent (negative correlation)
                'C3': 0.3,    # Penetration exponent (positive correlation - higher pen = faster wear?) - Check sign convention
                'C4': 0.2,    # RPM exponent (positive correlation)
                'C5': 0.1,    # Diameter exponent (weak correlation?)
                'C6': 1.0     # CAI exponent (strong positive correlation)
            }
            logging.warning("Using placeholder default constants for cutter life prediction. Results are not calibrated.")
        required_const = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        if not all(k in constants for k in required_const):
             raise ValueError(f"Constants dictionary must contain keys: {', '.join(required_const)}")

        # Calculate Cutter Life (CL) in m³/cutter
        numerator = constants['C1'] * (ucs ** constants['C2'])
        denominator = (penetration_mm_rev ** constants['C3']) * \
                      (rpm ** constants['C4']) * \
                      (tbm_diameter_m ** constants['C5']) * \
                      (cai ** constants['C6'])

        if denominator == 0:
             cutter_life_m3 = float('inf') # Avoid division by zero
        else:
             cutter_life_m3 = numerator / denominator
             cutter_life_m3 = max(0, cutter_life_m3) # Ensure non-negative


        return {
            "prediction_method": "Generic Empirical Formula (Requires Calibration)",
            "predicted_cutter_life_m3_per_cutter": round(cutter_life_m3, 2) if cutter_life_m3 != float('inf') else "Infinite",
            "constants_used": constants,
            "inputs": {
                 "ucs_mpa": ucs, "penetration_mm_rev": penetration_mm_rev, "rpm": rpm,
                 "tbm_diameter_m": tbm_diameter_m, "cai": cai
            }
        }
    except ValueError as ve:
         logging.warning(f"Cutter life prediction input error: {ve}")
         return {"error": str(ve)}
    except Exception as e:
        logging.error("Error predicting cutter life", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}


# --- Agent Initialization ---
# @st.cache_resource # Caching models
def initialize_agents_and_models():
    """Loads models and initializes agents. Handles Hugging Face login."""
    logging.info("Initializing agents and models...")
    vlm_processor = None
    vlm_model = None
    text_model = None
    web_agent = None
    geotech_agent = None
    manager_agent = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        # --- Hugging Face Login ---
        hf_key = None
        try:
            hf_key = st.secrets.get("huggingface", {}).get("HUGGINGFACE_API_KEY")
        except Exception as e:
            logging.warning(f"Could not access Streamlit secrets: {e}")

        if not hf_key:
            hf_key = os.environ.get("HUGGINGFACE_API_KEY")

        if hf_key:
            try:
                login(token=hf_key, add_to_git_credential=False)
                logging.info("Hugging Face login successful.")
            except Exception as e:
                st.error(f"Hugging Face login failed: {e}. Check your API key.")
                logging.error(f"Hugging Face login failed: {e}", exc_info=True)
                # Proceed without login? Might fail later depending on models.
        else:
            st.warning("""
                **Hugging Face API key not found.**
                Provide it via Streamlit secrets (`secrets.toml`) or environment variable (`HUGGINGFACE_API_KEY`).
                Model loading might fail for gated models.
            """)
            logging.warning("Hugging Face API key not found. Proceeding without login.")

        # --- Load Text Model (Qwen) ---
        try:
            logging.info("Loading text model: Qwen/Qwen2.5-Coder-32B-Instruct")
            # Note: HfApiModel might handle model loading internally.
            # If direct loading needed: model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct", trust_remote_code=True).to(device)
            text_model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", device=device) # Pass device if supported
            logging.info("Text model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load text model (Qwen): {str(e)}")
            logging.error(f"Failed to load text model (Qwen): {e}", exc_info=True)
            # Cannot proceed without text model for agents

        # --- Load Vision Model (SmolVLM2) ---
        try:
            vlm_model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
            logging.info(f"Loading VLM processor: {vlm_model_id}")
            vlm_processor = AutoProcessor.from_pretrained(vlm_model_id, trust_remote_code=True)
            logging.info(f"Loading VLM model: {vlm_model_id}")
            vlm_model = AutoModelForCausalLM.from_pretrained(
                vlm_model_id,
                trust_remote_code=True,
                # torch_dtype=torch.bfloat16, # Optional: Use bfloat16 for faster inference if supported
                # low_cpu_mem_usage=True, # Optional: If memory constrained
            ).to(device)
            vlm_model.eval() # Set model to evaluation mode
            logging.info("VLM model and processor loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load vision model (SmolVLM2): {str(e)}. Document/image analysis will be unavailable.")
            logging.error(f"Failed to load VLM model/processor: {e}", exc_info=True)
            vlm_model = None # Ensure it's None if loading failed
            vlm_processor = None


        # --- Initialize Agents (Only if text_model loaded) ---
        if text_model:
            logging.info("Initializing agents...")
            # Web search agent
            web_agent = ToolCallingAgent(
                name="WebSearchAgent", # Give agents names
                description="Agent for searching the web and visiting webpages for information.",
                tools=[search_geotechnical_data, visit_webpage],
                model=text_model,
                max_steps=5 # Reduced steps for focused tasks
            )

            # Geotech calculation agent
            geotech_tools = [
                classify_soil, calculate_tunnel_support, calculate_rmr,
                calculate_q_system, estimate_tbm_performance, analyze_face_stability,
                import_borehole_data, visualize_3d_results, calculate_tbm_penetration,
                calculate_cutter_specs, calculate_specific_energy, predict_cutter_life
            ]
            geotech_agent = ToolCallingAgent(
                name="GeotechCalculationAgent",
                description="Agent specialized in performing geotechnical calculations and analyses.",
                tools=geotech_tools,
                model=text_model,
                max_steps=8 # Allow slightly more steps for complex calcs
            )

            # Manager agent
            manager_agent = CodeAgent(
                 # Removed search tool from manager, let web_agent handle it.
                 # Manager focuses on coordination and code execution if needed.
                name="ManagerAgent",
                description="Orchestrator agent that can coordinate tasks between other agents and execute Python code for complex logic or data manipulation.",
                model=text_model,
                # Allowed imports for potential data manipulation/plotting if needed
                additional_authorized_imports=["time", "numpy", "pandas", "math", "json", "plotly.graph_objects"]
            )
            logging.info("Agents initialized.")
        else:
             st.error("Text model failed to load. Agents cannot be initialized.")
             logging.error("Text model failed to load. Agents not initialized.")


        return text_model, web_agent, geotech_agent, manager_agent, vlm_processor, vlm_model, device

    except Exception as e:
        st.error(f"Critical error during initialization: {str(e)}")
        logging.error(f"Fatal error during initialization: {e}", exc_info=True)
        return None, None, None, None, None, None, None


# --- VLM Processing Functions ---

# @st.cache_data(max_entries=10) # Cache VLM results for same image/prompt
def run_vlm_inference(_vlm_processor, _vlm_model, image: Image.Image, prompt: str, device: str) -> Optional[str]:
    """Runs inference using the loaded VLM model and processor."""
    if not _vlm_processor or not _vlm_model:
        logging.warning("VLM model or processor not available for inference.")
        return "Error: Vision model not loaded."
    if image.mode == 'RGBA':
        image = image.convert('RGB') # Ensure 3 channels

    logging.info(f"Running VLM inference with prompt: '{prompt}'")
    try:
        # Prepare inputs using the processor
        messages = [{"role": "user", "content": f"{prompt}\n<image>"}]
        prompt_inputs = _vlm_processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _vlm_processor(text=prompt_inputs, images=image, return_tensors="pt").to(device)

        # Generate output
        with torch.no_grad(): # Disable gradient calculations for inference
            # Adjust generation parameters as needed
            output = _vlm_model.generate(
                **inputs,
                max_new_tokens=512, # Limit output length
                do_sample=True,   # Use sampling for potentially more creative descriptions
                temperature=0.6,
                top_p=0.9,
                # num_beams=3, # Use beam search for potentially higher quality (slower)
            )

        # Decode the output
        generated_text = _vlm_processor.batch_decode(output, skip_special_tokens=True)[0].strip()

        # Post-process: Extract the assistant's response part
        # The exact format depends on the model's chat template. Look for standard markers.
        assistant_response = generated_text
        user_prompt_marker = f"user\n{prompt}\n<image>\nassistant\n" # Common format
        if user_prompt_marker in assistant_response:
             assistant_response = assistant_response.split(user_prompt_marker, 1)[-1]
        # Fallback if exact marker isn't found - try simple split
        elif "assistant\n" in assistant_response:
             assistant_response = assistant_response.split("assistant\n", 1)[-1]


        logging.info(f"VLM generated description (cleaned): {assistant_response[:200]}...") # Log snippet
        return assistant_response

    except Exception as e:
        logging.error(f"Error during VLM inference: {e}", exc_info=True)
        st.error(f"Error during image analysis: {e}")
        return f"Error during image analysis: {str(e)}"


def process_uploaded_file(uploaded_file, vlm_processor, vlm_model, device):
    """Processes an uploaded image or PDF file, extracting features using VLM."""
    if not vlm_processor or not vlm_model:
         st.warning("Vision model not loaded. Cannot analyze file.")
         return None, None

    file_info = {"name": uploaded_file.name, "type": uploaded_file.type, "size_mb": round(uploaded_file.size / (1024*1024), 2)}
    logging.info(f"Processing uploaded file: {file_info['name']} ({file_info['type']})")
    st.session_state.vlm_description = None # Reset previous description
    st.session_state.vlm_search_results = None # Reset previous search results

    all_descriptions = []
    prompt = "Describe the key visual elements in this image, focusing on any geological, structural, site conditions, or engineering aspects visible. What can be inferred about the context?"

    try:
        if uploaded_file.type.startswith("image/"):
            logging.info("Processing as image file.")
            try:
                img = Image.open(uploaded_file)
                with st.spinner(f"Analyzing image: {file_info['name']}..."):
                    description = run_vlm_inference(vlm_processor, vlm_model, img, prompt, device)
                if description:
                    all_descriptions.append({"source": file_info['name'], "description": description})
                else:
                     st.warning(f"Could not generate description for image: {file_info['name']}")
            except Exception as img_e:
                st.error(f"Error opening or processing image {file_info['name']}: {img_e}")
                logging.error(f"Error opening/processing image {file_info['name']}: {img_e}", exc_info=True)

        elif uploaded_file.type == "application/pdf":
            logging.info("Processing as PDF file. Extracting images...")
            pdf_images_processed = 0
            try:
                # Read PDF content into memory
                pdf_bytes = uploaded_file.getvalue()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

                if pdf_doc.page_count == 0:
                    st.warning("PDF appears to be empty or corrupted.")
                    return file_info, "PDF contains no pages."

                st.info(f"Found {pdf_doc.page_count} pages in PDF. Extracting and analyzing images...")

                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc.load_page(page_num)
                    image_list = page.get_images(full=True) # Get full image info

                    if not image_list:
                        logging.info(f"No images found on page {page_num + 1}")
                        continue

                    logging.info(f"Found {len(image_list)} images on page {page_num + 1}")

                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        try:
                            img = Image.open(BytesIO(image_bytes))
                            img_filename = f"{file_info['name']} (Page {page_num + 1}, Img {img_index + 1}.{image_ext})"
                            with st.spinner(f"Analyzing image: {img_filename}..."):
                                 description = run_vlm_inference(vlm_processor, vlm_model, img, prompt, device)
                            if description:
                                all_descriptions.append({"source": img_filename, "description": description})
                                pdf_images_processed += 1
                            else:
                                st.warning(f"Could not generate description for image: {img_filename}")
                        except Exception as pdf_img_e:
                            st.error(f"Error processing image from PDF (Page {page_num + 1}, Img {img_index + 1}): {pdf_img_e}")
                            logging.error(f"Error processing image from PDF: {pdf_img_e}", exc_info=True)
                        # Optional: Limit number of images processed per PDF?
                        # if pdf_images_processed >= MAX_IMAGES_PER_PDF: break
                    # if pdf_images_processed >= MAX_IMAGES_PER_PDF: break

                pdf_doc.close()
                if pdf_images_processed == 0:
                     st.info("No images suitable for analysis were found or extracted from the PDF.")
                else:
                     st.success(f"Analyzed {pdf_images_processed} images from the PDF.")

            except fitz.fitz.FileDataError:
                st.error("Failed to open PDF. The file might be corrupted or password-protected.")
                logging.error("PyMuPDF FileDataError opening PDF.", exc_info=True)
            except Exception as pdf_e:
                st.error(f"An error occurred while processing the PDF: {pdf_e}")
                logging.error(f"Error processing PDF {file_info['name']}: {pdf_e}", exc_info=True)

        else:
            st.error(f"Unsupported file type: {uploaded_file.type}. Please upload an image (PNG, JPG) or a PDF.")
            return file_info, "Unsupported file type."

        # --- Combine Descriptions ---
        if not all_descriptions:
            final_description = "No visual features could be extracted or described from the uploaded file."
            logging.warning("No descriptions generated from file.")
        elif len(all_descriptions) == 1:
            final_description = all_descriptions[0]["description"]
        else:
            # Combine descriptions from multiple images (e.g., in a PDF)
            final_description = f"Analysis results from '{file_info['name']}':\n\n"
            for item in all_descriptions:
                final_description += f"--- Source: {item['source']} ---\n{item['description']}\n\n"
            # Optional: Ask LLM to summarize the combined descriptions?
            # summary_prompt = f"Summarize the key geotechnical and structural observations from the following descriptions extracted from a document:\n\n{final_description}"
            # summary = run_llm_inference(text_model, summary_prompt) # Needs text model inference function
            # final_description = summary if summary else final_description # Use summary if available

        st.session_state.vlm_description = final_description
        logging.info(f"Completed file processing. Description generated (length: {len(final_description)}).")
        return file_info, final_description

    except Exception as e:
        st.error(f"A critical error occurred during file processing: {e}")
        logging.error(f"Critical error processing file {file_info.get('name', 'N/A')}: {e}", exc_info=True)
        return file_info, f"Critical error: {str(e)}"


# --- Main Application Logic ---

# Initialize models and agents
# Use @st.cache_resource for the initialization function
@st.cache_resource(show_spinner="Initializing AI models and agents...")
def cached_initialize():
    return initialize_agents_and_models()

text_model, web_agent, geotech_agent, manager_agent, vlm_processor, vlm_model, device = cached_initialize()


# --- Chat Processing Function ---
def process_chat_request(request: str):
    """Handles user chat input, potentially using agents."""
    logging.info(f"Processing chat request: '{request}'")
    if not text_model or not web_agent or not geotech_agent or not manager_agent:
         # Check if initialization failed
         st.error("Agents are not initialized. Cannot process request.")
         logging.error("Attempted chat request processing, but agents are not initialized.")
         return "Error: The AI agents could not be initialized. Please check the logs or configuration."

    try:
        # Simple keyword check for direct answers (optional)
        if request.lower().strip() in ["what is ucs", "ucs", "ucs definition"]:
             logging.info("Providing direct answer for UCS definition.")
             return """UCS (Uniaxial Compressive Strength) is a fundamental geotechnical parameter measuring a rock or soil sample's maximum compressive strength when subjected to axial stress without lateral constraints. Expressed typically in MPa or kPa, it's a critical input for rock mass classification (like RMR), foundation design, tunnel design, slope stability, and excavation assessment."""

        # Determine intent - Is it a calculation, search, or general query?
        # This could be done with another LLM call, or simple heuristics.
        # For now, assume manager agent can delegate or handle.

        # Use the manager agent to handle the request, potentially delegating
        # to web_agent or geotech_agent.
        # Provide context if available (e.g., previous analysis results)
        context = {
            "previous_analysis": st.session_state.current_analysis,
            "chat_history": st.session_state.chat_history[-5:] # Last 5 interactions
        }

        logging.info("Running manager agent for the request...")
        # Note: manager_agent.run might expect specific input format or yield multiple steps. Adapt as needed.
        # Using `CodeAgent` directly might involve executing code based on the request.
        # If simple delegation is needed, `ToolCallingAgent` as manager might be better.
        # Let's assume the CodeAgent can decide to call tools or generate code/text.
        # We pass tools available to other agents for context, but let manager decide.

        # Simplified approach: Try geotech agent first if it looks like a calculation,
        # then web agent, then manager as fallback/coordinator.
        # Let's try letting the manager decide based on tools descriptions.

        # Re-instantiate manager with all tools for potential delegation lookup?
        # Or provide context about other agents' tools.
        # Let's assume manager primarily uses its own tools/code execution,
        # and complex delegation isn't built into smolagents' CodeAgent directly.

        # Fallback: Simple routing based on keywords (less robust)
        calc_keywords = ["calculate", "estimate", "analyze", "rmr", "q-system", "tbm", "stability", "support pressure", "classify soil"]
        search_keywords = ["search", "find", "what is", "information on", "lookup"]

        response = None
        if any(keyword in request.lower() for keyword in calc_keywords) and geotech_agent:
            logging.info("Request seems calculation-related. Trying Geotech Agent.")
            try:
                # Geotech agent expects 'task' argument
                response = geotech_agent(task=request)
                logging.info(f"Geotech Agent Response Type: {type(response)}")
                # Ensure response is stringifiable
                response = json.dumps(response, indent=2) if isinstance(response, (dict, list)) else str(response)
            except Exception as geo_e:
                logging.warning(f"Geotech agent failed: {geo_e}. Falling back.")
                response = None # Reset response to allow fallback

        if response is None and any(keyword in request.lower() for keyword in search_keywords) and web_agent:
             logging.info("Request seems search-related. Trying Web Agent.")
             try:
                 # Web agent expects 'task' argument
                 response = web_agent(task=request)
                 logging.info(f"Web Agent Response Type: {type(response)}")
                 response = str(response) # Ensure string
             except Exception as web_e:
                 logging.warning(f"Web agent failed: {web_e}. Falling back.")
                 response = None

        # Fallback to Manager Agent / direct Text Model call if others fail or don't match
        if response is None:
            logging.info("Falling back to Manager Agent / Direct LLM call.")
            try:
                 # Simple generation using the base text model if manager setup is complex
                 # For simplicity, let's use the HfApiModel directly for general queries
                 if isinstance(text_model, HfApiModel):
                     # Format prompt for chat model
                     formatted_prompt = f"User: {request}\nAssistant:" # Adjust based on model's expected format
                     # Direct call to model's generate method (may need specific parameters)
                     # This bypasses agent logic, direct generation.
                     # response = text_model.generate(formatted_prompt, max_length=512) # Example, syntax depends on HfApiModel implementation
                     # --- OR ---
                     # If manager agent is robust enough, use it:
                     manager_response_generator = manager_agent.run(goal=request, context=context)
                     final_response = ""
                     for step_output in manager_response_generator:
                          # Process step output - could be tool call result or text
                          if hasattr(step_output, 'content'):
                               final_response += step_output.content + "\n"
                          else:
                               final_response += str(step_output) + "\n"
                     response = final_response.strip() if final_response else "Manager agent did not produce a result."

                 else:
                      response = "Manager agent or direct LLM call is not configured correctly."

            except Exception as mgr_e:
                 logging.error(f"Manager agent / Direct LLM call failed: {mgr_e}", exc_info=True)
                 response = f"Error processing request: {str(mgr_e)}"


        # Final check for response content
        if not response:
             response = "I could not process your request using the available tools. Please try rephrasing."
             logging.warning("No response generated after trying agents/fallback.")

        # Limit response length
        max_resp_len = 4000
        if len(response) > max_resp_len:
             logging.warning(f"Response truncated ({len(response)} > {max_resp_len})")
             response = response[:max_resp_len] + "... (response truncated)"

        return response

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Error processing chat request '{request}': {e}", exc_info=True)
        # Provide specific traceback in logs but generic message in UI
        # print(traceback.format_exc()) # Print full traceback to console/log
        return f"An unexpected error occurred while processing your request. Please check the application logs."


# --- Streamlit UI Layout ---

# --- Sidebar ---
with st.sidebar:
    st.title("🏗️ Geotechnical AI")
    st.markdown("Qwen2.5 + SmolVLM2")
    st.markdown("---")

    # --- Document/Image Analysis Section ---
    st.header("📄 Document/Image Analysis")
    uploaded_file = st.file_uploader(
        "Upload PDF or Image (PNG, JPG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=False, # Process one file at a time for simplicity
        key="file_uploader"
    )

    # Use columns for buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_button_disabled = uploaded_file is None or vlm_model is None
        if st.button("📊 Analyze File", key="analyze_file", disabled=analyze_button_disabled, help="Analyze the uploaded file using the vision model." if not analyze_button_disabled else "Upload a file and ensure VLM model is loaded."):
            if uploaded_file is not None and vlm_processor and vlm_model:
                file_info, description = process_uploaded_file(uploaded_file, vlm_processor, vlm_model, device)
                st.session_state.uploaded_file_info = file_info # Store info about the analyzed file
                # Description is stored in st.session_state.vlm_description inside the function
                if description and "error" not in description.lower():
                    st.success("File analysis complete.")
                else:
                    st.error("File analysis failed or produced no description.")
            elif vlm_model is None:
                 st.error("Vision model is not loaded. Cannot analyze.")

    with col2:
        search_button_disabled = st.session_state.vlm_description is None or "error" in str(st.session_state.vlm_description).lower() or not web_agent
        if st.button("🌐 Search Web", key="search_web_vlm", disabled=search_button_disabled, help="Search the web based on the file analysis results." if not search_button_disabled else "Analyze a file first or check agent status."):
             if st.session_state.vlm_description and web_agent:
                 with st.spinner("Searching web based on analysis..."):
                     # Create a concise query from the description (or use full description)
                     # Option 1: Use full description
                     # query = st.session_state.vlm_description
                     # Option 2: Ask LLM to create a query (might be slow)
                     # query_prompt = f"Generate a concise web search query (max 10 words) based on the key findings in this text:\n\n{st.session_state.vlm_description[:1000]}"
                     # query = run_llm_inference(text_model, query_prompt) or "Geotechnical features from image"
                     # Option 3: Simple keyword extraction or use first sentence (simplest)
                     query = f"Information related to: {st.session_state.vlm_description[:150]}" # Use first part as query basis
                     logging.info(f"Using VLM description snippet for web search query: {query}")

                     try:
                        search_results = search_geotechnical_data(query) # Call the tool directly
                        st.session_state.vlm_search_results = search_results
                        st.success("Web search complete.")
                     except Exception as search_e:
                         st.error(f"Web search failed: {search_e}")
                         logging.error(f"Web search based on VLM failed: {search_e}", exc_info=True)
                         st.session_state.vlm_search_results = f"Error during web search: {str(search_e)}"
             elif not web_agent:
                  st.error("Web search agent not available.")


    # Display VLM Analysis results in sidebar
    if st.session_state.uploaded_file_info:
         with st.expander(f"Analysis: {st.session_state.uploaded_file_info['name']}", expanded=st.session_state.vlm_description is not None):
             if st.session_state.vlm_description:
                 st.markdown("**VLM Description:**")
                 st.markdown(st.session_state.vlm_description)
             else:
                 st.info("Analysis pending or failed.")

    if st.session_state.vlm_search_results:
        with st.expander(f"Web Search Results for {st.session_state.uploaded_file_info['name']}", expanded=True):
             st.markdown(st.session_state.vlm_search_results)

    st.markdown("---") # Separator


    # --- Manual Analysis Tools Section ---
    st.header("🔧 Manual Analysis")
    analysis_type = st.selectbox(
        "Select Analysis Tool",
        ["Soil Classification", "Tunnel Support", "RMR Analysis", "Q-System Analysis", "TBM Performance", "Face Stability"] # Added more options
    )

    with st.expander("Tool Parameters", expanded=True):
        params = {}
        # Use analysis_type to show relevant inputs
        if analysis_type == "Soil Classification":
            params = {
                "soil_type": st.selectbox("Soil Type", ["clay", "silt", "sand", "gravel"], key="sc_type"),
                "plasticity_index": st.number_input("Plasticity Index (PI)", 0.0, 100.0, 15.0, key="sc_pi"),
                "liquid_limit": st.number_input("Liquid Limit (LL)", 0.0, 150.0, 40.0, key="sc_ll")
            }
        elif analysis_type == "Tunnel Support":
            params = {
                "depth": st.number_input("Depth to Centerline (m)", 1.0, 1000.0, 50.0, key="ts_depth"),
                "soil_density": st.number_input("Soil/Rock Unit Weight (kN/m³)", 10.0, 30.0, 20.0, key="ts_density"),
                "k0": st.number_input("K₀ Coefficient (at-rest)", 0.1, 3.0, 0.8, step=0.1, key="ts_k0"),
                "tunnel_diameter": st.number_input("Tunnel Diameter (m)", 1.0, 25.0, 8.0, key="ts_dia")
            }
        elif analysis_type == "RMR Analysis":
             params = {
                "ucs": st.number_input("UCS (MPa)", 0.0, 500.0, 80.0, key="rmr_ucs"),
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 70.0, key="rmr_rqd"),
                "spacing": st.number_input("Discontinuity Spacing (m)", 0.01, 5.0, 0.5, format="%.2f", key="rmr_spacing"),
                "condition": st.slider("Discontinuity Condition Rating (R4)", 0, 30, 15, key="rmr_cond"),
                "groundwater": st.slider("Groundwater Condition Rating (R5)", 0, 15, 10, key="rmr_gw"),
                "orientation": st.slider("Orientation Adjustment Rating (R6)", -12, 0, -2, key="rmr_orient")
            }
        elif analysis_type == "Q-System Analysis":
             params = {
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 75.0, key="q_rqd"),
                "jn": st.number_input("Joint Set Number (Jn)", 0.5, 20.0, 9.0, step=0.5, format="%.1f", key="q_jn"),
                "jr": st.number_input("Joint Roughness Number (Jr)", 0.5, 4.0, 1.5, step=0.5, format="%.1f", key="q_jr"),
                "ja": st.number_input("Joint Alteration Number (Ja)", 0.75, 20.0, 4.0, step=0.25, format="%.2f", key="q_ja"),
                "jw": st.number_input("Joint Water Reduction Factor (Jw)", 0.05, 1.0, 1.0, step=0.05, format="%.2f", key="q_jw"),
                "srf": st.number_input("Stress Reduction Factor (SRF)", 0.5, 400.0, 2.5, step=0.5, format="%.1f", key="q_srf"),
                "span": st.number_input("Tunnel Span/Height (m, Optional)", 1.0, 50.0, 10.0, key="q_span"),
                "esr": st.number_input("Excavation Support Ratio (ESR, Optional)", 0.5, 5.0, 1.6, step=0.1, key="q_esr")
             }
        elif analysis_type == "TBM Performance":
             params = {
                "ucs": st.number_input("UCS (MPa)", 1.0, 500.0, 100.0, key="tbm_ucs"),
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 75.0, key="tbm_rqd"),
                "joint_spacing": st.number_input("Joint Spacing (m)", 0.01, 5.0, 0.6, format="%.2f", key="tbm_spacing"),
                "abrasivity_index": st.number_input("Abrasivity Index (e.g., CAI)", 0.0, 6.0, 2.5, step=0.1, format="%.1f", key="tbm_cai"),
                "tbm_diameter": st.number_input("TBM Diameter (m)", 1.0, 20.0, 6.0, key="tbm_dia")
                # Add optional inputs later if needed (brittleness, thrust)
            }
        elif analysis_type == "Face Stability":
             params = {
                "depth": st.number_input("Depth to Centerline (m)", 1.0, 1000.0, 20.0, key="fs_depth"),
                "diameter": st.number_input("Tunnel Diameter (m)", 1.0, 25.0, 10.0, key="fs_dia"),
                "unit_weight": st.number_input("Soil/Rock Unit Weight (kN/m³)", 10.0, 30.0, 19.0, key="fs_unitw"),
                "cohesion": st.number_input("Cohesion (c') (kPa)", 0.0, 200.0, 10.0, key="fs_coh"),
                "friction_angle": st.number_input("Friction Angle (phi') (deg)", 0.0, 50.0, 28.0, key="fs_phi"),
                "water_table_depth": st.number_input("Water Table Depth from Surface (m, enter high value like 999 if dry)", 0.0, 1000.0, 15.0, key="fs_wt"),
                "support_pressure": st.number_input("Applied Face Support Pressure (kPa)", 0.0, 500.0, 0.0, key="fs_supp")
            }
        st.session_state.analysis_params = params # Store current params

    # Run Analysis Button
    run_analysis_disabled = not geotech_agent # Disable if agent not loaded
    if st.button("🚀 Run Manual Analysis", key="run_manual_analysis", disabled=run_analysis_disabled):
        if geotech_agent:
            with st.spinner(f"Running {analysis_type}..."):
                try:
                    result = None
                    # Map analysis type to the correct tool function call
                    if analysis_type == "Soil Classification":
                        result = classify_soil(**st.session_state.analysis_params)
                    elif analysis_type == "Tunnel Support":
                        result = calculate_tunnel_support(**st.session_state.analysis_params)
                    elif analysis_type == "RMR Analysis":
                        result = calculate_rmr(**st.session_state.analysis_params)
                    elif analysis_type == "Q-System Analysis":
                         # Handle optional None for span/esr if they are 0 or defaults
                         q_params = st.session_state.analysis_params.copy()
                         q_params["span"] = q_params["span"] if q_params.get("span", 0) > 0 else None
                         q_params["esr"] = q_params["esr"] if q_params.get("esr", 0) > 0 else None
                         result = calculate_q_system(**q_params)
                    elif analysis_type == "TBM Performance":
                        result = estimate_tbm_performance(**st.session_state.analysis_params)
                    elif analysis_type == "Face Stability":
                         fs_params = st.session_state.analysis_params.copy()
                         # Handle optional water table depth
                         fs_params["water_table_depth"] = fs_params["water_table_depth"] if fs_params.get("water_table_depth", 999) < 999 else None
                         result = analyze_face_stability(**fs_params) # Returns JSON string

                    st.session_state.current_analysis = result
                    st.session_state.current_analysis_type = analysis_type # Store type for display logic
                    st.success(f"{analysis_type} complete.")
                    logging.info(f"Manual analysis result for {analysis_type}: {result}")

                except Exception as tool_e:
                    st.error(f"Error during {analysis_type}: {tool_e}")
                    logging.error(f"Error running tool {analysis_type}: {tool_e}", exc_info=True)
                    st.session_state.current_analysis = {"error": f"Failed to run tool: {str(tool_e)}"}
        else:
            st.error("Geotechnical calculation agent is not available.")


    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ by Kilic Intelligence")

# --- Main Content Area ---
st.title("🏗️ Geotechnical AI Agent")
st.markdown("Powered by Qwen2.5-Coder-32B & SmolVLM2-2.2B")

# --- Display Manual Analysis Results ---
st.subheader("📊 Manual Analysis Results")
if st.session_state.current_analysis:
    current_result = st.session_state.current_analysis
    analysis_title = st.session_state.get("current_analysis_type", "Analysis")

    with st.expander(f"Details: {analysis_title}", expanded=True):
        # If the result is a JSON string (like from analyze_face_stability), parse it
        if isinstance(current_result, str):
            try:
                current_result = json.loads(current_result)
                st.json(current_result)
            except json.JSONDecodeError:
                st.markdown("```\n" + current_result + "\n```") # Display as code block if not valid JSON
        else:
             st.json(current_result) # Display dictionary directly

        # --- Add specific visualizations for results ---
        try:
            if isinstance(current_result, dict) and "error" not in current_result:
                 if analysis_title == "Tunnel Support" and "vertical_stress_total_kPa" in current_result:
                      fig = go.Figure()
                      params = st.session_state.analysis_params
                      depth = params.get('depth', 0)
                      diameter = params.get('tunnel_diameter', 1)
                      vert_stress = current_result.get('vertical_stress_total_kPa', 0)
                      horz_stress = current_result.get('horizontal_stress_total_kPa', 0)
                      support_p = current_result.get('estimated_support_pressure_kPa', 0)

                      # Conceptual plot - show stresses around tunnel outline
                      fig.add_shape(type="circle", xref="x", yref="y",
                                    x0=-diameter/2, y0=depth-diameter/2,
                                    x1=diameter/2, y1=depth+diameter/2,
                                    line_color="blue", name="Tunnel")
                      # Add arrows for stress (conceptual)
                      fig.add_annotation(x=0, y=depth-diameter*0.7, text=f"σv={vert_stress:.1f} kPa", showarrow=True, arrowhead=2, ax=0, ay=-30)
                      fig.add_annotation(x=-diameter*0.7, y=depth, text=f"σh={horz_stress:.1f} kPa", showarrow=True, arrowhead=2, ay=0, ax=30)
                      fig.add_annotation(x=diameter*0.3, y=depth+diameter*0.3, text=f"Support P≈{support_p:.1f} kPa", showarrow=True, arrowhead=5, ax=-40, ay=-40, bordercolor="red", borderwidth=1)


                      fig.update_layout(
                          title="Tunnel Stresses (Conceptual)",
                          xaxis_title="Horizontal Distance (m)", yaxis_title="Depth (m)",
                          yaxis_range=[depth + diameter, max(0, depth - diameter*1.5)], # Reversed y-axis (depth increases downwards)
                          xaxis_range=[-diameter, diameter],
                          width=600, height=400
                      )
                      st.plotly_chart(fig, use_container_width=True)

                 elif analysis_title == "RMR Analysis" and "component_ratings" in current_result:
                      ratings = current_result["component_ratings"]
                      labels = list(ratings.keys())
                      values = list(ratings.values())
                      fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color='skyblue')])
                      fig.update_layout(
                          title=f"RMR Component Ratings (Total RMR = {current_result.get('rmr_adjusted (RMR_89)', 'N/A')})",
                          xaxis_title="RMR Parameter", yaxis_title="Rating Value",
                          yaxis_range=[min(values)-5, max(values)+5] # Adjust range
                      )
                      st.plotly_chart(fig, use_container_width=True)

                 elif analysis_title == "Q-System Analysis" and "calculation_components" in current_result:
                     comps = current_result["calculation_components"]
                     labels = list(comps.keys())
                     values = [abs(v) for v in comps.values()] # Use absolute for log scale
                     texts = [f"{v:.3f}" for v in comps.values()] # Display actual value

                     fig = go.Figure(data=[go.Bar(x=labels, y=values, text=texts, textposition='auto', marker_color='lightgreen')])
                     fig.update_layout(
                         title=f"Q-System Components (Q = {current_result.get('q_value', 'N/A'):.3f})",
                         xaxis_title="Component", yaxis_title="Value (Log Scale)",
                         yaxis_type="log", # Use log scale for wide range
                         yaxis_tickformat=".2e"
                     )
                     st.plotly_chart(fig, use_container_width=True)

                 elif analysis_title == "TBM Performance" and "estimated_penetration_rate_m_hr" in current_result:
                     labels = ['Penetration (m/hr)', 'Advance (m/day)', 'Utilization (%)']
                     values = [
                         current_result.get('estimated_penetration_rate_m_hr', 0),
                         current_result.get('estimated_advance_rate_m_day', 0),
                         current_result.get('estimated_utilization_percent', 0)
                     ]
                     fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color='coral')])
                     fig.update_layout(title="Estimated TBM Performance Metrics", yaxis_title="Value")
                     st.plotly_chart(fig, use_container_width=True)

                 elif analysis_title == "Face Stability" and "simplified_stability_assessment" in current_result:
                      stab_assess = current_result["simplified_stability_assessment"]
                      fs = stab_assess.get("factor_of_safety_estimated", 0)
                      status = stab_assess.get("status", "Unknown")
                      if isinstance(fs, str): fs = 1.0 # Handle 'Infinite' case for plotting

                      fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = fs,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Estimated Face Stability FS ({status})"},
                            gauge = {
                                'axis': {'range': [0, 3], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps' : [
                                    {'range': [0, 1.0], 'color': 'red'},
                                    {'range': [1.0, 1.5], 'color': 'yellow'},
                                    {'range': [1.5, 3], 'color': 'lightgreen'}],
                                'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 1.5} # Target FS
                            }))
                      fig.update_layout(height=300, margin={'t':50, 'b':10, 'l':10, 'r':10})
                      st.plotly_chart(fig, use_container_width=True)

        except Exception as plot_e:
            st.warning(f"Could not generate plot for {analysis_title}: {plot_e}")
            logging.warning(f"Plotting error for {analysis_title}: {plot_e}", exc_info=True)

else:
    st.info("Run a manual analysis using the tools in the sidebar to see results here.")

st.markdown("---")

# --- Chat Interface Section ---
st.subheader("💬 Chat with Geotechnical AI")

# Chat message display function
def display_chat_message(msg):
    role_icon = "🧑" if msg["role"] == "user" else "🤖"
    try:
        content = msg["content"]
        # Try to pretty-print if content is a JSON string
        try:
            parsed_json = json.loads(content)
            st.markdown(f"{role_icon} **{msg['role'].title()}:**")
            st.json(parsed_json)
        except (json.JSONDecodeError, TypeError):
            # If not valid JSON or not a string, display as markdown
             if isinstance(content, (dict, list)): # Handle dict/list directly if not json string
                  st.markdown(f"{role_icon} **{msg['role'].title()}:**")
                  st.json(content)
             else: # Display as markdown for strings or other types
                  st.markdown(f"{role_icon} **{msg['role'].title()}:**\n\n{str(content)}")

    except Exception as e:
        st.error(f"Error displaying message: {str(e)}")
        logging.error(f"Error displaying chat message: {e}", exc_info=True)


# Display chat history
chat_container = st.container()
with chat_container:
    # Use st.session_state.chat_history directly
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = DEFAULT_CHAT_HISTORY # Initialize if empty

    for msg in st.session_state.chat_history:
        display_chat_message(msg)


# User input at the bottom
user_input = st.chat_input("Ask a geotechnical question or give a command...")

if user_input:
    # Add user message to history and display it immediately
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with chat_container: # Redisplay user message in the container
         display_chat_message({"role": "user", "content": user_input})

    # Process request and get assistant response
    with st.spinner("Thinking..."):
        assistant_response = process_chat_request(user_input)

    # Add assistant message to history
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Rerun the script to update the chat display with the new assistant message
    st.rerun()
