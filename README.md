# Geotechnical Engineering Multi-Agent System

A comprehensive AI-powered application for geotechnical engineers, combining advanced language models, multi-agent architecture, and specialized tools to extract, analyze, and visualize geotechnical data from unstructured sources.

![Geotechnical Engineering](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/TBM_cutterhead.jpg/320px-TBM_cutterhead.jpg)

## 🚀 Features

- **Multi-Agent System**: Leverages three specialized AI agents powered by Qwen2.5-Coder-32B-Instruct:
  - **Web Agent**: Searches for and retrieves geotechnical information from online sources
  - **Geotechnical Agent**: Performs specialized calculations and engineering analyses
  - **Manager Agent**: Coordinates tasks and synthesizes results from other agents

- **PDF Analysis**: Extract geotechnical parameters from PDFs using ColPali vision-language model
  - Processes engineering drawings, borehole logs, and technical reports
  - Provides semantic search within documents
  - Highlights relevant sections based on query

- **Calculation Tools**:
  - Soil classification (USCS)
  - Tunnel support pressure calculations
  - Rock Mass Rating (RMR) analysis
  - Q-system rock quality assessment
  - TBM performance estimation
  - Tunnel face stability analysis
  - Cutter head specifications and life prediction

- **Visualization**:
  - Interactive Plotly visualizations of geotechnical data
  - 3D modeling of tunnel alignments with geological layers
  - Component rating charts for rock classification systems

- **Chat Interface**: Natural language interaction for engineering queries and analysis

## 🛠️ Architecture

The system employs a multi-agent architecture:

```
┌─────────────────────────────────────────┐
│            Streamlit Frontend           │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│           Manager Agent                 │
│    (Qwen2.5-Coder-32B-Instruct)         │
└───┬───────────────────┬─────────────────┘
    │                   │
┌───▼───────────┐   ┌───▼───────────────┐
│  Web Agent    │   │  Geotechnical     │
│  (Search,     │   │  Agent            │
│  Retrieval)   │   │  (Calculations)   │
└───────────────┘   └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │ RunPod Serverless │
                    │ PDF Analysis      │
                    │ (ColPali VLM)     │
                    └───────────────────┘
```

## 📋 Requirements

### Base Requirements
- Python 3.9+ (3.10 recommended)
- 16GB RAM minimum for local deployment (32GB+ recommended)
- CUDA-compatible GPU with 16GB+ VRAM for full model functionality

### API Access Requirements
- Hugging Face API access to Qwen2.5-Coder-32B-Instruct model
- RunPod account for PDF analysis with ColPali VLM (optional but recommended)

## 🔧 Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/geotechnical-ai-system.git
   cd geotechnical-ai-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install system packages for PDF processing:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install poppler-utils
   
   # macOS
   brew install poppler
   
   # Windows
   # Download and install poppler from https://github.com/oschwartz10612/poppler-windows/releases
   ```

5. Create a `.streamlit/secrets.toml` file:
   ```toml
   [huggingface]
   HUGGINGFACE_API_KEY = "your_huggingface_api_key"

   [runpod]
   endpoint_url = "your_runpod_endpoint_url"
   api_key = "your_runpod_api_key"
   ```

6. Run the application:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t geotechnical-ai-system .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -e HUGGINGFACE_API_KEY=your_api_key geotechnical-ai-system
   ```

### Streamlit Cloud Deployment

1. Fork or clone this repository to your GitHub account
2. Log in to [Streamlit Cloud](https://share.streamlit.io/)
3. Click on "New app" and select your repository
4. Configure the app settings:
   - Main file path: `app.py`
   - Python version: 3.10
5. Add secrets in the Streamlit Cloud dashboard:
   - Go to "Advanced Settings" > "Secrets"
   - Add your Hugging Face API key and RunPod configuration

## 📊 Usage Examples

### PDF Analysis

Upload geotechnical reports, borehole logs, or engineering drawings to extract parameters:

1. Navigate to the "PDF Analysis" tab
2. Upload a PDF document using the file uploader
3. Enter a query like "What is the soil composition at 15m depth?"
4. View extracted information with highlighted relevant sections

### Geotechnical Calculations

Perform engineering calculations with the specialized tools:

1. Select an analysis type from the sidebar (Soil Classification, Tunnel Support, etc.)
2. Input required parameters
3. Click "Run Analysis" to view results and visualizations

### Chat Interface

Interact with the system using natural language:

- Ask general questions: "What factors affect TBM performance in hard rock?"
- Request calculations: "Calculate the support pressure for a 6m diameter tunnel at 100m depth"
- Analyze specific scenarios: "What would be the RMR value for granite with 75% RQD?"

## 🔍 Advanced Features

### Custom TBM Performance Prediction

The system can estimate Tunnel Boring Machine (TBM) performance based on multiple geological parameters:

```python
tbm_results = estimate_tbm_performance(
    ucs=120,             # Uniaxial compressive strength (MPa)
    rqd=65,              # Rock quality designation (%)
    joint_spacing=0.3,   # Average joint spacing (m)
    abrasivity=2.5,      # Cerchar abrasivity index
    diameter=8.2         # TBM diameter (m)
)
```

### 3D Visualization of Geological Data

Create interactive 3D models of tunnel paths with geological layers:

1. Import tunnel coordinates and geology data
2. Run the `visualize_3d_results` tool
3. Interact with the 3D model to identify critical sections

### Integrated Web Search

The system can search for geotechnical information online:

1. Ask a question about an unfamiliar term or concept
2. The Web Agent will search and retrieve relevant information
3. Results are integrated with local calculations and analysis

## 🛠️ RunPod Serverless Configuration

The system uses RunPod for serverless processing of PDF documents:

1. Create a RunPod account at [runpod.io](https://www.runpod.io/)
2. Create a new serverless endpoint using the Dockerfile in this repository
3. Configure GPU preferences in `runpod.toml` (default: RTX 4090)
4. Add your RunPod endpoint URL and API key to `.streamlit/secrets.toml`

## ❓ Troubleshooting

### PDF Analysis Issues

- **Error: "Failed to access RunPod secrets"**  
  Ensure your RunPod credentials are correctly configured in `.streamlit/secrets.toml`

- **Error: "PDF analysis timed out"**  
  Increase the timeout value in `runpod.toml` for large documents

### Model Loading Issues

- **Error: "Failed to initialize agents"**  
  Check your Hugging Face API key and ensure you have access to the Qwen2.5-Coder-32B-Instruct model

- **Warning: "Hugging Face API key not found"**  
  Add your API key to `.streamlit/secrets.toml` or as an environment variable

### Performance Optimization

- For large PDFs, consider enabling caching:
  ```python
  @st.cache_data
  def process_large_pdf(pdf_bytes, query):
      # Processing logic here
  ```

- For slow calculations, use parallel processing:
  ```python
  import concurrent.futures
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results = executor.map(calculate_function, parameter_list)
  ```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Qwen2.5 model
- [RunPod](https://www.runpod.io/) for serverless GPU infrastructure
- [Streamlit](https://streamlit.io/) for the web application framework
- [ColPali](https://github.com/vidore/colpali) for the vision-language model used in PDF analysis
