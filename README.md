# Geotechnical Engineering Multi-Agent System

This Streamlit application implements a multi-agent system for geotechnical engineering, leveraging Hugging Face's smolagents and the Qwen2.5-VL-72B-Instruct vision-language model to extract geotechnical parameters from unstructured datasets.

## Features

- **Data Extraction**: Extract geotechnical parameters from PDFs and images
- **Data Analysis**: Generate correlation panels to understand parameter relationships
- **Visualization**: Create interactive Plotly visualizations of geotechnical data
- **Web Search**: Search for additional geotechnical information

## Deployment on Streamlit Cloud

### Prerequisites

- A Streamlit Cloud account
- A Hugging Face account with API access to the Qwen2.5-VL-72B-Instruct model

### Deployment Steps

1. Fork or clone this repository to your GitHub account
2. Log in to [Streamlit Cloud](https://share.streamlit.io/)
3. Click on "New app" and select your repository
4. Configure the app settings:
   - Main file path: `app.py`
   - Python version: 3.9 or higher
5. Add your Hugging Face API token as a secret:
   - Go to "Advanced Settings" > "Secrets"
   - Add a secret with the key `HF_TOKEN` and your token as the value

### Local Development

To run the app locally:

1. Clone the repository
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Hugging Face API token:
   ```
   HF_TOKEN=your_token_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## System Requirements

- Python 3.9 or higher
- 4GB+ RAM recommended
- Internet connection for API access to Hugging Face models

## Dependencies

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
