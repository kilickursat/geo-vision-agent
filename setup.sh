#!/bin/bash

# Make setup.sh executable
chmod +x setup.sh

# Create .streamlit directory if it doesn't exist
mkdir -p ~/.streamlit

# Configure Streamlit settings
echo "\
[general]
email = \"\"

[server]
headless = true
enableCORS = false
port = $PORT
" > ~/.streamlit/config.toml

# Create secrets.toml file structure for storing sensitive information
echo "\
# This file is used for storing sensitive information
# Do not commit this file to version control
# Use Streamlit Sharing's secrets management instead

[huggingface]
api_token = \"\"
" > ~/.streamlit/secrets.toml

echo "Setup completed successfully."
