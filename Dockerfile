FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Install poppler-utils for pdf2image
RUN apt-get update && apt-get install -y poppler-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies with enhanced cleanup
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed blinker -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip/*

# Clear cache before downloading the model
RUN rm -rf /root/.cache/huggingface/hub/* && \
    mkdir -p /root/.cache/huggingface/hub/

# Copy the handler code
COPY handler.py .

# Set the entrypoint to execute the handler on container start
ENTRYPOINT ["python", "-u", "handler.py"]
