FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working dir
WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy source
COPY src /app/src
COPY models /app/models

# Expose API port
EXPOSE 8000

# Default command runs inference server (uvicorn)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
