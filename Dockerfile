FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV and PaddleOCR
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app/ ./app/
COPY config/ ./config/
COPY main.py .

# Models volume mount point
RUN mkdir -p models

EXPOSE 8000

CMD ["python", "main.py"]