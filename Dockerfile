FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data and models
RUN mkdir -p data/train data/val data/test models checkpoints metrics

# Set environment variables
ENV PYTHONPATH=/app

# Create a non-root user
RUN useradd -m -u 1000 appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for API
EXPOSE 8000

# Create entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command (can be overridden)
CMD ["api"]