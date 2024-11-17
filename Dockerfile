FROM continuumio/miniconda3:latest AS builder

# Copy environment file
COPY environment.yml .

# Create conda environment and clean up
RUN conda env create -f environment.yml && conda clean -afy

# Activate the environment and save it as the base Python environment
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > /etc/profile.d/conda.sh

# Target image
FROM continuumio/miniconda3:latest

# Copy the conda environment from builder
COPY --from=builder /opt/conda/envs/ /opt/conda/envs/

# Activate the environment in every shell
ENV PATH /opt/conda/envs/mlops_env/bin:$PATH

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data models checkpoints metrics

# Set environment variables
ENV PYTHONPATH=/app:/app/src:$PYTHONPATH

# Expose port for API
EXPOSE 8000

# Copy entrypoint script and set executable permissions
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["api"]
