FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY server.py .
COPY interactive_rag_3d.py .
COPY static/ ./static/

# Create a user to run the application (security best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
# 5000 for Three.js version
# 8050 for Dash version
EXPOSE 5000 8050

# Default command (can be overridden in docker-compose)
CMD ["python", "server.py"]
