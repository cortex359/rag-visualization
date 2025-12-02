# Docker Deployment Guide

This guide explains how to run the RAG Visualizer using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose 2.0 or later

## Quick Start

### Option 1: Three.js Version (Recommended)

Launch the Three.js visualizer with UMAP:

```bash
docker-compose up threejs-umap
```

Access at: **http://localhost:5000**

### Option 2: Three.js Version with PCA

```bash
docker-compose --profile pca up threejs-pca
```

Access at: **http://localhost:5001**

### Option 3: Dash Version

```bash
docker-compose --profile dash up dash-umap
```

Access at: **http://localhost:8050**

### Option 4: Dash Version with PCA

```bash
docker-compose --profile dash-pca up dash-pca
```

Access at: **http://localhost:8051**

## Running Multiple Services

Run both Three.js versions simultaneously:

```bash
docker-compose up threejs-umap
docker-compose --profile pca up threejs-pca
```

Or run everything:

```bash
docker-compose --profile pca --profile dash --profile dash-pca up
```

## Build Options

### Build from scratch

```bash
docker-compose build
```

### Build without cache

```bash
docker-compose build --no-cache
```

### Pull and rebuild

```bash
docker-compose pull
docker-compose up --build
```

## Running in Background

Run as daemon (detached mode):

```bash
docker-compose up -d threejs-umap
```

Check logs:

```bash
docker-compose logs -f threejs-umap
```

Stop services:

```bash
docker-compose down
```

## Service Overview

| Service | Port | Description | Profile |
|---------|------|-------------|---------|
| `threejs-umap` | 5000 | Three.js + UMAP | Default |
| `threejs-pca` | 5001 | Three.js + PCA | `pca` |
| `dash-umap` | 8050 | Dash + UMAP | `dash` |
| `dash-pca` | 8051 | Dash + PCA | `dash-pca` |

## Docker Image Details

### Base Image
- Python 3.11 slim (minimal footprint)

### Installed Packages
- sentence-transformers
- Flask + flask-cors (Three.js backend)
- Dash + dash-bootstrap-components (Dash UI)
- UMAP + scikit-learn (dimensionality reduction)
- NumPy (numerical computing)

### Size
- Built image: ~2.5 GB (includes ML models)
- Base image: ~150 MB
- ML models downloaded at first run

### Security
- Non-root user (`appuser`)
- Minimal attack surface
- No unnecessary packages

## Health Checks

All services include health checks:
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Start period**: 40 seconds (allows time for model loading)

Check health status:

```bash
docker-compose ps
```

## Troubleshooting

### Container fails to start

Check logs:
```bash
docker-compose logs threejs-umap
```

### Out of memory

Increase Docker memory limit in Docker Desktop settings or:
```bash
docker-compose up --memory="4g" threejs-umap
```

### Port already in use

Change port mapping in `docker-compose.yml`:
```yaml
ports:
  - "5555:5000"  # Map host port 5555 to container port 5000
```

### Models not downloading

Ensure internet connectivity. Models are downloaded on first run:
- `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)

### Slow initialization

First run is slower due to:
1. Model download
2. Embedding generation (79 chunks)
3. UMAP/PCA dimensionality reduction

Subsequent runs use cached models and are faster.

## Production Deployment

### Environment Variables

Add to `docker-compose.yml`:

```yaml
environment:
  - FLASK_ENV=production
  - WORKERS=4
  - THREADS=2
```

### Use Production WSGI Server

Modify command in `docker-compose.yml`:

```yaml
command: gunicorn --bind 0.0.0.0:5000 --workers 4 server:app
```

Add `gunicorn` to `requirements.txt`.

### Reverse Proxy

Use Nginx or Traefik as reverse proxy:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - threejs-umap
```

### SSL/TLS

Add certificates and configure HTTPS in reverse proxy.

### Resource Limits

Add resource constraints:

```yaml
services:
  threejs-umap:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Cleanup

Remove containers:
```bash
docker-compose down
```

Remove containers and images:
```bash
docker-compose down --rmi all
```

Remove everything including volumes:
```bash
docker-compose down --rmi all --volumes
```

## Custom Dockerfile

To customize, modify `Dockerfile`:

```dockerfile
# Example: Add custom Python packages
RUN pip install custom-package

# Example: Copy custom data
COPY my_documents.json /app/data/
```

Rebuild:
```bash
docker-compose build
```

## Networking

Services can communicate via Docker network:

```yaml
networks:
  rag-network:
    driver: bridge
```

Add to each service:
```yaml
networks:
  - rag-network
```

## Volumes (Optional)

Persist data across container restarts:

```yaml
volumes:
  model-cache:

services:
  threejs-umap:
    volumes:
      - model-cache:/home/appuser/.cache/torch
```

## Development Mode

For development, mount source code:

```yaml
services:
  threejs-umap:
    volumes:
      - ./static:/app/static
      - ./server.py:/app/server.py
    command: python server.py --method umap --port 5000
```

Changes to files will be reflected without rebuilding.

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build image
        run: docker-compose build
      - name: Push to registry
        run: docker-compose push
```

## Support

For issues:
1. Check logs: `docker-compose logs -f [service]`
2. Verify health: `docker-compose ps`
3. Inspect container: `docker exec -it [container] bash`
4. Review GitHub issues: [project-url]

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Three.js Documentation](https://threejs.org/docs/)
- [Dash Documentation](https://dash.plotly.com/)
