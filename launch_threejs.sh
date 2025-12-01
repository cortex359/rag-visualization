#!/usr/bin/env bash
# Launch script for Three.js RAG Visualizer

echo "======================================"
echo "  RAG 3D Visualizer (Three.js)"
echo "======================================"
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting Flask backend server..."
echo "Once loaded, open your browser to: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Pass all arguments to Python script (supports --method umap or --method pca)
python server.py "$@"
