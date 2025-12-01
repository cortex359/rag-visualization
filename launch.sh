#!/bin/bash
# Launch script for Interactive RAG 3D Visualizer

echo "======================================"
echo "  Interactive RAG 3D Visualizer"
echo "======================================"
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Starting visualization server..."
echo "Once loaded, open your browser to: http://localhost:8050"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python interactive_rag_3d.py
