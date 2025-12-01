# RAG Embedding Visualizer

Interactive 3D visualization of document chunk embeddings in semantic space for demonstrating Retrieval Augmented Generation (RAG) concepts.

## 🌟 Interactive 3D Demo (Recommended!)

**NEW:** Real-time interactive 3D visualization with dark mode, perfect for full-screen presentations!

## Features

- **Document Chunking**: Splits documents into overlapping chunks with configurable size
- **Semantic Embeddings**: Uses sentence-transformers to generate embeddings
- **Dimensionality Reduction**: Supports both UMAP and t-SNE for 2D/3D visualization
- **Interactive Visualization**: Hover over points to see chunk text and document source
- **Color-coded by Document**: Visually distinguish different source documents

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Interactive 3D Web App (Recommended for Presentations)

Launch the real-time interactive visualization:

```bash
# Simple launch (default: UMAP)
./launch.sh

# Or with PCA dimensionality reduction
./launch.sh --method pca

# Or manually
source venv/bin/activate
python interactive_rag_3d.py              # Uses UMAP (default)
python interactive_rag_3d.py --method pca # Uses PCA
```

Then open your browser to **http://localhost:8050**

**Dimensionality Reduction Methods:**
- `--method umap` (default): Non-linear reduction that preserves both local and global structure, better for visualizing semantic clusters
- `--method pca`: Linear reduction that maximizes variance, faster and more deterministic but may not capture complex relationships as well

**Features:**
- 🌑 **Dark mode theme** - perfect for presentations
- ⚡ **Real-time query embedding** - type and see embeddings instantly
- 📊 **79+ data points** from extensive document collection
- 🎯 **3D interactive plot** - rotate, zoom, explore
- 💎 **Query visualization** - your queries appear as golden diamonds
- 🎨 **Color-coded categories** - ML, cooking, climate, history, quantum, etc.

**Usage Tips:**
- Type at least 3 characters to see your query embedded
- Hover over points to see the chunk text
- Rotate the 3D plot to explore semantic relationships
- Full-screen (F11) for best presentation experience

### Option 2: Static HTML Export

Run the demo to generate a static HTML file:

```bash
python rag_visualizer.py
```

This will generate `rag_visualization.html` which you can open in your browser.

## Using Your Own Documents

```python
from rag_visualizer import RAGVisualizer

# Create visualizer
visualizer = RAGVisualizer()

# Add your documents
documents = {
    "Document 1": "Your text here...",
    "Document 2": "More text...",
}

# Generate visualization
visualizer.visualize(
    documents=documents,
    chunk_size=200,      # Characters per chunk
    overlap=50,          # Overlapping characters
    method="umap",       # or "tsne"
    dimensions=2,        # 2D or 3D
    output_file="my_visualization.html"
)
```

## Parameters

- **chunk_size**: Number of characters per chunk (default: 200)
- **overlap**: Number of overlapping characters between chunks (default: 50)
- **method**: Dimensionality reduction method - "umap" or "tsne" (default: "umap")
- **dimensions**: 2 for 2D plot, 3 for 3D plot (default: 2)

## What to Expect

The visualization will show:
- Documents about similar topics clustering together in semantic space
- Different colored clusters for different documents
- Interactive hover tooltips showing the actual chunk text
- Semantic relationships between document chunks

## Tips for Your Talk

1. **Start with diverse documents**: The sample includes ML, cooking, climate, history, and quantum computing to show clear separation
2. **Show the clustering**: Point out how chunks from the same topic cluster together
3. **Demonstrate similarity**: Show how related concepts (e.g., ML and quantum computing) are closer than unrelated ones
4. **Interactive exploration**: During the talk, hover over points to show the actual text
5. **Explain the implications**: This is how RAG retrieves relevant context - by finding nearest neighbors in this space

## Advanced Usage

For more control, use the class methods directly:

```python
visualizer = RAGVisualizer(model_name="all-MiniLM-L6-v2")
visualizer.add_documents(documents, chunk_size=200, overlap=50)
visualizer.generate_embeddings()
visualizer.reduce_dimensions(method="umap", n_components=2)
fig = visualizer.create_interactive_plot()
fig.show()  # Display in Jupyter
fig.write_html("output.html")  # Save to file
```

## Requirements

- Python 3.8+
- sentence-transformers: For generating embeddings
- plotly: For interactive visualization
- umap-learn: For UMAP dimensionality reduction
- scikit-learn: For t-SNE and utilities
- numpy: For numerical operations
