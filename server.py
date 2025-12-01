#!/usr/bin/env python3
"""
Flask backend for Three.js RAG Visualizer
Serves embedding data and handles real-time queries
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from sklearn.decomposition import PCA
import argparse
from pathlib import Path

# Import the sample documents from the Dash version
from interactive_rag_3d import EXTENSIVE_DOCUMENTS


class RAGBackend:
    """Backend for managing embeddings and queries"""

    def __init__(self, reduction_method='umap'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.reduced_embeddings = None
        self.metadata = []
        self.reducer = None
        self.reduction_method = reduction_method.lower()

        self._prepare_data()

    def _prepare_data(self):
        """Prepare document chunks and embeddings"""
        print("Preparing document chunks...")

        # Create chunks
        for doc_name, text in EXTENSIVE_DOCUMENTS.items():
            words = text.split()
            chunk_size = 40
            overlap = 10

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 10:
                    continue

                chunk_text = ' '.join(chunk_words)
                self.chunks.append(chunk_text)
                self.metadata.append({
                    'document': doc_name,
                    'text': chunk_text,
                    'type': 'document'
                })

        print(f"Created {len(self.chunks)} document chunks")
        print("Generating embeddings...")

        # Generate embeddings
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)

        print(f"Reducing dimensions with {self.reduction_method.upper()}...")

        # Choose dimensionality reduction method
        if self.reduction_method == 'pca':
            self.reducer = PCA(n_components=3, random_state=42)
        elif self.reduction_method == 'umap':
            self.reducer = umap.UMAP(
                n_components=3,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")

        self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        print(f"Data preparation complete using {self.reduction_method.upper()}!")

    def get_all_data(self):
        """Get all document data for visualization"""
        data_points = []

        for i, meta in enumerate(self.metadata):
            pos = self.reduced_embeddings[i]
            data_points.append({
                'id': i,
                'position': [float(pos[0]), float(pos[1]), float(pos[2])],
                'document': meta['document'],
                'text': meta['text'],
                'type': 'document'
            })

        return data_points

    def embed_query(self, query_text):
        """Embed a query and return 3D position"""
        if not query_text or len(query_text.strip()) < 3:
            return None

        query_embedding = self.model.encode([query_text])
        query_3d = self.reducer.transform(query_embedding)

        return {
            'position': [float(query_3d[0][0]), float(query_3d[0][1]), float(query_3d[0][2])],
            'text': query_text
        }

    def find_nearest_neighbors(self, query_text, n=5):
        """Find n nearest neighbors in 3D space"""
        query_result = self.embed_query(query_text)
        if not query_result:
            return []

        query_pos = np.array(query_result['position'])

        # Calculate Euclidean distances in 3D space
        distances = np.sqrt(np.sum((self.reduced_embeddings - query_pos) ** 2, axis=1))

        # Get top n indices
        nearest_indices = np.argsort(distances)[:n]

        results = []
        for idx in nearest_indices:
            distance = distances[idx]
            similarity = 1.0 / (1.0 + distance)

            results.append({
                'id': int(idx),
                'distance': float(distance),
                'similarity': float(similarity),
                'document': self.metadata[idx]['document'],
                'text': self.metadata[idx]['text'],
                'position': [
                    float(self.reduced_embeddings[idx][0]),
                    float(self.reduced_embeddings[idx][1]),
                    float(self.reduced_embeddings[idx][2])
                ]
            })

        return results


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Flask backend for Three.js RAG Visualizer')
parser.add_argument('--method', type=str, choices=['umap', 'pca'], default='umap',
                   help='Dimensionality reduction method')
parser.add_argument('--port', type=int, default=5000, help='Port to run server on')

args = parser.parse_args()

# Initialize backend
print("\n" + "=" * 60)
print("Initializing RAG Backend...")
print(f"Using dimensionality reduction method: {args.method.upper()}")
print("=" * 60 + "\n")

backend = RAGBackend(reduction_method=args.method)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Ensure static directory exists
Path('static').mkdir(exist_ok=True)


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/app.js')
def serve_js():
    """Serve the JavaScript file"""
    return send_from_directory('static', 'app.js', mimetype='application/javascript')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.route('/api/data')
def get_data():
    """Get all document data"""
    return jsonify({
        'points': backend.get_all_data(),
        'method': backend.reduction_method
    })


@app.route('/api/query', methods=['POST'])
def query():
    """Embed a query and return position"""
    data = request.get_json()
    query_text = data.get('query', '')

    result = backend.embed_query(query_text)

    if result is None:
        return jsonify({'error': 'Query too short'}), 400

    return jsonify(result)


@app.route('/api/neighbors', methods=['POST'])
def neighbors():
    """Find nearest neighbors to query"""
    data = request.get_json()
    query_text = data.get('query', '')
    n = data.get('n', 5)

    results = backend.find_nearest_neighbors(query_text, n)

    return jsonify({
        'query': query_text,
        'neighbors': results
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting RAG Backend Server...")
    print("=" * 60)
    print(f"\n📍 Open your browser to: http://localhost:{args.port}")
    print("⌨️  Three.js visualization with full camera control!")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=args.port)
