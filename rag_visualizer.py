#!/usr/bin/env python3
"""
Interactive RAG Embedding Visualizer
Demonstrates document chunking and semantic similarity in vector space
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import umap
from typing import List, Tuple, Dict
import textwrap


class RAGVisualizer:
    """Visualize document embeddings in 2D space"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the visualizer with an embedding model

        Args:
            model_name: HuggingFace sentence-transformers model name
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None
        self.reduced_embeddings = None
        self.metadata = []

    def chunk_document(self, text: str, chunk_size: int = 200,
                       overlap: int = 50, doc_name: str = "Document") -> List[str]:
        """
        Split document into overlapping chunks

        Args:
            text: Document text
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters
            doc_name: Name/identifier for the document

        Returns:
            List of text chunks
        """
        chunks = []
        words = text.split()

        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                self.metadata.append({
                    'document': doc_name,
                    'chunk_id': len(self.chunks),
                    'text': chunk_text
                })

                # Keep overlap words
                overlap_words = int(len(current_chunk) * (overlap / chunk_size))
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_length = sum(len(w) + 1 for w in current_chunk)

        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            self.metadata.append({
                'document': doc_name,
                'chunk_id': len(self.chunks),
                'text': chunk_text
            })

        self.chunks.extend(chunks)
        return chunks

    def add_documents(self, documents: Dict[str, str], chunk_size: int = 200,
                     overlap: int = 50):
        """
        Add multiple documents and chunk them

        Args:
            documents: Dict mapping document names to their text
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters
        """
        for doc_name, text in documents.items():
            print(f"Chunking document: {doc_name}")
            self.chunk_document(text, chunk_size, overlap, doc_name)

        print(f"Total chunks created: {len(self.chunks)}")

    def generate_embeddings(self):
        """Generate embeddings for all chunks"""
        print("Generating embeddings...")
        self.embeddings = self.model.encode(self.chunks, show_progress_bar=True)
        print(f"Embeddings shape: {self.embeddings.shape}")

    def reduce_dimensions(self, method: str = "umap", n_components: int = 2,
                         random_state: int = 42):
        """
        Reduce embeddings to 2D or 3D for visualization

        Args:
            method: 'umap' or 'tsne'
            n_components: 2 or 3 dimensions
            random_state: Random seed for reproducibility
        """
        print(f"Reducing dimensions using {method.upper()}...")

        if method.lower() == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
        elif method.lower() == "tsne":
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, len(self.chunks) - 1),
                metric='cosine'
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        print(f"Reduced embeddings shape: {self.reduced_embeddings.shape}")

    def create_interactive_plot(self, title: str = "RAG Document Embeddings Visualization",
                               width: int = 1200, height: int = 800):
        """
        Create an interactive plotly visualization

        Args:
            title: Plot title
            width: Plot width in pixels
            height: Plot height in pixels

        Returns:
            Plotly figure object
        """
        if self.reduced_embeddings is None:
            raise ValueError("Must call reduce_dimensions() first")

        # Create color mapping for documents
        unique_docs = list(set(m['document'] for m in self.metadata))
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        doc_colors = {doc: color_palette[i % len(color_palette)]
                     for i, doc in enumerate(unique_docs)}

        # Prepare hover text
        hover_texts = []
        for meta in self.metadata:
            wrapped_text = '<br>'.join(textwrap.wrap(meta['text'], width=50))
            hover_text = (
                f"<b>Document:</b> {meta['document']}<br>"
                f"<b>Chunk:</b> {meta['chunk_id']}<br>"
                f"<b>Text:</b><br>{wrapped_text}"
            )
            hover_texts.append(hover_text)

        # Create figure
        if self.reduced_embeddings.shape[1] == 2:
            fig = go.Figure()

            # Add trace for each document
            for doc_name in unique_docs:
                indices = [i for i, m in enumerate(self.metadata)
                          if m['document'] == doc_name]

                fig.add_trace(go.Scatter(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    mode='markers',
                    name=doc_name,
                    marker=dict(
                        size=12,
                        color=doc_colors[doc_name],
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    ),
                    text=[hover_texts[i] for i in indices],
                    hovertemplate='%{text}<extra></extra>'
                ))

            fig.update_layout(
                title=title,
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                width=width,
                height=height,
                hovermode='closest',
                plot_bgcolor='#f8f9fa',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            )

        else:  # 3D plot
            fig = go.Figure()

            for doc_name in unique_docs:
                indices = [i for i, m in enumerate(self.metadata)
                          if m['document'] == doc_name]

                fig.add_trace(go.Scatter3d(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    z=self.reduced_embeddings[indices, 2],
                    mode='markers',
                    name=doc_name,
                    marker=dict(
                        size=8,
                        color=doc_colors[doc_name],
                        line=dict(width=1, color='white'),
                        opacity=0.8
                    ),
                    text=[hover_texts[i] for i in indices],
                    hovertemplate='%{text}<extra></extra>'
                ))

            fig.update_layout(
                title=title,
                width=width,
                height=height,
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3"
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.8)"
                )
            )

        return fig

    def visualize(self, documents: Dict[str, str], chunk_size: int = 200,
                 overlap: int = 50, method: str = "umap", dimensions: int = 2,
                 output_file: str = "rag_visualization.html"):
        """
        Complete pipeline: chunk, embed, reduce, and visualize

        Args:
            documents: Dict mapping document names to their text
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters
            method: Dimensionality reduction method ('umap' or 'tsne')
            dimensions: 2 or 3
            output_file: HTML file to save the visualization
        """
        self.add_documents(documents, chunk_size, overlap)
        self.generate_embeddings()
        self.reduce_dimensions(method, dimensions)

        fig = self.create_interactive_plot()
        fig.write_html(output_file)
        print(f"\n✓ Visualization saved to: {output_file}")
        print(f"Open this file in your browser to explore the interactive plot!")

        return fig


# Sample documents for demonstration
SAMPLE_DOCUMENTS = {
    "Machine Learning Basics": """
    Machine learning is a subset of artificial intelligence that focuses on building systems
    that can learn from and make decisions based on data. Supervised learning uses labeled
    data to train models, while unsupervised learning finds patterns in unlabeled data.
    Deep learning uses neural networks with multiple layers to extract high-level features
    from raw data. Common algorithms include decision trees, support vector machines, and
    gradient boosting. Model evaluation uses metrics like accuracy, precision, recall, and F1 score.
    Cross-validation helps prevent overfitting by testing on multiple data splits. Feature
    engineering is crucial for improving model performance. Regularization techniques like
    L1 and L2 help prevent overfitting. Ensemble methods combine multiple models for better predictions.
    """,

    "Cooking Italian Pasta": """
    Cooking perfect pasta requires attention to detail and timing. Start by bringing a large
    pot of salted water to a rolling boil. Use about 4-6 quarts of water per pound of pasta.
    Add the pasta and stir immediately to prevent sticking. Cook according to package directions,
    but taste a minute or two before the suggested time. Perfect pasta should be al dente,
    meaning it has a slight bite to it. Always reserve a cup of pasta water before draining.
    This starchy water is essential for creating silky sauces. Carbonara is made with eggs,
    pecorino cheese, guanciale, and black pepper. Marinara sauce combines tomatoes, garlic,
    olive oil, and fresh basil. Aglio e olio is a simple dish with garlic, olive oil, and chili flakes.
    """,

    "Climate Change Science": """
    Climate change refers to long-term shifts in global temperatures and weather patterns.
    Human activities, particularly burning fossil fuels, release greenhouse gases into the
    atmosphere. Carbon dioxide, methane, and nitrous oxide trap heat and warm the planet.
    The global average temperature has increased by about 1.1°C since pre-industrial times.
    Rising temperatures cause glaciers to melt, sea levels to rise, and extreme weather events
    to become more frequent. The Paris Agreement aims to limit warming to well below 2°C.
    Renewable energy sources like solar and wind power can help reduce emissions. Forests
    act as carbon sinks, absorbing CO2 from the atmosphere. Ocean acidification threatens
    marine ecosystems. Adaptation strategies help communities cope with climate impacts.
    """,

    "Ancient Roman History": """
    Ancient Rome began as a small settlement on the Tiber River around 753 BCE. The Roman
    Kingdom was followed by the Roman Republic in 509 BCE. Julius Caesar played a crucial
    role in the transition to the Roman Empire. Augustus became the first Roman Emperor in
    27 BCE, beginning the Pax Romana. The Roman Empire expanded across Europe, North Africa,
    and the Middle East. Roman engineering marvels included aqueducts, roads, and the Colosseum.
    Latin was the language of government and education. Roman law influenced modern legal
    systems worldwide. The empire split into Eastern and Western halves in 395 CE. The Western
    Roman Empire fell in 476 CE, while the Eastern Byzantine Empire continued for another millennium.
    """,

    "Quantum Computing Fundamentals": """
    Quantum computing harnesses quantum mechanical phenomena to process information. Unlike
    classical bits that are either 0 or 1, qubits can exist in superposition of both states.
    Quantum entanglement allows qubits to be correlated in ways impossible for classical bits.
    Quantum gates manipulate qubits to perform calculations. Shor's algorithm can factor
    large numbers exponentially faster than classical algorithms. Grover's algorithm provides
    quadratic speedup for searching unsorted databases. Quantum computers excel at optimization
    problems, cryptography, and simulating quantum systems. Quantum decoherence is a major
    challenge, as qubits are sensitive to environmental noise. Error correction codes help
    maintain quantum information. Companies like IBM, Google, and IonQ are developing quantum hardware.
    """
}


def main():
    """Run the RAG visualization demo"""
    print("=" * 60)
    print("RAG Embedding Visualizer")
    print("=" * 60)
    print()

    # Create visualizer
    visualizer = RAGVisualizer()

    # Generate visualization
    visualizer.visualize(
        documents=SAMPLE_DOCUMENTS,
        chunk_size=200,
        overlap=50,
        method="umap",  # or "tsne"
        dimensions=3,   # or 3 for 3D
        output_file="rag_visualization.html"
    )

    print("\n" + "=" * 60)
    print("You can now open 'rag_visualization.html' in your browser!")
    print("Hover over points to see the chunk text.")
    print("Notice how semantically similar chunks cluster together!")
    print("=" * 60)


if __name__ == "__main__":
    main()
