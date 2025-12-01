#!/usr/bin/env python3
"""
Query Visualization Demo
Shows how a user query embeds in the same semantic space as document chunks
"""

from rag_visualizer import RAGVisualizer, SAMPLE_DOCUMENTS
import plotly.graph_objects as go
import numpy as np


def visualize_with_query(query: str, output_file: str = "rag_query_visualization.html"):
    """
    Visualize documents with a query point showing semantic similarity

    Args:
        query: User query string
        output_file: Output HTML file
    """
    print("=" * 60)
    print("RAG Query Visualization")
    print("=" * 60)
    print(f"\nQuery: '{query}'\n")

    # Create visualizer and process documents
    visualizer = RAGVisualizer()
    visualizer.add_documents(SAMPLE_DOCUMENTS, chunk_size=200, overlap=50)

    # Store original chunk count
    num_doc_chunks = len(visualizer.chunks)

    # Add query as a special "chunk"
    visualizer.chunks.append(query)
    visualizer.metadata.append({
        'document': 'USER_QUERY',
        'chunk_id': -1,
        'text': query
    })

    # Generate embeddings (including query)
    visualizer.generate_embeddings()

    # Reduce dimensions
    visualizer.reduce_dimensions(method="umap", n_components=2)

    # Separate query embedding from document embeddings
    doc_embeddings = visualizer.reduced_embeddings[:num_doc_chunks]
    query_embedding = visualizer.reduced_embeddings[num_doc_chunks:]

    # Calculate distances from query to all chunks
    distances = np.linalg.norm(doc_embeddings - query_embedding, axis=1)
    nearest_indices = np.argsort(distances)[:5]  # Top 5 nearest chunks

    print("Top 5 most similar chunks to query:")
    print("-" * 60)
    for i, idx in enumerate(nearest_indices, 1):
        meta = visualizer.metadata[idx]
        distance = distances[idx]
        print(f"\n{i}. Document: {meta['document']}")
        print(f"   Distance: {distance:.3f}")
        print(f"   Text: {meta['text'][:100]}...")

    # Create visualization
    fig = go.Figure()

    # Color mapping
    unique_docs = list(set(m['document'] for m in visualizer.metadata[:num_doc_chunks]))
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    doc_colors = {doc: color_palette[i % len(color_palette)]
                 for i, doc in enumerate(unique_docs)}

    # Plot document chunks
    for doc_name in unique_docs:
        indices = [i for i, m in enumerate(visualizer.metadata[:num_doc_chunks])
                  if m['document'] == doc_name]

        fig.add_trace(go.Scatter(
            x=doc_embeddings[indices, 0],
            y=doc_embeddings[indices, 1],
            mode='markers',
            name=doc_name,
            marker=dict(
                size=10,
                color=doc_colors[doc_name],
                line=dict(width=1, color='white'),
                opacity=0.7
            ),
            text=[f"<b>Document:</b> {visualizer.metadata[i]['document']}<br>"
                  f"<b>Text:</b> {visualizer.metadata[i]['text'][:100]}..."
                  for i in indices],
            hovertemplate='%{text}<extra></extra>'
        ))

    # Highlight nearest neighbors
    fig.add_trace(go.Scatter(
        x=doc_embeddings[nearest_indices, 0],
        y=doc_embeddings[nearest_indices, 1],
        mode='markers',
        name='Top 5 Matches',
        marker=dict(
            size=20,
            color='rgba(255, 0, 0, 0)',
            line=dict(width=3, color='red'),
        ),
        text=[f"<b>Match #{i+1}</b><br>"
              f"<b>Document:</b> {visualizer.metadata[idx]['document']}<br>"
              f"<b>Distance:</b> {distances[idx]:.3f}<br>"
              f"<b>Text:</b> {visualizer.metadata[idx]['text'][:100]}..."
              for i, idx in enumerate(nearest_indices)],
        hovertemplate='%{text}<extra></extra>',
        showlegend=True
    ))

    # Plot query point
    fig.add_trace(go.Scatter(
        x=query_embedding[:, 0],
        y=query_embedding[:, 1],
        mode='markers+text',
        name='Query',
        marker=dict(
            size=20,
            color='gold',
            symbol='star',
            line=dict(width=2, color='darkgoldenrod'),
        ),
        text=['QUERY'],
        textposition='top center',
        textfont=dict(size=14, color='darkgoldenrod', family='Arial Black'),
        hovertext=f"<b>Query:</b><br>{query}",
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    # Draw lines from query to nearest neighbors
    for idx in nearest_indices:
        fig.add_trace(go.Scatter(
            x=[query_embedding[0, 0], doc_embeddings[idx, 0]],
            y=[query_embedding[0, 1], doc_embeddings[idx, 1]],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=f"RAG Query Visualization<br><sub>Query: '{query}'</sub>",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=1200,
        height=800,
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)"
        )
    )

    fig.write_html(output_file)
    print(f"\n{'=' * 60}")
    print(f"✓ Visualization saved to: {output_file}")
    print("The gold star shows where your query sits in semantic space!")
    print("Red circles highlight the 5 nearest chunks that would be retrieved.")
    print("=" * 60)

    return fig


if __name__ == "__main__":
    # Example queries
    queries = [
        "What is machine learning?",
        "How do I cook pasta?",
        "Tell me about quantum computers",
        "What caused the fall of Rome?",
        "Explain climate change"
    ]

    print("Available example queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")

    choice = input("\nSelect a query (1-5) or type your own: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(queries):
        query = queries[int(choice) - 1]
    else:
        query = choice if choice else queries[0]

    visualize_with_query(query)
