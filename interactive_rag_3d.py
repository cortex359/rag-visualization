#!/usr/bin/env python3
"""
Interactive 3D RAG Embedding Visualizer with Real-time Query Embedding
Dark mode, full-screen suitable, with extensive sample data
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
from typing import List, Dict
import textwrap

# Extensive sample documents for more data points
EXTENSIVE_DOCUMENTS = {
    "Machine Learning - Basics": """
    Machine learning is a subset of artificial intelligence that focuses on building systems
    that can learn from and make decisions based on data. Supervised learning uses labeled
    data to train models, while unsupervised learning finds patterns in unlabeled data.
    Deep learning uses neural networks with multiple layers to extract high-level features
    from raw data. Common algorithms include decision trees, support vector machines, and
    gradient boosting. Model evaluation uses metrics like accuracy, precision, recall, and F1 score.
    Cross-validation helps prevent overfitting by testing on multiple data splits.
    """,

    "Machine Learning - Neural Networks": """
    Neural networks are computational models inspired by biological neurons. They consist of
    interconnected layers of nodes that process and transform data. Backpropagation is used
    to update weights during training. Activation functions like ReLU, sigmoid, and tanh
    introduce non-linearity. Convolutional neural networks excel at image processing.
    Recurrent neural networks handle sequential data. Transformers use attention mechanisms
    for natural language processing. Batch normalization stabilizes training. Dropout prevents
    overfitting by randomly disabling neurons during training.
    """,

    "Machine Learning - Training": """
    Training machine learning models requires careful optimization. Gradient descent iteratively
    updates model parameters to minimize loss. Learning rate determines step size during updates.
    Adam optimizer combines momentum with adaptive learning rates. Mini-batch training balances
    computational efficiency with convergence stability. Early stopping prevents overfitting.
    Data augmentation increases training set diversity. Transfer learning leverages pre-trained
    models for new tasks. Hyperparameter tuning optimizes model architecture and training settings.
    """,

    "Cooking - Italian Pasta": """
    Cooking perfect pasta requires attention to detail and timing. Start by bringing a large
    pot of salted water to a rolling boil. Use about 4-6 quarts of water per pound of pasta.
    Add the pasta and stir immediately to prevent sticking. Cook according to package directions,
    but taste a minute or two before the suggested time. Perfect pasta should be al dente,
    meaning it has a slight bite to it. Always reserve a cup of pasta water before draining.
    This starchy water is essential for creating silky sauces. Fresh pasta cooks in 2-3 minutes.
    """,

    "Cooking - Sauces": """
    Carbonara is made with eggs, pecorino cheese, guanciale, and black pepper. The heat from
    the pasta cooks the eggs to create a creamy sauce. Marinara sauce combines tomatoes, garlic,
    olive oil, and fresh basil. Simmer for 20-30 minutes to develop flavors. Aglio e olio is
    a simple dish with garlic, olive oil, and chili flakes. Pesto uses fresh basil, pine nuts,
    parmesan, garlic, and olive oil blended until smooth. Bolognese is a meat-based sauce
    that simmers for hours. Arrabbiata adds red chili peppers for heat.
    """,

    "Cooking - Techniques": """
    Sautéing involves cooking food quickly in a small amount of fat over high heat. Braising
    combines searing with slow cooking in liquid. Roasting uses dry heat in an oven to caramelize
    sugars and develop flavor. Blanching briefly boils vegetables then plunges them into ice
    water to preserve color. Deglazing uses liquid to lift browned bits from the pan bottom.
    Emulsification combines oil and water-based ingredients. Reduction concentrates flavors
    by evaporating liquid. Mise en place means preparing all ingredients before cooking begins.
    """,

    "Climate Change - Causes": """
    Climate change refers to long-term shifts in global temperatures and weather patterns.
    Human activities, particularly burning fossil fuels, release greenhouse gases into the
    atmosphere. Carbon dioxide, methane, and nitrous oxide trap heat and warm the planet.
    Deforestation reduces carbon absorption capacity. Industrial processes emit various
    pollutants. Agriculture produces methane from livestock. Transportation sector contributes
    significantly to emissions. Power generation from coal and natural gas releases CO2.
    Land use changes alter carbon cycles. Feedback loops amplify warming effects.
    """,

    "Climate Change - Effects": """
    The global average temperature has increased by about 1.1°C since pre-industrial times.
    Rising temperatures cause glaciers to melt, sea levels to rise, and extreme weather events
    to become more frequent. Hurricanes intensify with warmer ocean temperatures. Droughts
    become longer and more severe. Flooding increases in coastal areas. Coral reefs experience
    bleaching. Wildlife migration patterns shift. Growing seasons change affecting agriculture.
    Arctic sea ice diminishes. Permafrost thawing releases stored carbon. Species extinction
    rates accelerate with habitat loss.
    """,

    "Climate Change - Solutions": """
    The Paris Agreement aims to limit warming to well below 2°C. Renewable energy sources
    like solar and wind power can help reduce emissions. Electric vehicles decrease transportation
    emissions. Energy efficiency improvements reduce consumption. Carbon capture technology
    removes CO2 from atmosphere. Reforestation increases carbon absorption. Sustainable
    agriculture practices reduce methane. Green building design minimizes energy use.
    Policy changes incentivize clean energy. International cooperation coordinates global efforts.
    """,

    "Ancient Rome - Foundation": """
    Ancient Rome began as a small settlement on the Tiber River around 753 BCE. According to
    legend, Romulus and Remus founded the city. The Roman Kingdom was followed by the Roman
    Republic in 509 BCE. The Senate governed with elected officials called consuls. Patricians
    held power while plebeians struggled for rights. The Twelve Tables established law code.
    Rome expanded through military conquest. The Punic Wars against Carthage secured dominance.
    Julius Caesar played a crucial role in the transition to the Roman Empire.
    """,

    "Ancient Rome - Empire": """
    Augustus became the first Roman Emperor in 27 BCE, beginning the Pax Romana. This peaceful
    period lasted two centuries. The Roman Empire expanded across Europe, North Africa, and
    the Middle East. Roman engineering marvels included aqueducts, roads, and the Colosseum.
    Latin was the language of government and education. Roman law influenced modern legal
    systems worldwide. The empire reached its greatest extent under Trajan. Trade networks
    connected distant provinces. Military legions maintained order. Provincial governors
    administered territories.
    """,

    "Ancient Rome - Decline": """
    The empire split into Eastern and Western halves in 395 CE. Economic troubles plagued
    the Western Empire. Inflation devalued currency. Military spending exceeded revenue.
    Barbarian invasions pressured borders. Political instability weakened central authority.
    The Western Roman Empire fell in 476 CE when Odoacer deposed the last emperor. The
    Eastern Byzantine Empire continued for another millennium. Christianity became the
    official religion. Cultural legacy persisted through medieval period. Latin evolved
    into Romance languages. Roman architecture influenced later styles.
    """,

    "Quantum Computing - Fundamentals": """
    Quantum computing harnesses quantum mechanical phenomena to process information. Unlike
    classical bits that are either 0 or 1, qubits can exist in superposition of both states.
    Quantum entanglement allows qubits to be correlated in ways impossible for classical bits.
    Quantum gates manipulate qubits to perform calculations. Measurement collapses superposition
    to a definite state. Quantum interference amplifies correct answers while canceling wrong
    ones. The no-cloning theorem prevents copying quantum states. Quantum teleportation transfers
    states between distant qubits.
    """,

    "Quantum Computing - Algorithms": """
    Shor's algorithm can factor large numbers exponentially faster than classical algorithms.
    This threatens current encryption methods. Grover's algorithm provides quadratic speedup
    for searching unsorted databases. Quantum annealing solves optimization problems. Variational
    quantum eigensolvers find molecular ground states. Quantum Fourier transform underpins
    many algorithms. Phase estimation determines eigenvalues. Quantum walks explore graph
    structures. Amplitude amplification enhances success probability. These algorithms demonstrate
    quantum advantage over classical computing.
    """,

    "Quantum Computing - Hardware": """
    Quantum computers excel at optimization problems, cryptography, and simulating quantum
    systems. Quantum decoherence is a major challenge, as qubits are sensitive to environmental
    noise. Error correction codes help maintain quantum information. Superconducting qubits
    operate at near absolute zero. Ion traps use electromagnetic fields to confine qubits.
    Topological qubits promise better stability. Companies like IBM, Google, and IonQ are
    developing quantum hardware. Quantum supremacy demonstrates tasks impossible for classical
    computers. Current systems have 50-100 qubits.
    """,

    "Data Science - Analysis": """
    Data science combines statistics, programming, and domain expertise to extract insights
    from data. Exploratory data analysis reveals patterns and anomalies. Statistical inference
    draws conclusions from samples. Hypothesis testing validates assumptions. Regression analysis
    models relationships between variables. Classification assigns categories to observations.
    Clustering groups similar data points. Dimensionality reduction simplifies complex data.
    Time series analysis examines temporal patterns. A/B testing compares alternatives.
    """,

    "Data Science - Tools": """
    Python is the primary language for data science. Pandas provides data manipulation tools.
    NumPy handles numerical computations. Matplotlib and Seaborn create visualizations.
    Scikit-learn offers machine learning algorithms. TensorFlow and PyTorch build neural
    networks. Jupyter notebooks enable interactive analysis. SQL queries databases. Apache
    Spark processes big data. Git version controls code. Docker containerizes applications.
    Cloud platforms provide scalable computing resources.
    """,

    "Data Science - Workflow": """
    The data science workflow begins with problem definition. Data collection gathers relevant
    information. Data cleaning removes errors and inconsistencies. Feature engineering creates
    useful variables. Model selection chooses appropriate algorithms. Training fits models
    to data. Validation assesses performance. Hyperparameter tuning optimizes settings.
    Testing evaluates final model. Deployment puts models into production. Monitoring tracks
    performance over time. Iteration refines the solution.
    """,

    "Artificial Intelligence - History": """
    Artificial intelligence research began in the 1950s. Alan Turing proposed the Turing test
    to measure machine intelligence. Early AI used symbolic reasoning and rule-based systems.
    Expert systems captured human knowledge. The AI winter saw reduced funding and interest.
    Machine learning emerged as a new approach. Deep learning breakthrough in 2012 revolutionized
    computer vision. Natural language processing advanced with transformers. Reinforcement
    learning mastered games. Current AI shows impressive capabilities.
    """,

    "Artificial Intelligence - Applications": """
    AI powers virtual assistants like Siri and Alexa. Recommendation systems suggest products
    and content. Computer vision enables facial recognition and autonomous vehicles. Natural
    language processing translates languages and answers questions. Medical AI diagnoses
    diseases and discovers drugs. Financial AI detects fraud and trades stocks. Manufacturing
    AI optimizes production and quality control. Agricultural AI monitors crops and predicts
    yields. Entertainment AI creates art, music, and games. Scientific AI accelerates research
    and discovery.
    """,

    "Artificial Intelligence - Ethics": """
    AI ethics addresses bias, fairness, and accountability. Algorithmic bias can perpetuate
    discrimination. Explainable AI makes decisions transparent. Privacy concerns arise from
    data collection. Job displacement worries workers. Autonomous weapons raise safety issues.
    Deepfakes spread misinformation. Surveillance threatens civil liberties. AI safety ensures
    systems behave as intended. Value alignment matches AI goals with human values. Regulation
    balances innovation with protection.
    """,

    "Space Exploration - Solar System": """
    Our solar system contains eight planets orbiting the Sun. Mercury is closest and hottest.
    Venus has a thick toxic atmosphere. Earth supports life with liquid water. Mars shows
    evidence of ancient rivers. Jupiter is a gas giant with dozens of moons. Saturn's rings
    are made of ice and rock. Uranus rotates on its side. Neptune has the strongest winds.
    Dwarf planets include Pluto. Asteroids populate the asteroid belt. Comets originate from
    the Kuiper Belt and Oort Cloud.
    """,

    "Space Exploration - Missions": """
    Space exploration began with Sputnik in 1957. Apollo 11 landed humans on the Moon in 1969.
    Voyager probes explored outer planets and left the solar system. Mars rovers search for
    signs of life. The Hubble telescope revolutionized astronomy. The International Space
    Station hosts continuous human presence. SpaceX develops reusable rockets. NASA's Artemis
    program returns to the Moon. James Webb telescope observes distant galaxies. Private
    companies pursue space tourism. Future missions target Mars colonization.
    """,

    "Space Exploration - Challenges": """
    Space exploration faces technical and biological challenges. Radiation exposure threatens
    astronaut health. Microgravity causes bone loss and muscle atrophy. Long missions require
    life support systems. Communication delays increase with distance. Launch costs remain
    high despite reusability. Space debris threatens satellites and spacecraft. Extreme
    temperatures require thermal protection. Propulsion systems limit speed and distance.
    Psychological effects of isolation concern mission planners. Resource scarcity demands
    recycling systems.
    """,

    "Biotechnology - Genetics": """
    Biotechnology manipulates living organisms for useful purposes. Genetic engineering modifies
    DNA to change traits. CRISPR-Cas9 enables precise gene editing. Recombinant DNA technology
    produces insulin and vaccines. Gene therapy treats inherited diseases. Cloning creates
    genetically identical organisms. Transgenic organisms contain genes from other species.
    Genomics sequences and analyzes entire genomes. Personalized medicine tailors treatments
    to individuals. Pharmacogenomics predicts drug responses.
    """,

    "Biotechnology - Applications": """
    Agricultural biotechnology improves crop yields and nutrition. GMO crops resist pests
    and herbicides. Golden rice provides vitamin A. Drought-resistant varieties help farmers.
    Industrial biotechnology produces biofuels and biodegradable plastics. Enzymes catalyze
    chemical reactions. Fermentation creates foods and beverages. Bioremediation cleans
    pollution. Medical biotechnology develops new treatments. Stem cells regenerate tissues.
    Immunotherapy fights cancer. Synthetic biology designs novel organisms.
    """,

    "Cybersecurity - Threats": """
    Cybersecurity protects systems from digital attacks. Malware includes viruses, worms,
    and trojans. Ransomware encrypts files demanding payment. Phishing tricks users into
    revealing credentials. Social engineering manipulates people. DDoS attacks overwhelm
    servers. SQL injection exploits database vulnerabilities. Zero-day exploits target unknown
    flaws. Advanced persistent threats conduct long-term espionage. Insider threats come from
    authorized users. Supply chain attacks compromise trusted software.
    """,

    "Cybersecurity - Defense": """
    Defense strategies employ multiple layers. Firewalls filter network traffic. Antivirus
    software detects malware. Encryption protects data confidentiality. Authentication verifies
    user identity. Multi-factor authentication adds security. Access control limits permissions.
    Intrusion detection systems monitor for attacks. Security information and event management
    analyzes logs. Penetration testing finds vulnerabilities. Security awareness training
    educates users. Incident response plans handle breaches. Backup systems enable recovery.
    """,

    "Renewable Energy - Solar": """
    Renewable energy comes from naturally replenishing sources. Solar power converts sunlight
    into electricity using photovoltaic cells. Solar panels contain semiconductor materials
    that generate current when struck by photons. Concentrated solar power uses mirrors to
    focus sunlight and heat fluid. Solar energy is abundant and clean. Efficiency continues
    improving with new materials. Storage batteries address intermittency. Solar installations
    range from rooftop panels to utility-scale farms. Costs have declined dramatically.
    """,

    "Renewable Energy - Wind": """
    Wind power harnesses kinetic energy from moving air. Wind turbines convert mechanical
    rotation into electricity. Onshore wind farms are cost-effective. Offshore wind captures
    stronger consistent winds. Turbine sizes have increased for greater capacity. Wind resources
    vary by location and season. Grid integration requires managing variability. Hybrid systems
    combine wind with storage. Environmental impacts include bird collisions and noise. Wind
    energy provides significant electricity generation globally.
    """,

    "Blockchain - Technology": """
    Blockchain is a distributed ledger technology. Blocks contain transaction records linked
    cryptographically. Consensus mechanisms validate new blocks. Proof of work requires
    computational effort. Proof of stake bases validation on ownership. Smart contracts
    execute automatically when conditions are met. Decentralization removes central authority.
    Immutability prevents altering past records. Transparency makes transactions public.
    Cryptography ensures security. Distributed networks provide resilience.
    """,

    "Blockchain - Applications": """
    Cryptocurrency uses blockchain for digital money. Bitcoin was the first cryptocurrency.
    Ethereum enables smart contracts and decentralized applications. Supply chain management
    tracks products from origin to consumer. Digital identity verifies credentials without
    central database. Voting systems ensure transparent elections. Healthcare records maintain
    patient data securely. Real estate transfers simplify property transactions. Intellectual
    property protects creative works. Financial services settle transactions faster and cheaper.
    """
}


class InteractiveRAG3D:
    """Interactive 3D RAG visualization with real-time query embedding"""

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.reduced_embeddings = None
        self.metadata = []
        self.reducer = None

        # Prepare initial data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare document chunks and embeddings"""
        print("Preparing document chunks...")

        # Create more chunks with smaller size for more data points
        for doc_name, text in EXTENSIVE_DOCUMENTS.items():
            words = text.split()
            chunk_size = 40  # Smaller chunks = more data points
            overlap = 10

            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 10:  # Skip very small chunks
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

        print("Reducing dimensions with UMAP...")
        # Fit UMAP on document embeddings
        self.reducer = umap.UMAP(
            n_components=3,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        print("Data preparation complete!")

    def embed_query(self, query: str):
        """Embed a query and transform to 3D space"""
        if not query or len(query.strip()) < 3:
            return None

        query_embedding = self.model.encode([query])
        query_3d = self.reducer.transform(query_embedding)
        return query_3d[0]

    def create_3d_plot(self, query: str = ""):
        """Create 3D plotly figure with dark theme"""

        # Color mapping for documents
        unique_docs = list(set(m['document'] for m in self.metadata))
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B500', '#6C5CE7',
            '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7', '#00B894',
            '#FF7675', '#74B9FF', '#A29BFE', '#FD79A8', '#FDCB6E'
        ]
        doc_colors = {doc: colors[i % len(colors)] for i, doc in enumerate(unique_docs)}

        fig = go.Figure()

        # Add document chunks
        for doc_name in unique_docs:
            indices = [i for i, m in enumerate(self.metadata) if m['document'] == doc_name]

            hover_texts = []
            for i in indices:
                wrapped = textwrap.fill(self.metadata[i]['text'], width=60)
                hover_texts.append(
                    f"<b>{self.metadata[i]['document']}</b><br><br>{wrapped}"
                )

            fig.add_trace(go.Scatter3d(
                x=self.reduced_embeddings[indices, 0],
                y=self.reduced_embeddings[indices, 1],
                z=self.reduced_embeddings[indices, 2],
                mode='markers',
                name=doc_name,
                marker=dict(
                    size=6,
                    color=doc_colors[doc_name],
                    line=dict(width=0.5, color='#1a1a1a'),
                    opacity=0.8
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                legendgroup=doc_name,
            ))

        # Add query point if exists
        if query and len(query.strip()) >= 3:
            query_3d = self.embed_query(query)
            if query_3d is not None:
                fig.add_trace(go.Scatter3d(
                    x=[query_3d[0]],
                    y=[query_3d[1]],
                    z=[query_3d[2]],
                    mode='markers+text',
                    name='Your Query',
                    marker=dict(
                        size=20,
                        color='#FFD700',
                        symbol='diamond',
                        line=dict(width=3, color='#FFA500'),
                    ),
                    text=['QUERY'],
                    textposition='top center',
                    textfont=dict(size=16, color='#FFD700', family='Arial Black'),
                    hovertext=f"<b>Your Query:</b><br>{query}",
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=True
                ))

        # Dark theme layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            scene=dict(
                xaxis=dict(
                    backgroundcolor='#0a0a0a',
                    gridcolor='#2a2a2a',
                    showbackground=True,
                    title='',
                    showticklabels=False
                ),
                yaxis=dict(
                    backgroundcolor='#0a0a0a',
                    gridcolor='#2a2a2a',
                    showbackground=True,
                    title='',
                    showticklabels=False
                ),
                zaxis=dict(
                    backgroundcolor='#0a0a0a',
                    gridcolor='#2a2a2a',
                    showbackground=True,
                    title='',
                    showticklabels=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(20, 20, 20, 0.8)",
                font=dict(size=11, color='#ffffff'),
                bordercolor='#444444',
                borderwidth=1
            ),
            font=dict(color='#ffffff'),
            hovermode='closest',
        )

        return fig


# Initialize the app
print("\n" + "=" * 60)
print("Initializing Interactive 3D RAG Visualizer...")
print("=" * 60 + "\n")

rag_viz = InteractiveRAG3D()

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(
                "🔮 Interactive RAG Embedding Visualization",
                className="text-center mb-3 mt-3",
                style={'color': '#00D9FF', 'fontWeight': 'bold'}
            ),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.InputGroupText("🔍", style={'backgroundColor': '#1a1a1a', 'borderColor': '#444'}),
                dbc.Input(
                    id='query-input',
                    placeholder='Type your search query here... (real-time embedding)',
                    type='text',
                    debounce=False,  # No debounce for real-time
                    style={
                        'backgroundColor': '#1a1a1a',
                        'color': '#ffffff',
                        'borderColor': '#444',
                        'fontSize': '18px'
                    }
                ),
            ], className="mb-3"),
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(
                id='query-info',
                className="text-center mb-2",
                style={'color': '#00D9FF', 'fontSize': '14px', 'minHeight': '20px'}
            )
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id='3d-plot',
                style={'height': '80vh'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                }
            )
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.P(
                    f"📊 {len(rag_viz.chunks)} document chunks embedded in 3D semantic space",
                    className="text-center mb-1",
                    style={'color': '#888', 'fontSize': '13px'}
                ),
                html.P(
                    "💡 Tip: Type in the search box to see your query embedded in real-time!",
                    className="text-center",
                    style={'color': '#888', 'fontSize': '13px'}
                ),
            ])
        ], width=12)
    ])

], fluid=True, style={'backgroundColor': '#0a0a0a', 'minHeight': '100vh'})


@app.callback(
    [Output('3d-plot', 'figure'),
     Output('query-info', 'children')],
    [Input('query-input', 'value')]
)
def update_plot(query):
    """Update plot with query embedding in real-time"""
    fig = rag_viz.create_3d_plot(query if query else "")

    if query and len(query.strip()) >= 3:
        info = f"✨ Query embedded: '{query}' - Watch the golden diamond in 3D space!"
    elif query and len(query.strip()) < 3:
        info = "⏳ Type at least 3 characters to embed your query..."
    else:
        info = "💬 Start typing to see real-time query embedding..."

    return fig, info


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Interactive RAG Visualizer...")
    print("=" * 60)
    print("\n📍 Open your browser to: http://localhost:8050")
    print("⌨️  Type queries in real-time to see embeddings!")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=8050)
