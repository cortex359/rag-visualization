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
from sklearn.decomposition import PCA
from typing import List, Dict
import textwrap
import argparse

# Extensive sample documents for more data points
SAMPLE_DOCUMENTS = {
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

RWTH_DOCUMENTS = {
    "Teaching - Research-Oriented Curriculum": """
    RWTH integrates cutting-edge research topics directly into teaching through a two-phase approach. In 'Scientific Principles', all students learn foundational methodology, scientific writing, ethics, and academic integrity. The second phase, 'Research Orientation', offers individualized opportunities such as research assistantships and voluntary project modules. Top students are recognized via Dean’s Lists and may be nominated for the German Academic Scholarship Foundation. A coherent quality management system with measurable objectives ensures regular curriculum evaluation and adaptation, aligning teaching with RWTH’s status as an Excellence University.
    """,
    "Teaching - Digitalization Strategy for Learning": """
    RWTH’s digitalization strategy aims to scale individualized research-oriented teaching with a rich mix of online formats. Courses are supplemented with Open Educational Resources and Creative Commons-licensed research material. Learning Analytics Services and educational data mining support optimized learning and teaching strategies, integrated via platforms like RWTHmoodle and Dynexite. The approach prioritizes student autonomy and accessibility, leveraging technology to enhance the visibility and reach of RWTH’s research while maintaining high quality standards in content delivery.
    """,
    "Teaching - Lifelong Learning": """
    In line with the Excellence Strategy, RWTH promotes lifelong learning for alumni, professionals, and non-traditional learners. Programs include tailored continuing education modules, industry collaborations, and hybrid learning environments. The infrastructure and curriculum reforms support flexible pathways, facilitating skill acquisition in areas such as AI ethics, sustainability, and global mobility. Lifelong engagement strengthens RWTH’s societal impact and creates enduring connections between academic knowledge and real-world application.
    """,
    "Teaching - Challenge-Based Learning": """
    RWTH implements modern pedagogical approaches, such as challenge-based and co-creation models, where students tackle real societal or industry problems in interdisciplinary teams. Courses integrate sustainability themes, digital tools, and inclusivity into collaborative problem-solving. These innovations encourage creativity, agency, and resilience while fostering technical and interpersonal competencies. Industry partners and civic actors are embedded within learning environments to enhance relevance and practical impact.
    """,
    "Teaching - Internationalization in Academic Programs": """
    By 2024, 60% of new master programs at RWTH are taught partially or fully in English, increasing accessibility to international students. Existing courses have expanded English-language offerings to 43%. These efforts, aligned with RWTH’s Internationalization Strategy, exceed national averages for international student enrollment. The globalizing curriculum prepares students for cross-border collaboration and participation in multinational research networks, reinforcing RWTH’s reputation for attracting diverse academic talent.
    """,
    "Research - Excellence Clusters": """
    RWTH hosts high-profile Excellence Clusters such as the 'Internet of Production', merging Materials Science, Production Engineering, and digital innovation. These collaborative structures enable interdisciplinary breakthroughs and tightly align with societal grand challenges, including Industry 4.0 integration, AI ethics, and sustainable production systems. They exemplify RWTH’s commitment to mission-driven research agendas and positioning within Germany’s top research institutions.
    """,
    "Research - Sustainability in Research": """
    RWTH aligns research priorities with the UN’s Sustainable Development Goals (SDGs) guided by the Stockholm Resilience Center’s 'Wedding Cake' model. Strong sustainability principles place ecological integrity at the foundation for societal well-being and economic stability. Focus areas include SDGs related to health, water, energy, climate action, marine and terrestrial ecosystems, and sustainable cities. Interdisciplinary projects address both immediate environmental challenges and long-term societal transformations.
    """,
    "Research - AI and Data Science Initiatives": """
    Strategic appointments boost RWTH’s capacity in data science and artificial intelligence research. The AI Center catalyzes collaborations across faculties, addressing complex questions in computing, machine learning ethics, and AI applications in education. These investments intertwine fundamental research with teaching innovations, ensuring AI developments benefit both academic knowledge and society.
    """,
    "Research - Computational Life Sciences": """
    RWTH’s Center for Computational Life Sciences merges biology, data science, and engineering to accelerate discoveries in healthcare and bioinformatics. This interdisciplinary structure supports scalable collaborative projects, leveraging computational models and simulations to tackle molecular, cellular, and systems-level challenges with societal implications.
    """,
    "Research - Open Science Practices": """
    RWTH advances open science through shared infrastructures, FAIR data principles, and accessible publication formats. Collaborative platforms and open educational resources extend research reach, enhance reproducibility, and reinforce transparency. These measures democratize access to RWTH’s research outputs and enable inclusive global participation in scientific progress.
    """,
    "Infrastructure - Digital Learning Platforms": """
    RWTH invests in advanced digital infrastructure such as RWTHmoodle, Dynexite, and Learning Analytics systems to enhance both research-led and personalized teaching. The goal is seamless integration across teaching, assessment, and research dissemination, ensuring data-driven optimization of student learning experiences.
    """,
    "Infrastructure - Innovation Zones": """
    Hybrid and experimental learning spaces are integrated into campus design. These innovation zones support student-led projects, interdisciplinary collaboration, and co-creation with industry partners. The campus serves as a living lab, bolstering RWTH’s position as a hub for applied research and societal transformation.
    """,
    "Infrastructure - Sustainability in Campus Operations": """
    RWTH incorporates sustainability goals into infrastructure planning, prioritizing resource efficiency, renewable energy integration, and low-carbon operations. Green buildings, energy-efficient labs, and mobility solutions reflect the ecological priority model. Campus design supports both inclusivity and environmental stewardship.
    """,
    "Transfer & Innovation - Technology Transfer": """
    RWTH transforms lab innovations into market-ready products through targeted technology transfer programs. Patents serve as key indicators of research success, complemented by startup incubation and industry partnerships. These measures embed innovation into RWTH’s economic and societal contributions.
    """,
    "Transfer & Innovation - Societal Co-Creation": """
    Beyond traditional tech transfer, RWTH engages in societal co-creation, involving civic actors in problem-solving for urban development, climate strategies, and public policy. Collaborative models generate responsible innovations that align with community needs and global sustainability.
    """,
    "Transfer & Innovation - Entrepreneurship Education": """
    RWTH fosters inventiveness and entrepreneurial skills through structured programs, competitions, and startup mentoring. These initiatives connect students and researchers to investor networks, translating academic insights into scalable ventures.
    """,
    "Diversity & Inclusion - Gender Equality": """
    RWTH’s diversity strategy outlines clear targets for gender equality in professorships and leadership roles. Institutional frameworks align with Horizon Europe’s gender equality measures, ensuring equitable career pathways and representation.
    """,
    "Diversity & Inclusion - Intersectional Diversity": """
    RWTH embraces diversity beyond gender, considering ethnicity, socio-economic status, disability, and career pathways. Intersectional policies aim to create inclusive environments in learning, research, and administration, strengthening RWTH’s cultural and intellectual breadth.
    """,
    "Diversity & Inclusion - Inclusion Action Plan": """
    The Inclusion Action Plan (2021–2026) addresses accessibility, disability support, and barrier-free participation across RWTH. Investments in infrastructure, resources, and awareness programs ensure a universally inclusive campus experience.
    """,
    "Sustainability - Responsible Research Innovation": """
    RWTH’s Responsible Research Innovation Hub fosters interdisciplinary teams to design sustainable solutions with ecological priority. Collaborative frameworks integrate ethics, inclusivity, and societal impact into research agendas, reinforcing long-term resilience.
    """,
    "Sustainability - SDG Integration in Governance": """
    RWTH embeds SDG priorities into institutional governance, aligning faculty objectives with sustainability benchmarks. Metrics track contributions to climate action, clean energy, and sustainable industry, ensuring accountability and continuous progress.
    """,
    "Sustainability - Green Energy Networks": """
    Research projects at the E.ON Energy Research Center and FEN campus focus on decentralized renewable energy networks, power electronics, and DC technology adoption. These innovations underpin RWTH’s contribution to the European Green Deal’s 2050 targets.
    """,
    "Teaching - Honors College": """
    RWTH’s Honors College nurtures exceptional students through specialized seminars, mentorship, and funding opportunities. By connecting top talent to leading research projects, the program cultivates academic excellence and leadership within the university’s diverse community.
    """,
    "Research - Interdisciplinary Integration": """
    The RWTH strategic framework structures interdisciplinarity through scalable centers, fostering long-term collaborations across engineering, sciences, medicine, and humanities. This integration ensures responsiveness to emerging societal challenges.
    """,
    "Transfer & Innovation - International Collaboration": """
    RWTH strengthens partnerships with institutions in the Global South and European networks, committing to equitable, resilient forms of international cooperation. These relationships enhance diversity in knowledge systems and research impact.
    """,
    "Infrastructure - Shared Facilities": """
    RWTH optimizes access to high-quality shared facilities, including advanced laboratories and collaborative workspaces. Transparent governance structures ensure equitable usage and foster interdisciplinary research efficiency.
    """,
    "Teaching - Safe Learning Environments": """
    Instruments are in place to create safe, inclusive, and equitable learning environments. These acknowledge differences among students and promote respectful engagement, aligning with national and international educational goals.
    """,
}

EXTENSIVE_DOCUMENTS = RWTH_DOCUMENTS

class InteractiveRAG3D:
    """Interactive 3D RAG visualization with real-time query embedding"""

    def __init__(self, reduction_method='umap'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None
        self.reduced_embeddings = None
        self.metadata = []
        self.reducer = None
        self.reduction_method = reduction_method.lower()

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
            raise ValueError(f"Unknown reduction method: {self.reduction_method}. Use 'umap' or 'pca'.")

        self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        print(f"Data preparation complete using {self.reduction_method.upper()}!")

    def embed_query(self, query: str):
        """Embed a query and transform to 3D space"""
        if not query or len(query.strip()) < 3:
            return None

        query_embedding = self.model.encode([query])
        query_3d = self.reducer.transform(query_embedding)
        return query_3d[0], query_embedding[0]

    def find_nearest_neighbors(self, query: str, n: int = 5):
        """Find n nearest neighbors to query based on Euclidean distance in 3D space"""
        if not query or len(query.strip()) < 3:
            return []

        # Embed and transform query to 3D space
        query_result = self.embed_query(query)
        if query_result is None:
            return []

        query_3d, _ = query_result

        # Calculate Euclidean distances in 3D space
        distances = np.sqrt(np.sum((self.reduced_embeddings - query_3d) ** 2, axis=1))

        # Get top n indices (smallest distances)
        nearest_indices = np.argsort(distances)[:n]

        results = []
        for idx in nearest_indices:
            # Convert distance to similarity score (closer = higher score)
            # Using inverse distance normalized to 0-1 range
            distance = distances[idx]
            similarity = 1.0 / (1.0 + distance)  # Closer points have higher similarity

            results.append({
                'index': int(idx),
                'similarity': float(similarity),
                'distance': float(distance),
                'document': self.metadata[idx]['document'],
                'text': self.metadata[idx]['text']
            })

        return results

    def create_3d_plot(self, query: str = "", show_neighbors: bool = False,
                       n_neighbors: int = 5, show_legend: bool = False):
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

        # Find nearest neighbors if query and show_neighbors enabled
        neighbor_indices = set()
        if query and len(query.strip()) >= 3 and show_neighbors:
            neighbors = self.find_nearest_neighbors(query, n_neighbors)
            neighbor_indices = {n['index'] for n in neighbors}

        # Add document chunks
        for doc_name in unique_docs:
            indices = [i for i, m in enumerate(self.metadata) if m['document'] == doc_name]

            hover_texts = []
            for i in indices:
                # Create wrapped text with fixed width for consistent display
                text = self.metadata[i]['text']
                # Break into lines of ~50 characters at word boundaries
                words = text.split()
                lines = []
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > 50 and current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1

                if current_line:
                    lines.append(' '.join(current_line))

                wrapped = '<br>'.join(lines)
                hover_texts.append(
                    f"<b>{self.metadata[i]['document']}</b><br><br>"
                    f"<span style='display: inline-block; max-width: 400px; white-space: normal;'>"
                    f"{wrapped}</span>"
                )

            # Separate normal points from nearest neighbors
            normal_indices = [i for i in indices if i not in neighbor_indices]
            neighbor_in_group = [i for i in indices if i in neighbor_indices]

            # Add normal points
            if normal_indices:
                normal_hover = [hover_texts[indices.index(i)] for i in normal_indices]
                fig.add_trace(go.Scatter3d(
                    x=self.reduced_embeddings[normal_indices, 0],
                    y=self.reduced_embeddings[normal_indices, 1],
                    z=self.reduced_embeddings[normal_indices, 2],
                    mode='markers',
                    name=doc_name,
                    marker=dict(
                        size=6,
                        color=doc_colors[doc_name],
                        line=dict(width=0.5, color='#1a1a1a'),
                        opacity=0.8 if not show_neighbors else 0.3
                    ),
                    text=normal_hover,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=doc_name,
                    showlegend=show_legend
                ))

            # Add nearest neighbor points (highlighted)
            if neighbor_in_group:
                neighbor_hover = [hover_texts[indices.index(i)] for i in neighbor_in_group]
                fig.add_trace(go.Scatter3d(
                    x=self.reduced_embeddings[neighbor_in_group, 0],
                    y=self.reduced_embeddings[neighbor_in_group, 1],
                    z=self.reduced_embeddings[neighbor_in_group, 2],
                    mode='markers',
                    name=f'{doc_name} (Neighbor)',
                    marker=dict(
                        size=12,
                        color='#FF00FF',  # Bright magenta for neighbors
                        line=dict(width=2, color='#FFFFFF'),
                        opacity=1.0,
                        symbol='circle'
                    ),
                    text=neighbor_hover,
                    hovertemplate='%{text}<extra></extra>',
                    legendgroup=doc_name,
                    showlegend=False
                ))

        # Add query point if exists
        if query and len(query.strip()) >= 3:
            result = self.embed_query(query)
            if result is not None:
                query_3d, _ = result
                fig.add_trace(go.Scatter3d(
                    x=[query_3d[0]],
                    y=[query_3d[1]],
                    z=[query_3d[2]],
                    mode='markers+text',
                    name='Your Query',
                    marker=dict(
                        size=15,
                        color='#FFD700',
                        symbol='diamond',
                        line=dict(width=3, color='#FFA500'),
                    ),
                    text=['QUERY'],
                    textposition='top center',
                    textfont=dict(size=16, color='#FFD700', family='Arial Black'),
                    hovertext=(
                        f"<b>Your Query:</b><br><br>"
                        f"<span style='display: inline-block; max-width: 400px; white-space: normal;'>"
                        f"{query}</span>"
                    ),
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=True
                ))

        # Dark theme layout
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0a0a0a',
            plot_bgcolor='#0a0a0a',
            uirevision='constant',  # Preserve camera angle and zoom while typing
            modebar={'orientation': 'v'},  # Vertical modebar
            dragmode='orbit',  # Default to orbit mode
            scene=dict(
                dragmode='orbit',  # Keep scene in orbit mode by default
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


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Interactive 3D RAG Embedding Visualizer',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python interactive_rag_3d.py --method umap    # Use UMAP (default)
  python interactive_rag_3d.py --method pca     # Use PCA
    """
)
parser.add_argument(
    '--method',
    type=str,
    choices=['umap', 'pca'],
    default='umap',
    help='Dimensionality reduction method: "umap" for UMAP (default) or "pca" for PCA'
)

args = parser.parse_args()

# Initialize the app
print("\n" + "=" * 60)
print("Initializing Interactive 3D RAG Visualizer...")
print(f"Using dimensionality reduction method: {args.method.upper()}")
print("=" * 60 + "\n")

rag_viz = InteractiveRAG3D(reduction_method=args.method)

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True
)

# App layout
app.layout = dbc.Container([
    # Hidden div for keyboard events
    html.Div(id='keyboard-listener', tabIndex=0, style={
        'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%',
        'zIndex': -1, 'outline': 'none'
    }),

    # Store components for state
    dcc.Store(id='camera-state'),
    dcc.Store(id='show-neighbors-state', data=False),
    dcc.Store(id='show-legend-state', data=False),

    dbc.Row([
        dbc.Col([
            html.H2(
                "✨ Interactive RAG Embedding Visualization 📚",
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
                    debounce=False,
                    style={
                        'backgroundColor': '#1a1a1a',
                        'color': '#ffffff',
                        'borderColor': '#444',
                        'fontSize': '18px'
                    }
                ),
            ], className="mb-2"),
        ], width=12)
    ]),

    # Control panel
    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(
                    "🎯 Show Neighbors",
                    id='toggle-neighbors-btn',
                    color="primary",
                    size="sm",
                    outline=True,
                    style={'fontSize': '13px'}
                ),
                dbc.Button(
                    "📋 Toggle Legend",
                    id='toggle-legend-btn',
                    color="secondary",
                    size="sm",
                    outline=True,
                    style={'fontSize': '13px'}
                ),
            ], className="mb-2"),
        ], width="auto"),
        dbc.Col([
            html.Div([
                html.Span("N neighbors:", style={'color': '#888', 'fontSize': '13px', 'marginRight': '10px', 'flexShrink': '0'}),
                html.Div([
                    dcc.Slider(
                        id='n-neighbors-slider',
                        min=3,
                        max=15,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(3, 16, 3)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={'flex': '1', 'minWidth': '200px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'width': '100%', 'gap': '10px'})
        ], width=5),
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
        # Main 3D plot
        dbc.Col([
            dcc.Graph(
                id='3d-plot',
                style={'height': '75vh'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'rag_visualization',
                        'height': 1080,
                        'width': 1920,
                        'scale': 2
                    }
                }
            )
        ], id='plot-column', width=12),

        # Sidebar for nearest neighbors (conditional)
        dbc.Col([
            html.Div(id='neighbors-sidebar', style={
                'backgroundColor': '#1a1a1a',
                'padding': '15px',
                'borderRadius': '8px',
                'height': '75vh',
                'overflowY': 'auto',
                'border': '1px solid #444'
            })
        ], id='sidebar-column', width=4, style={'display': 'none'})
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.P(
                    f"📊 {len(rag_viz.chunks)} chunks | ⌨️  Keyboard: W/S (forward/back), A/D (left/right), Q/E (up/down), Arrow keys (rotate)",
                    className="text-center mb-1",
                    style={'color': '#888', 'fontSize': '12px'}
                ),
            ])
        ], width=12)
    ])

], fluid=True, style={'backgroundColor': '#0a0a0a', 'minHeight': '100vh'})


# Callback for toggling neighbors state
@app.callback(
    Output('show-neighbors-state', 'data'),
    Input('toggle-neighbors-btn', 'n_clicks'),
    State('show-neighbors-state', 'data'),
    prevent_initial_call=True
)
def toggle_neighbors(n_clicks, current_state):
    return not current_state


# Callback for toggling legend state
@app.callback(
    Output('show-legend-state', 'data'),
    Input('toggle-legend-btn', 'n_clicks'),
    State('show-legend-state', 'data'),
    prevent_initial_call=True
)
def toggle_legend(n_clicks, current_state):
    return not current_state


# Main callback for updating plot and info
@app.callback(
    [Output('3d-plot', 'figure'),
     Output('query-info', 'children'),
     Output('neighbors-sidebar', 'children'),
     Output('toggle-neighbors-btn', 'outline'),
     Output('toggle-legend-btn', 'outline'),
     Output('plot-column', 'width'),
     Output('sidebar-column', 'style')],
    [Input('query-input', 'value'),
     Input('show-neighbors-state', 'data'),
     Input('show-legend-state', 'data'),
     Input('n-neighbors-slider', 'value')],
    [State('3d-plot', 'relayoutData')]
)
def update_visualization(query, show_neighbors, show_legend, n_neighbors, relayout_data):
    """Update plot, info, and sidebar with query embedding in real-time"""

    # Create figure with current settings
    fig = rag_viz.create_3d_plot(
        query if query else "",
        show_neighbors=show_neighbors,
        n_neighbors=n_neighbors,
        show_legend=show_legend
    )

    # Preserve camera position if it exists in relayout_data
    # This ensures camera state is maintained across all updates including dragmode changes
    if relayout_data:
        # First check for complete camera object
        if 'scene.camera' in relayout_data:
            fig.update_layout(scene_camera=relayout_data['scene.camera'])
        else:
            # Build camera from individual properties if available
            camera_update = {}

            # Eye position (camera location)
            if 'scene.camera.eye.x' in relayout_data:
                camera_update['eye'] = {
                    'x': relayout_data.get('scene.camera.eye.x', 1.5),
                    'y': relayout_data.get('scene.camera.eye.y', 1.5),
                    'z': relayout_data.get('scene.camera.eye.z', 1.3)
                }

            # Center position (where camera is looking)
            if 'scene.camera.center.x' in relayout_data:
                camera_update['center'] = {
                    'x': relayout_data.get('scene.camera.center.x', 0),
                    'y': relayout_data.get('scene.camera.center.y', 0),
                    'z': relayout_data.get('scene.camera.center.z', 0)
                }

            # Up vector (camera orientation)
            if 'scene.camera.up.x' in relayout_data:
                camera_update['up'] = {
                    'x': relayout_data.get('scene.camera.up.x', 0),
                    'y': relayout_data.get('scene.camera.up.y', 0),
                    'z': relayout_data.get('scene.camera.up.z', 1)
                }

            # Projection settings
            if 'scene.camera.projection.type' in relayout_data:
                camera_update['projection'] = {
                    'type': relayout_data.get('scene.camera.projection.type', 'perspective')
                }

            if camera_update:
                fig.update_layout(scene_camera=camera_update)

        # Preserve dragmode if it changed
        if 'dragmode' in relayout_data or 'scene.dragmode' in relayout_data:
            dragmode = relayout_data.get('dragmode') or relayout_data.get('scene.dragmode', 'orbit')
            fig.update_layout(dragmode=dragmode, scene_dragmode=dragmode)

    # Update info text
    if query and len(query.strip()) >= 3:
        if show_neighbors:
            info = f"✨ Query: '{query}' | Showing top {n_neighbors} nearest neighbors in magenta"
        else:
            info = f"✨ Query embedded: '{query}' - Watch the golden diamond in 3D space!"
    elif query and len(query.strip()) < 3:
        info = "⏳ Type at least 3 characters to embed your query..."
    else:
        info = "💬 Start typing to see real-time query embedding..."

    # Update neighbors sidebar
    sidebar_content = []
    if query and len(query.strip()) >= 3 and show_neighbors:
        neighbors = rag_viz.find_nearest_neighbors(query, n_neighbors)

        sidebar_content.append(html.H5(
            f"Top {n_neighbors} Nearest Neighbors",
            style={'color': '#00D9FF', 'marginBottom': '15px'}
        ))

        for i, neighbor in enumerate(neighbors, 1):
            similarity_pct = neighbor['similarity'] * 100
            distance = neighbor.get('distance', 0)

            sidebar_content.append(html.Div([
                html.Div([
                    html.Strong(f"#{i}", style={'color': '#FF00FF', 'marginRight': '8px'}),
                    html.Span(f"{similarity_pct:.1f}% similar", style={'color': '#888', 'fontSize': '12px'}),
                    html.Span(f" • ", style={'color': '#444', 'fontSize': '12px', 'margin': '0 4px'}),
                    html.Span(f"dist: {distance:.2f}", style={'color': '#666', 'fontSize': '11px'})
                ], style={'marginBottom': '5px'}),
                html.Div(
                    neighbor['document'],
                    style={'color': '#00D9FF', 'fontSize': '13px', 'fontWeight': 'bold', 'marginBottom': '5px'}
                ),
                html.Div(
                    neighbor['text'][:200] + ('...' if len(neighbor['text']) > 200 else ''),
                    style={'color': '#ccc', 'fontSize': '12px', 'marginBottom': '15px', 'lineHeight': '1.4'}
                ),
                html.Hr(style={'borderColor': '#333', 'margin': '15px 0'})
            ]))
    else:
        sidebar_content.append(html.Div([
            html.H5("Nearest Neighbors", style={'color': '#888', 'marginBottom': '15px'}),
            html.P(
                "Type a query and click 'Show Neighbors' to see the most similar document chunks.",
                style={'color': '#666', 'fontSize': '13px', 'textAlign': 'center', 'marginTop': '50px'}
            )
        ]))

    # Button outline states (False = filled, True = outline)
    neighbors_btn_outline = not show_neighbors
    legend_btn_outline = not show_legend

    # Control column widths and sidebar visibility based on show_neighbors
    if show_neighbors:
        plot_width = 8
        sidebar_style = {
            'backgroundColor': '#1a1a1a',
            'padding': '15px',
            'borderRadius': '8px',
            'height': '75vh',
            'overflowY': 'auto',
            'border': '1px solid #444',
            'display': 'block'
        }
    else:
        plot_width = 12
        sidebar_style = {'display': 'none'}

    return fig, info, sidebar_content, neighbors_btn_outline, legend_btn_outline, plot_width, sidebar_style


# Clientside callback for keyboard navigation
app.clientside_callback(
    """
    function(id) {
        // Camera movement parameters
        const moveSpeed = 0.3;
        const rotateSpeed = 0.1;

        // Get the 3D plot element
        const plotDiv = document.getElementById('3d-plot');
        if (!plotDiv || !plotDiv.layout || !plotDiv.layout.scene) {
            return window.dash_clientside.no_update;
        }

        // Listen for keyboard events
        document.addEventListener('keydown', function(event) {
            if (!plotDiv.layout.scene.camera) return;

            const camera = JSON.parse(JSON.stringify(plotDiv.layout.scene.camera));
            const eye = camera.eye || {x: 1.5, y: 1.5, z: 1.3};

            // Check if user is typing in an input field
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
                return;
            }

            let updated = false;

            switch(event.key.toLowerCase()) {
                // Forward/Backward
                case 'w':
                    eye.z -= moveSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
                case 's':
                    eye.z += moveSpeed;
                    updated = true;
                    event.preventDefault();
                    break;

                // Left/Right
                case 'a':
                    eye.x -= moveSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
                case 'd':
                    eye.x += moveSpeed;
                    updated = true;
                    event.preventDefault();
                    break;

                // Up/Down
                case 'q':
                    eye.y += moveSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
                case 'e':
                    eye.y -= moveSpeed;
                    updated = true;
                    event.preventDefault();
                    break;

                // Rotation with arrow keys
                case 'arrowup':
                    eye.y += rotateSpeed;
                    eye.z -= rotateSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
                case 'arrowdown':
                    eye.y -= rotateSpeed;
                    eye.z += rotateSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
                case 'arrowleft':
                    eye.x -= rotateSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
                case 'arrowright':
                    eye.x += rotateSpeed;
                    updated = true;
                    event.preventDefault();
                    break;
            }

            if (updated) {
                camera.eye = eye;
                Plotly.relayout(plotDiv, {'scene.camera': camera});
            }
        });

        return window.dash_clientside.no_update;
    }
    """,
    Output('keyboard-listener', 'children'),
    Input('keyboard-listener', 'id')
)


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Interactive RAG Visualizer...")
    print("=" * 60)
    print("\n📍 Open your browser to: http://localhost:8050")
    print("⌨️  Type queries in real-time to see embeddings!")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=False, host='0.0.0.0', port=8050)
