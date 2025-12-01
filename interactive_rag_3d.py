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
    # Cluster 1: Research Excellence
    "Research Excellence - Interdisciplinary Research Initiatives": """
RWTH Aachen University fosters interdisciplinary research initiatives as a core part of its Excellence Strategy. These initiatives aim to break traditional academic boundaries by integrating expertise from engineering, natural sciences, medicine, and humanities. They enhance innovation by pooling diverse perspectives and methodologies to address complex global challenges such as climate change, digitalization, and health technology. Emphasis is placed on creating collaborative environments that encourage experimental and applied research, aiming to heighten RWTH’s global competitiveness and impact. Funding and support structures are designed to facilitate project development, joint appointments, and resource sharing across faculties.
""",
    "Research Excellence - Sustainable Energy Systems": """
Sustainable energy systems represent a priority research area aligned with RWTH’s commitment to societal impact and innovation. The university integrates advanced technological research with environmental science to develop renewable energy solutions and smart grid technologies. RWTH’s research in solar power, wind energy, and energy storage aims to contribute to the global transition to carbon-neutral economies. Collaboration with industry partners and international research networks accelerates technology transfer. The Excellence Strategy ensures sustained investment in state-of-the-art laboratories and interdisciplinary teams to keep RWTH at the forefront of energy research.
""",
    "Research Excellence - Digital Transformation and AI": """
Digital transformation and artificial intelligence are central to RWTH's research agenda within the Excellence Strategy framework. Research efforts emphasize algorithm development, machine learning, robotics, and data science applications across disciplines. RWTH promotes innovation ecosystems that integrate academia, startups, and industrial partners to accelerate AI-driven technologies. Ethical considerations and societal impacts of AI are analyzed through multidisciplinary projects combining technology with social sciences and law. Persistent funding and strategic partnerships enable RWTH to expand its leadership role in Germany’s digital innovation landscape.
""",
    "Research Excellence - Biomedical Engineering Innovations": """
Biomedical engineering is a strategic research focus at RWTH Aachen, advancing healthcare technologies and personalized medicine. The Excellence Strategy supports projects that develop cutting-edge diagnostic tools, biomaterials, and medical devices through collaborative efforts between engineering and medical faculties. Research groups emphasize translational approaches that rapidly move innovations from lab to clinical application. Partnerships with hospitals, biotech companies, and international research centers strengthen RWTH’s position as a nexus for biomedical innovation. Investments in infrastructure and talent development sustain growth in this high-impact field.
""",
    "Research Excellence - Material Science and Nanotechnology": """
RWTH excels in material science and nanotechnology research by exploring novel materials for energy storage, electronics, and structural applications. The Excellence Strategy prioritizes high-risk, high-reward projects that pioneer new synthesis methods and characterization techniques at the nanoscale. Multidisciplinary teams work on applications ranging from lightweight composites to quantum materials. The university leverages its central location and infrastructure to form research clusters integrating chemistry, physics, and engineering disciplines. Extensive collaboration with industry partners drives innovation and facilitates commercialization of advanced materials technologies.
""",

    # Cluster 2: Teaching and Education Innovation
    "Teaching and Education - Curriculum Modernization": """
One of the pillars of the RWTH Excellence Strategy focuses on curriculum modernization to equip students for a rapidly changing labor market. This involves integrating digital competencies, interdisciplinary content, and practical project work into existing degree programs. RWTH actively engages faculty and students in redesigning courses to foster critical thinking, creativity, and collaboration. The modernization project includes enhanced use of blended learning tools, flexible modular structures, and incorporation of sustainability topics across disciplines. These efforts aim to strengthen RWTH’s educational quality and relevance on the global stage.
""",
    "Teaching and Education - Internationalization of Degree Programs": """
Internationalization is a core element of RWTH’s teaching strategy, aiming to attract global talent and offer students intercultural competence. RWTH promotes joint and dual degree programs in collaboration with top-tier international universities. Language support, international internships, and study abroad opportunities are systematically expanded to increase students’ global readiness. Digital technologies support virtual exchange formats, enabling cross-border learning experiences. The Excellence Strategy supports these initiatives to make RWTH a preferred destination for international students and scholars.
""",
    "Teaching and Education - Digital Learning Environments": """
The transformation to digital learning environments is a key focus area for RWTH under its Excellence Strategy. The university invests in state-of-the-art digital platforms, interactive teaching tools, and virtual laboratories to enhance student engagement and accessibility. Faculty development programs encourage innovative pedagogies including flipped classrooms and adaptive learning. RWTH also addresses digital equity by providing infrastructure support to all students. These initiatives aim to improve learning outcomes while preparing students for a digital-first world in academia and industry.
""",
    "Teaching and Education - Student-Centered Learning Approaches": """
RWTH prioritizes student-centered learning by promoting active learning, peer collaboration, and personalized feedback. The Excellence Strategy integrates new assessment models that emphasize critical thinking and problem-solving over rote memorization. Mentorship programs and learning analytics are utilized to monitor and support individual student progress. Efforts to foster inclusive and supportive learning environments ensure that diverse student needs are addressed. This approach aligns with RWTH’s mission to produce graduates who are adaptive, innovative, and socially responsible.
""",
    "Teaching and Education - Lifelong Learning and Continuing Education": """
Recognizing the importance of lifelong learning, RWTH offers a broad portfolio of continuing education courses tailored to professionals and industry partners. These programs cover emerging technologies, management skills, and interdisciplinary topics to support career development and knowledge transfer. The Excellence Strategy promotes flexible formats including online and hybrid courses to increase accessibility. Cooperation with local and global enterprises enhances relevance and practical application. This commitment strengthens RWTH’s role as a hub for ongoing professional education and innovation dissemination.
""",

    # Cluster 3: Infrastructure & Innovation Ecosystem
    "Infrastructure & Innovation - Research Facilities Modernization": """
Modern, cutting-edge research facilities form a cornerstone of RWTH Aachen’s Excellence Strategy. Investments focus on upgrading laboratories, creating high-tech centers, and expanding shared equipment platforms across faculties. The modernization effort supports advanced experimental research and large-scale collaborative projects that demand specialized infrastructure. Emphasis on sustainability and energy efficiency reflects RWTH’s commitment to responsible campus development. By enhancing its physical infrastructure, RWTH aims to attract top researchers and foster innovative breakthroughs.
""",
    "Infrastructure & Innovation - Innovation Hubs and Technology Transfer": """
Innovation hubs at RWTH foster close collaboration between academia and industry to accelerate technology transfer and entrepreneurship. The Excellence Strategy endorses creation and expansion of interdisciplinary spaces like startup incubators, co-working labs, and maker spaces. Dedicated support services assist researchers in patenting, business development, and networking. These hubs strengthen RWTH’s ecosystem by promoting knowledge exchange, commercialization of research, and regional economic growth. They embody the university’s ambition to translate scientific excellence into societal value.
""",
    "Infrastructure & Innovation - Digital Campus Initiatives": """
The digital campus initiative at RWTH integrates cutting-edge ICT infrastructure with smart services to optimize teaching, research, and administration. The Excellence Strategy promotes campus-wide deployment of high-speed networks, cloud computing resources, and digital identity management. Smart building technology enhances energy efficiency and user comfort. Integration of virtual reality tools supports immersive learning and remote collaboration. The digital campus creates a connected environment that increases productivity and quality of campus life while fostering innovation.
""",
    "Infrastructure & Innovation - Sustainability in Campus Development": """
Sustainability principles are embedded in RWTH’s campus development and infrastructure projects under the Excellence Strategy. Green building standards, renewable energy installations, and waste reduction practices guide construction and maintenance. The university actively monitors carbon footprint and implements measures to achieve climate neutrality. Sustainable mobility concepts, including bike-sharing and electric vehicle infrastructure, reduce environmental impact. These initiatives showcase RWTH’s dedication to ecological responsibility and set an example for the academic community.
""",
    "Infrastructure & Innovation - Collaborative Research Networks": """
RWTH Aachen strategically builds collaborative research networks at local, national, and international levels. These networks enable resource sharing, joint grant applications, and shared scientific agendas. The Excellence Strategy supports the creation of centers of excellence and thematic alliances that bring together multiple stakeholders. Collaboration with industry partners, governmental agencies, and other universities strengthens RWTH’s research capacity and global visibility. These networks foster interdisciplinary synergies and accelerate innovation cycles.
""",

    # Cluster 4: Talent and Diversity
    "Talent and Diversity - Female Leadership Advancement": """
Promoting female leadership within RWTH Aachen is a priority dimension of the Excellence Strategy. The university implements mentoring, career development, and networking programs tailored to women academics and researchers. Specific measures include targeted hiring initiatives, flexible working arrangements, and gender-sensitive evaluation processes. By increasing female representation in leadership positions, RWTH aims to foster diversity that enhances innovation and decision-making quality. These efforts contribute to creating an inclusive academic culture that supports equal opportunity.
""",
    "Talent and Diversity - International Faculty Recruitment": """
International recruitment of top academic talent is actively pursued to enhance RWTH’s global research and teaching profile. The Excellence Strategy encompasses strategies such as attractive employment packages, relocation support, and integration programs. Multilingual and intercultural competence training helps new faculty feel welcomed and effective. Recruitment efforts target emerging fields and strategic research areas, ensuring a diverse and highly qualified workforce. Internationalization of faculty strengthens RWTH’s competitiveness and global networks.
""",
    "Talent and Diversity - Support for Early Career Researchers": """
Supporting early career researchers is essential for sustainable excellence at RWTH Aachen. The university provides structured programs including fellowships, career workshops, and mentorship to promote academic independence. The Excellence Strategy focuses on transparent recruitment, balanced workload, and networking opportunities. Doctoral candidates receive tailored training in research skills, responsible conduct, and interdisciplinary collaboration. These initiatives help nurture the next generation of leaders in science and innovation.
""",
    "Talent and Diversity - Inclusive Academic Culture": """
RWTH promotes an inclusive academic culture that respects and values diversity across all dimensions including ethnicity, disability, and socio-economic background. The Excellence Strategy targets bias reduction through awareness campaigns, training programs, and accessible infrastructure improvements. Support services such as counseling, language assistance, and peer groups ensure equitable participation. The university fosters a welcoming environment that maximizes the contributions of all community members to academic success and societal engagement.
""",
    "Talent and Diversity - Intercultural Competence Development": """
Intercultural competence is fostered at RWTH through curricular and extracurricular activities, supporting both international and domestic community members. Language courses, cultural workshops, and international networking events enhance mutual understanding and collaboration. The Excellence Strategy embeds intercultural communication skills as part of student and staff development. These efforts enable RWTH to thrive as a culturally diverse institution prepared for global challenges and cooperation.
""",

    # Cluster 5: Societal Impact and Sustainability
    "Societal Impact - Regional Innovation and Economic Development": """
RWTH Aachen plays a vital role in driving regional innovation and economic development, aligning with its Excellence Strategy priorities. The university collaborates with local businesses, government, and civil society to foster technology-driven startups and enhance workforce skills. Innovation parks and knowledge transfer offices serve as catalysts for entrepreneurship and commercialization. Social responsibility and sustainable economic growth are promoted through inclusive policies and community engagement. RWTH’s impact extends beyond academia, contributing significantly to the prosperity of the region.
""",
    "Societal Impact - Sustainable Urban Development Research": """
Research on sustainable urban development is a key societal challenge addressed by RWTH through interdisciplinary approaches. Combining expertise in architecture, civil engineering, environmental science, and social sciences, RWTH develops innovative solutions for smart cities, green infrastructures, and mobility systems. The Excellence Strategy supports projects that emphasize resilience, climate adaptation, and citizen participation. Collaborations with city administrations and planners facilitate real-world implementation. This focus enhances RWTH’s contributions to global urban sustainability goals.
""",
    "Societal Impact - Public Engagement and Science Communication": """
RWTH prioritizes public engagement and science communication as part of its mission in the Excellence Strategy. The university organizes events, open lectures, and outreach programs designed to increase science literacy and dialogue with diverse audiences. Digital platforms and media collaborations extend reach and inclusivity. Scientists are encouraged and trained to communicate their research in accessible language, fostering trust and societal relevance. These initiatives strengthen the relationship between RWTH and the wider community, promoting mutual benefit.
""",
    "Societal Impact - Ethics and Responsibility in Research": """
Ethics and social responsibility are embedded in RWTH’s research activities under its Excellence Strategy. The university maintains rigorous review processes for research involving human subjects, animal welfare, and data protection. Interdisciplinary ethics committees provide guidance on emerging issues such as AI ethics and sustainability. Training programs raise awareness among staff and students about responsible research conduct. RWTH aims to balance scientific advancement with ethical considerations to ensure positive societal outcomes.
""",
    "Societal Impact - Climate Action and Environmental Responsibility": """
Climate action is a foundational component of RWTH Aachen’s strategy for sustainable development. The university integrates research, teaching, and operations to reduce environmental impact and foster climate resilience. Projects address renewable energy, climate modeling, sustainable materials, and policy analysis. Student initiatives and collaborations with NGOs amplify awareness and action on campus and beyond. RWTH commits to measurable targets for carbon neutrality and resource efficiency, contributing actively to global climate protection efforts.
""",

    # Cluster 6: Governance and Strategic Development
    "Governance and Strategy - Excellence Strategy Implementation": """
RWTH Aachen employs a comprehensive governance structure to implement its Excellence Strategy, involving multiple stakeholder groups. Transparent decision-making processes, clear responsibilities, and regular evaluation ensure alignment with strategic goals. Steering committees, working groups, and advisory boards coordinate activities related to research, teaching, and infrastructure. The strategy emphasizes agility to adapt to emerging trends and funding opportunities. Effective governance underpins RWTH’s sustained trajectory toward academic and institutional excellence.
""",
    "Governance and Strategy - Strategic Partnerships and Collaborations": """
Strategic partnerships form a key element in RWTH’s approach to strengthening research and education under the Excellence Strategy. Collaborations with leading universities, research institutions, and industry players worldwide enhance knowledge exchange and resource mobilization. International consortia and joint projects widen RWTH’s impact and visibility. The university actively pursues long-term alliances that align with its priority fields and values. These partnerships create synergies that drive innovation and global competitiveness.
""",
    "Governance and Strategy - Data-Driven Decision Making": """
Data-driven decision making is increasingly integrated into RWTH’s governance practices to optimize institutional performance. The Excellence Strategy promotes adoption of analytics tools for monitoring research outputs, teaching quality, and resource utilization. Data transparency supports evidence-based policy development and strategic planning. Dashboards and reporting systems enable real-time insights for leadership and faculty. This approach enhances accountability, efficiency, and continuous improvement within the university.
""",
    "Governance and Strategy - Risk Management in Research and Education": """
Proactive risk management is embedded in RWTH’s strategic planning aligned with its Excellence Strategy. The university identifies and mitigates risks related to funding fluctuations, regulatory changes, and technological disruptions. Risk assessment frameworks guide project selection and resource allocation to ensure sustainable results. Crisis preparedness plans address scenarios including data breaches, safety incidents, and reputational risks. Robust management of risks strengthens RWTH’s resilience and capacity to maintain excellence under changing conditions.
""",
    "Governance and Strategy - Quality Assurance and Accreditation": """
Quality assurance mechanisms at RWTH ensure high standards in research and education consistent with the Excellence Strategy. Periodic internal and external reviews assess teaching effectiveness, research impact, and administrative processes. Accreditation procedures validate degree programs meeting national and international criteria. Continuous feedback loops engage faculty, students, and stakeholders to foster improvement. These processes support RWTH’s commitment to excellence, transparency, and competitiveness in a dynamic academic environment.
""",

    # Cluster 7: Digital Innovation and Industry 4.0
    "Digital Innovation - Industry 4.0 and Smart Manufacturing": """
Industry 4.0 represents a transformative focus area within RWTH’s research and innovation agenda. The university drives advances in smart manufacturing, cyber-physical systems, and automation technologies that enhance industrial productivity and flexibility. Cross-disciplinary collaborations connect electrical engineering, computer science, and mechanical engineering. RWTH’s facilities enable prototyping and testing of innovative solutions alongside industrial partners. The Excellence Strategy supports scaling these technologies for practical deployment, strengthening the manufacturing sector regionally and globally.
""",
    "Digital Innovation - Internet of Things (IoT) Research and Application": """
The Internet of Things (IoT) ecosystem is a priority research domain at RWTH Aachen, addressing connectivity, sensor technology, and data analytics. The Excellence Strategy fosters integrated projects that apply IoT solutions in smart cities, healthcare, logistics, and energy management. RWTH encourages collaboration with startups and large enterprises to accelerate market-ready innovations. Emphasis on security, interoperability, and sustainability guides development efforts. IoT research contributes to RWTH’s ambition to lead in digital transformation and societal progress.
""",
    "Digital Innovation - Cybersecurity and Data Protection": """
Cybersecurity and data protection are critical components of RWTH’s digital innovation initiatives within the Excellence Strategy. Research focuses on cryptography, secure software development, and privacy-preserving technologies. RWTH maintains robust IT infrastructure to protect academic and administrative data integrity. Collaboration with government agencies and industry enhances response capacity to emerging cyber threats. Training programs build awareness among students and staff, ensuring a resilient digital ecosystem aligned with ethical and legal standards.
""",
    "Digital Innovation - Big Data Analytics and Cloud Computing": """
RWTH leverages big data analytics and cloud computing to enable cutting-edge research and flexible educational delivery. The Excellence Strategy supports development of scalable data platforms and high-performance computing resources. Multidisciplinary projects utilize large datasets for insights in areas such as physics, social sciences, and urban planning. Cloud services facilitate virtual labs, collaborative tools, and remote access. These digital capabilities expand RWTH’s research horizons and enhance operational efficiency across the institution.
""",
    "Digital Innovation - Robotics and Autonomous Systems": """
Robotics and autonomous systems research at RWTH advances automation in manufacturing, mobility, and service sectors. The Excellence Strategy funds projects integrating AI, control theory, and sensor technologies to create intelligent machines capable of complex tasks. RWTH’s robotic labs provide environments for testing and collaboration with industry. Ethical implications and human-robot interaction studies complement technical development. This research area strengthens RWTH’s position as a leader in next-generation intelligent systems.
""",

    # Cluster 8: Health Sciences and Medical Research
    "Health Sciences - Translational Medicine and Clinical Research": """
RWTH emphasizes translational medicine to bridge basic research with clinical applications under the Excellence Strategy. Interdisciplinary teams integrate molecular biology, imaging, and patient care to accelerate development of new diagnostics and therapies. The university collaborates with university hospitals and biotech companies for clinical trials and innovation transfer. Infrastructure investments facilitate data sharing and advanced clinical research methodologies. This approach enhances RWTH’s impact on health outcomes and medical advancement.
""",
    "Health Sciences - Precision Medicine and Genomics": """
Precision medicine is prioritized within RWTH’s health sciences research to tailor treatments based on individual genetic profiles. Projects focus on genomics, bioinformatics, and biomarker discovery. The Excellence Strategy supports integration of big data from patient cohorts and experimental studies to drive personalized healthcare solutions. RWTH collaborates with national and international consortia to enhance research scale and quality. These efforts contribute to breakthroughs in cancer therapy, rare diseases, and preventive medicine.
""",
    "Health Sciences - Public Health and Epidemiology": """
Public health and epidemiology research at RWTH addresses population health challenges and disease prevention strategies. The Excellence Strategy promotes data-driven studies on health behavior, environmental impacts, and health policy evaluation. Interdisciplinary cooperation with social sciences and engineering enriches research methodologies. Results inform evidence-based policy making and community health interventions. RWTH aims to strengthen its role as a contributor to improving public health regionally and globally.
""",
    "Health Sciences - Biomedical Informatics and Health IT": """
Biomedical informatics and health IT are at the forefront of RWTH's efforts to modernize healthcare data management and analysis. The Excellence Strategy supports development of electronic health records, decision support systems, and telemedicine platforms. RWTH engages experts in computer science, medicine, and law to ensure security, interoperability, and usability of healthcare technologies. Collaboration with hospitals enhances practical application and impacts patient care quality. This domain exemplifies the integration of digital innovation with medicine.
""",
    "Health Sciences - Neurosciences and Cognitive Research": """
Neurosciences and cognitive research represent a dynamic field at RWTH, investigating brain function, neurodegenerative diseases, and cognition. The Excellence Strategy promotes cutting-edge experimental techniques including neuroimaging, electrophysiology, and computational modeling. Multidisciplinary teams combine psychology, biology, engineering, and computer science perspectives. Research outcomes contribute to clinical treatments, rehabilitation, and artificial intelligence developments inspired by human cognition. RWTH fosters international collaboration and state-of-the-art facilities to sustain leadership in neuroscience.
""",

    # Additional Topics
    "Sustainability - Circular Economy Research": """
RWTH Aachen advances circular economy research aimed at reducing waste and promoting sustainable resource management. The Excellence Strategy supports projects developing innovative recycling technologies, sustainable materials, and product life-cycle analysis. Interdisciplinary collaborations connect engineering, economics, and environmental sciences. The university works with industry and policymakers to implement circular economy principles in practice. This research not only fosters sustainability but also creates new economic opportunities aligned with global environmental goals.
""",
    "Sustainability - Water Resource Management": """
Water resource management is a critical area of RWTH’s research focusing on sustainable use, pollution control, and climate adaptation strategies. The Excellence Strategy promotes integrated approaches combining hydrology, engineering, and environmental policy. RWTH develops advanced monitoring technologies, modeling tools, and treatment systems. Partnerships support implementation of sustainable water infrastructure in urban and rural contexts. This work contributes to global efforts addressing water scarcity and quality challenges.
""",
    "Sustainability - Environmental Policy and Governance": """
RWTH conducts leading research in environmental policy and governance aimed at supporting sustainable development transitions. The Excellence Strategy encourages analysis of regulatory frameworks, stakeholder participation, and economic incentives. Interdisciplinary teams link social sciences, law, and natural sciences to evaluate policy effectiveness and design innovative governance models. This research informs decision-making at local, national, and international levels. RWTH’s expertise strengthens societal capacity to address environmental challenges.
""",
    "Sustainability - Renewable Materials Innovation": """
Innovation in renewable materials is a strategic research area at RWTH supporting the transition away from fossil-based resources. The Excellence Strategy funds projects on bio-based polymers, composites, and sustainable manufacturing processes. Collaboration with industry partners accelerates development and commercialization. RWTH integrates materials science, chemistry, and engineering to create high-performance, eco-friendly products. These innovations contribute to sustainable production and consumption patterns worldwide.
""",
    "Sustainability - Energy Efficiency in Buildings": """
RWTH focuses on improving energy efficiency in buildings through research on insulation materials, smart control systems, and renewable energy integration. The Excellence Strategy promotes interdisciplinary approaches involving architecture, engineering, and computer science. Pilot projects demonstrate new technologies in real-world environments. RWTH’s sustainable building research supports efforts to reduce carbon emissions and operational costs. This field exemplifies the university’s commitment to practical sustainability solutions.
""",
    "Communication - Academic Networking and Conferences": """
Academic networking and conferences are vital tools RWTH employs to promote collaboration and knowledge dissemination. The Excellence Strategy supports organizing international symposia, workshops, and seminars that bring together experts from various fields. These events facilitate exchange of ideas, foster partnerships, and enhance RWTH’s visibility in the global academic community. Digital conference formats expand participation and accessibility. Strong networking underpins RWTH’s interdisciplinary research ambitions and reputation.
""",
    "Communication - Alumni Engagement and Outreach": """
Alumni engagement is a key aspect of RWTH’s strategy to build lasting relationships with graduates and leverage their expertise and networks. The Excellence Strategy funds initiatives such as alumni clubs, career mentoring programs, and fundraising campaigns. Regular communication and involvement opportunities keep alumni connected to campus developments. Their contributions enrich academic programs and provide valuable feedback. Active alumni networks enhance RWTH’s social capital and global reach.
""",
    "Communication - Media Relations and Publicity": """
RWTH cultivates proactive media relations and publicity to highlight its achievements and strategic priorities. The Excellence Strategy supports dedicated communication teams that produce press releases, feature articles, and social media content. Clear and timely communication boosts RWTH’s public image and helps attract students, faculty, and partners. Crisis communication protocols ensure preparedness for managing reputational risks. Effective media relations are essential to projecting RWTH as a top-tier research university.
""",
    "Communication - Internal Communication and Staff Engagement": """
Internal communication at RWTH seeks to foster transparency, collaboration, and motivation among staff and faculty. The Excellence Strategy promotes use of digital platforms, newsletters, and regular town-hall meetings. Feedback mechanisms allow staff input into decision-making and strategic development. Engagement initiatives support wellbeing and professional growth. Strong internal communication strengthens RWTH’s institutional culture and operational effectiveness.
""",
    "Communication - Science Education and Outreach for Schools": """
Teaching science to school students and promoting STEM education is a strategic outreach activity for RWTH. The Excellence Strategy funds programs that provide workshops, lab visits, and mentoring aimed at young learners. Partnerships with schools and educational organizations create pathways to higher education and research careers. These activities increase interest in science and technology and foster inclusion of underrepresented groups. RWTH’s school outreach efforts contribute to long-term societal innovation capacity.
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
