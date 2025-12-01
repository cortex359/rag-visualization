// Three.js RAG Visualizer
// Complete camera control and smooth interactions

class RAGVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.points = [];
        this.queryPoint = null;
        this.neighborPoints = [];
        this.allData = [];
        this.showNeighbors = false;
        this.showLegend = true;
        this.nNeighbors = 5;
        this.queryText = '';
        this.legendGroup = null;

        // Camera movement
        this.cameraVelocity = new THREE.Vector3();
        this.cameraRotation = { yaw: 0, pitch: 0 };
        this.keys = {};

        // Mouse interaction
        this.isDragging = false;
        this.previousMousePosition = { x: 0, y: 0 };
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();

        // Colors for different documents
        this.docColors = {};
        this.colorPalette = [
            0xFF6B6B, 0x4ECDC4, 0x45B7D1, 0xFFA07A, 0x98D8C8,
            0xF7DC6F, 0xBB8FCE, 0x85C1E2, 0xF8B500, 0x6C5CE7,
            0xA29BFE, 0xFD79A8, 0xFDCB6E, 0x00B894, 0xFF7675
        ];

        // FPS counter
        this.lastTime = performance.now();
        this.frames = 0;

        // Tooltip
        this.tooltip = null;
        this.hoveredPoint = null;

        this.init();
    }

    async init() {
        this.setupScene();
        this.setupEventListeners();
        await this.loadData();
        this.animate();
    }

    setupScene() {
        const container = document.getElementById('canvas-container');

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0xffffff, 0.8);
        pointLight.position.set(10, 10, 10);
        this.scene.add(pointLight);

        // Grid helper
        const gridHelper = new THREE.GridHelper(20, 20, 0x2a2a2a, 0x1a1a1a);
        this.scene.add(gridHelper);

        // Axes helper
        const axesHelper = new THREE.AxesHelper(5);
        this.scene.add(axesHelper);

        // Window resize
        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    setupEventListeners() {
        // Get tooltip element
        this.tooltip = document.getElementById('tooltip');

        // Keyboard
        window.addEventListener('keydown', (e) => {
            // Don't capture keys when typing in input
            if (document.activeElement.tagName === 'INPUT') return;
            this.keys[e.key.toLowerCase()] = true;
        });

        window.addEventListener('keyup', (e) => {
            this.keys[e.key.toLowerCase()] = false;
        });

        // Mouse orbit
        this.renderer.domElement.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.previousMousePosition = { x: e.clientX, y: e.clientY };
        });

        window.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const deltaX = e.clientX - this.previousMousePosition.x;
                const deltaY = e.clientY - this.previousMousePosition.y;

                this.cameraRotation.yaw -= deltaX * 0.005;
                this.cameraRotation.pitch -= deltaY * 0.005;

                // Clamp pitch
                this.cameraRotation.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.cameraRotation.pitch));

                this.previousMousePosition = { x: e.clientX, y: e.clientY };
            }

            // Update mouse for raycasting
            this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

            // Check for hover over points
            this.checkHover(e.clientX, e.clientY);
        });

        window.addEventListener('mouseup', () => {
            this.isDragging = false;
        });

        // Mouse zoom
        this.renderer.domElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomSpeed = 0.1;
            const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
            this.camera.position.addScaledVector(forward, e.deltaY > 0 ? -zoomSpeed : zoomSpeed);
        });

        // UI Controls
        const queryInput = document.getElementById('query-input');
        queryInput.addEventListener('input', (e) => {
            this.queryText = e.target.value;
            this.handleQuery();
        });

        document.getElementById('show-neighbors-btn').addEventListener('click', () => {
            this.showNeighbors = !this.showNeighbors;
            const btn = document.getElementById('show-neighbors-btn');
            btn.classList.toggle('active', this.showNeighbors);
            document.getElementById('sidebar').classList.toggle('visible', this.showNeighbors);
            if (this.showNeighbors) {
                this.handleQuery();
            } else {
                this.clearNeighbors();
            }
        });

        document.getElementById('toggle-legend-btn').addEventListener('click', () => {
            this.showLegend = !this.showLegend;
            const btn = document.getElementById('toggle-legend-btn');
            btn.classList.toggle('active', this.showLegend);
            if (this.legendGroup) {
                this.legendGroup.visible = this.showLegend;
            }
        });

        document.getElementById('reset-camera-btn').addEventListener('click', () => {
            this.resetCamera();
        });

        const slider = document.getElementById('n-neighbors-slider');
        slider.addEventListener('input', (e) => {
            this.nNeighbors = parseInt(e.target.value);
            document.getElementById('n-neighbors-value').textContent = this.nNeighbors;
            if (this.showNeighbors && this.queryText.length >= 3) {
                this.handleQuery();
            }
        });
    }

    async loadData() {
        try {
            const response = await fetch('/api/data');
            const data = await response.json();

            this.allData = data.points;

            // Assign colors to documents
            const uniqueDocs = [...new Set(data.points.map(p => p.document))];
            uniqueDocs.forEach((doc, i) => {
                this.docColors[doc] = this.colorPalette[i % this.colorPalette.length];
            });

            // Create points
            this.createPoints(data.points);

            // Create legend
            this.createLegend(uniqueDocs);

            // Hide loading
            document.getElementById('loading').style.display = 'none';

            // Update stats
            document.getElementById('point-count').textContent = data.points.length;

        } catch (error) {
            console.error('Error loading data:', error);
            document.getElementById('loading').innerHTML = '<div style="color: #ff4444;">Error loading data</div>';
        }
    }

    createPoints(data) {
        data.forEach(point => {
            // Use BoxGeometry for square-ish appearance
            const geometry = new THREE.BoxGeometry(0.12, 0.12, 0.12);
            const material = new THREE.MeshLambertMaterial({
                color: this.docColors[point.document],
                emissive: this.docColors[point.document],
                emissiveIntensity: 0.2
            });

            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(point.position[0], point.position[1], point.position[2]);
            cube.userData = point;

            // Random rotation for variety
            cube.rotation.set(
                Math.random() * Math.PI,
                Math.random() * Math.PI,
                Math.random() * Math.PI
            );

            this.scene.add(cube);
            this.points.push(cube);
        });
    }

    createLegend(uniqueDocs) {
        this.legendGroup = new THREE.Group();

        uniqueDocs.forEach((doc, i) => {
            // This is a simple legend - in a real implementation you might want
            // to render text sprites or use HTML overlay
            const geometry = new THREE.SphereGeometry(0.15, 16, 16);
            const material = new THREE.MeshBasicMaterial({
                color: this.docColors[doc]
            });

            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(-8, 8 - i * 0.5, 0);

            this.legendGroup.add(sphere);
        });

        this.scene.add(this.legendGroup);
        this.legendGroup.visible = this.showLegend;
    }

    async handleQuery() {
        if (this.queryText.length < 3) {
            this.clearQuery();
            document.getElementById('info').textContent = '⏳ Type at least 3 characters to embed your query...';
            return;
        }

        try {
            // Embed query
            const queryResponse = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: this.queryText })
            });

            const queryData = await queryResponse.json();

            // Create/update query point
            this.createQueryPoint(queryData.position);

            // Get neighbors if enabled
            if (this.showNeighbors) {
                const neighborsResponse = await fetch('/api/neighbors', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: this.queryText, n: this.nNeighbors })
                });

                const neighborsData = await neighborsResponse.json();
                this.highlightNeighbors(neighborsData.neighbors);
                this.displayNeighborsList(neighborsData.neighbors);

                document.getElementById('info').textContent =
                    `✨ Query: '${this.queryText}' | Showing top ${this.nNeighbors} nearest neighbors`;
            } else {
                document.getElementById('info').textContent =
                    `✨ Query embedded: '${this.queryText}' - Golden diamond in 3D space!`;
            }

        } catch (error) {
            console.error('Error handling query:', error);
        }
    }

    createQueryPoint(position) {
        // Remove old query point
        if (this.queryPoint) {
            this.scene.remove(this.queryPoint);
        }

        // Create diamond shape (octahedron)
        const geometry = new THREE.OctahedronGeometry(0.25, 0);
        const material = new THREE.MeshLambertMaterial({
            color: 0xFFD700,
            emissive: 0xFFD700,
            emissiveIntensity: 0.5
        });

        this.queryPoint = new THREE.Mesh(geometry, material);
        this.queryPoint.position.set(position[0], position[1], position[2]);

        // Add glow effect
        const glowGeometry = new THREE.OctahedronGeometry(0.35, 0);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: 0xFFD700,
            transparent: true,
            opacity: 0.3
        });
        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        this.queryPoint.add(glow);

        this.scene.add(this.queryPoint);

        // Animate rotation
        this.queryPoint.userData.animate = true;
    }

    highlightNeighbors(neighbors) {
        // Clear previous highlights
        this.clearNeighbors();

        neighbors.forEach(neighbor => {
            // Find the corresponding point mesh
            const point = this.points.find(p => p.userData.id === neighbor.id);
            if (point) {
                // Store original color
                if (!point.userData.originalColor) {
                    point.userData.originalColor = point.material.color.getHex();
                }

                // Change to magenta
                point.material.color.setHex(0xFF00FF);
                point.material.emissive.setHex(0xFF00FF);
                point.material.emissiveIntensity = 0.5;

                // Scale up
                point.scale.set(1.5, 1.5, 1.5);

                this.neighborPoints.push(point);
            }
        });

        // Dim non-neighbor points
        this.points.forEach(point => {
            if (!this.neighborPoints.includes(point)) {
                point.material.opacity = 0.3;
                point.material.transparent = true;
            }
        });
    }

    clearNeighbors() {
        // Restore neighbor points
        this.neighborPoints.forEach(point => {
            if (point.userData.originalColor) {
                point.material.color.setHex(point.userData.originalColor);
                point.material.emissive.setHex(point.userData.originalColor);
                point.material.emissiveIntensity = 0.2;
            }
            point.scale.set(1, 1, 1);
        });

        // Restore all points opacity
        this.points.forEach(point => {
            point.material.opacity = 1.0;
            point.material.transparent = false;
        });

        this.neighborPoints = [];
    }

    clearQuery() {
        if (this.queryPoint) {
            this.scene.remove(this.queryPoint);
            this.queryPoint = null;
        }
        this.clearNeighbors();
    }

    displayNeighborsList(neighbors) {
        const list = document.getElementById('neighbors-list');
        list.innerHTML = '';

        neighbors.forEach((neighbor, i) => {
            const item = document.createElement('div');
            item.className = 'neighbor-item';

            const similarity = (neighbor.similarity * 100).toFixed(1);
            const distance = neighbor.distance.toFixed(2);

            item.innerHTML = `
                <div class="neighbor-header">
                    <span class="neighbor-rank">#${i + 1}</span>
                    <span class="neighbor-similarity">${similarity}% • dist: ${distance}</span>
                </div>
                <div class="neighbor-doc">${neighbor.document}</div>
                <div class="neighbor-text">${neighbor.text.substring(0, 150)}${neighbor.text.length > 150 ? '...' : ''}</div>
            `;

            list.appendChild(item);
        });
    }

    resetCamera() {
        this.camera.position.set(5, 5, 5);
        this.camera.lookAt(0, 0, 0);
        this.cameraRotation = { yaw: 0, pitch: 0 };
        this.cameraVelocity.set(0, 0, 0);
    }

    checkHover(mouseX, mouseY) {
        // Don't show tooltip while dragging
        if (this.isDragging) {
            this.hideTooltip();
            return;
        }

        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Check intersections with regular points only (not query or neighbor highlights)
        const intersects = this.raycaster.intersectObjects(this.points);

        if (intersects.length > 0) {
            const point = intersects[0].object;

            // Only show tooltip if it's a different point
            if (this.hoveredPoint !== point) {
                this.hoveredPoint = point;
                this.showTooltip(point, mouseX, mouseY);
            } else {
                // Update position if same point
                this.updateTooltipPosition(mouseX, mouseY);
            }
        } else {
            this.hideTooltip();
        }
    }

    showTooltip(point, mouseX, mouseY) {
        if (!this.tooltip || !point.userData) return;

        const data = point.userData;
        const docElement = this.tooltip.querySelector('.tooltip-doc');
        const textElement = this.tooltip.querySelector('.tooltip-text');

        docElement.textContent = data.document;

        // Truncate text if too long
        const text = data.text;
        textElement.textContent = text.length > 200 ? text.substring(0, 200) + '...' : text;

        this.tooltip.style.display = 'block';
        this.updateTooltipPosition(mouseX, mouseY);
    }

    updateTooltipPosition(mouseX, mouseY) {
        if (!this.tooltip) return;

        // Position tooltip near cursor but avoid edges
        const offset = 15;
        let left = mouseX + offset;
        let top = mouseY + offset;

        // Keep tooltip within viewport
        const tooltipRect = this.tooltip.getBoundingClientRect();
        if (left + tooltipRect.width > window.innerWidth) {
            left = mouseX - tooltipRect.width - offset;
        }
        if (top + tooltipRect.height > window.innerHeight) {
            top = mouseY - tooltipRect.height - offset;
        }

        this.tooltip.style.left = left + 'px';
        this.tooltip.style.top = top + 'px';
    }

    hideTooltip() {
        if (this.tooltip) {
            this.tooltip.style.display = 'none';
        }
        this.hoveredPoint = null;
    }

    updateCamera(delta) {
        const speed = 3.0 * delta;
        const rotSpeed = 1.5 * delta;

        // Keyboard movement (WASD, QE)
        const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
        const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
        const up = new THREE.Vector3(0, 1, 0);

        if (this.keys['w']) this.camera.position.addScaledVector(forward, speed);
        if (this.keys['s']) this.camera.position.addScaledVector(forward, -speed);
        if (this.keys['a']) this.camera.position.addScaledVector(right, -speed);
        if (this.keys['d']) this.camera.position.addScaledVector(right, speed);
        if (this.keys['q']) this.camera.position.addScaledVector(up, speed);
        if (this.keys['e']) this.camera.position.addScaledVector(up, -speed);

        // Arrow key rotation
        if (this.keys['arrowleft']) this.cameraRotation.yaw += rotSpeed;
        if (this.keys['arrowright']) this.cameraRotation.yaw -= rotSpeed;
        if (this.keys['arrowup']) this.cameraRotation.pitch += rotSpeed;
        if (this.keys['arrowdown']) this.cameraRotation.pitch -= rotSpeed;

        // Clamp pitch
        this.cameraRotation.pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, this.cameraRotation.pitch));

        // Apply rotation
        const quaternion = new THREE.Quaternion();
        const euler = new THREE.Euler(this.cameraRotation.pitch, this.cameraRotation.yaw, 0, 'YXZ');
        quaternion.setFromEuler(euler);
        this.camera.quaternion.copy(quaternion);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const currentTime = performance.now();
        const delta = (currentTime - this.lastTime) / 1000;
        this.lastTime = currentTime;

        // Update camera
        this.updateCamera(delta);

        // Animate query point
        if (this.queryPoint && this.queryPoint.userData.animate) {
            this.queryPoint.rotation.y += delta * 2;
            this.queryPoint.children[0].rotation.y -= delta * 1.5;
        }

        // FPS counter
        this.frames++;
        if (currentTime % 1000 < 20) {
            document.getElementById('fps').textContent = this.frames;
            this.frames = 0;
        }

        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new RAGVisualizer());
} else {
    new RAGVisualizer();
}
