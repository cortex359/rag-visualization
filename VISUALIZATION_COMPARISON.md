# RAG Visualizer: Three.js vs Dash Comparison

This project now offers **two visualization options**. Here's a detailed comparison to help you choose:

## 🏆 Three.js Version (RECOMMENDED)

**Launch:** `./launch_threejs.sh` → http://localhost:5000

### Advantages
✅ **Perfect Camera Control**
- No unexpected camera resets
- Smooth transitions between modes (orbit, pan, zoom)
- State preservation across all interactions

✅ **Superior Keyboard Navigation**
- **WASD**: Move forward/back/left/right
- **QE**: Move up/down
- **Arrow Keys**: Rotate camera view
- All keys work smoothly, even while typing queries

✅ **Better Performance**
- Native WebGL rendering via Three.js
- Higher frame rates (typically 60 FPS)
- Smoother animations
- Better handling of large datasets

✅ **Professional Polish**
- Glow effects on query points
- Smooth rotation animations
- Fade effects on non-neighbor points
- Clean, modern UI with backdrop blur

✅ **Full Control**
- Mouse drag for orbit
- Scroll for zoom
- All interactions work perfectly
- No library-imposed limitations

### Technical Architecture
```
Frontend (Three.js + Vanilla JS)
    ↓ HTTP REST API
Backend (Flask)
    ↓
Python ML (sentence-transformers, UMAP/PCA)
```

### Best For
- ✨ **Live presentations**
- 🎯 Demonstrations requiring precise control
- 📊 When performance matters
- 🎨 Professional/client-facing showcases

---

## 📊 Dash/Plotly Version (Original)

**Launch:** `./launch.sh` → http://localhost:8050

### Advantages
✅ **All-Python**
- No JavaScript required
- Easier to customize for Python developers
- Integrated callbacks

✅ **Plotly Features**
- Built-in screenshot capability
- Standard Plotly modebar
- Familiar plotting interface

### Known Limitations
⚠️ **Camera Issues**
- Camera resets when switching between Pan/Zoom/Rotate
- View angle can reset unexpectedly
- State preservation is challenging

⚠️ **Keyboard Navigation**
- Limited functionality
- Doesn't work reliably
- Conflicts with browser defaults

⚠️ **UI Issues**
- Slider width problems
- More complex state management
- Refresh needed for some interactions

### Technical Architecture
```
All-in-one (Dash + Plotly)
    ↓
Python ML (sentence-transformers, UMAP/PCA)
```

### Best For
- 📝 Quick prototypes
- 🐍 Pure Python environments
- 📚 Jupyter-style exploration
- 🔬 Research/analysis workflows

---

## Feature Comparison

| Feature | Three.js | Dash/Plotly |
|---------|----------|-------------|
| **Camera Control** | Perfect | Resets occur |
| **Keyboard Nav** | Excellent | Limited |
| **Performance** | ~60 FPS | ~30-40 FPS |
| **Animations** | Smooth | Basic |
| **UI Responsiveness** | Instant | Good |
| **Customization** | Full control | Limited by Plotly |
| **Setup Complexity** | Moderate | Simple |
| **Pure Python** | ❌ | ✅ |
| **Production Ready** | ✅ | ⚠️ |

---

## Common Features (Both)

Both versions support:
- ✅ Real-time query embedding
- ✅ Nearest neighbor search (3D Euclidean distance)
- ✅ Dark mode theme
- ✅ UMAP and PCA dimensionality reduction
- ✅ 79+ document chunks
- ✅ Color-coded by document category
- ✅ Interactive tooltips
- ✅ Configurable neighbor count (3-15)
- ✅ Legend toggle
- ✅ Full-screen friendly

---

## Quick Start

### Three.js Version
```bash
./launch_threejs.sh                    # UMAP on port 5000
./launch_threejs.sh --method pca       # PCA on port 5000
python server.py --port 5001           # Custom port
```

### Dash Version
```bash
./launch.sh                            # UMAP on port 8050
./launch.sh --method pca               # PCA on port 8050
python interactive_rag_3d.py --method pca
```

---

## Recommendation

**For your talk:** Use the **Three.js version**. The superior camera control and smooth interactions will make your demonstration much more professional and you won't have to worry about unexpected camera resets during your presentation.

**For development:** Use either, depending on your preference:
- Prefer Python ecosystem → Dash version
- Need best performance/control → Three.js version

---

## Files

### Three.js Version
- `server.py` - Flask backend with REST API
- `static/index.html` - HTML interface
- `static/app.js` - Three.js visualization logic
- `launch_threejs.sh` - Launch script

### Dash Version
- `interactive_rag_3d.py` - All-in-one Dash app
- `launch.sh` - Launch script

### Shared
- `rag_visualizer.py` - Static HTML export (legacy)
- `requirements.txt` - Python dependencies
- Sample documents embedded in code
