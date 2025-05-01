# safetensor-visualizer

A Python-first CLI tool to instantly spin up a local visual UI for `.safetensors` files.

## Installation

```bash
pip install -e .
```

## Usage

```python
from safetensor_visualizer import safetensorvisualizer

safetensorvisualizer("path/to/your/model.safetensors")
```

This will open a web interface (usually at `http://localhost:7860/`) in your browser, allowing you to inspect the tensors within the file.
