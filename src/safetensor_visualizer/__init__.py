# Main entry point for safetensor-visualizer
import safetensors
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import os
import pandas as pd

# Helper function to estimate tensor size in MB
def get_tensor_size_mb(tensor):
    if hasattr(tensor, 'nbytes'):
        return tensor.nbytes / (1024 * 1024)
    # Fallback for tensors without nbytes (less common)
    return (np.prod(tensor.shape) * tensor.element_size()) / (1024 * 1024)

def create_gradio_ui(tensors):
    """Builds the Gradio UI components."""
    # Prepare data for the overview table
    overview_data = []
    for name, tensor in tensors.items():
        try:
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            memory_mb = f"{get_tensor_size_mb(tensor):.2f}"
            overview_data.append({
                "Name": name,
                "Shape": str(shape),
                "Dtype": dtype,
                "Memory (MB)": memory_mb,
                # "Layer Type": "TODO" # Placeholder
            })
        except Exception as e:
            print(f"Error processing tensor {name}: {e}")
            overview_data.append({
                "Name": name,
                "Shape": "Error",
                "Dtype": "Error",
                "Memory (MB)": "Error",
            })

    overview_df = pd.DataFrame(overview_data)

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Safetensor Visualizer")
        with gr.Tabs():
            with gr.TabItem("Tensor Overview"):
                gr.DataFrame(
                    overview_df,
                    headers=["Name", "Shape", "Dtype", "Memory (MB)"],
                    # row_count=(20, "dynamic"),
                    # col_count=(len(overview_df.columns), "fixed"),
                    wrap=True,
                    interactive=False # For now, make it non-interactive
                )
            # TODO: Add other tabs for Detail View, Summaries etc.

    return demo

def safetensorvisualizer(file_path: str):
    """Loads a .safetensors file and launches the Gradio UI."""
    if not os.path.exists(file_path):
        gr.Error(f"Error: File not found at {file_path}")
        return

    print(f"Loading {file_path}...")
    tensors = {}
    try:
        # Note: Device set to 'cpu'. Visualizations work on CPU tensors.
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            num_tensors = len(keys)
            print(f"Found {num_tensors} tensors. Loading...")
            # Add progress if needed, Gradio handles some of this
            for i, key in enumerate(keys):
                 # Basic progress printing
                if (i + 1) % 100 == 0 or i == num_tensors - 1:
                    print(f"  Loaded {i+1}/{num_tensors} tensors...")
                tensors[key] = f.get_tensor(key)

        print(f"Successfully loaded {len(tensors)} tensors.")

        # Build and launch Gradio UI
        print("Building Gradio UI...")
        demo = create_gradio_ui(tensors)
        print("Launching Gradio UI...")
        demo.launch()

    except Exception as e:
        print(f"Error loading or processing file: {e}")
        gr.Error(f"Error loading or processing file: {e}")

# Example usage for basic testing (commented out)
# if __name__ == '__main__':
#     # --- IMPORTANT --- 
#     # Replace with a REAL .safetensors file path on your system for testing
#     # test_file = '/path/to/your/model.safetensors'
#     test_file = None # Set to None to avoid accidental runs without a path
#     # --------------- 

#     if test_file and os.path.exists(test_file):
#         safetensorvisualizer(test_file)
#     elif test_file:
#         print(f"Test file '{test_file}' not found. Please provide a valid path.")
#     else:
#          print("Please uncomment and set 'test_file' in __init__.py to a valid .safetensors path for testing.")
