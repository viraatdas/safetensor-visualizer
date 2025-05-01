from safetensor_visualizer import safetensorvisualizer

# --- IMPORTANT ---
# Replace this with the actual path to your .safetensors file
safetensors_file_path = "/Users/viraat/Documents/projects/exla/bitnet-accelerator/bitnet_model_artifacts_b1.58-2B-4T/model.safetensors"
# ---------------

if safetensors_file_path:
    try:
        print(f"Attempting to visualize: {safetensors_file_path}")
        safetensorvisualizer(safetensors_file_path)
        print("Gradio UI should be launching in your browser (check terminal for URL like http://127.0.0.1:7860).")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("Please update 'safetensors_file_path' with a valid path to test.")
