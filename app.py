import gradio as gr
from main import app as sentinel_app

# Create a minimal Gradio interface to satisfy Hugging Face's SDK requirement
dummy_ui = gr.Blocks()
with dummy_ui:
    gr.Markdown("### Sentinel X Backend")
    gr.Markdown("The AI Cloud API is currently online and processing requests.")

# Mount your fully-functional FastAPI app onto the Gradio server
app = gr.mount_gradio_app(sentinel_app, dummy_ui, path="/")