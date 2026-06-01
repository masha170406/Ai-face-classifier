import gradio as gr
import requests
import io
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

def send_api_inference_request(input_image):
    if input_image is None:
        return "Please upload an image.", "N/A", "N/A"
    
    try:
        # Convert Gradio numpy array wrapper to a PIL Image object
        pil_image = Image.fromarray(input_image).convert("RGB")
        
        # Downscale defense line: limits client payload before pushing to requests
        max_size = 600
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Compress and save to JPEG layout to reduce network overhead dramatically
        buffered_stream = io.BytesIO()
        pil_image.save(buffered_stream, format="JPEG", quality=85)
        image_bytes = buffered_stream.getvalue()
        
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        api_response = requests.post(API_URL, files=files)
        
        if api_response.status_code == 200:
            data = api_response.json()
            verdict = f"### Classification Verdict: **{data['prediction']}**"
            confidence_str = f"Confidence Score: **{data['confidence']}%**"
            
            breakdown_metrics = (
                f"**Client-Server Metadata Log:**\n"
                f"- API Processed Resolution: {data['input_metadata']['dimensions']} ({data['input_metadata']['category']})\n\n"
                f"**Probability Threshold Distribution:**\n"
                f"- Real Evaluation Index: {data['probabilities'].get('Real', 0.0)}%\n"
                f"- Fake Evaluation Index: {data['probabilities'].get('Fake', 0.0)}%"
            )
            return verdict, confidence_str, breakdown_metrics
        else:
            return f"Error: Server responded with code {api_response.status_code}", "N/A", "N/A"
            
    except requests.exceptions.ConnectionError:
        return "### Critical Connection Failure:\nEnsure the FastAPI backend script is executing concurrently on port 8000.", "N/A", "N/A"
# CLIENT INTERFACE DESIGN
with gr.Blocks(title="Deepfake Detection Sandbox Portal") as project_gui:
    gr.Markdown("# 🛡️ Deepfake Biometric Verification Interface")
    gr.Markdown(
        "**Academic Project Deliverable Submission**\n\n"
        "This client UI validates requirement compliance by preprocessing client image tensors "
        "and dispatching standardized multi-part image streams to an autonomous backend REST API."
    )
    
    with gr.Row():
        with gr.Column():
            input_source_ui = gr.Image(label="Drop Target File Here")
            submit_action_button = gr.Button("Verify Authenticity Index", variant="primary")
            
        with gr.Column():
            output_verdict_field = gr.Markdown("### Classification Verdict: Waiting for processing...")
            output_confidence_field = gr.Markdown("Confidence Score: N/A")
            output_metadata_field = gr.Markdown("**Metadata Log:** No session transactions logged.")

    submit_action_button.click(
        fn=send_api_inference_request,
        inputs=input_source_ui,
        outputs=[output_verdict_field, output_confidence_field, output_metadata_field]
    )

if __name__ == "__main__":
    project_gui.launch(share=True)