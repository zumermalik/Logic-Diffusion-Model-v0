import gradio as gr
import torch
from logic_diffusion.modeling import SimpleUNet
from logic_diffusion.pipeline import LogicGuidedPipeline
from logic_diffusion.config import TrainingConfig
from logic_diffusion.logic import LogicConstraint, DifferentiableLogic

# 1. Load Model (Mock loader for v0 demo)
def load_model():
    config = TrainingConfig()
    model = SimpleUNet(image_channels=config.channels)
    # In a real scenario, you would load weights here:
    # model.load_state_dict(torch.load("logic_diffusion_v0.pt"))
    return model, config

model, config = load_model()
pipeline = LogicGuidedPipeline(model, config)

# 2. Define Generation Function
def generate_image(prompt, guidance_scale, logic_strictness):
    # For v0, we are simulating the logic constraint based on the slider
    # In v1, 'prompt' would be parsed into logical rules
    
    # Define a dummy constraint for the demo
    # Rule: "Pixel intensity must be independent of bias" (Simulated)
    def dummy_rule(x):
        return DifferentiableLogic.and_op(x.mean(), torch.tensor(0.5))

    constraint = LogicConstraint(dummy_rule, weight=logic_strictness)
    
    # Run Sampling
    # Note: 'n_samples=1' for demo speed
    generated_tensor = pipeline.sample(n_samples=1, constraints=[constraint])
    
    # Convert tensor to visible image (Heatmap for v0)
    img_data = generated_tensor[0].permute(1, 2, 0).numpy()
    
    # Normalize for display
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    return img_data

# 3. Build UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Logic Diffusion v0")
    gr.Markdown("Generate data distributions that adhere to strict logical constraints.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Logical Prompt (e.g., 'Ensure independence between X and Y')")
            slider_strictness = gr.Slider(0.0, 5.0, value=1.0, label="Logic Strictness (Lambda)")
            btn_generate = gr.Button("Generate Unbiased Sample", variant="primary")
        
        with gr.Column():
            output_plot = gr.Image(label="Generated Manifold Heatmap")

    btn_generate.click(fn=generate_image, inputs=[prompt_input, slider_strictness, slider_strictness], outputs=output_plot)

if __name__ == "__main__":
    demo.launch()
