import torch
import gradio as gr
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

print("Loading model...")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print("Model loaded successfully")


def generate_image(
    prompt,
    negative_prompt,
    steps,
    guidance_scale
):

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale
    ).images[0]

    return image


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(
            label="Prompt",
            placeholder="A dog wearing black sunglasses sitting on a beach"
        ),
        gr.Textbox(
            label="Negative Prompt",
            value="blurry, low quality"
        ),
        gr.Slider(
            minimum=10,
            maximum=50,
            value=30,
            step=1,
            label="Inference Steps"
        ),
        gr.Slider(
            minimum=1,
            maximum=15,
            value=7.5,
            step=0.5,
            label="Guidance Scale"
        )
    ],
    outputs=gr.Image(label="Generated Image"),
    title="AI Image Generator",
    description="Generate images using Stable Diffusion XL"
)

demo.launch()
