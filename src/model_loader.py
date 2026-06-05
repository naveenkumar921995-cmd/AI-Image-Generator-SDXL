import torch
from diffusers import DiffusionPipeline

@st.cache_resource
def load_model():

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    pipe.to("cuda")

    return pipe
