import streamlit as st
from diffusers import DiffusionPipeline
import torch

st.set_page_config(
    page_title="AI Image Generator",
    layout="wide"
)

st.title("AI Image Generator")
st.write("Generate Images using Stable Diffusion XL")

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

pipe = load_model()

prompt = st.text_area(
    "Enter Prompt",
    height=120
)

if st.button("Generate Image"):

    with st.spinner("Generating..."):

        image = pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        st.image(image)

        image.save("generated.png")

        with open("generated.png", "rb") as file:
            st.download_button(
                label="Download Image",
                data=file,
                file_name="generated.png",
                mime="image/png"
            )
