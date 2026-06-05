import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.title("🎨 AI Image Generator")

@st.cache_resource
def load_model():

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float32
    )

    return pipe

pipe = load_model()

prompt = st.text_input(
    "Enter Prompt",
    "A dog wearing black sunglasses"
)

if st.button("Generate"):

    with st.spinner("Generating..."):

        image = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=1
        ).images[0]

        st.image(image)
