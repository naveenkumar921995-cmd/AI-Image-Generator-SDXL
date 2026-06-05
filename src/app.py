import streamlit as st
import torch
from diffusers import DiffusionPipeline

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI Image Generator")
st.write("Generate images using Stable Diffusion XL")

MODEL_ID = "stabilityai/sdxl-turbo"

@st.cache_resource
def load_model():

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipe.to(device)

    return pipe


pipe = load_model()

prompt = st.text_area(
    "Enter Prompt",
    placeholder="A dog wearing black sunglasses sitting on a beach"
)

steps = st.slider(
    "Inference Steps",
    1,
    10,
    4
)

if st.button("Generate Image"):

    with st.spinner("Generating image..."):

        image = pipe(
            prompt=prompt,
            num_inference_steps=steps
        ).images[0]

        st.image(
            image,
            caption="Generated Image",
            use_container_width=True
        )

        image.save("generated.png")

        with open("generated.png", "rb") as file:
            st.download_button(
                "Download Image",
                file,
                "generated.png",
                "image/png"
            )
