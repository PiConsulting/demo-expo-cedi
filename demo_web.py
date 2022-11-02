# -- coding: utf-8 --`
import argparse
import os
import random
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from  PIL import Image, ImageEnhance
import numpy as np
# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import PNDMScheduler

image_logo = Image.open("logo/pi_logo_blanco.png")

def run(engine):
    st.title("Demo: Texto a Imagen")
    with st.form(key="request"):
        with st.sidebar:
            
            st.image(image_logo)

            prompt = st.text_area(label="Ingrese el texto")

            num_inference_steps = st.select_slider(
                label='iteraciones',
                options=range(1, 150),
                value=10
            )

            guidance_scale = st.select_slider(
                label='orientación',
                options=range(1, 21),
                value=5
            )

            strength = st.slider(
                label='robustez',
                min_value = 0.0,
                max_value = 1.0,
                value = 0.3
            )

            seed = st.number_input(
                label='semilla',
                min_value = 0,
                max_value = 2 ** 31,
                value = random.randint(0, 2 ** 31)
            )

            generate = st.form_submit_button(label = 'Generate')

        if prompt:
            np.random.seed(seed)
            image = engine(
                prompt = prompt,
                strength = strength,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale
            )
            st.image(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), width=512)

@st.cache(allow_output_mutation=True)
def load_engine(args):
    scheduler = PNDMScheduler(
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        skip_prk_steps = True,
        tensor_format="np"
    )
    engine = StableDiffusionEngine(
        model = args.model,
        scheduler = scheduler,
        tokenizer = args.tokenizer
    )
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # pipeline configure
    parser.add_argument("--model", type=str, default="bes-dev/stable-diffusion-v1-4-openvino", help="model name")
    # scheduler params
    parser.add_argument("--beta-start", type=float, default=0.00085, help="LMSDiscreteScheduler::beta_start")
    parser.add_argument("--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end")
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="LMSDiscreteScheduler::beta_schedule")
    # tokenizer
    parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="tokenizer")

    try:
        args = parser.parse_args()
    except SystemExit as e:
        # This exception will be raised if --help or invalid command line arguments
        # are used. Currently streamlit prevents the program from exiting normally
        # so we have to do a hard exit.
        os._exit(e.code)

    engine = load_engine(args)
    run(engine)
