import json
import os
import numpy as np
import pandas as pd
import streamlit as st
# import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from svgpathtools import parse_path
from pathlib import Path

def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        "Basic example": full_app,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()
def full_app():
    with st.echo("below"):
        drawing_mode = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "point"),
        )
        stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
        if drawing_mode == 'point':
            point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
        stroke_color = st.sidebar.color_picker("Stroke color hex: ")
        bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
        bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])


        genre = st.radio(
        "Choose to use model",
        ('Draw by hand', 'Upload image'))

        if genre == 'Draw by hand':


            st.markdown('''
            Try to write a digit!
            ''')

            SIZE = 192
            canvas_result = st_canvas(
                fill_color='#000000',
                stroke_width=3,
                stroke_color='#FFFFFF',
                background_color='#000000',
                width=SIZE,
                height=SIZE,
                drawing_mode="freedraw",
                key='canvas')

            if canvas_result.image_data is not None:
                img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
                rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
                st.write('Model Input')
                st.image(rescaled)
        else:
            img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                img_array = np.array(image)

if __name__ == "__main__":
    main()

