import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random

def leaves_visualizer_body():
    st.markdown("---")
    
    # Page title and introduction
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 30px;"> Leaves Visualizer </h1>
    </div>
    """,
    unsafe_allow_html=True)

    st.write("\n")

    st.info(
        f"This page presents the findings from a study aimed at visually differentiating a healthy cherry leaf from one affected by powdery mildew."
    )

    st.markdown(
    """
    <div style="text-align: center;">
        <ul style="list-style-type: disc; padding-left: 0; display: inline-block;">
            <li>For additional information, please visit and <strong>read</strong> the 
            <a href="https://github.com/Edgarkimbugwe/CLMP-Detector/blob/main/README.md" target="_blank">Project README file</a>.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True)

    st.write("\n")
    
    # Hypothesis based on image inspection
    st.warning(
        f"Our hypothesis is that cherry leaves affected by powdery mildew develop a white, cotton-like growth, "
        f"typically appearing as light-green circular lesions that later form a white powdery texture. "
        f"To capture these visual differences, images must be processed properly before they are used for machine learning, "
        f"especially ensuring proper normalization for optimal feature extraction and model training."
    )

    st.write("\n")
    
