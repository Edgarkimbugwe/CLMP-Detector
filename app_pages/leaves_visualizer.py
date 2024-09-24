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
    
    version = 'v1'

    # Average and variability images
    if st.checkbox("Difference between average and variability images"):
        avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            f"From these images, we observe that powdery mildew-infected leaves show more white stripes in the center. "
            f"However, the variability patterns alone do not offer a clear visual distinction between healthy and infected leaves."
        )
        st.image(avg_powdery_mildew, caption='Powdery Mildew - Average and Variability', use_column_width=True)
        st.image(avg_healthy, caption='Healthy Leaf - Average and Variability', use_column_width=True)
        st.write("---")

    # Differences between average infected and healthy leaves
    if st.checkbox("Differences between average infected and average healthy leaves"):
        avg_diff_image = plt.imread(f"outputs/{version}/avg_diff.png")
        
        st.warning(
            f"The difference between the average images of infected and healthy leaves doesn't show an obvious pattern "
            f"that could be used to distinguish between the two categories."
        )
        st.image(avg_diff_image, caption='Difference between Average Images', use_column_width=True)
    
