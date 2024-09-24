import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Importing necessary functions from your modules
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def powdery_mildew_detector_body():

    st.markdown("---")
    
    # Page title and introduction
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 30px;"> Upload and Detect </h1>
    </div>
    """,
    unsafe_allow_html=True)

    st.write("\n")


    st.info(
        f"Upload images of cherry leaves to determine if they are affected by powdery mildew, and download a report of the analysis."
    )

    st.write("\n")

    st.markdown(
    """
    <div style="text-align: center;">
        <ul style="list-style-type: disc; padding-left: 0; display: inline-block;">
            <li>You can download a set of infected and healthy leaves for live prediction from 
            <a href="https://www.kaggle.com/datasets/codeinstitute/cherry-leaves" target="_blank">here</a>.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True)

    st.write("---")

    st.write(
        f"**Upload a clear image of a cherry leaf. Multiple selections are allowed.**"
    )
    images_buffer = st.file_uploader(' ', type=['png', 'jpg'], accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:
            img_pil = Image.open(image)
            st.info(f"Cherry leaf Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)  # Resize the image for the model
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)  # Predict

            plot_predictions_probabilities(pred_proba, pred_class)  # Visualize the prediction probabilities

            # Append results to the DataFrame
            df_report = df_report.append({"Name": image.name, 'Result': pred_class}, ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)  # Download report

