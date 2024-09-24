import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file

# Function to plot the prediction probabilities for each class (Healthy, Infected) as a bar chart using Plotly
def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results
    """

    prob_per_class = pd.DataFrame(
        data=[0, 0],
        index={'Healthy': 0, 'Infected': 1}.keys(),
        columns=['Probability']
    )
    prob_per_class.loc[pred_class] = pred_proba
    for x in prob_per_class.index.to_list():
        if x not in pred_class:
            prob_per_class.loc[x] = 1 - pred_proba
    prob_per_class = prob_per_class.round(3)
    prob_per_class['Diagnostic'] = prob_per_class.index

    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y=prob_per_class['Probability'],
        range_y=[0, 1],
        width=600, height=300, template='seaborn')
    st.plotly_chart(fig)

# Function to resize input images to the average dimensions used during model training, ensuring compatibility for prediction
def resize_input_image(img, version):
    """
    Reshape image to average image size
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    my_image = np.expand_dims(img_resized, axis=0)/255

    return my_image