import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread

def ml_performance_metrics():
    version = 'v1'
    
    st.info(
        f"This page provides a detailed overview of how the dataset was divided, how the model performed on that data, and an explanation of the results."
    )

    st.write("\n")

    st.write("### Image Distribution per Set and Label")

    st.write("\n")

    st.warning(
        f"The dataset was divided into three sets:\n\n"
        f"- **Train set** (70%): Used to train the model by learning general patterns and making predictions.\n\n"
        f"- **Validation set** (10%): Helps tune the model's parameters after each epoch.\n\n"
        f"- **Test set** (20%): Used to evaluate the final performance of the model on new, unseen data."
    )
    
    st.write("\n")
    
    labels_distribution = plt.imread(f"outputs/{version}/leaves_dataset.png")
    st.image(labels_distribution, caption='Dataset Labels Distribution')

    st.write("\n")

    labels_distribution_pie = plt.imread(f"outputs/{version}/sets_distribution_pie.png")
    st.image(labels_distribution_pie, caption='Dataset Distribution Across Sets on a Pie Chart')

       
    st.write("---")