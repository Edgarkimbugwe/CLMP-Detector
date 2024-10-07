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
        unsafe_allow_html=True,
    )

    st.write("\n")

    st.info(
        "This page presents the findings from a study aimed at visually "
        "differentiating a healthy cherry leaf from one affected by powdery "
        "mildew."
    )

    st.markdown(
        """
        <div style="text-align: center;">
            <ul style="list-style-type: disc; padding-left: 0; display: inline-block;">
                <li>For additional information, please visit and <strong>read</strong> the
                <a href="https://github.com/Edgarkimbugwe/CLMP-Detector/blob/main/README.md"
                target="_blank">Project README file</a>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("\n")

    # Hypothesis based on image inspection
    st.warning(
        "Our hypothesis is that cherry leaves affected by powdery mildew "
        "develop a white, cotton-like growth, typically appearing as light-green "
        "circular lesions that later form a white powdery texture. To capture "
        "these visual differences, images must be processed properly before "
        "they are used for machine learning, especially ensuring proper "
        "normalization for optimal feature extraction and model training."
    )

    st.write("\n")

    version = 'v1'

    # Average and variability images
    if st.checkbox("Difference between average and variability images"):
        avg_powdery_mildew = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png"
        )
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            "From these images, we observe that powdery mildew-infected leaves "
            "show more white stripes in the center. However, the variability "
            "patterns alone do not offer a clear visual distinction between "
            "healthy and infected leaves."
        )
        st.image(
            avg_powdery_mildew,
            caption='Powdery Mildew - Average and Variability',
            use_column_width=True,
        )
        st.image(
            avg_healthy,
            caption='Healthy Leaf - Average and Variability',
            use_column_width=True,
        )
        st.write("---")

    # Differences between average infected and healthy leaves
    if st.checkbox("Differences between average infected and average healthy leaves"):
        avg_diff_image = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "The difference between the average images of infected and healthy "
            "leaves doesn't show an obvious pattern that could be used to "
            "distinguish between the two categories."
        )
        st.image(avg_diff_image, caption='Difference between Average Images', use_column_width=True)

    # Image Montage of validation data
    if st.checkbox("Image Montage"):
        st.write("To refresh the montage, click on the 'Create Montage' button.")
        data_directory = 'inputs/cherryleaves_dataset/cherry-leaves'
        labels = os.listdir(data_directory + '/validation')
        label_to_display = st.selectbox(label="Select label", options=labels, index=0)

        if st.button("Create Montage"):
            image_montage(
                dir_path=data_directory + '/validation',
                label_to_display=label_to_display,
                nrows=6,
                ncols=3,
                figsize=(12, 27),
            )
        st.write("---")


# Image montage function
def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(dir_path + '/' + label_to_display)

        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.warning(
                f"Not enough images. There are {len(images_list)} images available."
            )
            return

        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(nrows * ncols):
            img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
            img_shape = img.shape
            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                f"{img_shape[1]}px x {img_shape[0]}px"
            )
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])

        plt.tight_layout()
        st.pyplot(fig=fig)
    else:
        st.error("The selected label doesn't exist.")
