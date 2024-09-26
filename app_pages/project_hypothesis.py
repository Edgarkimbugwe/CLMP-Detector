import streamlit as st
import matplotlib.pyplot as plt

def project_hypothesis_body():
    st.write("---")
    
    st.write("### Hypothesis 1 and Validation")

    st.success(
        f"Infected cherry leaves exhibit distinctive visual symptoms that can be identified through image analysis."
    )
    st.info(
        f"Our hypothesis is that leaves affected by powdery mildew show a progression of symptoms, "
        f"starting with light-green circular lesions that evolve into a prominent white, cotton-like growth. "
        f"These visual markers are crucial for accurate classification and detection by the model."
    )
    st.write("To see examples of infected versus healthy leaves, please refer to the Leaves Visualizer tab.")

    st.warning(
        f"The model demonstrated an ability to learn these visual differences, allowing it to generalize effectively "
        f"to unseen data. By training on a diverse dataset, the model avoids overfitting and enhances its predictive performance."
    )

    st.write("\n")

    st.write("### Hypothesis 2 and Validation")

    st.success(
        f"Utilizing image normalization techniques enhances model performance in distinguishing between classes."
    )
    st.info(
        f"Properly normalizing images is essential for optimizing feature extraction and ensuring that the model learns relevant patterns. "
        f"This involves adjusting pixel values to a common scale, which can significantly improve classification accuracy."
    )
    st.warning(
        f"Through experimentation, we found that models trained with normalized images outperformed those trained on raw images, "
        f"highlighting the importance of preprocessing in machine learning workflows."
    )

    st.write("\n")

    st.write("### Hypothesis 3 and Validation")

    st.success(
        f"Incorporating augmented training data leads to improved model robustness."
    )
    st.info(
        f"We hypothesize that augmenting the training dataset with transformed images (e.g., rotations, flips, and brightness adjustments) "
        f"can enhance the model's ability to generalize to real-world conditions. This approach helps in preventing overfitting and boosts accuracy."
    )
    st.warning(
        f"Model performance metrics indicated a noticeable improvement when augmentation was applied, validating our hypothesis "
        f"that diverse training data is key to achieving higher classification accuracy."
    )

    st.write("---")

    st.markdown(
        f"<div style='text-align: center;'>"
        f"For further details, refer to the "
        f"<a href='https://github.com/Edgarkimbugwe/CLMP-Detector/blob/main/README.md'>Project README file</a>."
        f"</div>",
    unsafe_allow_html=True)