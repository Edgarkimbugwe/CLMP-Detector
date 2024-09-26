import streamlit as st
import matplotlib.pyplot as plt

def project_hypothesis_body():
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