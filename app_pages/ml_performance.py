import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread


def ml_performance_metrics():
    st.write("---")

    version = 'v1'

    st.info(
        "This page provides a detailed overview of how the dataset was divided, "
        "how the model performed on that data, and an explanation of the results."
    )

    st.write("\n")

    st.write("### Image Distribution per Set and Label")

    st.write("\n")

    st.warning(
        "The dataset was divided into three sets:\n\n"
        "- **Train set** (70%): Used to train the model by learning general patterns "
        "and making predictions.\n\n"
        "- **Validation set** (10%): Helps tune the model's parameters after each epoch.\n\n"
        "- **Test set** (20%): Used to evaluate the final performance of the model "
        "on new, unseen data."
    )

    st.write("\n")

    labels_distribution = plt.imread(f"outputs/{version}/leaves_dataset.png")
    st.image(labels_distribution, caption='Dataset Labels Distribution')

    st.write("\n")

    labels_distribution_pie = plt.imread(
        f"outputs/{version}/sets_distribution_pie.png"
    )
    st.image(labels_distribution_pie, caption='Dataset Distribution Across Sets '
                                                'on a Pie Chart')

    st.write("---")

    st.write("### Model Performance Metrics")

    st.write("\n")

    model_clf_report = plt.imread(f"outputs/{version}/clf_report.png")
    st.image(model_clf_report, caption='Classification Report')

    st.write("\n")

    st.warning(
        "**Classification Report**\n\n"
        "- **Precision**: Measures the percentage of true positives out of all "
        "predicted positives.\n\n"
        "- **Recall**: Measures the percentage of actual positives correctly "
        "identified by the model.\n\n"
        "- **F1 Score**: Harmonic mean of precision and recall, balancing both "
        "metrics.\n\n"
        "- **Support**: Indicates the number of actual occurrences of each class "
        "in the dataset."
    )

    st.write("\n")

    model_roc_curve = plt.imread(f"outputs/{version}/roccurve.png")
    st.image(model_roc_curve, caption='ROC Curve')

    st.write("\n")

    st.warning(
        "**ROC Curve**\n\n"
        "The ROC curve visualizes the model's performance in distinguishing "
        "between classes. It plots the true positive rate (TPR) against the "
        "false positive rate (FPR). A good model achieves a high TPR with a low FPR."
    )

    st.write("\n")

    model_confusion_matrix = plt.imread(f"outputs/{version}/confusion_matrix.png")
    st.image(model_confusion_matrix, caption='Confusion Matrix')

    st.write("\n")

    st.warning(
        "**Confusion Matrix**\n\n"
        "A confusion matrix presents a table with predicted vs. actual values. "
        "A good model has a high count of true positives (TP) and true negatives "
        "(TN), and a low count of false positives (FP) and false negatives (FN)."
    )

    st.write("\n")

    model_loss_accuracy = plt.imread(f"outputs/{version}/model_history.png")
    st.image(model_loss_accuracy, caption='Model Loss/Accuracy Over Time')

    st.write("\n")

    st.warning(
        "**Model Performance (Loss & Accuracy)**\n\n"
        "- **Loss**: Measures the error for each prediction. Lower loss values "
        "indicate better model performance.\n\n"
        "- **Accuracy**: Measures how well the model's predictions match the "
        "true labels. Higher accuracy on validation data indicates the model "
        "generalizes well to unseen data."
    )

    st.write("\n")

    st.write("### Generalized Performance on Test Set")

    st.write("\n")

    test_evaluation = {'Loss': 0.0322, 'Accuracy': 0.9917}
    st.dataframe(pd.DataFrame(test_evaluation, index=['Metric']))

    st.write("---")

    st.markdown(
        "<div style='text-align: center;'>"
        "For further details, refer to the "
        "<a href='https://github.com/Edgarkimbugwe/CLMP-Detector/blob/main/README.md'>"
        "Project README file</a>.</div>",
        unsafe_allow_html=True,
    )
