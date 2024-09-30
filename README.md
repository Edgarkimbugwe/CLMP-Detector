<p align="center">
<img src="./images/readme/clpm_banner.png" alt="clpm Logo">
</p>

## Table of Contents
1. [Dataset Content](#dataset-content)
2. [Business Requirements](#business-requirements)





### Deployed version at [clmp-detector-fc49f90c032e.herokuapp.com/](https://clmp-detector-fc49f90c032e.herokuapp.com/)

## Dataset Content
The dataset used for this project consists of 4208 featured images of cherry leaves sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves). They are categorized into two classes: **healthy leaves** and **leaves infected by powdery mildew**. The data was collected to assist in training a machine learning model capable of detecting the presence of powdery mildew on cherry leaves by examining visual symptoms.

### Structure of the Dataset

- **Images:** The dataset contains high-quality images of cherry leaves in RGB format.
- **Classes:** There are two distinct classes:
    - __Healthy:__ Cherry leaves that show no signs of infection.
    - __Powdery mildew:__ Leaves displaying visual signs of white powdery growth and circular lesions.

### Data Distribution

The dataset is split into three subsets for model training and evaluation:
- **Training Set:** Comprising 70% of the data, this set is used for training the model, allowing it to learn the patterns associated with healthy and infected leaves.
- **Validation Set:** 10% of the dataset is used to fine-tune the model during training, helping prevent overfitting.
- **Test Set:** The remaining 20% is reserved for testing the final model, ensuring that it can generalize to new, unseen data.

### Preprocessing Steps

Before using the images for model training, the following preprocessing steps were applied:

- **Resizing:** All images were resized to ensure uniform input size for the neural network.
- **Normalization:** The pixel values were normalized to improve the model’s ability to learn patterns effectively.
- **Augmentation:** Various data augmentation techniques such as flipping, rotating, and zooming were used to increase the robustness of the model by simulating different image perspectives.

## Business Requirements

On conducting an [interview with Marianne McGuineys](https://github.com/Edgarkimbugwe/CLMP-Detector/wiki/Business-understanding-interview) the head of IT and Innovation at Farmy & Foods, the primary business requirement for this project was to build a robust system for early detection of powdery mildew on cherry leaves. The system must accurately differentiate between healthy and infected leaves, enabling farmers and agriculturalists to take timely actions to protect crops. To ensure reliability in real-world applications, the model must achieve a target accuracy of at least 97%, which is critical for minimizing false positives and negatives.

Additionally, the system should be integrated into a user-friendly dashboard where users, with little to no technical background, can easily upload images and receive real-time predictions. The solution must be scalable to handle other crops or plant diseases in the future, and the platform must be web-based to allow access across various devices, including mobile and desktop platforms, with minimal maintenance or technical intervention required.

In otherwords the business requirements summarised are: 
- Accurate Detection of Powdery Mildew on Cherry Leaves
- High Model Performance
- User-Friendly Dashboard for Prediction