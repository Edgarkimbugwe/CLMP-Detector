import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.markdown("---")
  
    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="font-size: 30px;"> Quick Project Summary </h1>
    </div>
    """,
    unsafe_allow_html=True)

    st.info(
        f"**General Information**\n\n"
        f"Cherry powdery mildew is a fungal disease caused by *Podosphaera clandestina*, affecting cherry trees primarily during warm, dry conditions. "
        f"The disease manifests as a white, powdery coating on the surfaces of leaves, young shoots, and sometimes fruit.\n\n"
        f"When left unchecked, powdery mildew can hinder plant growth, reduce photosynthetic efficiency, and lead to crop loss due to compromised fruit quality.\n\n"
        f"Both infected and healthy leaves were collected and analyzed. "
        f"The visual criteria for identifying infected leaves include:\n\n"
        f"* A white, powdery layer appearing on the upper leaf surface, which can later progress to a yellowing or curling of the leaves.\n"
        f"* Distortion of young shoots and potential russeting on fruit, ultimately affecting yield and overall quality."
        f"\n\n")
    
    st.success(
        f"**Business Requirements**\n\n"
        f"1. **Accurate Detection of Powdery Mildew on Cherry Leaves:** The model must effectively differentiate between healthy and infected leaves, enabling farmers and agriculturalists to detect powdery mildew at an early stage.\n\n"
        f"2. **High Model Performance:** To ensure reliability in agricultural applications, the model must achieve a target accuracy of 97% or higher.\n\n"
        f"3. **User-Friendly Dashboard for Prediction:** The solution should be integrated into a user-friendly dashboard that allows users to upload images and receive real-time predictions regarding the health status of the leaves."
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

    st.warning(
        f"**Project Dataset**\n\n"
        f"It is worth noting that the dataset contains 2104 healthy leaves and 2104 affected leaves "
        f"")