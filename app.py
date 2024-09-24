import streamlit as st
from app_pages.multipage import MultiPage

# Import the functions that generate the content for corresponding pages
from app_pages.page_summary import page_summary_body
from app_pages.leaves_visualizer import leaves_visualizer_body
from app_pages.powdery_mildew_detector import powdery_mildew_detector_body


# Create an instance of the MultiPage app with the title "CLMP Detector"
app = MultiPage(app_name="CLMP Detector")


# Add pages to the app, associating them with their corresponding functions
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Leaves Visualiser", leaves_visualizer_body)
app.add_page("Powdery Mildew detector", powdery_mildew_detector_body)


app.run()  # Run the app