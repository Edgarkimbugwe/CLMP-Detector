import streamlit as st
import matplotlib.pyplot as plt

# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    # Initialize the MultiPage object with the app's name
    def __init__(self, app_name) -> None:

        # List to hold the pages (each page is a dictionary with title and function)
        self.pages = []
        self.app_name = app_name    # The name of the app

        # Set the page configuration (title and icon) for the Streamlit app
        st.set_page_config(
            page_title=self.app_name,
            page_icon="☘️")

    # Method to add a new page to the app
    def add_page(self, title, func) -> None:

        # Append a new dictionary with the page title and the function that will be run for that page
        self.pages.append({"title": title, "function": func})

    # Method to run the app and display the pages
    def run(self):

        # Display the app's main title
        st.title(self.app_name)

        # Create a sidebar with a radio button for navigation between the added pages
        # 'format_func' is used to show the 'title' of the page in the radio button
        page = st.sidebar.radio('Menu', self.pages,
                                format_func=lambda page: page['title'])
        
        # Call the function corresponding to the selected page to render its content
        page['function']()