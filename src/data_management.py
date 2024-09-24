import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime
import joblib


def download_dataframe_as_csv(dataframe):
    """Generate a CSV download link for the provided DataFrame."""
    timestamp = datetime.now().strftime("%d%b%Y_%Hh%Mmin%Ss")
    csv_content = dataframe.to_csv(index=False).encode()
    b64_encoded = base64.b64encode(csv_content).decode()
    download_link = (
        f'<a href="data:file/csv;base64,{b64_encoded}" download="Report_{timestamp}.csv" '
        f'target="_blank">Download Report</a>'
    )
    return download_link


def load_pkl_file(file_path):
    """Load a pickle file and return its content."""
    return joblib.load(file_path)
