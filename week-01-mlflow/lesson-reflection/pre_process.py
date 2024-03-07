# pre_process.py

import pandas as pd

def pre_process(input_data):
    # Assuming input_data is a DataFrame
    processed_data = pd.get_dummies(input_data)
    # Include any other preprocessing steps here
    return processed_data
