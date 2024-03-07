# post_process.py

def post_process(prediction_output):
    # Transform the model output into a more user-friendly format
    # For instance, convert probabilities to labels, apply thresholding, etc.
    processed_output = prediction_output.argmax(axis=1)  # Example for classification
    return processed_output
