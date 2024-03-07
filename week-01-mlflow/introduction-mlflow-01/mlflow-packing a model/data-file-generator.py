import pandas as pd
import numpy as np

# Generate synthetic data for the exercise
np.random.seed(42)  # For reproducibility
X = np.linspace(-10, 10, 100)
y = 3*X + np.random.randn(100) * 2  # y = 3x + noise

# Create a DataFrame and save to CSV
data = pd.DataFrame({'X': X, 'y': y})
data_file_path = "data-file.csv"
data.to_csv(data_file_path, index=False)

print(data_file_path)