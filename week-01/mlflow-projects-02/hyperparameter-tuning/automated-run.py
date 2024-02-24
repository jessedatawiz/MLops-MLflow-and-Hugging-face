import itertools
import subprocess

# Define the hyperparameters and their values to try
hyperparameters = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 20, None]
}

# Generate all combinations of hyperparameters
all_combinations = list(itertools.product(*(hyperparameters[name] for name in hyperparameters)))

# Loop over all combinations and run experiments
for combination in all_combinations:
    params = " ".join(f"-P {name}={value}" for name, value in zip(hyperparameters.keys(), combination))
    command = f"mlflow run . {params}"
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

print("All experiments completed.")
