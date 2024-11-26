import numpy as np

# Preprocess test data file to clean target values
def preprocess_test_data(test_file, clean_file):
    with open(test_file, "r") as infile, open(clean_file, "w") as outfile:
        for line in infile:
            # Remove the '*' character from the last column (target)
            parts = line.strip().split()
            parts[-1] = parts[-1].replace("*", "")  # Clean the target
            outfile.write(" ".join(parts) + "\n")

# Load test data (features and labels)
def load_test_data(clean_file):
    data = np.loadtxt(clean_file)  # Assumes space-separated values
    X_test = data[:, :-1]  # All columns except the last
    y_test = data[:, -1]   # Last column is the label
    return X_test, y_test

# Predict using the SVM decision function
def predict(X, w, gamma):
    # Compute decision function
    decision_values = np.dot(X, w) + gamma
    # Apply sign function to get predictions
    return np.sign(decision_values)

# Compute accuracy
def compute_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    accuracy = correct / len(y_true)
    return accuracy

# Load weights (w), bias (gamma), and slack variables (s) from results.txt
def load_ampl_results(results_file):
    with open(results_file, "r") as f:
        lines = f.readlines()

    # Initialize containers
    w = {}
    s = {}
    gamma = None

    # Parse the results file
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse weights (w)
        if line.startswith("w["):
            index, value = line.replace("w[", "").replace("]", "").split("=")
            w[int(index.strip())] = float(value.strip())
        
        # Parse bias (gamma)
        elif line.startswith("gamma"):
            gamma = float(line.split("=")[1].strip())

        # Parse slack variables (s)
        elif line.startswith("s["):
            index, value = line.replace("s[", "").replace("]", "").split("=")
            s[int(index.strip())] = float(value.strip())

    # Convert dictionaries to arrays (sorted by index)
    w_array = np.array([w[i] for i in sorted(w.keys())])
    s_array = np.array([s[i] for i in sorted(s.keys())])

    return w_array, gamma, s_array

# Main script
if __name__ == "__main__":
    # File paths
    results_file = "original_results.txt"      # File containing weights, gamma, and slack variables
    test_file = "test_data.csv"       # Original test data file
    clean_test_file = "clean_test_data.csv"  # Cleaned test data file without "*"

    # Preprocess the test data to remove "*" from the target column
    preprocess_test_data(test_file, clean_test_file)

    # Load results and cleaned test data
    w, gamma, s = load_ampl_results(results_file)
    X_test, y_test = load_test_data(clean_test_file)

    # print("Weights (w):", w)
    # print("Bias (gamma):", gamma)
    # print("Slack variables (s):", s)

    # Make predictions
    y_pred = predict(X_test, w, gamma)

    # Compute and display accuracy
    accuracy = compute_accuracy(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
