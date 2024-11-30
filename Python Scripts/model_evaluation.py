import numpy as np
import os

# Preprocess test data file to clean target values
def preprocess_test_data(test_file, clean_file):
    with open(test_file, "r") as infile, open(clean_file, "w") as outfile:
        for line in infile:
            # Remove the '*' character from the last column (target)
            parts = line.strip().split()
            parts[-1] = parts[-1].replace("*", "")  # Clean the target
            outfile.write(" ".join(parts) + "\n")

def load_test_data(clean_file):
    # Explicitly specify delimiter as comma
    data = np.loadtxt(clean_file, delimiter=",")  # Adjust delimiter based on your file format
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
    # Ask the user which case to run
    print("Choose the dataset to run:")
    print("1. Existing datasets (e.g., file_100_test.dat)")
    print("2. Diabetes dataset (diabetes_test.dat)")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        # Existing datasets
        test_files = {
            "100": "file_100_test.dat",
            "500": "file_500_test.dat",
            "1000": "file_1000_test.dat",
            "2000": "file_2000_test.dat",
        }

        results_files = [
            "100_0.1.txt", "100_1.txt", "100_10.txt", "100_100.txt",
            "500_0.1.txt", "500_1.txt", "500_10.txt", "500_100.txt",
            "1000_0.1.txt", "1000_1.txt", "1000_10.txt", "1000_100.txt",
            "2000_0.1.txt", "2000_1.txt", "2000_10.txt", "2000_100.txt",
        ]

        for results_file in results_files:
            dataset_size = results_file.split("_")[0]
            test_file = test_files[dataset_size]

            clean_test_file = f"clean_{test_file}"
            preprocess_test_data(test_file, clean_test_file)

            w, gamma, s = load_ampl_results(results_file)
            X_test, y_test = load_test_data(clean_test_file)

            y_pred = predict(X_test, w, gamma)

            accuracy = compute_accuracy(y_test, y_pred)
            print(f"Results file: {results_file}, Test file: {test_file}, Accuracy: {accuracy * 100:.2f}%")

    elif choice == "2":
        # Diabetes dataset
        test_file = "diabetes_test.dat"
        clean_test_file = "clean_diabetes_test.dat"
        preprocess_test_data(test_file, clean_test_file)

        results_files = ["diabetes_0.1_results.txt", "diabetes_1_results.txt", 
                         "diabetes_10_results.txt", "diabetes_100_results.txt"]

        for results_file in results_files:
            w, gamma, s = load_ampl_results(results_file)
            X_test, y_test = load_test_data(clean_test_file)

            y_pred = predict(X_test, w, gamma)

            accuracy = compute_accuracy(y_test, y_pred)
            print(f"Results file: {results_file}, Test file: {test_file}, Accuracy: {accuracy * 100:.2f}%")

    else:
        print("Invalid choice. Please restart the program and select either 1 or 2.")
