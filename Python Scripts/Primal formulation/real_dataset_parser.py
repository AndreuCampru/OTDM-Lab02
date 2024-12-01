import csv
import os

# Input and output folder paths
input_file = r"C:\Users\andre\Desktop\OTDM-Lab2\Data\real_dataet\diabetes_train.dat"
output_file_folder = r"C:\Users\andre\Desktop\OTDM-Lab2\AMPL files\Primal formulation"

# Regularization parameter values
nu_values = [0.1, 1, 10, 100]

# Ensure the output folder exists
os.makedirs(output_file_folder, exist_ok=True)

# Process the input file and generate outputs for each nu value
with open(input_file, "r") as infile:
    # Assuming the input file is comma-delimited
    reader = csv.reader(infile, delimiter=",")  
    lines = [row for row in reader if row]  # Read all rows, excluding empty lines

    # Initialize sets and parameters
    samples = range(1, len(lines) + 1)
    features = range(1, len(lines[0]) - 1 + 1)  # Features are all columns except the last

    # Generate a file for each nu value
    for nu in nu_values:
        output_file = os.path.join(output_file_folder, f"diabetes_train_nu_{nu}.dat")
        with open(output_file, "w") as outfile:
            # Write sets
            outfile.write("set M := " + " ".join(map(str, samples)) + ";\n")
            outfile.write("set N := " + " ".join(map(str, features)) + ";\n")

            # Write nu parameter
            outfile.write("param nu := " + str(nu) + ";\n")

            # Write X matrix (features)
            outfile.write("param X : " + " ".join(map(str, features)) + " :=\n")
            for i, row in enumerate(lines, start=1):
                values = row[:-1]  # Exclude the last column (target)
                outfile.write(f"  {i} " + " ".join(values) + "\n")
            outfile.write(";\n")

            # Write Y vector (targets)
            outfile.write("param Y :=\n")
            for i, row in enumerate(lines, start=1):
                label = row[-1]  # Last column is the target
                label = "-1" if label == "0" else label  # Replace '0' with '-1'
                outfile.write(f"  {i} {label}\n")
            outfile.write(";\n")

        print(f"Generated file: {output_file}")
