import csv

input_file = r"C:\Users\andre\Documents\Data Science\Master in Data Science\Second Year\First Semester\Subjects\Optimization Techniques for Data Mining\2. Constrained Nonlinear Optimization\Laboratory Assignment\diabetes.dat"
output_file = r"C:\Users\andre\Documents\Data Science\Master in Data Science\Second Year\First Semester\Subjects\Optimization Techniques for Data Mining\2. Constrained Nonlinear Optimization\Laboratory Assignment\svm_diabetes_data.dat"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    reader = csv.reader(infile)
    lines = list(reader)  # Read all rows into a list

    # Initialize sets and parameters
    samples = range(1, len(lines) + 1)
    features = range(1, len(lines[0]) - 1 + 1)  # Features are all columns except the last

    # Write sets
    outfile.write("set M := " + " ".join(map(str, samples)) + ";\n")
    outfile.write("set N := " + " ".join(map(str, features)) + ";\n")

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
