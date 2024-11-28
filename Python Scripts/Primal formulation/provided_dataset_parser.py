input_file = r"C:\Users\andre\Documents\Data Science\Master in Data Science\Second Year\First Semester\Subjects\Optimization Techniques for Data Mining\2. Constrained Nonlinear Optimization\Laboratory Assignment\file.dat"
output_file = r"C:\Users\andre\Documents\Data Science\Master in Data Science\Second Year\First Semester\Subjects\Optimization Techniques for Data Mining\2. Constrained Nonlinear Optimization\Laboratory Assignment\svm_data.dat"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    lines = infile.readlines()

    # Initialize sets and parameters
    samples = range(1, len(lines) + 1)
    features = range(1, len(lines[0].split()) - 1 + 1)

    # Write sets
    outfile.write("set M := " + " ".join(map(str, samples)) + ";\n")
    outfile.write("set N := " + " ".join(map(str, features)) + ";\n")

    # Write X matrix
    outfile.write("param X : " + " ".join(map(str, features)) + " :=\n")
    for i, line in enumerate(lines, start=1):
        values = line.strip().split()
        outfile.write(f"  {i} " + " ".join(values[:-1]) + "\n")
    outfile.write(";\n")

    # Write Y vector
    outfile.write("param Y :=\n")
    for i, line in enumerate(lines, start=1):
        label = line.strip().split()[-1].replace('*', '')  # Remove '*' from Y
        outfile.write(f"  {i} {label}\n")
    outfile.write(";\n")
