import os

# Input and output folder paths
input_file_folder = r"C:\Users\andre\Documents\Data Science\Master in Data Science\Second Year\First Semester\Subjects\Optimization Techniques for Data Mining\2. Constrained Nonlinear Optimization\Laboratory Assignment\OTDM-Lab02\Data\provided _dataset"
output_file_folder = r"C:\Users\andre\Documents\Data Science\Master in Data Science\Second Year\First Semester\Subjects\Optimization Techniques for Data Mining\2. Constrained Nonlinear Optimization\Laboratory Assignment\OTDM-Lab02\AMPL files\Primal formulation"

# Ensure the output folder exists
os.makedirs(output_file_folder, exist_ok=True)

# List of files and nu values
input_files = [f for f in os.listdir(input_file_folder) if f.startswith("file_") and f.endswith("_train.dat")]
nu_values = [0.1, 1, 10, 100]

# Process each file and each nu value
for input_file in input_files:
    input_file_path = os.path.join(input_file_folder, input_file)
    
    for nu in nu_values:
        # Create a unique output file name
        output_file_name = f"{os.path.splitext(input_file)[0]}_nu_{nu}.dat"
        output_file_path = os.path.join(output_file_folder, output_file_name)
        
        with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
            lines = infile.readlines()

            # Initialize sets and parameters
            samples = range(1, len(lines) + 1)
            features = range(1, len(lines[0].split()) - 1 + 1)

            # Write sets
            outfile.write("set M := " + " ".join(map(str, samples)) + ";\n")
            outfile.write("set N := " + " ".join(map(str, features)) + ";\n")

            # Write nu parameter
            outfile.write("param nu := " + str(nu) + ";\n")

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

print("Processing complete. Output files have been generated.")
