import numpy as np

# Load dataset (X: features, y: labels) from file svm_data.dat
with open('../../svm_data.dat', 'r') as f:
    data = f.readlines()

    # Initialize containers for X and y
X = []
y = []

# Parse the data
reading_X = False
reading_Y = False
for line in data:
    line = line.strip()
    if line.startswith("param X :"):
        reading_X = True
        continue
    if line.startswith("param Y :"):
        reading_X = False
        reading_Y = True
        continue
    if line == ";":
        reading_X = False
        reading_Y = False
        continue

    if reading_X:
        parts = line.split()
        X.append([float(value) for value in parts[1:]])
    if reading_Y:
        parts = line.split()
        y.append(float(parts[1]))

X = np.array(X)
y = np.array(y)

# Compute kernel matrix (linear kernel)
K = np.dot(X, X.T)

# Write to AMPL data file format
with open('kernel.dat', 'w') as f:
    f.write(f"param n := {len(X)};\n")
    f.write(f"param d := {X.shape[1]};\n")
    f.write("set POINTS := " + " ".join(map(str, range(1, len(X) + 1))) + ";\n")
    f.write("param y :=\n")
    for i, label in enumerate(y, start=1):
        f.write(f"{i} {label}\n")
    f.write(";\n")
    f.write("param X :=\n")
    for i, point in enumerate(X, start=1):
        f.write(f"[{i}, *] " + " ".join(map(str, point)) + "\n")
    f.write(";\n")
    f.write("param K :=\n")
    for i in range(len(X)):
        for j in range(len(X)):
            f.write(f"[{i+1}, {j+1}] {K[i, j]}\n")
    f.write(";\n")
