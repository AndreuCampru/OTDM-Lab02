import numpy as np
import os

def parse_file(file, data=None):
    # Initialize containers for X and y
    X = []
    y = []

    # Parse the data. Read for every line (0.124 0.007 0.389 0.267   -1.0) the last value is y and the rest are X
    for line in data:
        #delete the '*' character   
        line = line.replace('*', '')
        parts = line.split()
        X.append(list(map(float, parts[:-1])))
        y.append(int(float(parts[-1])))

    X = np.array(X)
    y = np.array(y)

    # Compute kernel matrix (linear kernel)
    K = X.dot(X.T)

    # Write to AMPL data file format
    with open(file, 'w') as f:
        print("X len",len(X))
        print("Y len",X.shape[1])
        f.write(f"param n := {len(X)};\n")
        f.write(f"param d := {X.shape[1]};\n")
        f.write("set POINTS := " + " ".join(map(str, range(1, len(X) + 1))) + ";\n")
        f.write("param y :=\n")
        for i, label in enumerate(y, start=1):
            f.write(f"{i} {label}\n")
        f.write(";\n")
        colums=" ".join(map(str, range(1, X.shape[1] + 1)))
        f.write("param X : "+colums+" :=\n")
        for i, point in enumerate(X, start=1):
            f.write(f"{i} " + " ".join(map(str, point)) + "\n")
            
        f.write(";\n")
        f.write("param K :=\n")
        for i in range(len(X)):
            for j in range(len(X)):
                f.write(f"[{i+1}, {j+1}] {K[i, j]}\n")
        f.write(";\n")


#read all files in the relative '../../Data/provided_dataset' folder with os
files = os.listdir('../../Data/provided_dataset')



#filter files that contain the word 'train'
train_files = [file for file in files if 'train' in file]

for file in train_files:
    with open(f'../../Data/provided_dataset/{file}', 'r') as f:
        data = f.readlines()

    parse_file(f'../../AMPL files/Dual formulation/parsed_{file}',data)


# now parse the real dataset
with open('../../Data/real_dataset/diabetes_train.dat', 'r') as f:
    data = f.readlines()

#substitute all the ',' for ' '
data = [line.replace(',', ' ') for line in data]
parse_file('../../AMPL files/Dual formulation/parsed_diabetes_train.dat',data)

    
