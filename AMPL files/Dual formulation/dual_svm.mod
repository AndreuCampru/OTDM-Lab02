# dual_svm.mod

# Parameters
set POINTS;                # Set of data points
param d;                   # Number of parameters
param n;                   # Number of points
param y {POINTS};          # Class labels (y_i = +1 or -1)
param X {POINTS,1..d};    # Data points in d-dimensional space
param K {POINTS, POINTS};  # Kernel matrix

# Variables
var lambda {POINTS} >= 0;   # Lagrange multipliers

# Objective Function (Dual SVM)
maximize Obj:
    sum {i in POINTS} lambda[i] - 
    0.5 * sum {i in POINTS, j in POINTS} lambda[i] * lambda[j] * y[i] * y[j] * K[i, j];

# Constraints
s.t. Equality_Constraint:
    sum {i in POINTS} lambda[i] * y[i] = 0;

# Upper bound constraint on lambda

param nu > 0;  
s.t. Upper_Bound {i in POINTS}: lambda[i] <= nu;