## SVM problem in primal formulation

# sets
set N; # set of variables
set M; # set of points


# parameters
param X {M, N};           # data matrix
param Y {M};              # classes
param nu > 0 default 1.0; # regularization parameter


# variables
var w {N};      # weight vector
var gamma;      # bias
var s {M} >= 0; # slack variables


# objective function
minimize fobj: 
    0.5 * sum{j in N} (w[j]^2) + nu * sum {i in M} s[i];


# constraints
subject to c1 {i in M}:
    Y[i] * (sum {j in N} w[j] * X[i, j] + gamma) + s[i] >= 1;
    
subject to c2 {i in M}:
    s[i] >= 0;