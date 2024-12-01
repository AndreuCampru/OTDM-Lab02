import numpy as np

def calculate_hyperplane(lambda_values, X_train, y_train):
    """
    Calculates the weight vector (w) and bias (b) for the hyperplane.
    Args:
        lambda_values (numpy.ndarray): Array of lambda values.
        X_train (numpy.ndarray): Training data points.
        y_train (numpy.ndarray): Class labels for training data.
    Returns:
        tuple: (w, b) where w is the weight vector, and b is the bias.
    """
    # Compute weight vector (w)
    w = np.sum(lambda_values[:, None] * y_train[:, None] * X_train, axis=0)
    
    # Compute bias (b) using a support vector (lambda > 0)
    support_vector_idx = np.where(lambda_values > 1e-6)[0][0]  # First support vector
    b = y_train[support_vector_idx] - np.dot(w, X_train[support_vector_idx])
    
    return w, b

def read_lambda_results(file_path):
    """
    Reads lambda values from the results file.
    Args:
        file_path (str): Path to the lambda_results.txt file.
    Returns:
        numpy.ndarray: Array of lambda values.
    """
    #the results have tho following format: 0.0999999996543774 0.09999999981715924 0.09999999953084629 0.09999999977899471 0.09999999528924594
    with open(file_path, 'r') as f:
        data = f.read().strip().split()
    data = np.array(list(map(float, data)))

    return data


def calculate_objective(lambda_values, y, K):
    """
    Calculates the dual objective function value.
    Args:
        lambda_values (numpy.ndarray): Array of lambda values.
        y (numpy.ndarray): Class labels (+1, -1).
        K (numpy.ndarray): Kernel matrix.
    Returns:
        float: The dual objective function value.
    """
    term1 = np.sum(lambda_values)
    term2 = 0.5 * np.sum(
        lambda_values[:, None] * lambda_values[None, :] * y[:, None] * y[None, :] * K
    )
    return term1 - term2


def calculate_test_accuracy(lambda_values, X_train, y_train, X_test, y_test):
    """
    Calculates the test accuracy of the SVM.
    Args:
        lambda_values (numpy.ndarray): Array of lambda values.
        X_train (numpy.ndarray): Training data points.
        y_train (numpy.ndarray): Class labels for training data.
        X_test (numpy.ndarray): Test data points.
        y_test (numpy.ndarray): Class labels for test data.
    Returns:
        float: Test accuracy (percentage).
    """
    # Compute w from lambda
    w = np.sum(lambda_values[:, None] * y_train[:, None] * X_train, axis=0)
    
    # Compute bias b using a support vector (where lambda > 0)
    support_vector_idx = np.where(lambda_values > 1e-6)[0][0]
    b = y_train[support_vector_idx] - np.dot(w, X_train[support_vector_idx])
    
    # Make predictions on the test set
    predictions = np.sign(np.dot(X_test, w) + b)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test) * 100
    return accuracy



if __name__ == "__main__":
    # Paths to input files
    results_path = '../../AMPL files/Dual formulation/'
    lambda_file_provided = ["lambda_results_100.txt", "lambda_results_500.txt", "lambda_results_1000.txt", "lambda_results_2000.txt"]
    lambda_file_real = "lambda_results_diabetes.txt"


    data_path_provided = '../../Data/provided_dataset/'
    data_path_real = '../../Data/real_dataset/'
    training_file_prvided = ["file_100_train.dat", "file_500_train.dat", "file_1000_train.dat", "file_2000_train.dat"]
    training_file_real = "diabetes_train.dat"
    test_file_provided = ["file_100_test.dat", "file_500_test.dat", "file_1000_test.dat", "file_2000_test.dat"]
    test_file_real = "diabetes_test.dat"

    for i in range(4):
        lambda_file = results_path + lambda_file_provided[i]
        training_file = data_path_provided + training_file_prvided[i]
        test_file = data_path_provided + test_file_provided[i]

        # Read lambda values
        lambda_values = read_lambda_results(lambda_file)

        # Load training data and kernel matrix. delete '*' character
        training_data = np.loadtxt(training_file, dtype=str)
        for j in range(len(training_data)):
            training_data[j][-1] = training_data[j][-1].replace('*', '')
        training_data = training_data.astype(float)
        X_train = training_data[:, :-1]
        y_train = training_data[:, -1]  # Labels (+1, -1)

        # Kernel matrix (linear kernel for simplicity)
        K = np.dot(X_train, X_train.T)

        #print("hola\n", lambda_values,"\n" ,y_train,"\n", K)    
        
        # Calculate the objective function value
        obj_value = calculate_objective(lambda_values, y_train, K)
        print(f"Dual Objective Function Value for {lambda_file_provided[i]}: {obj_value}")

        # Calculate the hyperplane
        w, b = calculate_hyperplane(lambda_values, X_train, y_train)
        print(f"Weight Vector (w): {w}")
        print(f"Bias (b): {b}")

        # Load test data and change the value from -1 to 1 or vicavesa of the last column if it has a '*' character
        test_data = np.loadtxt(test_file, dtype=str)
        for j in range(len(test_data)):
            if '*' in test_data[j][-1]:
                test_data[j][-1] = test_data[j][-1].replace('*', '')
                if test_data[j][-1] == '-1.0':
                    test_data[j][-1] = '1.0'
                else:
                    test_data[j][-1] = '-1.0'
        test_data = test_data.astype(float)
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]


        # test_data = np.loadtxt(test_file)
        # X_test = test_data[:, :-1]
        # y_test = test_data[:, -1]

        # Calculate test accuracy
        accuracy = calculate_test_accuracy(lambda_values, X_train, y_train, X_test, y_test)
        print(f"Test Accuracy for {lambda_file_provided[i]}: {accuracy:.2f}%")


    # now for the real dataset
    lambda_file = results_path + lambda_file_real
    training_file = data_path_real + training_file_real
    test_file = data_path_real + test_file_real

    # Read lambda values
    lambda_values = read_lambda_results(lambda_file)

    # Load training data and kernel matrix (the training data has this format: 2.0,197.0,70.0,45.0,543.0,30.5,0.158,53.0,1.0)
    training_data = np.loadtxt(training_file, delimiter=',')
    X_train = training_data[:, :-1]
    y_train = training_data[:, -1]  # Labels (+1, -1)

    # Kernel matrix (linear kernel for simplicity)
    K = np.dot(X_train, X_train.T)

    # Calculate the objective function value
    obj_value = calculate_objective(lambda_values, y_train, K)
    print(f"Dual Objective Function Value for {lambda_file_real}: {obj_value}")

    # Calculate the hyperplane
    w, b = calculate_hyperplane(lambda_values, X_train, y_train)
    print(f"Weight Vector (w): {w}")
    print(f"Bias (b): {b}")

    # Load test data
    test_data = np.loadtxt(test_file, delimiter=',')
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    # Calculate test accuracy
    accuracy = calculate_test_accuracy(lambda_values, X_train, y_train, X_test, y_test)
    print(f"Test Accuracy for {lambda_file_real}: {accuracy:.2f}%")