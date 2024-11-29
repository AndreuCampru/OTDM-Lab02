import numpy as np
from sklearn.model_selection import train_test_split

# Read the data from the file
data = np.loadtxt('diabetes_raw.dat', delimiter=',')

# Change 0s to -1s in the last column
data[:, -1] = np.where(data[:, -1] == 0, -1, data[:, -1])

# Shuffle the data
np.random.shuffle(data)

# Split the data into train and test sets (80/20) using the library sikit-learn

train_data, test_data = train_test_split(data, test_size=0.2)

# Function to convert each number to its significant digits representation



# Save the train and test data to separate files. only save the significant digits.
np.savetxt('diabetes_train.dat', train_data, delimiter=',', fmt='%s')
np.savetxt('diabetes_test.dat', test_data, delimiter=',', fmt='%s')
