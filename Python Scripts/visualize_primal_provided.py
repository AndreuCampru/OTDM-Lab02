import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data from the first dataset
data_1 = {
    "Dataset Size": [100, 100, 100, 100, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000],
    "Regularization Parameter (ν)": [0.1, 1, 10, 100, 0.1, 1, 10, 100, 0.1, 1, 10, 100, 0.1, 1, 10, 100],
    "Objective Function Value": [7.4451, 43.0065, 295.1790, 2672.9486, 28.8314, 196.2445, 1723.6300, 16909.7936,
                                 46.7932, 327.3729, 2926.8016, 28794.1560, 82.0406, 610.7647, 5652.2907, 55953.0776],
    "Test Accuracy (%)": [60, 90, 90, 85, 91, 95, 95, 96, 92, 94, 94, 94, 93.5, 93.5, 93.5, 93.5],
}

# Data from the second dataset (provided in the image)
data_2 = {
    "Dataset Size": [614, 614, 614, 614],
    "Regularization Parameter (ν)": [0.1, 1, 10, 100],
    "Objective Function Value": [32.6776, 324.7993, 3244.7780, 32444.5065],
    "Test Accuracy (%)": [78.57, 79.87, 79.87, 79.87],
}

# Create separate DataFrames
df_1 = pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)

# Pivot the DataFrames to format them for heatmaps
accuracy_pivot_1 = df_1.pivot(index="Dataset Size", columns="Regularization Parameter (ν)", values="Test Accuracy (%)")
objective_pivot_1 = df_1.pivot(index="Dataset Size", columns="Regularization Parameter (ν)", values="Objective Function Value")

accuracy_pivot_2 = df_2.pivot(index="Dataset Size", columns="Regularization Parameter (ν)", values="Test Accuracy (%)")
objective_pivot_2 = df_2.pivot(index="Dataset Size", columns="Regularization Parameter (ν)", values="Objective Function Value")

# Plot Test Accuracy Heatmap for data_1
plt.figure(figsize=(10, 6))
sns.heatmap(accuracy_pivot_1, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={'label': 'Test Accuracy (%)'})
plt.title("Test Accuracy Heatmap (Data 1)")
plt.xlabel("Regularization Parameter (ν)")
plt.ylabel("Dataset Size")
plt.show()

# Plot Objective Function Value Heatmap for data_1
plt.figure(figsize=(10, 6))
sns.heatmap(objective_pivot_1, annot=True, fmt=".2f", cmap="Greens", cbar_kws={'label': 'Objective Function Value'})
plt.title("Objective Function Value Heatmap (Data 1)")
plt.xlabel("Regularization Parameter (ν)")
plt.ylabel("Dataset Size")
plt.show()

# Plot Test Accuracy Heatmap for data_2
plt.figure(figsize=(8, 4))  # Adjusted size for smaller data_2
sns.heatmap(accuracy_pivot_2, annot=True, fmt=".2f", cmap="Oranges", cbar_kws={'label': 'Test Accuracy (%)'})
plt.title("Test Accuracy Heatmap")
plt.xlabel("Regularization Parameter (ν)")
plt.ylabel("Dataset Size")
plt.show()

# Plot Objective Function Value Heatmap for data_2
plt.figure(figsize=(8, 4))  # Adjusted size for smaller data_2
sns.heatmap(objective_pivot_2, annot=True, fmt=".2f", cmap="Greens", cbar_kws={'label': 'Objective Function Value'})
plt.title("Objective Function Value Heatmap")
plt.xlabel("Regularization Parameter (ν)")
plt.ylabel("Dataset Size")
plt.show()
