import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the table
dataset_sizes = [100, 500, 1000, 2000]
regularization_params = [0.1, 1, 10, 100]
objective_values = [
    [7.4451, 43.0065, 295.1790, 2672.9486],
    [28.8314, 196.2445, 1723.6300, 16909.7935],
    [46.7932, 327.3729, 2926.8016, 28794.1560],
    [82.0405, 610.7647, 5652.2907, 55953.0774],
]
test_accuracies = [
    [97, 95, 95, 85],
    [87, 87, 91, 91],
    [96, 83.5, 85, 85.5],
    [77.25, 86, 87.5, 89],
]

# 614
# 0.1
# 32.6776
# 77.92%
# 1
# 324.7992
# 79.87%
# 10
# 3244.7779
# 79.22%
# 100
# 32444.5064
# 79.22%
objective_values_real_dataset = [32.6776, 324.7992, 3244.7779, 32444.5064]
test_accuracies_real_dataset = [77.92, 79.87, 79.22, 79.22]


# Convert data into numpy arrays
objective_values = np.array(objective_values)
test_accuracies = np.array(test_accuracies)

objective_values_real_dataset = np.array(objective_values_real_dataset)
test_accuracies_real_dataset = np.array(test_accuracies_real_dataset)

# # Create heatmaps
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Heatmap for Objective Function Value in green
# sns.heatmap(objective_values, annot=True, fmt=".2f",  ax=axes[0], cmap= "Greens",
#             xticklabels=regularization_params, yticklabels=dataset_sizes, cbar_kws={'label': 'Objective Value'})
# axes[0].set_title('Objective Function Value')
# axes[0].set_xlabel('Regularization Parameter')
# axes[0].set_ylabel('Dataset Size')

# # Heatmap for Test Accuracy in red
# sns.heatmap(test_accuracies, annot=True, fmt=".2f", ax=axes[1], cmap="Reds",
#             xticklabels=regularization_params, yticklabels=dataset_sizes, cbar_kws={'label': 'Test Accuracy (%)'})
# axes[1].set_title('Test Accuracy')
# axes[1].set_xlabel('Regularization Parameter')
# axes[1].set_ylabel('Dataset Size')

# plt.tight_layout()
# plt.show()

# #create the heatmap for the real dataset
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Heatmap for Objective Function Value in green
# sns.heatmap(objective_values_real_dataset.reshape(1, -1), annot=True, fmt=".2f",  ax=axes[0], cmap= "Greens",
#             xticklabels=regularization_params, yticklabels=['Real Dataset'], cbar_kws={'label': 'Objective Value'})
# axes[0].set_title('Objective Function Value')
# axes[0].set_xlabel('Regularization Parameter')

# # Heatmap for Test Accuracy in red
# sns.heatmap(test_accuracies_real_dataset.reshape(1, -1), annot=True, fmt=".2f", ax=axes[1], cmap="Reds",
#             xticklabels=regularization_params, yticklabels=['Real Dataset'], cbar_kws={'label': 'Test Accuracy (%)'})
# axes[1].set_title('Test Accuracy')
# axes[1].set_xlabel('Regularization Parameter')

# plt.tight_layout()
# plt.show()








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data setup
data = {
    "Dataset Size": [100, 100, 100, 100, 500, 500, 500, 500, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 614, 614, 614, 614],
    "Regularization Parameter (ν)": [0.1, 1, 10, 100, 0.1, 1, 10, 100, 0.1, 1, 10, 100, 0.1, 1, 10, 100, 0.1, 1, 10, 100],
    "Primal Bias": [-0.967226, -4.187172, -6.499275, -9.020870, -3.587445, -6.513385, -8.373591, -8.617367,
                    -4.456323, -7.848251, -10.185140, -10.473812, -5.546322, -9.337953, -11.089879, -11.408520,
                    -6.420799, -6.672915, -6.623668, -6.623668],
    "Dual Bias": [-1.611306, -4.457711, -6.499275, -8.004129, -3.833337, -5.556602, -7.444707, -7.677421,
                  -4.483860, -6.866890, -9.139167, -9.438609, -4.559938, -8.379206, -10.118558, -10.471653,
                  -6.870837, -6.983583, -6.930853, -6.930854]
}

weights = {
    "Primal Weights": [
        [0.641387, 0.783201, 0.805510, 0.797560],
        [1.648296, 2.053491, 2.388358, 2.623347],
        [2.759229, 3.349606, 3.733657, 3.987136],
        [3.506082, 5.200176, 4.660327, 5.429731],
        [1.856082, 1.544807, 1.947366, 1.764534],
        [3.051942, 3.111740, 3.564217, 3.260354],
        [3.756450, 3.985380, 4.745922, 4.202102],
        [3.823921, 4.140492, 4.865816, 4.373788],
        [2.372662, 1.898109, 2.369030, 2.156118],
        [3.887087, 3.740574, 4.090786, 3.908880],
        [4.999520, 4.854610, 5.244930, 5.117318],
        [5.144338, 4.967634, 5.428724, 5.231902],
        [2.841253, 2.523767, 2.897158, 2.763014],
        [4.740339, 4.426736, 4.822507, 4.609249],
        [5.657895, 5.146297, 5.745967, 5.520159],
        [5.839736, 5.307043, 5.932569, 5.671006],
        [0.073320, 0.029378, -0.009609, -0.005678],
        [0.073402, 0.029595, -0.010341, -0.004464],
        [0.073511, 0.029391, -0.010176, -0.004523],
        [0.073511, 0.029391, -0.010176, -0.004523]
    ],
    "Dual Weights": [
        [0.64138755, 0.7832013, 0.80551003, 0.79755994],
        [1.64829691, 2.05349104, 2.38835759, 2.62334679],
        [2.7592293, 3.3496061, 3.73365707, 3.98713597],
        [3.50608243, 5.20017589, 4.66032724, 5.42973069],
        [1.85606587, 1.54480573, 1.94735297, 1.76456743],
        [3.05194227, 3.1117399, 3.56421736, 3.26035376],
        [3.75645024, 3.98537957, 4.74592184, 4.20210158],
        [3.82392123, 4.14049207, 4.8658163, 4.37378807],
        [2.37266155, 1.89810923, 2.36902978, 2.15611812],
        [3.88708686, 3.74057382, 4.09078642, 3.90888001],
        [4.99952048, 4.85460993, 5.24493048, 5.11731842],
        [5.14433791, 4.96763423, 5.42872431, 5.23190202],
        [2.84126013, 2.52377034, 2.8971674, 2.76300117],
        [4.74033914, 4.42673626, 4.82250715, 4.60924869],
        [5.65789475, 5.14629666, 5.74596714, 5.52015935],
        [5.83971845, 5.30701211, 5.93252461, 5.6709607],
        [0.073319, 0.029378, -0.009609, -0.005678],
        [0.073402, 0.029595, -0.010341, -0.004464],
        [0.073511, 0.029391, -0.010176, -0.004523],
        [0.073511, 0.029391, -0.010176, -0.004523]
    ]
}

df = pd.DataFrame(data)

# # Heatmap for primal bias
# plt.figure(figsize=(10, 6))
# pivot = df[df['Dataset Size'] != 614].pivot(index='Dataset Size', columns='Regularization Parameter (ν)', values='Primal Bias')
# sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Primal Bias Heatmap")
# plt.show()

# # Heatmap for dual bias
# plt.figure(figsize=(10, 6))
# pivot = df[df['Dataset Size'] != 614].pivot(index='Dataset Size', columns='Regularization Parameter (ν)', values='Primal Bias')

# sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Dual Bias Heatmap")
# plt.show()

# # Primal vs Dual Weight Comparison
# primal_weights = np.array(weights["Primal Weights"])
# dual_weights = np.array(weights["Dual Weights"])
# plt.scatter(primal_weights.flatten(), dual_weights.flatten(), c='blue', alpha=0.7)
# plt.xlabel("Primal Weights")
# plt.ylabel("Dual Weights")
# plt.title("Primal vs. Dual Weights")
# plt.grid()
# plt.show()


# # Assuming df and weights dictionary already exist

# # 1. Line Plot for Primal and Dual Weights
# weights_df = pd.DataFrame(weights)
# weights_df['Dataset Size'] = data['Dataset Size']
# weights_df['Regularization Parameter (ν)'] = data['Regularization Parameter (ν)']

# plt.figure(figsize=(10, 6))
# for size in weights_df['Dataset Size'].unique():
#     subset = weights_df[weights_df['Dataset Size'] == size]
#     plt.plot(subset['Regularization Parameter (ν)'], subset['Primal Weights'].apply(np.linalg.norm),
#              label=f"Primal Weights (Size {size})")
#     plt.plot(subset['Regularization Parameter (ν)'], subset['Dual Weights'].apply(np.linalg.norm),
#              linestyle='--', label=f"Dual Weights (Size {size})")

# plt.xscale('log')
# plt.xlabel('Regularization Parameter (ν)')
# plt.ylabel('Weight Norm')
# plt.title('Norm of Primal and Dual Weights Across Dataset Sizes')
# plt.legend()
# plt.show()

# # 2. Scatter Plot for Primal vs. Dual Biases
# plt.figure(figsize=(8, 6))
# plt.scatter(df['Primal Bias'], df['Dual Bias'], c=df['Dataset Size'], cmap='viridis', s=100, edgecolor='k')
# plt.colorbar(label='Dataset Size')
# plt.xlabel('Primal Bias')
# plt.ylabel('Dual Bias')
# plt.title('Comparison of Primal and Dual Biases')
# plt.axline((0, 0), slope=1, color='red', linestyle='--', label='y=x Line')
# plt.legend()
# plt.show()

# # 3. Line Plot for Biases Across Dataset Sizes
# plt.figure(figsize=(10, 6))
# for reg_param in df['Regularization Parameter (ν)'].unique():
#     subset = df[df['Regularization Parameter (ν)'] == reg_param]
#     plt.plot(subset['Dataset Size'], subset['Primal Bias'], label=f"Primal Bias (ν={reg_param})")
#     plt.plot(subset['Dataset Size'], subset['Dual Bias'], linestyle='--', label=f"Dual Bias (ν={reg_param})")

# plt.xlabel('Dataset Size')
# plt.ylabel('Bias')
# plt.title('Primal and Dual Biases Across Dataset Sizes')
# plt.legend()
# plt.show()

# # 4. Box Plot for Weights Distribution
# primal_weights = [np.linalg.norm(w) for w in weights['Primal Weights']]
# dual_weights = [np.linalg.norm(w) for w in weights['Dual Weights']]

# plt.figure(figsize=(8, 6))
# sns.boxplot(data=[primal_weights, dual_weights], palette="Set2")
# plt.xticks([0, 1], ['Primal Weights', 'Dual Weights'])
# plt.ylabel('Norm of Weights')
# plt.title('Distribution of Primal and Dual Weights')
# plt.show()

# # 5. Bar Plot for Real Dataset (Size = 614)
# real_dataset = df[df['Dataset Size'] == 614]

# plt.figure(figsize=(8, 6))
# bar_width = 0.35
# x = np.arange(len(real_dataset))

# plt.bar(x - bar_width/2, real_dataset['Primal Bias'], bar_width, label='Primal Bias', color='blue')
# plt.bar(x + bar_width/2, real_dataset['Dual Bias'], bar_width, label='Dual Bias', color='orange')

# plt.xticks(x, real_dataset['Regularization Parameter (ν)'])
# plt.xlabel('Regularization Parameter (ν)')
# plt.ylabel('Bias')
# plt.title('Bias Comparison for Real Dataset (Size = 614)')
# plt.legend()
# plt.show()



# Filter real dataset

# Create DataFrame
df = pd.DataFrame(data)
weights_df = pd.DataFrame(weights)

# Merge weights into the main DataFrame
df = pd.concat([df, weights_df], axis=1)

# Ensure weights are numeric
df['Primal Norm'] = df['Primal Weights'].apply(np.linalg.norm)
df['Dual Norm'] = df['Dual Weights'].apply(np.linalg.norm)

real_dataset = df[df['Dataset Size'] == 614]

# # 1. Line Plot of Primal and Dual Weights Norm
# plt.figure(figsize=(10, 6))
# plt.plot(real_dataset['Regularization Parameter (ν)'], 
#          real_dataset['Primal Weights'].apply(np.linalg.norm), 
#          marker='o', label='Primal Weights Norm', color='blue')
# plt.plot(real_dataset['Regularization Parameter (ν)'], 
#          real_dataset['Dual Weights'].apply(np.linalg.norm), 
#          marker='s', label='Dual Weights Norm', color='orange')

# plt.xscale('log')
# plt.xlabel('Regularization Parameter (ν)')
# plt.ylabel('Weights Norm')
# plt.title('Norm of Primal and Dual Weights for Real Dataset')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 2. Scatter Plot of Primal Bias vs. Regularization Parameter
# plt.figure(figsize=(8, 6))
# plt.scatter(real_dataset['Regularization Parameter (ν)'], real_dataset['Primal Bias'], 
#             color='blue', s=100, edgecolor='k')
# plt.xscale('log')
# plt.xlabel('Regularization Parameter (ν)')
# plt.ylabel('Primal Bias')
# plt.title('Primal Bias Across Regularization Parameters (Real Dataset)')
# plt.grid(True)
# plt.show()

# # 3. Scatter Plot of Dual Bias vs. Regularization Parameter
# plt.figure(figsize=(8, 6))
# plt.scatter(real_dataset['Regularization Parameter (ν)'], real_dataset['Dual Bias'], 
#             color='orange', s=100, edgecolor='k')
# plt.xscale('log')
# plt.xlabel('Regularization Parameter (ν)')
# plt.ylabel('Dual Bias')
# plt.title('Dual Bias Across Regularization Parameters (Real Dataset)')
# plt.grid(True)
# plt.show()

# # 4. Heatmap of Primal and Dual Weights Difference
# weights_diff = real_dataset.apply(lambda row: np.linalg.norm(
#     np.array(row['Primal Weights']) - np.array(row['Dual Weights'])), axis=1)

# plt.figure(figsize=(8, 6))
# sns.heatmap(weights_diff.values.reshape(1, -1), annot=True, fmt=".2f", cmap="YlGnBu", 
#             xticklabels=real_dataset['Regularization Parameter (ν)'], 
#             yticklabels=['Weight Difference'])
# plt.xlabel('Regularization Parameter (ν)')
# plt.title('Difference Between Primal and Dual Weights (Norm)')
# plt.show()

# # 5. Combined Plot for Bias and Weight Norms
# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Primal and Dual Bias
# ax1.set_xlabel('Regularization Parameter (ν)')
# ax1.set_xscale('log')
# ax1.plot(real_dataset['Regularization Parameter (ν)'], real_dataset['Primal Bias'], 
#          marker='o', label='Primal Bias', color='blue')
# ax1.plot(real_dataset['Regularization Parameter (ν)'], real_dataset['Dual Bias'], 
#          marker='s', label='Dual Bias', color='orange')
# ax1.set_ylabel('Bias')
# ax1.tick_params(axis='y')

# # Weight Norms on Secondary Axis
# ax2 = ax1.twinx()
# ax2.plot(real_dataset['Regularization Parameter (ν)'], 
#          real_dataset['Primal Weights'].apply(np.linalg.norm), 
#          marker='^', linestyle='--', label='Primal Weights Norm', color='green')
# ax2.plot(real_dataset['Regularization Parameter (ν)'], 
#          real_dataset['Dual Weights'].apply(np.linalg.norm), 
#          marker='v', linestyle='--', label='Dual Weights Norm', color='red')
# ax2.set_ylabel('Weights Norm')
# ax2.tick_params(axis='y')

# fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
# fig.suptitle('Biases and Weights Norms for Real Dataset (Size = 614)')
# fig.tight_layout()
# plt.show()

# 6. Box Plot for Primal and Dual Weights Norm
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=[real_dataset['Primal Norm'], real_dataset['Dual Norm']], palette="Set2")
# plt.xticks([0, 1], ['Primal Weights Norm', 'Dual Weights Norm'])
# plt.ylabel('Weights Norm')
# plt.title('Distribution of Primal and Dual Weights Norms')
# plt.show()

# 7. Bar Plot for Primal and Dual Biases
plt.figure(figsize=(8, 6))
bar_width = 0.35
x = np.arange(len(real_dataset))

plt.bar(x - bar_width/2, real_dataset['Primal Bias'], bar_width, label='Primal Bias', color='blue')
plt.bar(x + bar_width/2, real_dataset['Dual Bias'], bar_width, label='Dual Bias', color='orange')

plt.xticks(x, real_dataset['Regularization Parameter (ν)'])
plt.xlabel('Regularization Parameter (ν)')
plt.ylabel('Bias')
plt.title('Bias Comparison for Real Dataset (Size = 614)')
plt.legend()
plt.show()

# 8. Line Plot for Primal and Dual Weights Norm
plt.figure(figsize=(10, 6))

plt.plot(real_dataset['Regularization Parameter (ν)'], real_dataset['Primal Norm'],
         marker='o', label='Primal Weights Norm', color='blue')
plt.plot(real_dataset['Regularization Parameter (ν)'], real_dataset['Dual Norm'],
         marker='s', label='Dual Weights Norm', color='orange')

plt.xscale('log')
plt.xlabel('Regularization Parameter (ν)')
plt.ylabel('Weights Norm')
plt.title('Norm of Primal and Dual Weights for Real Dataset')
plt.legend()
plt.grid(True)
plt.show()

# 9. Scatter Plot for Primal and Dual Biases
plt.figure(figsize=(8, 6))
plt.scatter(real_dataset['Primal Bias'], real_dataset['Dual Bias'],
            color='purple', s=100, edgecolor='k')
plt.xlabel('Primal Bias')
plt.ylabel('Dual Bias')
plt.title('Primal vs. Dual Biases for Real Dataset')
plt.grid(True)
plt.show()

# 10. Line Plot for Primal and Dual Biases
plt.figure(figsize=(10, 6))
for size in real_dataset['Dataset Size'].unique():
    subset = real_dataset[real_dataset['Dataset Size'] == size]
    plt.plot(subset['Regularization Parameter (ν)'], subset['Primal Bias'],
             label=f"Primal Bias (Size {size})")
    plt.plot(subset['Regularization Parameter (ν)'], subset['Dual Bias'],
             linestyle='--', label=f"Dual Bias (Size {size})")

plt.xscale('log')
plt.xlabel('Regularization Parameter (ν)')
plt.ylabel('Bias')
plt.title('Primal and Dual Biases for Real Dataset')
plt.legend()
plt.grid(True)
plt.show()
