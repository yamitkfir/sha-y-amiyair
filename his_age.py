import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

try:
    # --- Make sure your file path is correct ---
    file_path = "proj/shii_t.csv"
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

# Ensure relevant columns are numeric, changing errors to Not a Number (NaN)
df['MHI'] = pd.to_numeric(df['MHI'], errors='coerce')
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Drop rows where MHI or age are invalid
df.dropna(subset=['MHI', 'age'], inplace=True)

# Filter out MHI values less than 3
df = df[df['MHI'] >= 3]


age_groups = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
colors = ["blue", "green", "red", "purple", "black"]
plt.figure(figsize=(12, 7))

#pletting histograms by age group but only if there are enough data points

for i, age in enumerate(age_groups):
    # Filter data for the current age group
    age_data = df[(df['age'] >= age) & (df['age'] < age + 5)]['MHI']
    
    if len(age_data) > 4:  # Only plot if there are more than 4 data points
        kde = gaussian_kde(age_data.values)
        vec_t = np.linspace(age_data.min() - 20, age_data.max() + 20, 1000)
        J = kde(vec_t)
        plt.plot(vec_t, J, color=colors[i], label=f"{age}-{age + 5} years", linewidth=2)
plt.xlabel("MHI (Mental Health Index)")
plt.ylabel("Density")
plt.title("Kernel Density Estimation of MHI by Age Group (using SciPy)")
plt.legend(title="Age Group")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()