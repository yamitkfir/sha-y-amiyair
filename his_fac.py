import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde 

file_path = "proj/shii_t.csv"
df = pd.read_csv(file_path)

# Ensure the 'MHI' column is numeric, coercing errors to NaN
df['MHI'] = pd.to_numeric(df['MHI'], errors='coerce')

df = df[df['Faculty'].map(df['Faculty'].value_counts()) > 4]
unique_faculties = df['Faculty'].unique().tolist()

plt.figure(figsize=(12, 8))
colors = ["blue", "green", "red", "purple", "orange", "black"]

for i, faculty in enumerate(unique_faculties):
    faculty_data = df[df["Faculty"] == faculty]["MHI"]
    
    kde = gaussian_kde(faculty_data.values)    
    vec_t = np.linspace(faculty_data.min() - 20, faculty_data.max() + 20, 1000)
    J = kde(vec_t)
    plt.plot(vec_t, J, color=colors[i], label=faculty, linewidth=2)


plt.xlabel("MHI (Mental Health Index)")
plt.ylabel("Density")
plt.title("Kernel Density Estimation of MHI by Faculty (using SciPy)")
plt.legend(title="Faculty")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

