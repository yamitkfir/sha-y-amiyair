import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

file_path = "proj/shii_t.csv"
df = pd.read_csv(file_path)

education_mental_health = pd.crosstab(df['education'], df['15. Have your learning experiences affected your mental health?'])
chi2, p, dof, expected = chi2_contingency(education_mental_health)
expected_df = pd.DataFrame(expected, index=education_mental_health.index, columns=education_mental_health.columns)

print(f"Chi-squared: {chi2}, p-value: {p}, degrees of freedom: {dof}")
print(education_mental_health)
print(expected_df)

alpha = 0.05
if p < alpha:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")


