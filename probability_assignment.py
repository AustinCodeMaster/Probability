import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LinearRegression
import io

# Data
raw_data = '''
Student\tHours_Coding\tNum_Bugs
1\t10.7\t25
2\t24.8\t50
3\t19.4\t38
4\t16.2\t31
5\t5.3\t3
6\t5.3\t7
7\t2.9\t3
8\t22.7\t51
9\t16.2\t34
10\t18.8\t29
11\t2\t6
12\t25.3\t49
13\t21.9\t40
14\t6.7\t16
15\t6\t17
16\t6\t17
17\t9\t14
18\t14.4\t27
19\t12.1\t26
20\t8.6\t22
21\t16.5\t31
22\t4.9\t9
23\t8.7\t12
24\t10.5\t15
25\t12.7\t29
26\t20.7\t48
27\t6.4\t12
28\t14.1\t33
29\t16\t34
30\t2.6\t2
31\t16.4\t35
32\t5.7\t19
33\t3.1\t6
34\t24.7\t57
35\t25.2\t37
36\t21.3\t47
37\t9\t18
38\t3.9\t6
39\t18.3\t37
40\t12.3\t15
41\t4.5\t8
42\t13.6\t29
43\t2.3\t12
44\t23.8\t45
45\t7.8\t12
46\t17.7\t33
47\t9.1\t23
48\t14.2\t30
49\t14.9\t27
50\t6\t15
'''

# Load data into DataFrame
data = pd.read_csv(io.StringIO(raw_data), sep='\t')

# i. Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x='Hours_Coding', y='Num_Bugs', data=data)
plt.title('Hours Coding vs Num Bugs')
plt.xlabel('Hours Coding')
plt.ylabel('Num Bugs')
plt.savefig('scatter_hours_vs_bugs.png')
plt.close()

# ii. Pearson correlation
r, p_value = pearsonr(data['Hours_Coding'], data['Num_Bugs'])
print(f"Pearson correlation coefficient (r): {r:.3f}")

# iii. Regression equation
X = data[['Hours_Coding']]
y = data['Num_Bugs']
reg = LinearRegression().fit(X, y)
a = reg.intercept_
b = reg.coef_[0]
print(f"Regression equation: Num_Bugs = {a:.2f} + {b:.2f} * Hours_Coding")

# Predict for 20 hours
pred_20 = reg.predict([[20]])[0]
print(f"Predicted Num_Bugs for 20 hours: {pred_20:.2f}")

# iv. Frequency distribution table
freq_table = data['Num_Bugs'].value_counts().sort_index()
print("\nFrequency distribution table for Num_Bugs:")
print(freq_table)

# v. Histogram
plt.figure(figsize=(8,6))
sns.histplot(data['Num_Bugs'], bins=10, kde=False, stat='density', color='skyblue', edgecolor='black')

# vi. Overlay normal curve
mean = data['Num_Bugs'].mean()
std = data['Num_Bugs'].std()
x = np.linspace(data['Num_Bugs'].min(), data['Num_Bugs'].max(), 100)
plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label='Normal Curve')
plt.title('Histogram of Num Bugs with Normal Curve')
plt.xlabel('Num Bugs')
plt.ylabel('Density')
plt.legend()
plt.savefig('histogram_num_bugs.png')
plt.close()

print("\nPlots saved as 'scatter_hours_vs_bugs.png' and 'histogram_num_bugs.png'.")
