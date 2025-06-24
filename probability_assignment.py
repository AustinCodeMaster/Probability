import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LinearRegression
import io

# --- Data Preparation ---
# The data below contains the number of hours spent coding and the number of coding bugs for 50 students.
# I am using a multi-line string to store the data, which will be loaded into a pandas DataFrame.
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

# Load the data into a pandas DataFrame using StringIO
# This allows us to treat the string as if it were a file
# The separator '\t' means the data is tab-separated
# This step is important for all further analysis
# ---
data = pd.read_csv(io.StringIO(raw_data), sep='\t')

# --- Scatter Plot (Question i) ---
# Plotting Hours Coding (x-axis) vs Num Bugs (y-axis)
# This helps visualize the relationship between the two variables
plt.figure(figsize=(8,6))
sns.scatterplot(x='Hours_Coding', y='Num_Bugs', data=data)
plt.title('Hours Coding vs Num Bugs')
plt.xlabel('Hours Coding')
plt.ylabel('Num Bugs')
plt.savefig('scatter_hours_vs_bugs.png')  # Save the plot as a PNG file
plt.close()

# --- Pearson Correlation (Question ii) ---
# Calculate Pearson's correlation coefficient to measure linear relationship
r, p_value = pearsonr(data['Hours_Coding'], data['Num_Bugs'])
print(f"Pearson correlation coefficient (r): {r:.3f}")

# --- Linear Regression (Question iii) ---
# Fit a linear regression model: Num_Bugs = a + b * Hours_Coding
# This will help us predict the number of bugs based on hours spent coding
X = data[['Hours_Coding']]
y = data['Num_Bugs']
reg = LinearRegression().fit(X, y)
a = reg.intercept_  # Intercept of the regression line
b = reg.coef_[0]    # Slope of the regression line
print(f"Regression equation: Num_Bugs = {a:.2f} + {b:.2f} * Hours_Coding")

# Predict the number of bugs for 20 hours of coding
pred_20 = reg.predict([[20]])[0]
print(f"Predicted Num_Bugs for 20 hours: {pred_20:.2f}")

# --- Frequency Distribution Table (Question iv) ---
# Count how many times each number of bugs appears in the data
freq_table = data['Num_Bugs'].value_counts().sort_index()
print("\nFrequency distribution table for Num_Bugs:")
print(freq_table)

# --- Histogram (Question v) ---
# Plot a histogram to show the distribution of the number of bugs
plt.figure(figsize=(8,6))
sns.histplot(data['Num_Bugs'], bins=10, kde=False, stat='density', color='skyblue', edgecolor='black')

# --- Overlay Normal Curve (Question vi) ---
# Calculate the mean and standard deviation of the number of bugs
mean = data['Num_Bugs'].mean()
std = data['Num_Bugs'].std()
# Generate x values for the normal curve
x = np.linspace(data['Num_Bugs'].min(), data['Num_Bugs'].max(), 100)
# Plot the normal distribution curve using the sample mean and std
plt.plot(x, norm.pdf(x, mean, std), 'r-', lw=2, label='Normal Curve')
plt.title('Histogram of Num Bugs with Normal Curve')
plt.xlabel('Num Bugs')
plt.ylabel('Density')
plt.legend()
plt.savefig('histogram_num_bugs.png')  # Save the histogram with normal curve
plt.close()

print("\nPlots saved as 'scatter_hours_vs_bugs.png' and 'histogram_num_bugs.png'.")
