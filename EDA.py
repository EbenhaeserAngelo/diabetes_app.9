# EDA.py
# Exploratory Data Analysis for Diabetes Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d1 = pd.read_csv('Diabetes (1) 8.csv')

d2=d1.loc[(d1['Glucose']!=0) & (d1['BloodPressure']!=0) & (d1['SkinThickness']!=0) & (d1['Insulin']!=0) & (d1['BMI']!=0)]

d1['Glucose'].replace(0,d2['Glucose'].mean(),inplace=True)
d1['BloodPressure'].replace(0,d2['BloodPressure'].mean(),inplace=True)
d1['SkinThickness'].replace(0,d2['SkinThickness'].mean(),inplace=True)
d1['Insulin'].replace(0,d2['Insulin'].mean(),inplace=True)
d1['BMI'].replace(0,d2['BMI'].mean(),inplace=True)

d1.to_csv('Diabetes_Cleaned_data_N.csv')  
#Cleaned data saved to 'Diabetes_Cleaned_data_N.csv'
d1.info()
d1.describe()
d1.head()
d1.tail()
d1.isnull().sum()
d1.duplicated().sum()
# Code starting from the second d1 is my own work
d1.to_csv('Diabetes_Cleaned_data_N.csv', index=False)
print ("Cleaned data saved to 'Diabetes_Cleaned_data_N.csv")
d1.hist(figsize=(12, 10))
plt.suptitle('Histograms of Cleaned Diabetes Dataset')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(d1.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Cleaned Diabetes Dataset')
plt.show()

# Boxplots for each feature numerical columns
plt.figure(figsize=(12, 8))
d1.boxplot(rot=45)
plt.title('Boxplots of Features in Cleaned Diabetes Dataset')
plt.show()

# Top Correlations with Outcome
corr_outcome =d1.corr()['Outcome'].sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=corr_outcome.values, y=corr_outcome.index)
plt.title("Correlations of Features with Outcome")
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()

#Scatter plot with trendline (Glucose vs.Outcome)
plt.figure(figsize=(8, 6))
sns.regplot(x='Glucose', y='Outcome', data=d1)
plt.title('Glucose vs Outcome with Trendline')
plt.show()   

# Visualizing feature averages grouped by Outcome
plt.figure(figsize=(12,6))
sns.pointplot(data=d1 ,x= 'Outcome', y='BMI')
plt.title('Average BMI by Outcome')
plt.show()

#Violin plot
plt.figure(figsize=(8,6))
sns.violinplot(data=d1, x='Outcome', y='Glucose')
plt.title('Violin Plot of Glucose by Outcome')
plt.show() 