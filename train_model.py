import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import joblib 

#Load dataset
data ="/home/ebeanski/Streamlit/Diabetes_app/Diabetes_Cleaned.csv"
df = pd.read_csv(data)

#split the dataset into X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

#Train the model
model = RandomForestClassifier()
model.fit(X, y)

#Save the model 
joblib.dump(model, 'diabetes_model.pkl')
print("The model is trained and saved as 'diabetes_model.pkl'") 

