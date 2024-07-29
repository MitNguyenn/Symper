import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv('suv_data.csv')
print(df.head())

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})


X = df[['User ID', 'Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

y_prob = model.predict_proba(X_test)[:, 1]

X_test_with_prob = pd.DataFrame(X_test, columns=['User ID', 'Gender', 'Age', 'EstimatedSalary'])
X_test_with_prob['Purchased_Prob'] = y_prob

print(X_test_with_prob.head())


