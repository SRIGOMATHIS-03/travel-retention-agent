import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import streamlit as st

# Load & Clean (your code ✅)
df = pd.read_csv("data/Travel details dataset.csv")
print(df.head())
print(df.info())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print("Shape after cleaning:", df.shape)

# **FIX: Convert string costs to numbers**
df['Accommodation cost'] = pd.to_numeric(df['Accommodation cost'], errors='coerce')
df['Transportation cost'] = pd.to_numeric(df['Transportation cost'], errors='coerce')
df['Traveler age'] = pd.to_numeric(df['Traveler age'], errors='coerce')
df['Duration (days)'] = pd.to_numeric(df['Duration (days)'], errors='coerce')

# Drop any remaining NaNs after conversion
df.dropna(inplace=True)
print("Shape after numeric conversion:", df.shape)

# **Feature Engineering for Churn**
df['total_cost'] = df['Accommodation cost'] + df['Transportation cost']
df['cost_per_day'] = df['total_cost'] / df['Duration (days)']
df['churn_risk'] = np.where(
    (df['Traveler age'] < 30) | 
    (df['cost_per_day'] > df['total_cost'].quantile(0.75)) | 
    (df['Accommodation type'].str.contains('Hostel|Airbnb', case=False, na=False)),
    1, 0  # 1=High risk
)
print("Churn distribution:\n", df['churn_risk'].value_counts())

# Prepare features
cat_cols = ['Traveler gender', 'Traveler nationality', 'Accommodation type', 'Transportation type', 'Destination']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df[['Traveler age', 'Duration (days)', 'total_cost', 'cost_per_day'] + cat_cols]
y = df['churn_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2%}")

# Save model + encoders
joblib.dump(model, 'churn_model.pkl')
joblib.dump(le_dict, 'label_encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("✅ Model saved! Accuracy:", accuracy)

# Streamlit preview
st.title("🛫 Travel Retention Agent - Model Training")
st.success(f"🚀 Model ready! Test Accuracy: {accuracy:.2%}")
st.write("Churn distribution:", df['churn_risk'].value_counts())
st.dataframe(df[['Traveler age', 'total_cost', 'churn_risk']].head())