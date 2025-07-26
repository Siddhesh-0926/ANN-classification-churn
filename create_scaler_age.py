import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Create a fake dataset with an 'Age' column
df = pd.DataFrame({
    'Age': [22, 25, 30, 35, 40, 45, 50, 55]
})

# Step 2: Fit a scaler on the 'Age' column
scaler_age = StandardScaler()
scaler_age.fit(df[['Age']])

# Step 3: Save the scaler to scaler_age.pkl
with open("scaler_age.pkl", "wb") as f:
    pickle.dump(scaler_age, f)

print("✅ scaler_age.pkl has been created successfully.")
