# Install required libraries
# pip install flask scikit-learn pandas numpy requests

# Import all required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify

# ------------------------------
# Step 1: Create Sample Dataset
# ------------------------------
data = {
    "login_time": [9, 10, 3, 22, 15, 2],
    "device_trusted": [1, 1, 0, 0, 1, 0],
    "location_match": [1, 1, 0, 0, 1, 0],
    "risk_level": ["low", "low", "high", "medium", "low", "high"]
}

df = pd.DataFrame(data)
print("Sample Data ✅")
print(df)

# ------------------------------
# Step 2: Train ML Model
# ------------------------------
encoder = LabelEncoder()
df['risk_level_enc'] = encoder.fit_transform(df['risk_level'])

X = df[['login_time','device_trusted','location_match']]
y = df['risk_level_enc']

model = DecisionTreeClassifier()
model.fit(X, y)

print("\nModel Trained ✅")

# ------------------------------
# Step 3: Create Flask API
# ------------------------------
app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        login_time = data.get('login_time')
        device = data.get('device_trusted')
        location = data.get('location_match')

        # Validate inputs
        if login_time is None or device is None or location is None:
            return jsonify({"error": "Missing required fields"}), 400

        prediction = model.predict([[login_time, device, location]])[0]
        risk = encoder.inverse_transform([prediction])[0]

        if risk == "low":
            return jsonify({"status": "success", "auth": "passwordless access"})
        elif risk == "medium":
            return jsonify({"status": "pending", "auth": "OTP required"})
        else:
            return jsonify({"status": "denied", "auth": "high risk - access blocked"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------------
# Step 4: Run Flask App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
