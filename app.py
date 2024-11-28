import numpy as np
import pandas as pd
import mysql.connector
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from flask import Flask, render_template, request, jsonify

# Flask application setup
app = Flask(__name__)

# SQL database connection
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "harsha@123",
    "database": "fraud_detection"
}

# Connect to the MySQL database
def connect_to_db():
    return mysql.connector.connect(
        host=db_config["host"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"]
    )

# Step 1: Create Advanced Synthetic Fraud Detection Data
# Step 1: Create Advanced Synthetic Fraud Detection Data
def create_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'transaction_amount': np.random.normal(100, 50, num_samples).clip(0, None),
        'transaction_type': np.random.choice([0, 1], num_samples),  # 0: purchase, 1: refund
        'is_fraud': np.random.choice([0, 1], num_samples, p=[0.98, 0.02])  # 2% fraud rate
    })
    data['transaction_amount_log'] = np.log1p(data['transaction_amount'])
    data['rolling_avg'] = data['transaction_amount'].rolling(window=5, min_periods=1).mean()
    data['time_diff'] = np.random.exponential(5, num_samples)  # Time difference in seconds

    # Insert data into MySQL database
    db_connection = connect_to_db()
    cursor = db_connection.cursor()

    for _, row in data.iterrows():
        # Convert float64 to Python float
        transaction_amount = float(row['transaction_amount'])
        transaction_type = int(row['transaction_type'])
        is_fraud = int(row['is_fraud'])
        transaction_amount_log = float(row['transaction_amount_log'])
        rolling_avg = float(row['rolling_avg'])
        time_diff = float(row['time_diff'])
        
        cursor.execute("""
            INSERT INTO transactions (transaction_amount, transaction_type, is_fraud, transaction_amount_log, rolling_avg, time_diff)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (transaction_amount, transaction_type, is_fraud, transaction_amount_log, rolling_avg, time_diff))
    
    db_connection.commit()  # Commit the transaction
    cursor.close()
    db_connection.close()

    return data.dropna()

# Step 2: Train Advanced Fraud Detection Model Ensemble
def train_advanced_fraud_detection(data):
    X = data.drop(columns=['is_fraud'])
    y = data['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    # Stacking multiple models
    stack = StackingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        final_estimator=RandomForestClassifier(n_estimators=50, max_depth=3)
    )
    
    # Train the stacking model
    stack.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = stack.predict(X_test)
    y_proba = stack.predict_proba(X_test)[:, 1]
    print("Advanced Model Evaluation:")
    print("AUC-ROC:", roc_auc_score(y_test, y_proba))
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(stack, 'advanced_fraud_detection_model.pkl')
    return stack

# Step 3: Real-Time Fraud Detection
@app.route("/predict", methods=["POST"])
def predict():
    model = joblib.load('advanced_fraud_detection_model.pkl')
    transaction = request.json
    txn_df = pd.DataFrame([transaction])
    is_fraud = model.predict(txn_df)[0]
    fraud_prob = model.predict_proba(txn_df)[0][1]
    return jsonify({"fraud": bool(is_fraud), "probability": fraud_prob})

# Web pages
@app.route("/", methods=["GET"])
def homepage():
    return render_template("index.html")

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")

@app.route("/frauddetection", methods=["GET"])
def fraud_detection():
    return render_template("fraud_detection.html")

@app.route("/security", methods=["GET"])
def security():
    return render_template("security_alerts.html")

@app.route("/website", methods=["GET"])
def website():
    return render_template("website_security.html")

# Run Flask app
if __name__ == "__main__":
    # Step 1: Generate synthetic data and insert it into MySQL database
    data = create_synthetic_data()
    # Step 2: Train the fraud detection model
    train_advanced_fraud_detection(data)

    # Step 3: Start the Flask application
    app.run(debug=True)
