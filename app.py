from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["GET"])
def homepage():
    return render_template("index.html")

@app.route("/dashboard", methods=["GET"])
def fraud():
    return render_template("dashboard.html")

@app.route("/frauddetection", methods=["GET"])
def frauddetection():
    return render_template("fraud_detection.html")

@app.route("/security", methods=["GET"])
def security():
    return render_template("security_alerts.html")

@app.route("/website", methods=["GET"])
def website():
    return render_template("website_security.html")

app.run(debug=True)