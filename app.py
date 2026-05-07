from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import numpy as np
import requests

import joblib

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import matplotlib.pyplot as plt
#from weasyprint import HTML

# 🔥 ADD THESE IMPORTS (NEW)
import math
from datetime import date

import matplotlib.pyplot as plt
import io
import base64

def generate_chart_image(labels, values):
    plt.figure(figsize=(8,4))
    plt.plot(labels, values, marker='o')
    plt.fill_between(labels, values, alpha=0.3)

    plt.title("24 Hour Solar Forecast")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()

    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

app = Flask(__name__)
app.secret_key = "secret123"

forecast_labels_24 = None

# ===============================
# 🔥 LOAD MODEL + SCALER
# ===============================
import tensorflow as tf

model = tf.keras.models.load_model(
    "solar_ann_model.keras",
    compile=False,
    safe_mode=False
)
sc_X = joblib.load("scaler_X.pkl")
sc_y = joblib.load("scaler_y.pkl")

# ===============================
# 🔥 24 HOUR FORECAST CACHE (NEW)
# ===============================
forecast_data_24 = None
forecast_date_24 = None

def generate_realistic_forecast(base_power):

    forecast = []
    labels = []

    data = session.get("last_data")

    # 🔥 FIX: fallback if no session
    if not data or "features" not in data:
        for h in range(24):
            labels.append(f"{h}:00")
            forecast.append(0)
        return labels, forecast

    base_features = data["features"]

    for h in range(0, 24):
        labels.append(f"{h}:00")

        temp, humidity, cloud, wind, rain, radiation, day, month = base_features

        if h < 6 or h > 18:
            radiation = 0
        else:
            radiation = radiation * math.sin((math.pi / 12) * (h - 6))

        radiation *= (1 - cloud / 100)

        features = np.array([[temp, humidity, cloud, wind, rain, radiation, day, month]])
        scaled = sc_X.transform(features)

        pred_scaled = model.predict(scaled)
        pred = sc_y.inverse_transform(pred_scaled)[0][0]

        forecast.append(float(round(max(pred, 0), 2)))

    return labels, forecast

# ===============================
# HOME
# ===============================
@app.route('/')
def home():
    data = session.pop('result', None)

    return render_template(
        'index.html',
        prediction_text=data["prediction_text"] if data else None,
        prediction_value=data["prediction_value"] if data else None,
        forecast_data=data["forecast_data"] if data else None,
        forecast_labels=data["forecast_labels"] if data else None
    )

# ===============================
# 🔮 PREDICTION ROUTE
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    features = [
        float(request.form["temp"]),
        float(request.form["humidity"]),
        float(request.form["cloud"]),
        float(request.form["wind"]),
        float(request.form["rain"]),
        float(request.form["radiation"]),
        float(request.form["day"]),
        float(request.form["month"])
    ]

    radiation = features[5]

    # 🔥 FIX: if no radiation → directly return 0
    if radiation <= 0:
        prediction = 0.0
    else:
        final = np.array(features).reshape(1, -1)
        final_scaled = sc_X.transform(final)

        prediction_scaled = model.predict(final_scaled)
        prediction = float(sc_y.inverse_transform(prediction_scaled)[0][0])

        prediction = round(abs(prediction), 2)

    # ===============================
    # 📊 FORECAST (YOUR ORIGINAL - NOT TOUCHED)
    # ===============================
    hours = list(range(6, 18))

    forecast = []
    for h in hours:
        peak = np.exp(-((h - 12) ** 2) / 10)
        value = prediction * peak
        forecast.append(float(round(value, 2)))

    session['last_data'] = {
        "features": features,
        "prediction": prediction,
        "forecast": forecast
    }

    session['result'] = {
        "prediction_text": f"Predicted Solar Power: {prediction} kW",
        "prediction_value": prediction,
        "forecast_data": forecast,
        "forecast_labels": [f"{h}:00" for h in hours]
    }

    return redirect(url_for('home'))

# ===============================
# GAUGE
# ===============================
@app.route("/gauge/<value>")
def gauge(value):
    return render_template("gauge.html", value=float(value))

# ===============================
# 🌤 24 HOUR FORECAST API (UPDATED)
# ===============================
@app.route("/api/forecast")
def api_forecast():

    global forecast_data_24, forecast_date_24, forecast_labels_24

    today = date.today()

    if forecast_date_24 != today:

        data = session.get("last_data")

        if data and "prediction" in data:
            base_power = float(data["prediction"])
        else:
            base_power = 1.0

        forecast_labels_24, forecast_data_24 = generate_realistic_forecast(base_power)
        forecast_date_24 = today

    return jsonify({
        "labels": forecast_labels_24,
        "data": forecast_data_24
    })

# ===============================
# FORECAST PAGE
# ===============================
@app.route("/forecast")
def forecast():
    return render_template("forecast.html")

# ===============================
# 📄 PDF DOWNLOAD
# ===============================
@app.route("/download_pdf")
def download_pdf():

    data = session.get("last_data")

    if data and data.get("forecast"):
        data["peak"] = max(data["forecast"])
        data["peak_time"] = f"{data['forecast'].index(data['peak'])}:00"

    # generate chart
    chart_img = generate_chart_image(
        list(range(len(data["forecast"]))),
        data["forecast"]
    )
    data["chart"] = chart_img

    # ✅ just render HTML (no PDF backend)
    return render_template("pdf_template.html", data=data)

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
