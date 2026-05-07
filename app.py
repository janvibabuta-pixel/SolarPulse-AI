from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import numpy as np
import joblib
import tensorflow as tf
import math
from datetime import date
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import base64

# ===============================
# 🔥 CHART GENERATION
# ===============================
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

# ===============================
# 🔥 APP INIT
# ===============================
app = Flask(__name__)
app.secret_key = "secret123"

# ===============================
# 🔥 LOAD MODEL + SCALERS
# ===============================

import tensorflow as tf

model = tf.keras.models.load_model("solar_ann_model.keras", compile=False)

sc_X = joblib.load("scaler_X.pkl")
sc_y = joblib.load("scaler_y.pkl")

# ===============================
# 🔥 FORECAST CACHE
# ===============================
forecast_data_24 = None
forecast_date_24 = None
forecast_labels_24 = None

# ===============================
# 🔥 REALISTIC FORECAST
# ===============================
def generate_realistic_forecast(base_power):

    forecast = []
    labels = []

    for h in range(24):
        labels.append(f"{h}:00")

        # realistic solar curve
        if h < 6 or h > 18:
            value = 0
        else:
            peak = math.sin((math.pi / 12) * (h - 6))
            value = base_power * peak

        forecast.append(round(max(value, 0), 2))

    return labels, forecast

# ===============================
# HOME
# ===============================
@app.route('/')
def home():
    data = session.pop('result', None)

    # 👇 IMPORTANT CHANGE
    form_values = session.pop('form_values', None)

    return render_template(
        'index.html',
        prediction_text=data["prediction_text"] if data else None,
        prediction_value=data["prediction_value"] if data else None,
        forecast_data=data["forecast_data"] if data else None,
        forecast_labels=data["forecast_labels"] if data else None,
        form_values=form_values
    )

# ===============================
# 🔮 PREDICT
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

        # 🔥 SAVE USER INPUTS
    session['form_values'] = {
        "temp": features[0],
        "humidity": features[1],
        "cloud": features[2],
        "wind": features[3],
        "rain": features[4],
        "radiation": features[5],
        "day": features[6],
        "month": features[7]
    }

    radiation = features[5]

    if radiation <= 0:
        prediction = 0.0
    else:
        final = np.array(features).reshape(1, -1)
        final_scaled = sc_X.transform(final)

        prediction_scaled = model.predict(final_scaled)
        prediction = float(sc_y.inverse_transform(prediction_scaled)[0][0])

        prediction = max(prediction, 0)
        prediction = round(prediction, 2)

    # ===============================
    # FORECAST
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
# API FORECAST
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


@app.route("/download_pdf")
def download_pdf():

    data = session.get("last_data")

    if not data:
        return "No data available"

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []

    # TITLE
    elements.append(Paragraph("SolarPulse AI Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # INPUTS
    labels = [
        "Temperature","Humidity","Cloud Cover","Wind Speed",
        "Rainfall","Radiation","Day","Month"
    ]

    elements.append(Paragraph("Input Parameters:", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    for label, val in zip(labels, data["features"]):
        elements.append(Paragraph(f"{label}: {val}", styles["Normal"]))

    elements.append(Spacer(1, 12))

    # PREDICTION
    elements.append(Paragraph("Prediction:", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"{data['prediction']} kW", styles["Normal"]))

    elements.append(Spacer(1, 12))

    # FORECAST
    elements.append(Paragraph("Forecast (Hourly):", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    for i, val in enumerate(data["forecast"]):
        elements.append(Paragraph(f"{i+6}:00 → {val} kW", styles["Normal"]))

    doc.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="solar_report.pdf",
        mimetype="application/pdf"
    )
# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
