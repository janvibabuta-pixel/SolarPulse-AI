from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
import numpy as np
import joblib
import tensorflow as tf
import math
from datetime import date
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image  # Added Image
from reportlab.lib.styles import getSampleStyleSheet
import io
import base64   
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# ===============================
# 🔥 CHART GENERATION
# ===============================
def generate_chart_image(labels, values):
    plt.figure(figsize=(8,4))
    plt.plot(labels, values, color='#f39c12', marker='o', linewidth=2)
    plt.fill_between(labels, values, color='#f39c12', alpha=0.3)

    plt.title("24 Hour Solar Forecast")
    plt.xlabel("Hour")
    plt.ylabel("Power (kW)")
    plt.grid(True, linestyle='--', alpha=0.6)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
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
model = tf.keras.models.load_model(
    "solar_ann_model.keras",
    compile=False,
    safe_mode=False
)

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

    session['form_values'] = {
        "temp": features[0], "humidity": features[1], "cloud": features[2],
        "wind": features[3], "rain": features[4], "radiation": features[5],
        "day": features[6], "month": features[7]
    }

    radiation = features[5]
    if radiation <= 0:
        prediction = 0.0
    else:
        final = np.array(features).reshape(1, -1)
        final_scaled = sc_X.transform(final)
        prediction_scaled = model.predict(final_scaled)
        prediction = float(sc_y.inverse_transform(prediction_scaled)[0][0])
        prediction = round(max(prediction, 0), 2)

    hours = list(range(6, 18))
    forecast = [float(round(prediction * np.exp(-((h - 12) ** 2) / 10), 2)) for h in hours]
    labels = [f"{h}:00" for h in hours]

    session['last_data'] = {
        "features": features,
        "prediction": prediction,
        "forecast": forecast,
        "forecast_labels": labels
    }

    session['result'] = {
        "prediction_text": f"Predicted Solar Power: {prediction} kW",
        "prediction_value": prediction,
        "forecast_data": forecast,
        "forecast_labels": labels
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
        base_power = float(data["prediction"]) if data and "prediction" in data else 1.0
        forecast_labels_24, forecast_data_24 = generate_realistic_forecast(base_power)
        forecast_date_24 = today
    return jsonify({"labels": forecast_labels_24, "data": forecast_data_24})

# ===============================
# FORECAST PAGE
# ===============================
@app.route("/forecast")
def forecast():
    return render_template("forecast.html")

# ===============================
# 📄 DOWNLOAD PDF
# ===============================
@app.route("/download_pdf")
def download_pdf():
    data = session.get("last_data")
    if not data:
        return "No data available. Please make a prediction first.", 400

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # --- Title ---
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#2c3e50")
    elements.append(Paragraph("SolarPulse AI: Energy Analysis Report", title_style))
    elements.append(Spacer(1, 20))

    # --- Add Chart Image to PDF ---
    chart_base64 = generate_chart_image(data["forecast_labels"], data["forecast"])
    chart_bytes = io.BytesIO(base64.b64decode(chart_base64))
    report_img = Image(chart_bytes, width=450, height=225)
    elements.append(report_img)
    elements.append(Spacer(1, 20))

    # --- Input Parameters Table ---
    elements.append(Paragraph("Input Parameters", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    input_labels = ["Temperature (°C)", "Humidity (%)", "Cloud Cover (%)", "Wind Speed (m/s)", 
                    "Rainfall (mm)", "Radiation (W/m²)", "Day of Month", "Month"]
    table_data = [["Parameter", "Value"]]
    for label, val in zip(input_labels, data["features"]):
        table_data.append([label, f"{val}"])

    t = Table(table_data, colWidths=[200, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f39c12")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 25))

    # --- Prediction Result ---
    elements.append(Paragraph(f"<b>Estimated Peak Solar Generation:</b> {data['prediction']} kW", styles["Normal"]))
    elements.append(Spacer(1, 25))

    # --- Hourly Forecast Table ---
    elements.append(Paragraph("Hourly Forecast Data", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    forecast_table = [["Time", "Output (kW)"]]
    for label, val in zip(data["forecast_labels"], data["forecast"]):
        forecast_table.append([label, f"{val} kW"])

    ft = Table(forecast_table, colWidths=[150, 150])
    ft.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ]))
    elements.append(ft)

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"Solar_Report_{date.today()}.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)
