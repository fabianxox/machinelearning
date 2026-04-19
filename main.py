from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# -----------------------------
# Training data (normal behavior)
# -----------------------------

X = np.array([
    [40, 60],
    [42, 62],
    [45, 65],
    [43, 63],
    [44, 64],
    [41, 61],
    [46, 66]
])

# Compute statistics
mu = np.mean(X, axis=0)
variance = np.var(X, axis=0)

epsilon = 0.00001


# -----------------------------
# Gaussian probability
# -----------------------------

def gaussian_prob(x, mu, var):

    coeff = 1 / np.sqrt(2 * np.pi * var)
    exp = np.exp(-((x - mu) ** 2) / (2 * var))

    return coeff * exp


# -----------------------------
# Detection logic
# -----------------------------

def detect_anomaly(temp, hum):

    p1 = gaussian_prob(
        temp,
        mu[0],
        variance[0]
    )

    p2 = gaussian_prob(
        hum,
        mu[1],
        variance[1]
    )

    probability = p1 * p2

    if probability < epsilon:
        return True, probability

    return False, probability


# -----------------------------
# Request schema
# -----------------------------

class Sensor(BaseModel):

    temperature: float
    humidity: float


# -----------------------------
# Path operator
# -----------------------------

@app.post("/sensor")
def post(data: Sensor):

    is_anomaly, probability = detect_anomaly(
        data.temperature,
        data.humidity
    )

    print(
        "Temp:",
        data.temperature,
        "Humidity:",
        data.humidity
    )

    return {
        "temperature": data.temperature,
        "humidity": data.humidity,
        "probability": float(probability),
        "anomaly": is_anomaly
    }