import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import paho.mqtt.client as mqtt
import json

# ===== Load model, scaler, dan label encoder =====
model = load_model(r"C:\Users\ferdi\OneDrive - Universitas Negeri Malang\Documents\Project traktor\programm server\content\programm\model_ann.h5")
scaler = joblib.load(r"C:\Users\ferdi\OneDrive - Universitas Negeri Malang\Documents\Project traktor\programm server\content\programm\scaler.pkl")
label_encoder = joblib.load(r"C:\Users\ferdi\OneDrive - Universitas Negeri Malang\Documents\Project traktor\programm server\content\programm\label_encoder.pkl")

# ===== Fungsi prediksi =====
def prediksi_tanaman(ph, n, p, k):
    input_data = pd.DataFrame([[ph, n, p, k]], columns=['PH', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'])
    input_scaled = scaler.transform(input_data)
    prediksi_prob = model.predict(input_scaled, verbose=0)
    prediksi_kelas = np.argmax(prediksi_prob, axis=1)
    return label_encoder.inverse_transform(prediksi_kelas)[0]

# ===== Variabel global =====
state_active = False
data_buffer = []  # Simpan semua pembacaan sensor untuk dirata-rata
BROKER = "broker.emqx.io"
PORT = 1883

# ===== Callback MQTT =====
def on_connect(client, userdata, flags, rc):
    print(f"Terhubung ke broker MQTT dengan kode: {rc}")
    client.subscribe("sensor/state")
    client.subscribe("sensor/tanah")

def on_message(client, userdata, msg):
    global state_active, data_buffer

    if msg.topic == "sensor/state":
        try:
            state_value = int(msg.payload.decode().strip())
            print(f"State diterima: {state_value}")

            if state_value == 1:
                state_active = True
                data_buffer.clear()
                print("Mulai pembacaan data sensor...")
            elif state_value == 0 and state_active:
                state_active = False
                print("Stop pembacaan, proses prediksi...")

                if data_buffer:
                    avg_data = np.mean(data_buffer, axis=0)  # Rata-rata PH, N, P, K
                    ph, n, p, k = avg_data
                    hasil_prediksi = prediksi_tanaman(ph, n, p, k)

                    # Publish hasil
                    client.publish("sensor/N", n)
                    client.publish("sensor/P", p)
                    client.publish("sensor/K", k)
                    client.publish("sensor/PH", ph)
                    client.publish("sensor/prediksi", hasil_prediksi)

                    print(f"Hasil prediksi: {hasil_prediksi}")
                else:
                    print("Tidak ada data untuk diproses.")
        except Exception as e:
            print(f"Error parsing state: {e}")

    elif msg.topic == "sensor/tanah" and state_active:
        try:
            data = json.loads(msg.payload.decode().strip())
            ph = float(data.get("PH", 0))
            n = float(data.get("N", 0))
            p = float(data.get("P", 0))
            k = float(data.get("K", 0))
            data_buffer.append([ph, n, p, k])
            print(f"Data sensor masuk: PH={ph}, N={n}, P={p}, K={k}")
        except Exception as e:
            print(f"Error parsing sensor data: {e}")

# ===== Setup MQTT client =====
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, keepalive=60)
print("Menunggu data MQTT...")
client.loop_forever()
