from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open("regmodel.pkl", "rb") as f:
    regmodel = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # must be a dict
    input_array = np.array(list(data.values())).reshape(1, -1)
    new_data = scaler.transform(input_array)
    output = regmodel.predict(new_data)
    return jsonify({"predicted_price": output[0]})

if __name__ == "__main__":
    app.run(debug=True)