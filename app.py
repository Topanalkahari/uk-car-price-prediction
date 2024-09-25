from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load(open("model_knn.jlb", "rb"))

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def prediksi():

    data1 = float(request.form['tahun']) 
    data2 = float(request.form['pajak'])
    data3 = float(request.form['engine'])

    arr = np.array([[data1, data2, data3]])
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    scaler.fit(arr)
    arr_scal = scaler.transform(arr)

    pred = model.predict(arr_scal)

    return render_template('index.html', prediction = "{}".format(pred[0]))

if __name__ == "__main__":
    app.run(debug=True)