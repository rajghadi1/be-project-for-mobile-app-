from flask import Flask, request, render_template
import numpy as np
from pickle4 import pickle
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)
model = pickle.load(open('beproject.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('page.html')


@app.route('/predict', methods=['POST'])
def predict():
 if request.method == 'POST':
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    output = model.predict(features_value)
    if output == 1:
        return render_template('detected.html', predict_text='Parkinson’s Disease Detected')
    elif output == 0:
        return render_template('notdetected.html', predict_text='Parkinson’s Disease Not Detected')
    else:
        return render_template('page.html', predict_text='Something went wrong')


if __name__ == '__main__':
    app.run(debug=True)