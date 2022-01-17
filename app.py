from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import json

app = Flask(__name__)

ENV = 'dev'

if ENV == 'dev':
    app.debug = True
else:
    app.debug = False

model = pickle.load(open('ClassificationModel.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html', max_condition=52)


@app.route('/api/predict', methods=["POST"])
def predict():
    age = request.form.get('age')
    bmi = request.form.get('bmi')
    urban = request.form.get('urban')
    smoker = request.form.get('smoker')
    condition = request.form.get('condition')
    prev_conditions = request.form.get('prev_conditions[]')
    drugs_treated_with = request.form.get('drugs_treated_with[]')

    A = 1 if "A" in prev_conditions else 0
    B = 1 if "B" in prev_conditions else 0
    C = 1 if "C" in prev_conditions else 0
    D = 1 if "D" in prev_conditions else 0
    E = 1 if "E" in prev_conditions else 0
    F = 1 if "F" in prev_conditions else 0
    Z = 1 if "Z" in prev_conditions else 0

    DX1 = 1 if "DX1" in drugs_treated_with else 0
    DX2 = 1 if "DX2" in drugs_treated_with else 0
    DX3 = 1 if "DX3" in drugs_treated_with else 0
    DX4 = 1 if "DX4" in drugs_treated_with else 0
    DX5 = 1 if "DX5" in drugs_treated_with else 0
    DX6 = 1 if "DX6" in drugs_treated_with else 0

    num_prev_cond = len(prev_conditions)

    prediction = model.predict(
        pd.DataFrame([[
            int(condition),
            int(age),
            float(bmi), A, B, C, D, E, F, Z, num_prev_cond,
            int(smoker),
            int(urban), 1, 0, 0, 1, 0, 0
        ]],
                     columns=[
                         'Diagnosed_Condition', 'Patient_Age',
                         'Patient_Body_Mass_Index', 'A', 'B', 'C', 'D', 'E',
                         'F', 'Z', 'Number_of_prev_cond', 'Smoker', 'URBAN',
                         'DX1', 'DX2', 'DX3', 'DX4', 'DX5', 'DX6'
                     ]))

    will_survive = bool(prediction[0] == 1)
    return json.dumps({'will_survive': will_survive})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
