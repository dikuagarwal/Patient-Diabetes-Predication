import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import logistic regresor model and standard scaler pickle

log_reg_model = pickle.load(open('Model/log_reg.pkl','rb'))
standard_scaler = pickle.load(open('Model/scaler.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

## Route for Diabetes predication
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():

    result= ""

    if request.method=='POST':
        
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))


        new_data= standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict= log_reg_model.predict(new_data)
        

        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        return render_template('home.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")