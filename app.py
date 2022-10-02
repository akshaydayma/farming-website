from flask import Flask, request, render_template
import pandas as pd
import joblib
import pickle


# Declare a Flask app
app = Flask(__name__)
@app.route("/",methods=["GET"])
def welcome():
    return render_template("website.html")

@app.route('/submit', methods=['POST'])
def submit():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        #clf = joblib.load("RandomForest.pkl")
        clf = pickle.load(open('RandomForest.pkl','rb'))
        
        # Get values through input bars
        N = int(request.form["N"])
        P = int(request.form["P"])
        K = int(request.form["K"])
        temp = float(request.form["temp"])
        humidity = float(request.form["humidity"])
        ph =  float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
        
        # Put inputs to dataframe
        #X = pd.DataFrame([N,P,K,temp,humidity,ph,rainfall], columns = ["N", "P","K","temp","humidity","ph","rainfall"])
        data = clf.predict([[N,P, K, temp, humidity, ph, rainfall]])
        ans = data[0]
    return render_template("website.html",output="Recommended Crop for you is {}".format(ans))
if __name__=="__main__":
    app.run()