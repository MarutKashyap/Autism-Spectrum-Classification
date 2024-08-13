from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('rand_forest.pkl')

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    A1_Score = request.form['A1_Score']
    A2_Score = request.form['A2_Score']
    A3_Score = request.form['A3_Score']
    A4_Score = request.form['A4_Score']
    A5_Score = request.form['A5_Score']
    A6_Score = request.form['A6_Score']
    A7_Score = request.form['A7_Score']
    A8_Score = request.form['A8_Score']
    A9_Score = request.form['A9_Score']
    A10_Score = request.form['A10_Score']
    age = request.form['age']
    result = request.form['result']
    m = request.form['m']
    Had_jaundice_yes = request.form['Had_jaundice_yes']
    Rel_had_yes = request.form['Rel_had_yes']

    
    X = [[int(A1_Score),int(A2_Score),int(A3_Score),int(A4_Score),int(A5_Score),int(A6_Score),int(A7_Score),int(A8_Score),int(A9_Score),int(A10_Score),int(age),int(result),int(m),int(Had_jaundice_yes),int(request.form['Rel_had_yes'])]]
    prediction = model.predict(X)[0]
    
    return render_template('predict.html', prediction = "Detected {}".format(prediction))
if __name__ == "__main__": 
    app.run(debug=True)


