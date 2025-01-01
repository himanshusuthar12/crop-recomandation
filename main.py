from flask import Flask,request,app,render_template,jsonify
import numpy as np
import pickle


# Now you can use the model for predictions, evaluation, etc.

app = Flask(__name__,template_folder='templets')
# Load the model
model=pickle.load(open('crop-recomandation.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_freature = [float(x)for x in request.form.values()]
    final_feature = [np.array(int_freature)]
    predction = model.predict(final_feature)
    output = round(predction[0])
    col = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    for i in range(22):
        if i+1 == output:
            val = col[i]

    return render_template("index.html",prediction_text="You will try: {}".format(val))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__=="__main__":
    app.run(debug=True)
   