import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict',methods=['POST'])
def predict():
    


    int_features = [int(x) for x in request.form.values()]
    
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index2.html', prediction_text='Tahmini Fiyatı ₺ {}'.format(output))



if __name__ == "__main__":
	#port = int(os.environ.get('PORT', 33507))
     app.run()
    
    