from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app


@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")



@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 




@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            rnd_spend =float(request.form['rnd_spend'])
            administration =float(request.form['administration'])
            marketing_spend =float(request.form['marketing_spend'])
            florida =int(request.form['florida'])
            newyork=int(request.form['newyork'])
            
         
            data = [rnd_spend,administration,marketing_spend,florida,newyork]
            #return data
            data = np.array(data).reshape(1, 5)
            data =pd.DataFrame(data)
            #states=pd.get_dummies(data[3],drop_first=True)
            #data=data.drop('state',axis=1)
            #data=pd.concat([data,states],axis=1)
        
            obj = PredictionPipeline()
            predict = obj.predict(data)

            #return str(predict)
            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')



if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)