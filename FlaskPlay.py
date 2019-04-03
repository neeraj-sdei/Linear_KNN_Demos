import pickle

import numpy as np
from flask import Flask, render_template, request

from kaggleKnClassifier import KaggleCustomerBasket as kg


classifier_f = open("basketbalLinear.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
app = Flask(__name__)
port=str(8009)
formUrl="http://localhost:"+port+"/points"

@app.route('/basketball')
def basketball():
    return render_template("inputfile.html",formURL=formUrl)

@app.route('/')
def hello():
    url1= "http://localhost:"+port+"/basketball"
    url2= "http://localhost:"+port+"/customer-basket"
    return render_template("welcome.html",url1=url1, url2=url2)


@app.route('/customer-basket')
def customerKNN():
    graph1_url = kg().getGraph()

    return render_template("graphfile.html",graph=graph1_url)

@app.route('/points',methods = ['POST'])
def points():
    height= float(request.form['height'])
    weight= float(request.form['weight'])
    fieldGoals= float(request.form['fieldGoals'])
    freeThrows= float(request.form['freeThrows'])
    X=np.array([height,weight,fieldGoals,freeThrows]).reshape(1,4)
    print(X.shape)
    print(classifier.predict(X))
    predictedValue=str(classifier.predict(X)[0])
    return "<h1>average points scored per game:</h1>"+predictedValue

if __name__ == '__main__':
   app.run(debug=True,port=8009)