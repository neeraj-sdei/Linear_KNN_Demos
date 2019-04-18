import os
import pickle

import numpy as np
from flask import Flask, render_template, request,jsonify
from kaggleKnClassifier import KaggleCustomerBasket as kg
import turicreate as tc
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader

print(os.path.dirname(__file__))
env = Environment(
    loader=PackageLoader(__name__, 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)
classifier_f = open("basketbalLinear.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
app = Flask(__name__)
port=str(8009)
host="http://35.165.235.204:"
#host="http://0.0.0.0:"
formUrl=host+port+"/points"

model = tc.load_model("recommend_movies")


@app.route('/basketball')
def basketball():
    return render_template("inputfile.html",formURL=formUrl)

@app.route('/')
def hello():
    url1= host+port+"/basketball"
    url2= host+port+"/customer-basket"
    url3= host+port+"/recommend"
    return render_template("welcome.html",url1=url1, url2=url2,url3=url3)


@app.route('/customer-basket')
def customerKNN():
    graph1_url = kg().getGraph()

    return render_template("graphfile.html",graph=graph1_url)

@app.route('/recommend')
def recommendationEngine():
    return render_template("recommendation.html")

@app.route('/recommend/<int:num>')
def recommendations(num):
    template = env.get_template('template.html')

    movies= list(model.recommend(users=[num],k=5))

    for movie in movies:
        print(movie)
    return template.render(movies=movies,user_id=num)

# @app.route('/count-people')
# def countPeople():
#     cap = cv2.VideoCapture('motion.mp4')
#     r, img = cap.read()
#     frame= yield (b'--frame\r\n'
#            b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
#
#     return render_template("countingpeople.html",vid=frame)



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
   app.run(host='0.0.0.0',debug=True,port=8009)
