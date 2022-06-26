from flask import Flask,render_template, request, url_for,redirect
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
 
import sqlalchemy as db
from sqlalchemy import create_engine, types
engine = db.create_engine("mysql+pymysql://root:root@127.0.0.1:3306/mushroom_db",
                            encoding='latin1', echo=False)
# App definition
app = Flask(__name__,template_folder='templates')
 
 
@app.route('/',methods = ['POST','GET'])
def index():
    return render_template('index.html')
 
# prediction function 

@app.route('/result',methods = ['POST','GET'])
def result():
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Edible, Yes. You have selected an edible mushroom ..Good for Health!'
        else: 
            prediction ='The selected mushroom is Poisonus. Please change the combination to get edible'            
        return render_template('result.html', prediction = prediction)
    return render_template('result.html')

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 21) 
    loaded_model = pickle.load(open("model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 

    # change the array to a list
    print(type(to_predict))
    print(type(result))
    feature_list=to_predict.tolist()
    print(feature_list)
    
    a_str = ','.join(str(x) for x in result)
    feature_list[0].insert(0,a_str)
   
    print(feature_list)
    #Convert it into string before adding to the database

    stringvalue=','.join([str(elem) for elem in feature_list[0]])
    print(stringvalue)
    #get the data from the db
    conn = engine.connect()

    table = 'new_records'

    trans = conn.begin()
    # Insert the user given data to new_records table
    insert_st = 'Insert into ' +table +' values (' + stringvalue +');'
    conn.execute(insert_st)
    trans.commit()
    conn.close()  
    return result[0] 
  

@app.route('/newvalues')
def values():
    conn = engine.connect()
    table = 'new_records'
    # select * from 'new table'
    select_st = 'Select * from ' +table +';'
    res = conn.execute(select_st)
    for v in res:
        for column, value in v.items():
            print('{0}: {1}'.format(column, value))
    return render_template('newvalues.html',details=res)
 
if __name__ == "__main__":
   app.run()