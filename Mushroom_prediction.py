
# This is the main file that is used for building the prediction model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split 




df_mushrooms = pd.read_csv('mushrooms.csv')
for col in df_mushrooms.columns.values:
    if len(df_mushrooms[col].unique()) <= 1:
        print("Removing column {}, which only contains the value: {}".format(col, df_mushrooms[col].unique()[0]))
        df_mushrooms.drop(col,axis='columns', inplace=True)

col_names = df_mushrooms.columns 
  
for c in col_names: 
    df = df_mushrooms.replace("?", np.NaN) 
df = df.apply(lambda x:x.fillna(x.value_counts().index[0])) 

df = df.astype('category')

category_col =['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
       'spore-print-color', 'population', 'habitat']  



from sklearn import preprocessing 
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 
for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
    

df['class'] = df['class'].apply(lambda x:1 if x=='e' else 0)

X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

#Using RandomForest Classifier as it gave the best accuracy
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))

features_list = X.columns.values

# Saving the model and column names for future use

# saving the model
import pickle
with open('model.pkl','wb') as file:
    pickle.dump(rf, file)



# saving the columns
model_columns = features_list
with open('model_columns.pkl','wb') as file:
    pickle.dump(model_columns, file)


# save the final dataframe as a db. We will use this db to do the selection and prediction
import sqlalchemy as db
from sqlalchemy import create_engine, types
engine = db.create_engine("mysql+pymysql://root:root@127.0.0.1:3306/mushroom_db",
                            encoding='latin1', echo=False)
df.to_sql(con=engine,index=False,  name='mushroom_attributes', if_exists='replace')


#get the data from the db
conn = engine.connect()

table = 'mushroom_attributes'

# select * from 'user'
select_st = 'Select * from ' +table +' limit 0,10;'
res = conn.execute(select_st)
for _row in res:
    print(_row)
