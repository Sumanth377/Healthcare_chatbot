from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
from datetime import datetime
from datetime import date
import random
import threading
import os
import time
import urllib.request
import urllib.parse
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import tensorflow as tf
from keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout,LeakyReLU,LSTM,GRU,MaxPooling1D,Bidirectional,LSTM,Input,BatchNormalization
from keras.models import Sequential
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
modeldata = tf.keras.models.load_model('Train.h5')
training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')

df_train = pd.read_csv('Data/Training.csv')
df_test = pd.read_csv('Data/Testing.csv')
print('=====Feature Extracted Data=========')
print(df_train)
print('Size of Data')
print('==================')
print(df_train.shape)
print('=====Label Data=========')
print(df_train.prognosis.value_counts())
import numpy as np
labelencoder = LabelEncoder()
df_train['prognosis'] = labelencoder.fit_transform(df_train['prognosis'])


Y_train = df_train['prognosis'].values.reshape(-1,1)
Y_train=np.ravel(Y_train)
df_test['prognosis'] = labelencoder.fit_transform(df_test['prognosis'])
Y_test = df_test['prognosis'].values.reshape(-1,1)
Y_test=np.ravel(Y_test)

X_train= df_train.drop(['prognosis'],axis=1).values
X_test= df_test.drop(['prognosis'],axis=1).values

from keras.layers import Flatten, Dense, Conv1D, MaxPool1D, Dropout,LeakyReLU,LSTM,GRU,MaxPooling1D,Bidirectional,LSTM,Input,BatchNormalization
from keras.models import Sequential
# Create sequential model
import numpy as np
X_train1 = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1], 1)
X_test1 = np.array(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
cnn_model1 = Sequential()
#First CNN layer  with 32 filters, conv window 3, relu activation and same padding
cnn_model1.add(LSTM(16,return_sequences=True,input_shape=(X_train.shape[1],1)))
cnn_model1.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation=LeakyReLU(alpha=0.001)))
cnn_model1.add(BatchNormalization())
#Second CNN layer  with 64 filters, conv window 3, relu activation and same padding
cnn_model1.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation=LeakyReLU(alpha=0.001)))
cnn_model1.add(BatchNormalization())
#Fourth CNN layer with Max pooling
cnn_model1.add(MaxPooling1D(pool_size=(3,), strides=2, padding='same'))
cnn_model1.add(Dropout(0.2))
#Flatten the output
cnn_model1.add(Flatten())
#Add a dense layer with 256 neurons
cnn_model1.add(Dense(units = 128, activation=LeakyReLU(alpha=0.001)))
#Add a dense layer with 512 neurons
cnn_model1.add(Dense(units = 256, activation=LeakyReLU(alpha=0.001)))
#Softmax as last layer with five outputs
cnn_model1.add(Dense(units = 41, activation='softmax'))
cnn_model1.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model1.summary()
##history = cnn_model1.fit(X_train1, Y_train, epochs=5, batch_size = 32, validation_data = (X_test1, Y_test))
### summarize history for accuracy
##tacc=history.history['accuracy']
##plt.plot(tacc)
##plt.plot(history.history['val_accuracy'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()
### summarize history for loss
##plt.plot(history.history['loss'])
##plt.plot(history.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()
##cnn_model1.save('/content/gdrive/MyDrive/Train.h5')


cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()
y_predict=modeldata.predict(X_test)
y_predict=np.argmax(y_predict,axis=1)
y_true=Y_test
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
test_accuracy=accuracy_score(y_true, y_predict)

print("Accuracy: ")
print(test_accuracy)

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

severityDictionary=dict()
precautionDictionary=dict()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Symres')
def Symres():
    return render_template('Symres.html')
#Adding translator
from googletrans import Translator

translator = Translator()
@app.route('/translate', methods=['POST'])
def translate_text():
    
    text = request.form['text']
    target = request.form['target']
    source = request.form['source']
    translated = translator.translate(text, dest=target, src=source)
    return translated.text

@app.route('/Symptoms',methods=['POST','GET'])
def Symptoms():
    second_prediction=''
    _description=''
    FData1=''
    FData2=''
    FData3=''
    FData4=''
    Ddata1=''
    Ddata2=''
    if request.method=='POST':
        sym1=request.form['sym1']
        sym2=request.form['sym2']
        sym3=request.form['sym3']
        sym4=request.form['sym4']
        sym5=request.form['sym5']
        sym6=request.form['sym6']
        sym7=request.form['sym7']
        sym8=request.form['sym8']
        sym9=request.form['sym9']
        sym10=request.form['sym10']
        symptoms_exp=[]
        if sym1=='yes':
            symptoms_exp.append('continuous_sneezing')
        if sym2=='yes':
            symptoms_exp.append('fatigue')
        if sym3=='yes':
            symptoms_exp.append('high_fever')
        if sym4=='yes':
            symptoms_exp.append('muscle_pain')
        if sym5=='yes':
            symptoms_exp.append('runny_nose')
        if sym6=='yes':
            symptoms_exp.append('skin_rash')
        if sym7=='yes':
            symptoms_exp.append('joint_pain')
        if sym8=='yes':
            symptoms_exp.append('vomiting')
        if sym9=='yes':
            symptoms_exp.append('back_pain')
        if sym10=='yes':
            symptoms_exp.append('burning_micturition')
        print(symptoms_exp)
        df = pd.read_csv('Data/Training.csv')
        X = df.iloc[:, :-1]
        symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1
        X_Res = np.array([input_vector]).reshape(1,-1,1)
##        print(X_Res)
        y_predict=modeldata.predict(X_Res)
        predict=np.argmax(y_predict,axis=1)
        if(predict==0):
            fpredict="(vertigo) Paroymsal  Positional Vertigo"
        elif(predict==1):
            fpredict="AIDS"
        elif(predict==2):
            fpredict="Acne"
        elif(predict==3):
            fpredict="Alcoholic hepatitis"
        elif(predict==4):
            fpredict="Allergy"
        elif(predict==5):
            fpredict="Arthritis"
        elif(predict==6):
            fpredict="Bronchial Asthma"
        elif(predict==7):
            fpredict="Cervical spondylosis"
        elif(predict==8):
            fpredict="Chicken pox"
        elif(predict==9):
            fpredict="Chronic cholestasis"
        elif(predict==10):
            fpredict="Common Cold"
        elif(predict==11):
            fpredict="Dengue"
        elif(predict==12):
            fpredict="Diabetes"
        elif(predict==13):
            fpredict="Dimorphic hemmorhoids(piles)"
        elif(predict==14):
            fpredict="Drug Reaction"
        elif(predict==15):
            fpredict="Fungal infection"
        elif(predict==16):
            fpredict="GERD"
        elif(predict==17):
            fpredict="Gastroenteritis"
        elif(predict==18):
            fpredict="Heart attack"
        elif(predict==19):
            fpredict="Hepatitis B"
        elif(predict==20):
            fpredict="Hepatitis C"
        elif(predict==21):
            fpredict="Hepatitis D"
        elif(predict==22):
            fpredict="Hepatitis E"
        elif(predict==23):
            fpredict="Hypertension"
        elif(predict==24):
            fpredict="Hyperthyroidism"
        elif(predict==25):
            fpredict="Hypoglycemia"
        elif(predict==26):
            fpredict="Hypothyroidism"
        elif(predict==27):
            fpredict="Impetigo"
        elif(predict==28):
            fpredict="Jaundice"
        elif(predict==29):
            fpredict="Malaria"
        elif(predict==30):
            fpredict="Migraine"
        elif(predict==31):
            fpredict="Osteoarthristis"
        elif(predict==32):
            fpredict="Paralysis (brain hemorrhage)"
        elif(predict==33):
            fpredict="Peptic ulcer diseae"
        elif(predict==34):
            fpredict="Pneumonia"
        elif(predict==35):
            fpredict="Psoriasis"
        elif(predict==36):
            fpredict="Tuberculosis"
        elif(predict==37):
            fpredict="Typhoid"
        elif(predict==38):
            fpredict="Urinary tract infection"
        elif(predict==39):
            fpredict="Varicose veins"
        elif(predict==40):
            fpredict="hepatitis A"
        second_prediction=fpredict
        print(second_prediction)
        
        symptom_dataset = pd.read_csv('MasterData/symptom_Description.csv', names = ['Name', 'Description'])
        symptom_Description = pd.DataFrame()
        symptom_Description['name'] = symptom_dataset['Name']
        symptom_Description['descp'] = symptom_dataset['Description']
        symptom_record = symptom_Description[symptom_Description['name'] == second_prediction]
        print(symptom_record['descp'])
        _description=symptom_record['descp'].tolist()
        _description=_description[0]
        print(_description)
        precautionDictionary=dict()
        with open('MasterData/symptom_precaution.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                _prec={row[0]:[row[1],row[2],row[3],row[4]]}
                precautionDictionary.update(_prec)
        precution_list=precautionDictionary[second_prediction]
        print("Take following measures : ")
        FData=[]
        for  i,j in enumerate(precution_list):
            print(i+1,")",j)
            strdata=j
            FData.append(strdata)
        FData1=FData[0]
        FData2=FData[1]
        FData3=FData[2]
        FData4=FData[3]
        dimensionality_reduction = training.groupby(training['prognosis']).max()
        doc_dataset = pd.read_csv('Data/doctors_dataset.csv', names = ['Name', 'Description'])
        diseases = dimensionality_reduction.index
        diseases = pd.DataFrame(diseases)
        doctors = pd.DataFrame()
        doctors['name'] = np.nan
        doctors['link'] = np.nan
        doctors['disease'] = np.nan
        doctors['disease'] = diseases['prognosis']
        doctors['name'] = doc_dataset['Name']
        doctors['link'] = doc_dataset['Description']
        record = doctors[doctors['disease'] == second_prediction]
        Ddata1=record['name'].tolist()
        Ddata1=Ddata1[0]
        Ddata2=record['link'].tolist()
        Ddata2=Ddata2[0]
        second_prediction=second_prediction
        print(record['name'])
        print(record['link'].tolist())
##        myresult=1
##        if myresult==1:
##            return redirect(url_for('Symres',second_prediction=second_prediction,description=_description,Ddata1=Ddata1,Ddata2=Ddata2,FData1=FData1,FData2=FData2,FData3=FData3,FData4=FData4))
    return render_template('Symptoms.html',second_prediction=second_prediction,description=_description,Ddata1=Ddata1,Ddata2=Ddata2,FData1=FData1,FData2=FData2,FData3=FData3,FData4=FData4)
#################################>>>>>>>>>>>>>>>>>>USER<<<<<<<<<<<<<<<<############################################################

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
