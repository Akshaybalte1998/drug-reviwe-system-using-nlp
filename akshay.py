import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from flask import Flask, render_template,request,redirect,session
import os
import joblib
import pandas as pd
import re
import numpy

#MODEL_PATH = ''

#TOKENIZER_PATH ='tfidfvectorizer.pkl'

DATA_PATH =pd.read_csv(r"C:\Users\balte\Desktop\Drug_reviews_condition_classification-main\drugsComTrain.csv")
#df = pd.read_csv(r"C:\Users\balte\Desktop\chaman\drugsComTrain_raw.csv")
# loading vectorizer
vectorizer = pickle.load(r"C:\Users\balte\Desktop\Drug_reviews_condition_classification-main\tfidfvectorizer.pkl",'rb')
# loading model
model = pickle.load(r"C:\Users\balte\Desktop\Drug_reviews_condition_classification-main\passmodel.pkl",'rb')

#getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()




st.title('Drug Review')
def predict():
    	if request.method == 'POST':
		    raw_text = request.form['rawtext']
		
		
	        if raw_text != "":
			     clean_text = cleanText(raw_text)
                    clean_lst = [clean_text]tfidf_vect = vectorizer.transform(clean_lst)
                    prediction = model.predict(tfidf_vect)
			        predicted_cond = prediction[0]
			        df = pd.read_csv(DATA_PATH)
			        top_drugs = top_drugs_extractor(predicted_cond,df)
			
					
			        return render_template('predict.html',rawtext=raw_text,result=predicted_cond,top_drugs=top_drugs)}
		    else:
			      raw_text ="There is no text to select"

   
	
    
    
    
    
def cleanText(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))




def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst


if st.button('Recommend'):
    recommendations = predict()
    for i in recommendations:
        st.write(i)