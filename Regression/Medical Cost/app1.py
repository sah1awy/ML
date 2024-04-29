import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

with open("pipeline.pkl","rb") as f:
    regressor = pickle.load(f)

def welcome():
    return "Welcome All"

def predict_medical_cost(age,sex,bmi,children,smoker):
    prediction=regressor.predict([[age,sex,bmi,children,smoker]])
    print(prediction)
    return prediction



def main():
    st.title("Medical Cost")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Medical Cost Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("age","Type Here")
    sex = st.text_input("sex","Type Here")
    bmi = st.text_input("bmi","Type Here")
    children = st.text_input("children","Type Here")
    smoker = st.text_input("smoker","Type Here")
    result= 0
    if st.button("Predict"):
        result=predict_medical_cost(age,sex,bmi,children,smoker)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    