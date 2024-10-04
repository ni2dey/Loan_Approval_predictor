import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import pickle

st.title("Loan Approval Predictor")
image = Image.open("images.png")
if image.mode != 'RGBA':
    image = image.convert('RGBA')

background = Image.new("RGBA", image.size, (255, 255, 255, 255))
background.paste(image, (0, 0), image)
final_image = background.convert("RGB")
st.sidebar.image(final_image, use_column_width=True)

#sidebar
st.sidebar.title("Welcome")
with st.sidebar:
    name = st.text_input("Enter your name:")


st.subheader("Fill the form below:📋")
"----"
def collect_data():

    # q1
    st.write("Q:1")
    option= {"Male": 1, "Female": 0}
    gender= st.radio('Pick your gender', option.keys(), index=None)


    # q2
    st.write("Q:2")
    status={"Yes":1,"No":0}
    marital=st.selectbox("Are you married?",("Yes","No"), index=None, placeholder="Choose an option",key="marital")


    # q3
    st.write("Q:3")
    depend=['0','1','2',"3+"]
    dependent=st.selectbox("How many number of dependent members you have?",depend, index=None, placeholder="Choose an option",key="depend")
    if dependent=="3+":
        dependent='3'


    # q4
    st.write("Q:4")
    educ1={"Graduate":0,"Not Graduate":1}
    education=st.selectbox("Select your education status:",educ1.keys(),index=None,placeholder="Choose an option",key='education')


    # q5
    st.write("Q:5")
    self={"Yes":1,"No":0}
    employed=st.radio("Are you Self-employed?",self.keys(), index=None,key="employed")


    # q6
    st.write("Q:6")
    income = st.number_input("Enter your Income:", min_value=0.0, max_value=10000000.0, value=0.0, step=1000.0, format="%.2f",key="income")

    # q7
    st.write("Q:7")
    co_income = st.number_input("Enter your Co-Applicant Income:", min_value=0.0, max_value=10000000.0, value=0.0, step=1000.0, format="%.2f",key="co_income")

    # q8
    st.write("Q:8")
    loan = st.number_input("Enter Loan Amount:", min_value=0.0, max_value=2000000.0, value=0.0, step=100.0, format="%.2f",key="loan")

    # q9
    st.write("Q:9")
    list1=[480,360,300,240,180,120,84,60,36,12]
    term=st.selectbox("Enter your loan term:",list1, index=None, placeholder="(in days)",key="term")

    # q10
    st.write("Q:10")
    credit=st.radio("What is your credit History?",['1','0'], index=None,key='credit')
    if credit=='1':
        credit=1
    else:
        credit=0

    #q11
    st.write("Q:11")
    area1={"Urban":2,"Semiurban":1,"Rural":0}
    propertyy=st.selectbox("What is your area of Property?",area1,index=None,placeholder="Choose an option",key='property')

    feature=[gender,marital,dependent,education,employed,income,co_income,loan,term,credit,propertyy]

    if all(feature):
        return feature
    else:
        st.warning("Fill all the questions!!!")

def scale_data(data):
    
    with open('scaler.pickle', 'rb') as f:
        loaded_scaler = pickle.load(f)
    with open('label.pickle', 'rb') as f:
        label = pickle.load(f)

    data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']] = loaded_scaler.transform(
        data[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']])

    # Apply label encoding
    list1 = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for i, feature in enumerate(list1):
        data[feature] = label[i].transform(data[feature])

    return data

def predict_result(data):

    with open('models.pickle', 'rb') as f:
        models = pickle.load(f)
        # Reshape the data to (1, -1) since it's a single sample with multiple features
        reshaped_data = np.array(data).reshape(1, -1)

        if 'result' not in st.session_state:
            st.session_state.result = models[0].predict(reshaped_data)  # Predict
            print(st.session_state.result)
            return st.session_state.result[0]
        else:
            st.warning("No result available.")
    return None

feature=collect_data()
print(feature)
if feature:
    data=pd.DataFrame([feature],columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])
    scaled_data=scale_data(data)
    print(scaled_data)

    if st.button("Submit"):
        prediction=str(predict_result(scaled_data))
        st.write(f"Prediction: {prediction}")
        if prediction=='1':
            st.image("accepted.jpeg")
        elif prediction=='0':
            st.image("loandenied.png")



