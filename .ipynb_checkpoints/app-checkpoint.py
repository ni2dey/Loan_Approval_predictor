import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
import numpy as np

import pickle

st.title("Loan Eligibility Predictor")
image = Image.open("images.png")
if image.mode != 'RGBA':
    image = image.convert('RGBA')

background = Image.new("RGBA", image.size, (255, 255, 255, 255))
background.paste(image, (0, 0), image)
final_image = background.convert("RGB")
st.sidebar.image(final_image, use_column_width=True)

st.sidebar.title("Welcome")
with st.sidebar:
    name = st.text_input("Enter your name:")
    gender=st.radio('Pick your gender', ["Male", "Female"],index=None)

st.subheader("Fill the form below:📋")
"----"
#q1
st.write("Q:1")
status={"Yes":1,"No":0}
marital=st.selectbox("Are you married?",("Yes","No"), index=None, placeholder="Choose an option")
if marital:
    marital=status[marital]
    print(f"marriage:{marital}")

#q2
st.write("Q:2")
depend=[0,1,2,"3+"]
dependent=st.selectbox("How many number of dependent members you have?",depend, index=None, placeholder="Choose an option")
if dependent==depend[3]:
    dependent=3

#q3
st.write("Q:3")
educ1={"Graduate":0,"Not Graduate":1}
education=st.selectbox("Select your education status:",educ1.keys(),index=None,placeholder="Choose an option")
if education:
    education=educ1[education]
    print(f"education:{education}")

#q4
st.write("Q:4")
self={"Yes":1,"No":0}
employed=st.radio("Are you Self-employed?",self.keys(), index=None)
if employed:
    employed=self[employed]
    print(f"Employed:{employed}")

#q5
st.write("Q:5")
income = st.number_input("Enter your Income:", min_value=0.0, max_value=10000000.0, value=0.0, step=1000.0, format="%.4f")

#q6
st.write("Q:6")
co_income = st.number_input("Enter your Co-Applicant Income:", min_value=0.0, max_value=10000000.0, value=0.0, step=1000.0, format="%.4f")

#q7
st.write("Q:7")
loan = st.number_input("Enter Loan Amount:", min_value=0.0, max_value=2000000.0, value=0.0, step=100.0, format="%.4f")

#q8
st.write("Q:8")
list1=[480,360,300,240,180,120,84,60,36,12]
term=st.selectbox("Enter your loan term:",list1,index=None,placeholder="Choose an option")

#q9
st.write("Q:9")
credit=st.radio("What is your credit History?",(1,0), index=None)

#q10
st.write("Q:10")
area1={"Urban":2,"SemiUrban":1,"Rural":0}
propertyy=st.selectbox("What is your area of Property?",area1,index=None,placeholder="Choose an option")
if propertyy:
    propertyy=area1[propertyy]

data=[[income,co_income,loan,term]]
with open('scaler.pickle', 'rb') as f:
    loaded_scaler = pickle.load(f)
scaled_data=loaded_scaler.transform(data)

total_data=[gender,marital,dependent,education,employed]
for value in scaled_data[0]:
    total_data.append(value)
total_data.append(credit)
total_data.append(propertyy)
features=[total_data]
print(features)

with open('models.pickle', 'rb') as f:
    models = pickle.load(f)
"------"
with st.container():
    # Model selection
    model = st.selectbox("Select your preferred model:", models, index=None)

    # Check if features are provided and valid
    if 'features' in st.session_state and (
            np.any(np.isnan(st.session_state.features)) or np.any(np.isnull(st.session_state.features))):
        st.warning("Enter the data first.")
    else:
        if model:
            # Ensure features is properly defined and in the correct format

            features = st.session_state.features
            result = model.predict(features)
            st.write(result)
        else:
            st.warning("Features not provided.")































