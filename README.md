# 📊 Loan Eligibility Prediction and Analysis using Machine Learning
This project focuses on analyzing and predicting loan eligibility based on user data using various machine learning algorithms. A dataset containing 600 rows of user financial and personal data has been thoroughly explored and preprocessed to build a robust predictive model for loan approval classification.

🔍 Project Overview
The goal of this project is to assist financial institutions in automating the loan approval process by accurately identifying whether a loan application is likely to be approved based on applicant information. The project involves:

->Exploratory Data Analysis (EDA) on 600 records

->Data cleaning and preprocessing using pandas and numpy

->Building and evaluating multiple machine learning models

->Deploying the model as an interactive web application using Streamlit

🛠️ Technologies & Libraries Used
Python

 Pandas – for data manipulation and preprocessing

NumPy– for numerical operations

**Matplotlib & Seaborn** – for data visualization

**Scikit-learn** – to implement:

->Linear Regression

->Logistic Regression

->Random Forest Classifier

**Streamlit** – for creating a user-friendly web interface

🔬 **Machine Learning Models**
After preprocessing the data (handling missing values, encoding categorical variables, scaling, etc.), three key models were trained and evaluated:

**Logistic Regression **– to classify whether a user is eligible for a loan

**Random Forest** – for improved accuracy and handling complex patterns

**Linear Regression** – used for baseline comparison (though logistic is more suited for binary classification)

Each model's performance was measured using accuracy, precision, recall, and F1-score to determine the best fit for real-world deployment.

🌐 Streamlit App
The final model is deployed using Streamlit, allowing users to input values and get real-time predictions on loan eligibility. This provides an intuitive and interactive experience for end users.

✅ Features
End-to-end machine learning workflow

Clean, modular code with comments

Interactive web app with dynamic input forms

Easy to customize and scale with more data

📂 Dataset
The dataset used contains 600 entries with various features such as:

Applicant income

Loan amount

Credit history

Education level

Property area

Marital status, etc.

🚀 How to Run
Clone the repository

Install dependencies: pip install -r requirements.txt

Run the Streamlit app: streamlit run app.py
