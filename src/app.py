import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Churn AI", layout="wide")

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction System")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input("Tenure (months)", 0, 72, 6)
    monthly = st.number_input("Monthly Charges (₹)", 18, 120, 80)
    total = st.number_input("Total Charges (₹)", 18, 9000, 500)

with col2:
    st.subheader("Services")

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No"])

    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No"])
    backup = st.selectbox("Online Backup", ["Yes", "No"])
    device = st.selectbox("Device Protection", ["Yes", "No"])
    support = st.selectbox("Tech Support", ["Yes", "No"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

input_data = pd.DataFrame(columns=columns)
input_data.loc[0] = 0

def set_feature(name):
    if name in input_data.columns:
        input_data[name] = 1

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly
input_data["TotalCharges"] = total
input_data["SeniorCitizen"] = 1 if senior == "Yes" else 0

if gender == "Male":
    set_feature("gender_Male")

if partner == "Yes":
    set_feature("Partner_Yes")

if dependents == "Yes":
    set_feature("Dependents_Yes")

if phone == "Yes":
    set_feature("PhoneService_Yes")

if multiple == "Yes":
    set_feature("MultipleLines_Yes")

if internet == "Fiber optic":
    set_feature("InternetService_Fiber optic")
elif internet == "No":
    set_feature("InternetService_No")

if online_sec == "Yes":
    set_feature("OnlineSecurity_Yes")

if backup == "Yes":
    set_feature("OnlineBackup_Yes")

if device == "Yes":
    set_feature("DeviceProtection_Yes")

if support == "Yes":
    set_feature("TechSupport_Yes")

if contract == "One year":
    set_feature("Contract_One year")
elif contract == "Two year":
    set_feature("Contract_Two year")

payment_map = {
    "Electronic check": "PaymentMethod_Electronic check",
    "Mailed check": "PaymentMethod_Mailed check",
    "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
    "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
}

set_feature(payment_map.get(payment, ""))

st.markdown("---")

if st.button("Predict"):

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("Churn Probability")
    st.progress(prob)

    st.write(f"Probability: {prob*100:.2f}%")

    if prob > 0.7:
        st.error("High Risk")
    elif prob > 0.4:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")