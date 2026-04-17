import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Churn AI", layout="wide")

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
h1 {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI-powered Customer Churn Intelligence System")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Customer Profile")

    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", 0, 72, 6)
    monthly = st.number_input("Monthly Charges (₹)", 18, 120, value=80, step=1, format="%d")
    total = st.number_input("Total Charges (₹)", 18, 9000, value=500, step=1, format="%d")

with col2:
    st.subheader("📡 Service Plan")

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

if monthly > 120 or total > 9000:
    st.warning("Values outside dataset range may reduce accuracy")

input_data = pd.DataFrame(columns=columns)
input_data.loc[0] = 0

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly
input_data["TotalCharges"] = total
input_data["SeniorCitizen"] = 1 if senior == "Yes" else 0

def set_feature(name):
    if name in input_data.columns:
        input_data[name] = 1

if contract == "One year":
    set_feature("Contract_One year")
elif contract == "Two year":
    set_feature("Contract_Two year")

if internet == "Fiber optic":
    set_feature("InternetService_Fiber optic")
elif internet == "No":
    set_feature("InternetService_No")

payment_map = {
    "Electronic check": "PaymentMethod_Electronic check",
    "Mailed check": "PaymentMethod_Mailed check",
    "Bank transfer (automatic)": "PaymentMethod_Bank transfer (automatic)",
    "Credit card (automatic)": "PaymentMethod_Credit card (automatic)"
}

set_feature(payment_map.get(payment, ""))

st.markdown("---")

if st.button("🔍 Analyze Customer"):

    prob = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Churn Risk Score")
    st.progress(prob)

    st.metric("Churn Probability", f"{prob*100:.1f}%")

    if prob > 0.6:
        st.error("🔴 High Risk Customer")
    elif prob > 0.4:
        st.warning("🟡 Medium Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")

    st.subheader("🧠 Key Insights")

    reasons = []

    if tenure < 6:
        reasons.append("Short tenure (new customer)")
    if contract == "Month-to-month":
        reasons.append("No long-term contract")
    if monthly > 80:
        reasons.append("High monthly charges")
    if internet == "Fiber optic":
        reasons.append("Premium internet plan")
    if payment == "Electronic check":
        reasons.append("Uses electronic check (higher churn rate)")
    elif payment == "Mailed check":
        reasons.append("Uses mailed check (moderate churn behavior)")

    if reasons:
        for r in reasons:
            st.write("- " + r)
    else:
        st.success("No major churn indicators found. Customer appears stable.")

    st.subheader("💡 Recommendation")

    if prob > 0.7:
        st.info("Offer discount or move customer to long-term plan")
    elif prob > 0.4:
        st.info("Engage with offers and support")
    else:
        st.info("Customer is stable, maintain service quality")