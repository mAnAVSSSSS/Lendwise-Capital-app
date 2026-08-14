"""
LendWise Capital — Streamlit Risk-Scoring App
================================================
Deploy: push this file + model_xgb.joblib to a GitHub repo,
then connect on share.streamlit.io. Or run locally:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="LendWise Capital — Default Risk Scorer", page_icon="💳")

@st.cache_resource
def load_model():
    return joblib.load("model_xgb.joblib")

model = load_model()

st.title("💳 LendWise Capital")
st.subheader("Loan Default Risk Scoring")
st.write(
    "Enter applicant and loan details to estimate the probability of default. "
    "Model trained on ~200K resolved LendingClub loans (XGBoost, AUC-ROC 0.72)."
)

with st.form("applicant_form"):
    col1, col2 = st.columns(2)

    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
        term = st.selectbox("Term (months)", [36, 60])
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
        installment = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=2000.0, value=300.0)
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox("Sub-Grade", [f"{grade}{i}" for i in range(1, 6)])
        purpose = st.selectbox(
            "Loan Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
             "small_business", "car", "medical", "other"],
        )
        application_type = st.selectbox("Application Type", ["Individual", "Joint App"])

    with col2:
        annual_inc = st.number_input("Annual Income ($)", min_value=0, max_value=1_000_000, value=60000, step=1000)
        emp_length = st.slider("Employment Length (years)", 0, 10, 5)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        verification_status = st.selectbox(
            "Income Verification", ["Verified", "Source Verified", "Not Verified"]
        )
        dti = st.slider("Debt-to-Income Ratio (DTI)", 0.0, 50.0, 18.0)
        open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=50, value=8)
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=8000)
        revol_util = st.slider("Revolving Credit Utilization (%)", 0.0, 150.0, 45.0)
        total_acc = st.number_input("Total Credit Accounts", min_value=0, max_value=100, value=20)
        credit_history_years = st.slider("Credit History Length (years)", 0.0, 40.0, 10.0)
        delinq_2yrs = st.number_input("Delinquencies (past 2 years)", min_value=0, max_value=20, value=0)
        pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
        pub_rec_bankruptcies = st.number_input("Public Record Bankruptcies", min_value=0, max_value=10, value=0)
        mort_acc = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=1)
        addr_state = st.selectbox("State", ["CA", "NY", "TX", "FL", "IL", "NJ", "Other"])

    submitted = st.form_submit_button("Predict Default Risk")

if submitted:
    dti_bucket = pd.cut([dti], bins=[-1, 10, 20, 30, 40, 100],
                         labels=["0-10", "10-20", "20-30", "30-40", "40+"])[0]

    input_df = pd.DataFrame([{
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "total_acc": total_acc,
        "addr_state": addr_state,
        "application_type": application_type,
        "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "credit_history_years": credit_history_years,
        "dti_bucket": dti_bucket,
    }])

    proba = model.predict_proba(input_df)[0, 1]
    risk_pct = proba * 100

    st.markdown("---")
    if proba < 0.15:
        st.success(f"**Predicted Default Risk: {risk_pct:.1f}%** — Low Risk")
    elif proba < 0.35:
        st.warning(f"**Predicted Default Risk: {risk_pct:.1f}%** — Moderate Risk")
    else:
        st.error(f"**Predicted Default Risk: {risk_pct:.1f}%** — High Risk")

    st.progress(min(proba, 1.0))
    st.caption(
        "This score reflects the model's estimated probability of default "
        "based on historical patterns. It should support, not replace, "
        "human underwriting judgment."
    )

st.markdown("---")
st.caption("LendWise Capital | Big Data & NLP Project | Group 4")
