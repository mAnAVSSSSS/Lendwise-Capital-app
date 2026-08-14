"""
LendWise Capital — Credit Risk Assessment Platform
=====================================================
Redesigned frontend, same backend: XGBoost model trained on 200,000
resolved LendingClub loans (sampled from a 2,260,668-row / 1.3M+
resolved-loan dataset). No changes to model, features, or prediction
logic — this file only changes presentation.

Run: streamlit run app.py
Needs in the same folder: model_xgb.joblib
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config + design tokens
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LendWise Capital | Risk Assessment", page_icon="◆", layout="wide")

INK = "#0F172A"          # primary text — dark navy/charcoal
INK_SOFT = "#475569"     # secondary text — slate
LINE = "#E2E8F0"         # borders
SURFACE = "#FFFFFF"
CANVAS = "#F8FAFC"       # page background
ACCENT = "#1E3A5F"       # single restrained accent — deep navy blue
ACCENT_SOFT = "#EEF2F7"
GREEN = "#15803D"
GREEN_BG = "#F0FDF4"
AMBER = "#B45309"
AMBER_BG = "#FFFBEB"
RED = "#B91C1C"
RED_BG = "#FEF2F2"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, sans-serif;
    color: {INK};
}}
.stApp {{ background-color: {CANVAS}; }}
#MainMenu, footer, header {{ visibility: hidden; }}

h1, h2, h3, .jakarta {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: {INK};
    letter-spacing: -0.01em;
}}

/* Top bar */
.topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 4px 14px 4px; border-bottom: 1px solid {LINE}; margin-bottom: 28px;
}}
.wordmark {{
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 20px;
    color: {ACCENT}; letter-spacing: -0.02em;
}}
.wordmark span {{ color: {INK}; font-weight: 500; font-size: 13px; margin-left: 10px; letter-spacing: 0.02em; }}
.status-pill {{
    font-size: 12px; color: {GREEN}; background: {GREEN_BG}; border: 1px solid #BBF7D0;
    padding: 4px 10px; border-radius: 20px; font-weight: 600;
}}

/* Cards */
.card {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 22px 24px; margin-bottom: 16px;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}}
.card-title {{
    font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 15px;
    color: {INK}; margin-bottom: 4px;
}}
.card-sub {{ font-size: 13px; color: {INK_SOFT}; margin-bottom: 14px; }}
.eyebrow {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: {ACCENT}; margin-bottom: 6px;
}}

/* Risk result */
.risk-card {{
    background: {SURFACE}; border: 1px solid {LINE}; border-radius: 16px;
    padding: 32px; text-align: center; margin-bottom: 18px;
}}
.risk-pct {{ font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 56px; line-height: 1; color: {INK}; }}
.risk-label {{
    display: inline-block; font-weight: 700; font-size: 13px; letter-spacing: 0.05em;
    padding: 5px 14px; border-radius: 20px; margin-top: 10px;
}}
.gauge-track {{
    position: relative; height: 10px; border-radius: 6px; margin: 26px 4px 8px 4px;
    background: linear-gradient(90deg, #16A34A 0%, #EAB308 50%, #DC2626 100%);
}}
.gauge-marker {{
    position: absolute; top: -7px; width: 4px; height: 24px; background: {INK};
    border-radius: 2px; transform: translateX(-2px);
}}
.gauge-labels {{ display: flex; justify-content: space-between; font-size: 11px; color: {INK_SOFT}; padding: 0 2px; font-weight: 600; letter-spacing: 0.03em; }}

/* Driver bars */
.driver-row {{ margin-bottom: 14px; }}
.driver-top {{ display: flex; justify-content: space-between; font-size: 13.5px; margin-bottom: 4px; }}
.driver-name {{ font-weight: 600; color: {INK}; }}
.driver-tag {{ font-size: 11px; font-weight: 700; }}
.driver-track {{ height: 6px; background: {ACCENT_SOFT}; border-radius: 4px; overflow: hidden; }}
.driver-fill {{ height: 100%; border-radius: 4px; }}
.driver-desc {{ font-size: 12.5px; color: {INK_SOFT}; margin-top: 3px; }}

/* Recommendation */
.rec-box {{ border-radius: 12px; padding: 16px 18px; margin-bottom: 16px; border: 1px solid; }}
.rec-title {{ font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 4px; }}
.rec-action {{ font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; font-size: 20px; }}

/* Snapshot table */
.snap-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {LINE}; font-size: 13.5px; }}
.snap-row:last-child {{ border-bottom: none; }}
.snap-label {{ color: {INK_SOFT}; }}
.snap-value {{ font-weight: 600; color: {INK}; }}

.disclaimer {{ font-size: 12px; color: {INK_SOFT}; border-top: 1px solid {LINE}; padding-top: 14px; margin-top: 8px; }}

div.stButton > button {{
    background: {ACCENT}; color: white; border-radius: 10px; border: none;
    padding: 12px 0; font-weight: 700; font-size: 14.5px; width: 100%;
}}
div.stButton > button:hover {{ background: #16283f; color: white; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CAT_COLS = ["grade", "sub_grade", "home_ownership", "verification_status",
            "purpose", "addr_state", "application_type", "dti_bucket"]

# Curated set of business-meaningful drivers to surface to the end user.
# addr_state / application_type are excluded here: with a limited per-state
# sample size their per-request attribution is noisy, and geography is
# already covered separately in the fairness audit rather than framed as
# a "why this borrower" driver.
EXPLAIN_CANDIDATES = {
    "int_rate": "Interest Rate", "dti": "Debt-to-Income Ratio", "grade": "Loan Grade",
    "sub_grade": "Loan Sub-Grade", "annual_inc": "Annual Income", "term": "Loan Term",
    "revol_util": "Revolving Credit Utilization", "emp_length": "Employment Length",
    "credit_history_years": "Credit History Length", "delinq_2yrs": "Past Delinquencies",
    "loan_amnt": "Loan Amount", "open_acc": "Open Credit Accounts",
    "home_ownership": "Home Ownership", "purpose": "Loan Purpose",
    "verification_status": "Income Verification", "mort_acc": "Mortgage Accounts",
    "pub_rec": "Public Records", "total_acc": "Total Credit Accounts",
}

@st.cache_resource
def load_model():
    return joblib.load("model_xgb.joblib")

model_pipe = load_model()
preprocessor = model_pipe.named_steps["preprocessor"]
classifier = model_pipe.named_steps["model"]


def group_name(fname):
    if fname.startswith("num__"):
        return fname[len("num__"):]
    if fname.startswith("cat__"):
        rest = fname[len("cat__"):]
        for c in sorted(CAT_COLS, key=len, reverse=True):
            if rest.startswith(c + "_"):
                return c
        return rest
    return fname


def predict_with_explanation(input_df):
    """Runs the real model and returns (probability, ranked driver list).
    Keeps the preprocessor output SPARSE throughout — XGBoost treats
    implicit zeros in a sparse matrix differently from explicit zeros in
    a dense array, so densifying here would silently change both the
    predicted probability and the explanation versus what the trained
    pipeline actually outputs. This mirrors exactly what model_pipe.predict_proba
    computes internally.
    """
    X_t = preprocessor.transform(input_df)
    feat_names = list(preprocessor.get_feature_names_out())
    booster = classifier.get_booster()
    dmat = xgb.DMatrix(X_t, feature_names=feat_names)

    proba = float(booster.predict(dmat)[0])
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    sv = contribs[:-1]

    groups = [group_name(f) for f in feat_names]
    imp = pd.DataFrame({"feature": feat_names, "group": groups, "shap": sv})
    grouped = imp.groupby("group")["shap"].sum().reset_index()
    grouped = grouped[grouped["group"].isin(EXPLAIN_CANDIDATES)]
    grouped["abs"] = grouped["shap"].abs()
    grouped = grouped.sort_values("abs", ascending=False).head(5)

    drivers = [
        {"feature": row["group"], "label": EXPLAIN_CANDIDATES[row["group"]],
         "shap": row["shap"], "abs": row["abs"]}
        for _, row in grouped.iterrows()
    ]
    return proba, drivers


def risk_band(proba):
    if proba < 0.15:
        return "Low", GREEN, GREEN_BG
    elif proba < 0.35:
        return "Moderate", AMBER, AMBER_BG
    else:
        return "High", RED, RED_BG


def recommended_action(proba):
    if proba < 0.15:
        return "Approve", GREEN, GREEN_BG
    elif proba < 0.25:
        return "Approve with Conditions", AMBER, AMBER_BG
    elif proba < 0.45:
        return "Manual Review", AMBER, AMBER_BG
    else:
        return "High Risk — Review Required", RED, RED_BG


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="topbar">
    <div><span class="wordmark">LENDWISE CAPITAL<span>Credit Risk Assessment Platform</span></span></div>
    <div class="status-pill">● Model Online</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Assess Borrower</div>', unsafe_allow_html=True)
st.markdown("### Evaluate Loan Default Risk")
st.markdown(
    f'<p style="color:{INK_SOFT}; font-size:14px; margin-top:-6px;">'
    'Enter the loan and borrower details below. The assessment is generated by a machine-learning '
    'model trained on historical LendingClub lending outcomes.</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Input sections (accordion)
# ---------------------------------------------------------------------------
with st.expander("**Loan Details**", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=40000, value=10000, step=500)
        term = st.selectbox("Loan Term (months)", [36, 60])
    with c2:
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
        installment = st.number_input("Monthly Installment ($)", min_value=10.0, max_value=2000.0, value=300.0)
    with c3:
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = st.selectbox("Sub-Grade", [f"{grade}{i}" for i in range(1, 6)])
    purpose = st.selectbox(
        "Loan Purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
         "small_business", "car", "medical", "other"],
    )

with st.expander("**Borrower Profile**", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        annual_inc = st.number_input("Annual Income ($)", min_value=0, max_value=1_000_000, value=60000, step=1000)
        emp_length = st.slider("Employment Length (years)", 0, 10, 5)
    with c2:
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])
    with c3:
        application_type = st.selectbox("Application Type", ["Individual", "Joint App"])
        addr_state = st.selectbox("State", ["CA", "NY", "TX", "FL", "IL", "NJ", "Other"])

with st.expander("**Credit Profile**", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0)
        open_acc = st.number_input("Open Credit Accounts", min_value=0, max_value=50, value=8)
        total_acc = st.number_input("Total Credit Accounts", min_value=0, max_value=100, value=20)
    with c2:
        revol_bal = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=8000)
        revol_util = st.slider("Revolving Utilization (%)", 0.0, 150.0, 45.0)
        credit_history_years = st.slider("Credit History Length (yrs)", 0.0, 40.0, 10.0)
    with c3:
        delinq_2yrs = st.number_input("Delinquencies (2 yrs)", min_value=0, max_value=20, value=0)
        pub_rec = st.number_input("Public Records", min_value=0, max_value=10, value=0)
        pub_rec_bankruptcies = st.number_input("Bankruptcies", min_value=0, max_value=10, value=0)
    mort_acc = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=1)

st.write("")
submitted = st.button("Assess Default Risk")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if submitted:
    if annual_inc == 0:
        st.error("Enter a valid annual income before running an assessment.")
    elif loan_amnt > annual_inc * 3 and annual_inc > 0:
        st.warning("Loan amount is unusually high relative to annual income — double-check these figures. Proceeding with assessment below.")

    with st.spinner("Evaluating financial profile and generating assessment…"):
        dti_bucket = pd.cut([dti], bins=[-1, 10, 20, 30, 40, 100],
                             labels=["0-10", "10-20", "20-30", "30-40", "40+"])[0]

        input_df = pd.DataFrame([{
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate, "installment": installment,
            "grade": grade, "sub_grade": sub_grade, "emp_length": emp_length,
            "home_ownership": home_ownership, "annual_inc": annual_inc,
            "verification_status": verification_status, "purpose": purpose, "dti": dti,
            "delinq_2yrs": delinq_2yrs, "open_acc": open_acc, "pub_rec": pub_rec,
            "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
            "addr_state": addr_state, "application_type": application_type, "mort_acc": mort_acc,
            "pub_rec_bankruptcies": pub_rec_bankruptcies, "credit_history_years": credit_history_years,
            "dti_bucket": dti_bucket,
        }])

        try:
            proba, drivers = predict_with_explanation(input_df)
        except Exception:
            st.markdown(f"""
            <div class="card" style="border-color:{RED};">
                <div class="card-title" style="color:{RED};">Unable to generate assessment</div>
                <div class="card-sub">Something went wrong while processing this application. Please check the entered information and try again.</div>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    band_label, band_color, band_bg = risk_band(proba)
    action_label, action_color, action_bg = recommended_action(proba)
    risk_pct = proba * 100
    marker_pos = min(max(risk_pct, 1), 99)

    st.markdown("---")
    st.markdown('<div class="eyebrow">Risk Assessment</div>', unsafe_allow_html=True)

    # Result card + gauge
    st.markdown(f"""
    <div class="risk-card">
        <div style="font-size:12px; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; color:{INK_SOFT};">Predicted Default Risk</div>
        <div class="risk-pct">{risk_pct:.1f}%</div>
        <div class="risk-label" style="color:{band_color}; background:{band_bg};">{band_label.upper()} RISK</div>
        <div class="gauge-track">
            <div class="gauge-marker" style="left:{marker_pos}%;"></div>
        </div>
        <div class="gauge-labels"><span>LOW</span><span>MODERATE</span><span>HIGH</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    st.markdown(f"""
    <div class="rec-box" style="border-color:{action_color}; background:{action_bg};">
        <div class="rec-title" style="color:{action_color};">Recommended Action</div>
        <div class="rec-action" style="color:{action_color};">{action_label}</div>
        <div style="font-size:12px; color:{INK_SOFT}; margin-top:6px;">Prediction ≠ decision — this is a decision-support signal, not an automated approval or denial.</div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Key Risk Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Top factors influencing this prediction, generated from the model\'s actual feature contributions for this application.</div>', unsafe_allow_html=True)

        max_abs = max([d["abs"] for d in drivers]) if drivers else 1
        for d in drivers:
            increases = d["shap"] > 0
            tag_color = RED if increases else GREEN
            tag_text = "↑ Increases Risk" if increases else "↓ Decreases Risk"
            width_pct = max(8, (d["abs"] / max_abs) * 100)
            bar_color = RED if increases else GREEN
            st.markdown(f"""
            <div class="driver-row">
                <div class="driver-top">
                    <span class="driver-name">{d['label']}</span>
                    <span class="driver-tag" style="color:{tag_color};">{tag_text}</span>
                </div>
                <div class="driver-track"><div class="driver-fill" style="width:{width_pct:.0f}%; background:{bar_color};"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Natural-language summary
        top_risk_factors = [d["label"] for d in drivers if d["shap"] > 0][:2]
        top_offset_factors = [d["label"] for d in drivers if d["shap"] < 0][:1]
        summary_parts = [f"This application is assessed as **{band_label.lower()} risk** ({risk_pct:.1f}% predicted probability of default)."]
        if top_risk_factors:
            summary_parts.append(f"The primary contributors are **{' and '.join(top_risk_factors).lower()}**.")
        if top_offset_factors:
            summary_parts.append(f"This is partially offset by **{top_offset_factors[0].lower()}**.")
        summary_text = " ".join(summary_parts)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Risk Assessment Summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:14px; line-height:1.6; color:{INK};">{summary_text}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Borrower Snapshot</div>', unsafe_allow_html=True)
        snap_items = [
            ("Loan Amount", f"${loan_amnt:,.0f}"), ("Term", f"{term} months"),
            ("Interest Rate", f"{int_rate:.1f}%"), ("Grade", f"{grade} ({sub_grade})"),
            ("Annual Income", f"${annual_inc:,.0f}"), ("DTI", f"{dti:.1f}%"),
            ("Employment", f"{emp_length} yrs"), ("Home Ownership", home_ownership),
        ]
        rows_html = "".join(
            f'<div class="snap-row"><span class="snap-label">{k}</span><span class="snap-value">{v}</span></div>'
            for k, v in snap_items
        )
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Model Information"):
            st.markdown(f"""
            - **Model:** XGBoost (Gradient Boosting)
            - **Training sample:** 200,000 resolved loans, stratified from 1.3M+ resolved
              records within a 2,260,668-row LendingClub dataset
            - **Test AUC-ROC:** 0.724
            - **Output:** Probability of default (0–1)
            - **Assessment generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)

    st.markdown(f"""
    <div class="disclaimer">
    This assessment is generated using a machine-learning model trained on historical lending
    data and should be used as a decision-support tool rather than as the sole basis for lending
    decisions. Developed as part of a Big Data &amp; NLP academic project.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="card" style="text-align:center; padding:36px 24px;">
        <div style="font-size:14px; color:{INK_SOFT};">
        Complete the loan, borrower, and credit details above, then select
        <strong>Assess Default Risk</strong> to generate a risk assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)
