import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ------------------------------------
# PAGE CONFIG + THEME
# ------------------------------------
st.set_page_config(page_title="PowerBI Insurance Dashboard", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
body { background-color:#0f1117; color:#e6eef6; }
.header { text-align:center; margin-bottom:20px; }

.kpi { 
    background: linear-gradient(135deg, #1c1f2b, #283044); 
    padding:16px; 
    border-radius:12px; 
    border:1px solid #2f3544; 
    text-align:center;
    color:#e6eef6;
    box-shadow:0 4px 12px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)


st.markdown("<div class='header'><h1>Insurance Claim – Anomaly Detection Dashboard</h1></div>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------
# LOAD MODEL
# ------------------------------------
@st.cache_resource
def load_model_and_meta():
    df = pd.read_csv("insurance_claim.csv")
    df["claim_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        np.random.randint(0, 365, len(df)), unit="D"
    )

    df_enc = pd.get_dummies(df, columns=["claim_type", "region", "income_bracket"], drop_first=False)

    drop_cols = ["anomaly", "claim_date"]
    X = df_enc.drop(columns=drop_cols, errors="ignore")

    REQUIRED_COLS = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X_scaled)

    scores = model.decision_function(X_scaled)
    return df, REQUIRED_COLS, scaler, model, float(scores.min()), float(scores.max())

DF_TRAIN, REQUIRED_COLS, SCALER, MODEL, SCORE_MIN, SCORE_MAX = load_model_and_meta()

# ------------------------------------
# HELPERS
# ------------------------------------
def align_dataframe(df_in):
    df = df_in.copy()
    cat_cols = [c for c in ["claim_type", "region", "income_bracket"] if c in df.columns]

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    for col in REQUIRED_COLS:
        if col not in df:
            df[col] = 0

    return df[REQUIRED_COLS]

def score_to_percent(s, smin, smax):
    if smin == smax:
        return 50
    scaled = (s - smin) / (smax - smin)
    pct = (1 - scaled) * 100
    return float(np.clip(pct, 0, 100))

def compute_risk(df_enc):
    X_scaled = SCALER.transform(df_enc)
    raw_score = MODEL.decision_function(X_scaled)
    risk = [score_to_percent(s, SCORE_MIN, SCORE_MAX) for s in raw_score]
    return raw_score, risk

# ------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose Page", ["Home", "Analytics", "Single Claim", "Batch Upload"])

# Filters
claim_types = sorted(DF_TRAIN["claim_type"].unique())
regions = sorted(DF_TRAIN["region"].unique())
income_list = sorted(DF_TRAIN["income_bracket"].unique())

f_type = st.sidebar.multiselect("Claim Type", claim_types, default=claim_types)
f_region = st.sidebar.multiselect("Region", regions, default=regions)
f_inc = st.sidebar.multiselect("Income Bracket", income_list, default=income_list)

min_amt, max_amt = st.sidebar.slider(
    "Claim Amount Range",
    float(DF_TRAIN["claim_amount"].min()),
    float(DF_TRAIN["claim_amount"].max()),
    (float(DF_TRAIN["claim_amount"].min()), float(DF_TRAIN["claim_amount"].max()))
)

# Filter data for visuals
DF_VIS = DF_TRAIN[
    (DF_TRAIN["claim_type"].isin(f_type)) &
    (DF_TRAIN["region"].isin(f_region)) &
    (DF_TRAIN["income_bracket"].isin(f_inc)) &
    (DF_TRAIN["claim_amount"].between(min_amt, max_amt))
]

# ------------------------------------
# PAGE 1 – HOME
# ------------------------------------
if page == "Home":

    # KPIs
    total = len(DF_VIS)
    anomalies = int(DF_VIS["anomaly"].sum())
    pct = round((anomalies / total) * 100, 2)
    avg = round(DF_VIS["claim_amount"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi'>Total Claims<br><b>{total}</b></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi'>Anomalies<br><b>{anomalies}</b></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi'>Anomaly %<br><b>{pct}%</b></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi'>Avg Claim<br><b>₹{avg:,.0f}</b></div>", unsafe_allow_html=True)

    # ------------------------------------------
    # SIDE-BY-SIDE CHARTS (FIXED)
    # ------------------------------------------
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.subheader("Claim Type Distribution")
        ct = DF_VIS["claim_type"].value_counts()

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(ct.values, labels=ct.index, autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

    with col2:
        st.subheader("Region: Normal vs Anomaly")

        df_enc = align_dataframe(DF_VIS)
        _, risk_vals = compute_risk(df_enc)
        DF_VIS["risk_pct"] = risk_vals
        DF_VIS["anomaly_risk"] = (DF_VIS["risk_pct"] >= 50).astype(int)

        pivot = DF_VIS.groupby("region")["anomaly_risk"].value_counts().unstack().fillna(0)
        pivot.columns = ["Normal", "Anomaly"]

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        pivot.plot(kind="bar", stacked=True, ax=ax2, color=["#7fb3ff", "#ff6b6b"])
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        st.pyplot(fig2)

    # HEATMAP
    st.markdown("---")
    st.subheader("Region-wise Anomaly Intensity (Risk %)")

    heat = DF_VIS.groupby("region")["risk_pct"].mean().reindex(regions).fillna(0)

    fig3, ax3 = plt.subplots(figsize=(8, 2.5))
    sns.heatmap(heat.to_frame().T, annot=True, fmt=".1f", cmap="YlOrRd", cbar=True)
    st.pyplot(fig3)

    # AVG RISK BAR CHART
    st.subheader("Region-wise Average Risk (%)")

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    heat.plot(kind="bar", ax=ax4, color="#ff6b6b", edgecolor="white")
    ax4.grid(axis="y", linestyle="--", alpha=0.3)
    st.pyplot(fig4)

# ------------------------------------
# PAGE 2 – ANALYTICS
# ------------------------------------
elif page == "Analytics":

    st.subheader("Analytics & Trends")

    DF_VIS["month"] = DF_VIS["claim_date"].dt.to_period("M").dt.to_timestamp()
    trend = DF_VIS.groupby("month")["claim_amount"].mean()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trend.index, trend.values, marker="o")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ------------------------------------
# PAGE 3 – SINGLE CLAIM
# ------------------------------------
elif page == "Single Claim":

    st.subheader("Evaluate a Single Claim")

    # ---------------------------
    # AUTO-FILL BUTTONS
    # ---------------------------
    colA, colB, colC = st.columns(3)

    if "auto_fill" not in st.session_state:
        st.session_state.auto_fill = None

    if colA.button("🔴 High Risk Example"):
        st.session_state.auto_fill = "high"

    if colB.button("🟡 Medium Risk Example"):
        st.session_state.auto_fill = "medium"

    if colC.button("🟢 Low Risk Example"):
        st.session_state.auto_fill = "low"

    # ---------------------------
    # DEFAULT VALUES FOR FORM
    # ---------------------------
    amount_default = 20000
    income_default = "Middle"
    prev_default = 0
    tenure_default = 5
    type_default = claim_types[0]
    region_default = regions[0]

    # APPLY AUTO-FILL
    if st.session_state.auto_fill == "high":
        amount_default = 250000
        income_default = "Low"
        prev_default = 3
        tenure_default = 0.5
        type_default = "Health"
        region_default = "East"

    elif st.session_state.auto_fill == "medium":
        amount_default = 80000
        income_default = "Middle"
        prev_default = 1
        tenure_default = 2
        type_default = "Vehicle"
        region_default = "North"

    elif st.session_state.auto_fill == "low":
        amount_default = 15000
        income_default = "High"
        prev_default = 0
        tenure_default = 6
        type_default = "Health"
        region_default = "South"

    st.markdown("---")

    # ---------------------------
    # CLAIM INPUT FORM
    # ---------------------------
    with st.form("single_claim"):

        col1, col2 = st.columns(2)

        with col1:
            claim_amount = st.number_input("Claim Amount (₹)", value=float(amount_default))
            claim_type = st.selectbox("Claim Type", claim_types, index=claim_types.index(type_default))
            region = st.selectbox("Region", regions, index=regions.index(region_default))

        with col2:
            income = st.selectbox("Income Bracket", income_list, index=income_list.index(income_default))
            previous_claims = st.number_input("Previous Claims", value=int(prev_default))
            policy_tenure = st.number_input("Policy Tenure (Years)", value=float(tenure_default))

        submitted = st.form_submit_button("Evaluate")

    # ---------------------------
    # RISK CALCULATION
    # ---------------------------
    if submitted:

        risk_score = 0

        # Claim amount factor
        if claim_amount > 200000:
            risk_score += 50
        elif claim_amount > 80000:
            risk_score += 35
        elif claim_amount > 40000:
            risk_score += 20
        else:
            risk_score += 10

        # Income factor
        if income == "Low":
            risk_score += 25
        elif income == "Middle":
            risk_score += 10
        else:
            risk_score += 5

        # Previous claims factor
        if previous_claims >= 3:
            risk_score += 40
        elif previous_claims == 2:
            risk_score += 25
        elif previous_claims == 1:
            risk_score += 10

        # Policy tenure factor
        if policy_tenure < 1:
            risk_score += 30
        elif policy_tenure < 3:
            risk_score += 15
        else:
            risk_score += 5

        risk_score = min(risk_score, 100)

        # ---------------------------
        # DISPLAY RESULT
        # ---------------------------
        st.markdown("---")
        st.subheader("🧠 Claim Risk Evaluation")

        if risk_score >= 65:
            st.error(f"High Risk — {risk_score:.1f}% (⚠ Fraud Likely)")
        elif risk_score >= 35:
            st.warning(f"Medium Risk — {risk_score:.1f}% (Review Recommended)")
        else:
            st.success(f"Low Risk — {risk_score:.1f}% (Normal Claim)")

        st.markdown("### Claim Details")
        st.write({
            "Claim Amount": claim_amount,
            "Claim Type": claim_type,
            "Region": region,
            "Income": income,
            "Previous Claims": previous_claims,
            "Policy Tenure": policy_tenure
        })






# ------------------------------------
# PAGE 4 – BATCH UPLOAD
# ------------------------------------
elif page == "Batch Upload":

    st.subheader("Batch Upload & Prediction")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        df_enc = align_dataframe(df)
        raw, risk = compute_risk(df_enc)
        df["risk_pct"] = risk
        df["prediction"] = (df["risk_pct"] >= 50).astype(int)

        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "predictions.csv")
