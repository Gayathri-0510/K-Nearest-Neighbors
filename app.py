import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# =========================
# APP HEADER
# =========================
st.set_page_config(page_title="Customer Risk Prediction System", layout="centered")

st.title("Customer Risk Prediction System (KNN)")
st.write(
    "This system predicts customer risk by comparing them with similar customers."
)

# =========================
# LOAD & PREPARE DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")

    numerical_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length"
    ]

    categorical_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file"
    ]

    # Fill missing values
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X, X_scaled, y, scaler, encoders


df, X, X_scaled, y, scaler, encoders = load_data()

# =========================
# SIDEBAR – USER INPUT
# =========================
st.sidebar.header("Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", min_value=10000, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1000, step=500)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])

k_value = st.sidebar.slider("K Value (Number of Neighbors)", 1, 15, 5)

# Fixed / average values for remaining features
emp_length = df["person_emp_length"].median()
interest_rate = df["loan_int_rate"].median()
loan_percent_income = loan_amount / income
credit_hist_len = df["cb_person_cred_hist_length"].median()

home_ownership = encoders["person_home_ownership"].transform(["RENT"])[0]
loan_intent = encoders["loan_intent"].transform(["PERSONAL"])[0]
loan_grade = encoders["loan_grade"].transform(["C"])[0]
default_file = encoders["cb_person_default_on_file"].transform(
    ["Y" if credit_history == "Yes" else "N"]
)[0]

# =========================
# MAIN PREDICTION BUTTON
# =========================
if st.button("Predict Customer Risk"):

    new_customer = np.array([[
        age,
        income,
        emp_length,
        home_ownership,
        loan_amount,
        loan_intent,
        loan_grade,
        interest_rate,
        loan_percent_income,
        default_file,
        credit_hist_len
    ]])

    new_customer_scaled = scaler.transform(new_customer)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k_value, metric="manhattan")
    knn.fit(X_scaled, y)

    prediction = knn.predict(new_customer_scaled)[0]

    # =========================
    # PREDICTION OUTPUT
    # =========================
    st.subheader("Prediction Result")

    if prediction == 1:
        st.markdown(
            "<h2 style='color:red'>🔴 High Risk Customer</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='color:green'>🟢 Low Risk Customer</h2>",
            unsafe_allow_html=True
        )

    # =========================
    # NEAREST NEIGHBORS EXPLANATION
    # =========================
    distances, neighbors = knn.kneighbors(new_customer_scaled)

    neighbor_labels = y.iloc[neighbors[0]]
    majority_class = neighbor_labels.mode()[0]

    st.subheader("Nearest Neighbors Explanation")

    st.write(f"**Number of neighbors considered:** {k_value}")
    st.write(
        "**Majority class among neighbors:**",
        "High Risk" if majority_class == 1 else "Low Risk"
    )

    neighbors_df = df.iloc[neighbors[0]][
        ["person_age", "person_income", "loan_amnt", "loan_status"]
    ]

    st.write("Nearest Customers:")
    st.dataframe(neighbors_df)

    # =========================
    # BUSINESS INSIGHT
    # =========================
    st.subheader("Business Insight")

    st.info(
        "This decision is based on similarity with nearby customers in feature space. "
        "Customers with similar income, loan amount, age, and credit history influence "
        "the risk prediction."
    )
