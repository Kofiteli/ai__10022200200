# Name: Kofi Boateng Index_number: 10022200200
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder


def run():
    st.subheader("📈 Regression Model")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv", key="regression_csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        # Handle missing values
        st.subheader("🧹 Handle Missing Values")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                strategy = st.selectbox(
                    f"How to handle missing values in '{col}'?",
                    options=["Drop rows", "Fill with Mean", "Fill with Median", "Fill with Mode"],
                    key=col
                )
                if strategy == "Drop rows":
                    df = df.dropna(subset=[col])
                elif strategy == "Fill with Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == "Fill with Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == "Fill with Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)

        # Encode categorical variables
        st.subheader("🔤 Encode Categorical Columns")
        encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        st.dataframe(df.head())

        # Target variable input
        st.subheader("🎯 Define the Target Variable")
        target_col = st.selectbox("Select the target column", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Standardization
            standardize = st.checkbox("⚙️ Standardize features", value=True)
            if standardize:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            test_size = st.slider("Test size (%)", 10, 50, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("📊 Model Evaluation")
            st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"**R² Score**: {r2_score(y_test, y_pred):.4f}")

            st.subheader("🔵 Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color='teal')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            selected_feature = st.selectbox("Feature to Visualize", X.columns)
            fig2, ax2 = plt.subplots()
            ax2.scatter(X_test[selected_feature], y_test, label="Actual")
            ax2.plot(X_test[selected_feature], y_pred, color='orange', linestyle='--', label="Predicted")
            ax2.legend()
            st.pyplot(fig2)

            st.subheader("📥 Make a Custom Prediction")
            input_data = {}
            for col in original_df.columns:
                if col != target_col:
                    if original_df[col].dtype == 'object':
                        input_data[col] = st.selectbox(col, sorted(original_df[col].dropna().unique()))
                    else:
                        input_data[col] = st.number_input(col, value=float(original_df[col].mean()))

            if st.button("Predict Value"):
                input_df = pd.DataFrame([input_data])
                for col in input_df.select_dtypes(include='object').columns:
                    if col in encoders:
                        input_df[col] = encoders[col].transform(input_df[col])

                if standardize:
                    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

                prediction = model.predict(input_df)[0]
                st.success(f"🔮 Predicted Value: {prediction:.4f}")
