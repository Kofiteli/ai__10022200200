# Name: Kofi Boateng Index_number: 10022200200
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def run():
    st.subheader("📈 Regression Model")

    # 1. Upload CSV dataset
    uploaded_file = st.file_uploader("Upload your CSV dataset for regression", type=["csv"])
    if not uploaded_file:
        st.info("Awaiting CSV file to be uploaded.")
        return

    # 2. Read dataset and preview
    df = pd.read_csv(uploaded_file)
    st.markdown("**Dataset Preview**")
    st.dataframe(df.head())

    # 3. Select target column
    target_col = st.selectbox("Select the target column to predict:", df.columns)
    if not target_col:
        st.warning("Please select a target column.")
        return

    # 4. Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error("Selected target column is not numeric. Please choose a numeric column for regression.")
        return

    # 5. Prepare data: drop missing and split
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 6. One-hot encode categorical features
    X = pd.get_dummies(X)
    y = y.loc[X.index]

    # 7. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. Fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 9. Predict and evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 10. Display metrics
    st.markdown("### Model Performance")
    st.write(f"- Mean Absolute Error (MAE): {mae:.3f}")
    st.write(f"- R² Score: {r2:.3f}")

    # 11. Plot Actual vs Predicted
    st.markdown("### Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)

    # 12. If only one feature, show regression line
    if X_test.shape[1] == 1:
        feature_name = X_test.columns[0]
        st.markdown("### Regression Line for Single Feature")
        fig2, ax2 = plt.subplots()
        ax2.scatter(X_test[feature_name], y_test, label='Actual')
        ax2.plot(X_test[feature_name], y_pred, color='red', label='Fit')
        ax2.set_xlabel(feature_name)
        ax2.set_ylabel(target_col)
        ax2.legend()
        st.pyplot(fig2)

    # 13. Custom input prediction
    st.markdown("### Make Your Own Prediction")
    input_data = {}
    for col in X.columns:
        # default to mean value
        default_val = float(X[col].mean()) if np.issubdtype(X[col].dtype, np.number) else 0.0
        input_data[col] = st.number_input(f"{col}", value=default_val)

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        try:
            pred_value = model.predict(input_df)[0]
            st.success(f"Predicted {target_col}: {pred_value:.3f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
