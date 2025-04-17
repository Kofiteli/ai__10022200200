# Name: Kofi Boateng Index_number: 10022200200
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

def run():
    st.subheader("🧠 Neural Network Classifier")

    uploaded_file = st.file_uploader("Upload a classification dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of dataset:")
        st.dataframe(df.head())

        target_col = st.selectbox("Select the target column", df.columns)

        if target_col:
            # Preprocessing
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Encode categorical target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Handle categorical features
            X = pd.get_dummies(X)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

            # Hyperparameters
            epochs = st.slider("Epochs", 5, 100, 20)
            learning_rate = st.slider("Learning Rate", 0.001, 0.01, 0.005)

            # Model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)

            # Plot training history
            st.markdown("### 📈 Training Progress")

            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            ax[0].plot(history.history['loss'], label='Train Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].set_title("Loss")
            ax[0].legend()

            ax[1].plot(history.history['accuracy'], label='Train Acc')
            ax[1].plot(history.history['val_accuracy'], label='Val Acc')
            ax[1].set_title("Accuracy")
            ax[1].legend()

            st.pyplot(fig)

            st.markdown("### 🧪 Predict on Custom Input")
            user_input = {}
            for col in df.drop(columns=[target_col]).columns:
                user_input[col] = st.number_input(f"{col}", value=0.0)

            if st.button("Predict"):
                user_df = pd.DataFrame([user_input])
                user_df = pd.get_dummies(user_df).reindex(columns=X.shape[1:], fill_value=0)
                user_scaled = scaler.transform(user_df)
                prediction = model.predict(user_scaled)
                pred_label = le.inverse_transform([np.argmax(prediction)])[0]
                st.success(f"Predicted class: {pred_label}")
