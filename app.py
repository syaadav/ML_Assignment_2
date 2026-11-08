import streamlit as st
import pandas as pd
import pickle # Changed from joblib to pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("Machine Learning Classification Evaluation App")
st.markdown("Upload your `test_data.csv`, select a trained model, and view the evaluation metrics.")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV format)", type=["csv"])

model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'KNN': 'model/knn.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Support Vector Machine': 'model/support_vector_machine.pkl'
}

selected_model_name = st.selectbox("Select Classification Model", list(model_options.keys()))

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df_test.head())

    if 'target' not in df_test.columns:
        st.error("The uploaded CSV must contain a 'target' column.")
    else:
        X_test = df_test.drop(columns=['target'])
        y_test = df_test['target']
        model_path = model_options[selected_model_name]

        try:
            # Load model using standard pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else "N/A"

            st.subheader(f"Evaluation Metrics for {selected_model_name}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.write("---")
            col_cm, col_cr = st.columns(2)

            with col_cm:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col_cr:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.background_gradient(cmap='Blues'))

        except FileNotFoundError:
            st.error(f"Model file not found at `{model_path}`.")
else:
    st.info("Please upload the test data CSV to begin evaluation.")
