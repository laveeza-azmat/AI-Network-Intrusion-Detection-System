import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(
    BASE_DIR, "dataset", "UNSW_NB15_training-set.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR, "dataset", "UNSW_NB15_testing-set.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "results", "ids_random_forest_model.pkl"
)

CLASS_DISTRIBUTION = os.path.join(
    BASE_DIR, "results", "class_distribution.png"
)

ATTACK_CATEGORIES = os.path.join(
    BASE_DIR, "results", "attack_categories.png"
)

CONFUSION_MATRIX = os.path.join(
    BASE_DIR, "results", "confusion_matrix.png"
)

FEATURE_IMPORTANCE = os.path.join(
    BASE_DIR, "results", "feature_importance.png"
)


st.set_page_config(
    page_title="AI-Powered Network IDS",
    page_icon="🛡️",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .attack-card {
        padding: 30px;
        border-radius: 18px;
        border: 2px solid #ef4444;
        background: linear-gradient(
            135deg,
            rgba(127, 29, 29, 0.25),
            rgba(30, 41, 59, 0.5)
        );
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    .normal-card {
        padding: 30px;
        border-radius: 18px;
        border: 2px solid #22c55e;
        background: linear-gradient(
            135deg,
            rgba(20, 83, 45, 0.25),
            rgba(30, 41, 59, 0.5)
        );
        text-align: center;
        margin-top: 25px;
        margin-bottom: 25px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 15px;
    }

    .result-value {
        font-size: 22px;
        font-weight: 700;
        margin: 8px;
    }

    .confidence {
        font-size: 18px;
        margin: 5px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    .risk-critical {
        color: #ef4444;
        font-weight: 800;
        font-size: 18px;
    }

    .risk-low {
        color: #22c55e;
        font-weight: 800;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_data():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    return train_data, test_data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


train_data, test_data = load_data()
model = load_model()


def prediction_card(prediction, probabilities, actual_label=None):

    attack_probability = float(probabilities[1]) * 100
    normal_probability = float(probabilities[0]) * 100

    if prediction == 1:
        st.markdown(
            """
            <div style="
                padding: 30px;
                border-radius: 18px;
                border: 2px solid #ef4444;
                background: rgba(127, 29, 29, 0.25);
                text-align: center;
                margin: 25px 0;
            ">
                <div style="
                    font-size: 30px;
                    font-weight: 800;
                ">
                    🚨 ATTACK DETECTED
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            """
            <div style="
                padding: 30px;
                border-radius: 18px;
                border: 2px solid #22c55e;
                background: rgba(20, 83, 45, 0.25);
                text-align: center;
                margin: 25px 0;
            ">
                <div style="
                    font-size: 30px;
                    font-weight: 800;
                ">
                    ✅ NORMAL TRAFFIC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.write(
            f"**Attack Probability — {attack_probability:.2f}%**"
        )

        st.progress(
            min(max(attack_probability / 100, 0.0), 1.0)
        )

    with col2:
        st.write(
            f"**Normal Probability — {normal_probability:.2f}%**"
        )

        st.progress(
            min(max(normal_probability / 100, 0.0), 1.0)
        )

    if actual_label is not None:

        actual_text = (
            "Attack"
            if int(actual_label) == 1
            else "Normal"
        )

        st.markdown(
            f"""
            <div style="
                padding: 15px;
                border-radius: 10px;
                background: rgba(100,116,139,0.12);
                margin-top: 15px;
            ">
                <b>Actual Dataset Label:</b> {actual_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        if int(actual_label) == 1 and prediction == 0:

            st.warning(
                "⚠️ False Negative: The actual traffic is an "
                "attack, but the model predicted Normal."
            )

        elif int(actual_label) == 0 and prediction == 1:

            st.warning(
                "⚠️ False Positive: The actual traffic is Normal, "
                "but the model predicted Attack."
            )


def get_model_columns():
    preprocessor = model.named_steps["preprocessor"]

    numerical_columns = list(
        preprocessor.transformers_[0][2]
    )

    categorical_columns = list(
        preprocessor.transformers_[1][2]
    )

    return numerical_columns, categorical_columns


st.sidebar.title("🛡️ AI Network IDS")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Data Explorer",
        "Visualizations",
        "Predictions",
        "Model Information"
    ]
)


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">🛡️ AI-Powered Network Intrusion Detection System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning based Network Traffic Classification</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Training Records",
            f"{len(train_data):,}"
        )

    with col2:
        st.metric(
            "Testing Records",
            f"{len(test_data):,}"
        )

    with col3:
        st.metric(
            "Features",
            len(test_data.columns) - 2
        )

    with col4:
        st.metric(
            "Model",
            "Random Forest"
        )

    st.markdown("---")

    st.subheader("Model Performance")

    metrics_path = os.path.join(
        BASE_DIR,
        "results",
        "model_metrics.csv"
    )

    if os.path.exists(metrics_path):

        metrics = pd.read_csv(metrics_path)

        if not metrics.empty:
            st.dataframe(
                metrics,
                width="stretch",
                hide_index=True
            )

    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("Dataset Loaded")

    with col2:
        st.success("Model Loaded")

    with col3:
        st.success("Prediction Ready")


# ==========================================================
# DATA EXPLORER
# ==========================================================

elif page == "Data Explorer":

    st.title("📊 Data Explorer")

    st.subheader("Testing Dataset")

    st.write(
        f"Rows: {len(test_data):,} | "
        f"Columns: {len(test_data.columns)}"
    )

    st.dataframe(
        test_data.head(100),
        width="stretch"
    )

    st.subheader("Class Distribution")

    class_counts = test_data["label"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Normal Traffic",
            f"{class_counts.get(0, 0):,}"
        )

    with col2:
        st.metric(
            "Attack Traffic",
            f"{class_counts.get(1, 0):,}"
        )

    st.bar_chart(class_counts)


# ==========================================================
# VISUALIZATIONS
# ==========================================================

elif page == "Visualizations":

    st.title("📈 Visualizations")

    if os.path.exists(CLASS_DISTRIBUTION):
        st.subheader("Class Distribution")
        st.image(CLASS_DISTRIBUTION)

    if os.path.exists(ATTACK_CATEGORIES):
        st.subheader("Attack Categories")
        st.image(ATTACK_CATEGORIES)

    if os.path.exists(CONFUSION_MATRIX):
        st.subheader("Confusion Matrix")
        st.image(CONFUSION_MATRIX)

    if os.path.exists(FEATURE_IMPORTANCE):
        st.subheader("Feature Importance")
        st.image(FEATURE_IMPORTANCE)


# ==========================================================
# PREDICTIONS
# ==========================================================

elif page == "Predictions":

    st.title("🔍 Network Traffic Predictions")

    st.write(
        "Choose a prediction mode to classify network traffic."
    )

    prediction_mode = st.radio(
        "Prediction Mode",
        [
            "Single Row Prediction",
            "Manual Network Flow",
            "Batch CSV Prediction"
        ],
        horizontal=True
    )


    # ======================================================
    # SINGLE ROW
    # ======================================================

    if prediction_mode == "Single Row Prediction":

        st.subheader("Single Row Prediction")

        st.write(
            "Select a row from the UNSW-NB15 testing dataset "
            "and let the trained model classify the network flow."
        )

        row_number = st.number_input(
            "Test Dataset Row",
            min_value=1,
            max_value=len(test_data),
            value=1,
            step=1
            )
        selected_row = test_data.iloc[
            int(row_number) - 1
            ]

        st.subheader("Selected Network Flow")

        display_row = selected_row.to_frame().T

        st.dataframe(
            display_row,
            width="stretch",
            hide_index=True
        )

        X_selected = selected_row.to_frame().T.drop(
            columns=["label", "attack_cat"],
            errors="ignore"
        )

        prediction = model.predict(X_selected)[0]
        probabilities = model.predict_proba(X_selected)[0]

        st.subheader("Prediction Result")

        prediction_card(
            prediction,
            probabilities,
            selected_row["label"]
        )


    # ======================================================
    # MANUAL NETWORK FLOW
    # ======================================================

    elif prediction_mode == "Manual Network Flow":

        st.subheader("Manual Network Flow")

        st.write(
            "Enter network traffic characteristics and "
            "let the trained model classify the flow."
        )

        numerical_columns, categorical_columns = get_model_columns()

        input_data = {}

        st.markdown("### Network Traffic Features")

        col1, col2 = st.columns(2)

        with col1:

            for i, column in enumerate(numerical_columns):

                default_value = test_data[column].median()

                input_data[column] = st.number_input(
                    column,
                    value=float(default_value),
                    key=f"manual_num_{column}"
                )

        with col2:

            for column in categorical_columns:

                values = (
                    test_data[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                values = sorted(values)

                default_value = str(
                    test_data[column].mode()[0]
                )

                if default_value in values:
                    default_index = values.index(default_value)
                else:
                    default_index = 0

                input_data[column] = st.selectbox(
                    column,
                    values,
                    index=default_index,
                    key=f"manual_cat_{column}"
                )

        manual_df = pd.DataFrame([input_data])

        st.markdown("### Entered Network Flow")

        st.dataframe(
            manual_df,
            width="stretch",
            hide_index=True
        )

        if st.button(
            "🔍 Detect Traffic",
            type="primary",
            width="stretch"
        ):

            prediction = model.predict(manual_df)[0]
            probabilities = model.predict_proba(manual_df)[0]

            st.subheader("Prediction Result")

            prediction_card(
                prediction,
                probabilities
            )


    # ======================================================
    # BATCH CSV
    # ======================================================

    elif prediction_mode == "Batch CSV Prediction":

        st.subheader("Batch CSV Prediction")

        st.write(
            "Upload a CSV file to classify multiple network "
            "traffic records at once."
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            key="batch_prediction_file"
        )

        if uploaded_file is not None:

            batch_data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.write(
                f"Records: {len(batch_data):,}"
            )

            st.dataframe(
                batch_data.head(20),
                width="stretch"
            )

            prediction_data = batch_data.drop(
                columns=["label", "attack_cat"],
                errors="ignore"
            )

            if st.button(
                "🚀 Run Batch Prediction",
                type="primary",
                width="stretch"
            ):

                predictions = model.predict(
                    prediction_data
                )

                probabilities = model.predict_proba(
                    prediction_data
                )

                results = batch_data.copy()

                results["Prediction"] = [
                    "Attack" if p == 1 else "Normal"
                    for p in predictions
                ]

                results["Attack Probability (%)"] = (
                    probabilities[:, 1] * 100
                ).round(2)

                results["Normal Probability (%)"] = (
                    probabilities[:, 0] * 100
                ).round(2)

                total_records = len(results)

                attack_count = int(
                    (predictions == 1).sum()
                )

                normal_count = int(
                    (predictions == 0).sum()
                )

                attack_rate = (
                    attack_count / total_records * 100
                )

                st.subheader("Batch Prediction Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Total Records",
                        f"{total_records:,}"
                    )

                with col2:
                    st.metric(
                        "Attacks Detected",
                        f"{attack_count:,}"
                    )

                with col3:
                    st.metric(
                        "Normal Traffic",
                        f"{normal_count:,}"
                    )

                with col4:
                    st.metric(
                        "Attack Rate",
                        f"{attack_rate:.2f}%"
                    )

                st.subheader("Prediction Results")

                st.dataframe(
                    results,
                    width="stretch",
                    hide_index=True
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Prediction Results",
                    csv_data,
                    "ids_prediction_results.csv",
                    "text/csv",
                    width="stretch"
                )


# ==========================================================
# MODEL INFORMATION
# ==========================================================

elif page == "Model Information":

    st.title("🤖 Model Information")

    st.subheader("Algorithm")

    st.write(
        "Random Forest Classifier"
    )

    st.subheader("Classification Task")

    st.write(
        "Binary Network Intrusion Detection"
    )

    st.write(
        "**0 = Normal Traffic**"
    )

    st.write(
        "**1 = Attack Traffic**"
    )

    st.subheader("Preprocessing")

    st.write(
        """
        • Missing value handling  
        • Numerical feature scaling  
        • Categorical feature encoding  
        • Unknown categorical values handled safely
        """
    )

    st.subheader("Security Focus")

    st.write(
        """
        The system evaluates network traffic as Normal or Attack.
        False positives and false negatives are important because
        incorrect classifications can affect network security.
        """
    )

    st.subheader("Prediction Modes")

    st.write(
        """
        **Single Row Prediction:** Test an existing network flow
        from the UNSW-NB15 testing dataset.

        **Manual Network Flow:** Enter network-flow features manually
        and receive an immediate prediction.

        **Batch CSV Prediction:** Upload multiple network flows and
        classify them together.
        """
    )