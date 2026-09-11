import streamlit as st

import pandas as pd

import numpy as np

import joblib

import psutil

import time

st.set_page_config(

    page_title="NetGuard AI",

    page_icon="🛡️",

    layout="wide"

)

# -----------------------------

# PAGE TITLE

# -----------------------------

st.title("🛡️ NetGuard AI")

st.subheader("Offline AI-Based Network Problem Detection Assistant")

st.caption(

    "AI-powered network anomaly detection and problem classification"

)

# -----------------------------

# MODEL LOADING

# -----------------------------

@st.cache_resource

def load_models():

    classifier = joblib.load("problem_classifier.joblib")

    anomaly_detector = joblib.load("anomaly_detector.joblib")

    return classifier, anomaly_detector

# -----------------------------

# SIDEBAR

# -----------------------------

st.sidebar.title("⚙️ Control Panel")

mode = st.sidebar.radio(

    "Select Mode",

    [

        "Live Network Monitor",

        "CSV Analysis"

    ]

)

# ============================================================

# LIVE NETWORK MONITOR

# ============================================================

if mode == "Live Network Monitor":

    st.header("🌐 Live Network Monitoring")

    st.write(

        "Monitor basic network activity on this computer "

        "without sending data to the internet."

    )

    col1, col2, col3 = st.columns(3)

    net_before = psutil.net_io_counters()

    time.sleep(1)

    net_after = psutil.net_io_counters()

    upload_speed = (

        net_after.bytes_sent - net_before.bytes_sent

    )

    download_speed = (

        net_after.bytes_recv - net_before.bytes_recv

    )

    with col1:

        st.metric(

            "Upload",

            f"{upload_speed / 1024:.2f} KB/s"

        )

    with col2:

        st.metric(

            "Download",

            f"{download_speed / 1024:.2f} KB/s"

        )

    with col3:

        connections = len(psutil.net_connections())

        st.metric(

            "Active Connections",

            connections

        )

    st.divider()

    # Network statistics

    st.header("📊 Network Statistics")

    network = psutil.net_io_counters()

    stats = {

        "Bytes Sent": network.bytes_sent,

        "Bytes Received": network.bytes_recv,

        "Packets Sent": network.packets_sent,

        "Packets Received": network.packets_recv,

        "Errors Sent": network.errout,

        "Errors Received": network.errin,

        "Dropped Sent": network.dropout,

        "Dropped Received": network.dropin

    }

    stats_df = pd.DataFrame(

        stats.items(),

        columns=["Metric", "Value"]

    )

    st.dataframe(

        stats_df,

        use_container_width=True,

        hide_index=True

    )

    # -----------------------------

    # SIMPLE LIVE RISK ANALYSIS

    # -----------------------------

    st.header("🤖 AI Network Assessment")

    risk_score = 0

    if connections > 100:

        risk_score += 30

    if network.errin > 100:

        risk_score += 20

    if network.errout > 100:

        risk_score += 20

    if network.dropin > 100:

        risk_score += 15

    if network.dropout > 100:

        risk_score += 15

    if risk_score >= 50:

        st.error("🚨 ABNORMAL NETWORK ACTIVITY")

        st.metric(

            "Risk Score",

            f"{risk_score}/100"

        )

        st.warning(

            "The system has detected unusual network indicators. "

            "Review active connections and network logs."

        )

    elif risk_score >= 25:

        st.warning("⚠️ MODERATE NETWORK RISK")

        st.metric(

            "Risk Score",

            f"{risk_score}/100"

        )

        st.info(

            "Some unusual network indicators were observed. "

            "Continue monitoring."

        )

    else:

        st.success("✅ NETWORK STATUS: NORMAL")

        st.metric(

            "Risk Score",

            f"{risk_score}/100"

        )

        st.info(

            "No major abnormal network indicators detected."

        )

    st.divider()

    st.header("🔍 Active Connections")

    try:

        connections = psutil.net_connections()

        connection_data = []

        for conn in connections[:100]:

            connection_data.append({

                "Local Address": str(conn.laddr),

                "Remote Address": str(conn.raddr),

                "Status": conn.status,

                "PID": conn.pid

            })

        if connection_data:

            connection_df = pd.DataFrame(connection_data)

            st.dataframe(

                connection_df,

                use_container_width=True,

                hide_index=True

            )

        else:

            st.info("No active connections found.")

    except Exception as e:

        st.warning(

            f"Unable to read all connections: {e}"

        )

# ============================================================

# CSV ANALYSIS

# ============================================================

else:

    st.header("📁 Network Dataset Analysis")

    st.write(

        "Upload a network traffic CSV file for analysis "

        "using the trained Random Forest and Isolation Forest models."

    )

    uploaded_file = st.file_uploader(

        "Upload Network CSV",

        type=["csv"]

    )

    if uploaded_file is not None:

        with st.spinner("Loading AI models..."):

            classifier, anomaly_detector = load_models()

        data = pd.read_csv(uploaded_file)

        st.success(

            f"Dataset loaded successfully: {len(data)} records"

        )

        st.write("### Dataset Preview")

        st.dataframe(

            data.head(10),

            use_container_width=True

        )

        original_data = data.copy()

        # Remove target columns

        for column in ["id", "attack_cat", "label"]:

            if column in data.columns:

                data = data.drop(columns=[column])

        # Categorical columns

        categorical_columns = [

            "proto",

            "service",

            "state"

        ]

        existing_columns = [

            col

            for col in categorical_columns

            if col in data.columns

        ]

        data = pd.get_dummies(

            data,

            columns=existing_columns

        )

        # Clean data

        data = data.replace(

            [np.inf, -np.inf],

            0

        )

        data = data.fillna(0)

        # Match training features

        expected_features = classifier.feature_names_in_

        data = data.reindex(

            columns=expected_features,

            fill_value=0

        )

        st.write(

            f"Prepared features: {data.shape[1]}"

        )

        # -----------------------------

        # AI PREDICTIONS

        # -----------------------------

        with st.spinner(

            "Running AI analysis..."

        ):

            problem_prediction = classifier.predict(data)

            probabilities = classifier.predict_proba(data)

            confidence = probabilities.max(axis=1)

            anomaly_prediction = anomaly_detector.predict(data)

        # Results

        results = pd.DataFrame()

        results["Problem Category"] = (

            problem_prediction

        )

        results["Confidence (%)"] = (

            confidence * 100

        ).round(2)

        results["Network Status"] = [

            "🚨 Abnormal"

            if x == -1

            else "✅ Normal"

            for x in anomaly_prediction

        ]

        # -----------------------------

        # SUMMARY

        # -----------------------------

        total = len(results)

        abnormal = np.sum(

            anomaly_prediction == -1

        )

        normal = np.sum(

            anomaly_prediction == 1

        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Total Records",

                total

            )

        with col2:

            st.metric(

                "Normal",

                normal

            )

        with col3:

            st.metric(

                "Abnormal",

                abnormal

            )

        # -----------------------------

        # MOST COMMON PROBLEM

        # -----------------------------

        problem_counts = (

            pd.Series(problem_prediction)

            .value_counts()

        )

        most_common = (

            problem_counts.index[0]

        )

        st.divider()

        st.header("🧠 AI Diagnosis")

        st.warning(

            f"Most common detected category: "

            f"**{most_common}**"

        )

        avg_confidence = (

            confidence.mean() * 100

        )

        st.metric(

            "Average AI Confidence",

            f"{avg_confidence:.2f}%"

        )

        # -----------------------------

        # RECOMMENDATION

        # -----------------------------

        recommendations = {

            "Normal":

                "Network behavior appears normal. Continue monitoring.",

            "Generic":

                "Review unusual traffic patterns and suspicious connections.",

            "Exploits":

                "Investigate affected hosts, exposed services and suspicious connections.",

            "Fuzzers":

                "Inspect unusual input traffic and review exposed network services.",

            "DoS":

                "Check traffic volume, affected services and possible network overload.",

            "Reconnaissance":

                "Investigate scanning activity and unusual connection attempts.",

            "Analysis":

                "Review detailed network logs and investigate unusual communication.",

            "Backdoor":

                "Investigate affected systems and unexpected remote connections.",

            "Shellcode":

                "Review suspicious hosts and endpoint security logs.",

            "Worms":

                "Investigate repeated host-to-host connections and possible propagation."

        }

        recommendation = recommendations.get(

            most_common,

            "Review network logs and investigate abnormal activity."

        )

        st.header("🔧 Recommended Action")

        st.info(recommendation)

        # -----------------------------

        # RESULTS TABLE

        # -----------------------------

        st.header("📋 Detailed AI Results")

        st.dataframe(

            results,

            use_container_width=True,

            hide_index=True

        )

        # -----------------------------

        # DOWNLOAD RESULTS

        # -----------------------------

        csv_output = results.to_csv(

            index=False

        )

        st.download_button(

            "⬇️ Download AI Results",

            csv_output,

            "netguard_ai_results.csv",

            "text/csv"

        )