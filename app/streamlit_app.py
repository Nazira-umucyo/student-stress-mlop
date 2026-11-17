"""Streamlit web UI for the Student Stress Level Prediction system."""
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COLORS = {"Low": "#4caf50", "Medium": "#ffb300", "High": "#e53935"}
LABELS = {0: "Low", 1: "Medium", 2: "High"}

# Feature groups with (feature, label, min, max, help text)
GROUPS = {
    "🧠 Psychological": [
        ("anxiety_level", "Anxiety level", 0, 21, "GAD-7 score: higher means more anxiety"),
        ("self_esteem", "Self-esteem", 0, 30, "Rosenberg score: higher means better self-esteem"),
        ("mental_health_history", "Mental health history", 0, 1, "1 = has a history of mental health issues"),
        ("depression", "Depression", 0, 27, "PHQ-9 score: higher means more depressive symptoms"),
    ],
    "💓 Physiological": [
        ("headache", "Headache frequency", 0, 5, ""),
        ("blood_pressure", "Blood pressure", 1, 3, "1 = low, 3 = high"),
        ("sleep_quality", "Sleep quality", 0, 5, "Higher means better sleep"),
        ("breathing_problem", "Breathing problems", 0, 5, ""),
    ],
    "🏠 Environmental": [
        ("noise_level", "Noise level", 0, 5, ""),
        ("living_conditions", "Living conditions", 0, 5, "Higher means better conditions"),
        ("safety", "Feeling of safety", 0, 5, "Higher means feeling safer"),
        ("basic_needs", "Basic needs met", 0, 5, "Higher means needs are better met"),
    ],
    "📚 Academic": [
        ("academic_performance", "Academic performance", 0, 5, ""),
        ("study_load", "Study load", 0, 5, ""),
        ("teacher_student_relationship", "Teacher-student relationship", 0, 5, "Higher means better"),
        ("future_career_concerns", "Future career concerns", 0, 5, ""),
    ],
    "👥 Social": [
        ("social_support", "Social support", 0, 3, "Higher means more support"),
        ("peer_pressure", "Peer pressure", 0, 5, ""),
        ("extracurricular_activities", "Extracurricular activities", 0, 5, ""),
        ("bullying", "Bullying", 0, 5, ""),
    ],
}
FEATURES = [f[0] for group in GROUPS.values() for f in group]
PRETTY = {f[0]: f[1] for group in GROUPS.values() for f in group}

st.set_page_config(page_title="Student Stress Predictor", page_icon="🧠", layout="wide")


# ---------- helpers ----------
def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(path, **kwargs):
    """POST to the API; returns the response, or None if the API is unreachable."""
    try:
        return requests.post(f"{API_URL}{path}", **kwargs)
    except requests.RequestException:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")
        return None


def show_error(response):
    try:
        st.error(response.json().get("detail", response.text))
    except Exception:
        st.error(f"Request failed ({response.status_code}).")


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_DIR / "StressLevelDataset.csv")


@st.cache_data
def load_demo_samples():
    path = DATA_DIR / "demo_samples.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def fill_example():
    """Copy the chosen example student into the input widgets."""
    choice = st.session_state.example_choice
    demos = load_demo_samples()
    if choice != "None" and not demos.empty:
        row = demos.iloc[int(choice.split("#")[1]) - 1]
        for f in FEATURES:
            st.session_state[f] = int(row[f])


# ---------- sidebar: model status ----------
with st.sidebar:
    st.title("🧠 Student Stress Predictor")
    st.caption("Educational screening tool, not a diagnosis. If you are struggling, "
               "please talk to someone you trust or a counsellor.")
    st.divider()
    info = api_get("/model-info")
    if info:
        st.success("API connected")
        st.metric("Model version", info.get("model_version", "?"))
        metrics = info.get("test_metrics", {})
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        c2.metric("F1 score", f"{metrics.get('f1_score', 0):.3f}")
        st.caption(f"Last trained: {info.get('trained_on', 'n/a')}")
        data = info.get("data", {})
        st.caption(f"Uploaded rows: {data.get('total_rows', 0)} "
                   f"({data.get('pending_rows', 0)} waiting for retraining)")
    else:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?")

tab_predict, tab_retrain, tab_insights = st.tabs(
    ["🔮 Predict", "🔁 Upload & Retrain", "📊 Data Insights"])

# ---------- tab 1: prediction ----------
with tab_predict:
    st.header("Predict a student's stress level")
    st.write("Fill in the 20 factors below, or load an example student, then click **Predict**.")

    demos = load_demo_samples()
    options = ["None"] + [f"{LABELS[int(r.true_label)]} stress example #{i + 1}"
                          for i, r in demos.iterrows()]
    st.selectbox("Load an example student (from the test set)", options,
                 key="example_choice", on_change=fill_example)

    for f in FEATURES:
        st.session_state.setdefault(f, 0 if f != "blood_pressure" else 1)

    cols = st.columns(len(GROUPS))
    for col, (group, items) in zip(cols, GROUPS.items()):
        with col:
            st.markdown(f"#### {group}")
            for feat, label, lo, hi, help_text in items:
                st.slider(label, lo, hi, key=feat, help=help_text or None)

    if st.button("Predict stress level", type="primary", width="stretch"):
        payload = {f: int(st.session_state[f]) for f in FEATURES}
        with st.spinner("Predicting..."):
            r = api_post("/predict", json=payload, timeout=30)
        if r is not None and r.ok:
            result = r.json()
            label = result["label"]
            st.markdown(
                f"<div style='padding:1.2rem;border-radius:10px;background:{COLORS[label]}22;"
                f"border-left:8px solid {COLORS[label]}'>"
                f"<h2 style='margin:0;color:{COLORS[label]}'>Predicted stress level: {label}</h2>"
                f"<p style='margin:0'>Confidence: {result['confidence']:.1%}</p></div>",
                unsafe_allow_html=True)
            probs = pd.DataFrame({"Stress level": list(result["probabilities"].keys()),
                                  "Probability": list(result["probabilities"].values())})
            st.altair_chart(alt.Chart(probs).mark_bar().encode(
                x=alt.X("Stress level", sort=["Low", "Medium", "High"]),
                y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Stress level", scale=alt.Scale(
                    domain=list(COLORS), range=list(COLORS.values())), legend=None),
            ).properties(height=250), width="stretch")
            if label == "High":
                st.info("High stress predicted. Reaching out to a counsellor, a trusted friend, "
                        "or family member can really help.")
        elif r is not None:
            show_error(r)

# ---------- tab 2: upload & retrain ----------
with tab_retrain:
    st.header("Upload new data and retrain the model")
    st.write("Upload a CSV with the 20 factor columns **and** a `stress_level` column "
             "(0 = Low, 1 = Medium, 2 = High). The data is cleaned and saved to the "
             "database, then used to fine-tune the current model.")

    st.subheader("Step 1: Upload data")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        preview = pd.read_csv(uploaded)
        st.write(f"Preview ({len(preview)} rows):")
        st.dataframe(preview.head(), width="stretch")
        if st.button("Upload and save to database"):
            uploaded.seek(0)
            with st.spinner("Cleaning and saving..."):
                r = api_post("/upload",
                             files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                             timeout=60)
            if r is not None and r.ok:
                res = r.json()
                st.success(f"Saved {res['rows_saved']} rows to the database "
                           f"({res['rows_removed_in_cleaning']} removed during cleaning).")
                st.write("Class counts:", res["class_counts"])
            elif r is not None:
                show_error(r)

    st.subheader("Step 2: Retrain")
    st.write("Fine-tunes the current model (as the pre-trained starting point) on the "
             "uploaded data mixed with a sample of the original training data. The new "
             "model replaces the old one only if test accuracy does not drop by more than 2 points.")
    if st.button("Retrain model", type="primary"):
        with st.spinner("Retraining... this takes about 10–30 seconds."):
            r = api_post("/retrain", timeout=600)
        if r is not None and r.ok:
            res = r.json()
            if res["accepted"]:
                st.success(f"New model accepted: now version {res['model_version']}.")
            else:
                st.warning("The retrained model performed worse, so the current model was kept.")
            st.write(f"Trained on {res['new_rows']} new rows + {res['replay_rows']} original rows "
                     f"for {res['epochs_run']} epochs.")
            names = ["accuracy", "loss", "precision", "recall", "f1_score"]
            st.table(pd.DataFrame({
                "Metric": names,
                "Before": [res[f"old_{n}"] for n in names],
                "After": [res[f"new_{n}"] for n in names],
            }).set_index("Metric"))
        elif r is not None:
            show_error(r)

    st.subheader("Retraining history")
    runs = api_get("/training-runs")
    if runs:
        st.dataframe(pd.DataFrame(runs), width="stretch")
    else:
        st.caption("No retraining runs yet.")

# ---------- tab 3: data insights ----------
with tab_insights:
    df = load_dataset()
    df["Stress"] = df["stress_level"].map(LABELS)
    order = ["Low", "Medium", "High"]
    st.header("Data insights")
    st.write(f"Based on the training dataset: {len(df):,} students, 20 factors.")

    st.subheader("1. Which factors are most linked to stress?")
    corr = (df[FEATURES + ["stress_level"]].corr()["stress_level"].drop("stress_level")
            .rename(index=PRETTY).rename("Correlation").rename_axis("Factor").reset_index())
    corr["Effect"] = corr["Correlation"].map(lambda v: "Raises stress" if v > 0 else "Lowers stress")
    st.altair_chart(alt.Chart(corr).mark_bar().encode(
        x=alt.X("Correlation", scale=alt.Scale(domain=[-0.8, 0.8])),
        y=alt.Y("Factor", sort="-x", title=None),
        color=alt.Color("Effect", scale=alt.Scale(
            domain=["Raises stress", "Lowers stress"], range=["#e53935", "#1e88e5"])),
        tooltip=["Factor", alt.Tooltip("Correlation", format=".2f")],
    ).properties(height=500), width="stretch")
    st.write("Every factor is clearly linked to stress. **Bullying, future career concerns, "
             "anxiety, and depression** raise stress the most, while **self-esteem, sleep "
             "quality, and academic performance** are the strongest protective factors. "
             "Blood pressure has the weakest link.")

    st.subheader("2. Explore any factor by stress level")
    feat = st.selectbox("Choose a factor", FEATURES, format_func=lambda f: PRETTY[f],
                        index=FEATURES.index("sleep_quality"))
    means = df.groupby("Stress")[feat].mean().reindex(order).round(2)
    st.altair_chart(alt.Chart(means.rename("Average").rename_axis("Stress").reset_index())
                    .mark_bar().encode(
        x=alt.X("Stress", sort=order, title="Stress level"),
        y=alt.Y("Average", title=f"Average {PRETTY[feat].lower()}"),
        color=alt.Color("Stress", scale=alt.Scale(domain=order,
                        range=[COLORS[c] for c in order]), legend=None),
        tooltip=["Stress", "Average"],
    ).properties(height=300), width="stretch")
    st.write(f"Average {PRETTY[feat].lower()}: **{means['Low']}** for low-stress students, "
             f"**{means['Medium']}** for medium, and **{means['High']}** for high. "
             "For example, sleep quality drops from about 4.1 to 1.3, while bullying rises "
             "from about 1.3 to 4.1.")

    st.subheader("3. Sleep quality and stress")
    sleep = pd.crosstab(df["sleep_quality"], df["Stress"])[order]
    st.bar_chart(sleep, color=[COLORS[c] for c in order], stack=False)
    st.write("Students who rate their sleep 4–5 are mostly in the low-stress group, while "
             "those rating it 2–3 are mostly medium, and those rating it 1 are almost all highly stressed (314 of 328). Since poor sleep both causes "
             "and results from stress, sleep is a useful early warning sign.")