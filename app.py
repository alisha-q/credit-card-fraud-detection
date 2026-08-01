"""
Explainable Credit Card Fraud Detection System — Streamlit App
Module 4 (Interim Report, Section 4.5): interactive real-time fraud prediction with a
SHAP-based explanation for every prediction.

This app is written to work with the OFFICIAL pipeline (sklearn/XGBoost model +
`shap.TreeExplainer`, produced by notebooks 03-05) as its primary path.

If those official artifacts aren't present yet (e.g. you're running this straight out of
the box, before installing scikit-learn/XGBoost/SHAP and re-running the notebooks), the app
automatically falls back to the lightweight NumPy reference model + a simplified KernelSHAP
explainer that ships with this repository, so the demo always works end-to-end. A banner in
the sidebar tells you which mode is active.
"""

import os
import sys
import json
import pickle
from math import comb

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DATA_PATH = os.path.join(HERE, "data", "creditcard.csv")
sys.path.insert(0, os.path.join(HERE, "src"))

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ============================================================
# THEME — dark / pink, shared between the CSS and the matplotlib charts
# ============================================================
BG_PRIMARY = "#120814"
BG_SECONDARY = "#1C0F22"
BG_CARD = "#241432"
BG_INPUT = "#2C1A3B"
BORDER = "#3E2650"
TEXT_PRIMARY = "#F6EEF9"
TEXT_MUTED = "#B79CC7"
PINK = "#FF2E88"
PINK_SOFT = "#FF7AB8"
PINK_DEEP = "#C81E64"
MINT = "#2DD4BF"
AMBER = "#FFB020"


# ============================================================
# MODEL / EXPLAINER LOADING (official pipeline, with reference-mode fallback)
# ============================================================
@st.cache_resource
def load_pipeline():
    official_model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    official_shap_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    splits_path = os.path.join(MODELS_DIR, "preprocessed_splits.pkl")

    if os.path.exists(official_model_path) and os.path.exists(splits_path):
        with open(official_model_path, "rb") as f:
            model = pickle.load(f)
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        scaler = splits["scaler"]
        explainer = None
        if os.path.exists(official_shap_path):
            with open(official_shap_path, "rb") as f:
                explainer = pickle.load(f)
        return {
            "mode": "official (scikit-learn / XGBoost / SHAP)",
            "model": model,
            "scaler": scaler,
            "explainer": explainer,
            "predict_proba": lambda X: model.predict_proba(X)[:, 1],
        }

    # ---- Fallback: lightweight NumPy reference pipeline ----
    sys.path.insert(0, HERE)
    from models_lib import LogisticRegressionNP  # noqa: F401  (needed for unpickling)
    import tree_lib  # noqa: F401  (needed for unpickling)

    ref_model_path = os.path.join(MODELS_DIR, "xgboost_reference.pkl")
    scaler_params_path = os.path.join(MODELS_DIR, "scaler_params.json")
    with open(ref_model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_params_path) as f:
        scaler_params = json.load(f)

    class _ScalerAdapter:
        """Mimics the .transform() interface used for Time/Amount scaling."""
        def transform_row(self, row):
            row = row.copy()
            t_idx, a_idx = FEATURE_COLUMNS.index("Time"), FEATURE_COLUMNS.index("Amount")
            row[t_idx] = (row[t_idx] - scaler_params["time_mean"]) / scaler_params["time_std"]
            row[a_idx] = (row[a_idx] - scaler_params["amount_mean"]) / scaler_params["amount_std"]
            return row

    return {
        "mode": "reference (NumPy fallback — official artifacts not found)",
        "model": model,
        "scaler": _ScalerAdapter(),
        "explainer": None,
        "predict_proba": lambda X: model.predict_proba(X),
    }


@st.cache_data
def load_background_sample(n=200):
    """A small background sample used by the fallback KernelSHAP explainer."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    sample = df[FEATURE_COLUMNS].sample(n=min(n, len(df)), random_state=42).values
    return sample


@st.cache_data
def load_model_performance():
    """Real metrics for the deployed model, read straight from the notebook 04 output
    so the sidebar always reflects whatever's actually in results/."""
    path = os.path.join(HERE, "results", "model_comparison_f1_optimal.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_example_transactions(n=25):
    """Random real transactions (legit + fraud) so users can demo the app without
    typing 30 numbers by hand."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    fraud = df[df.Class == 1].sample(n=min(n // 2, (df.Class == 1).sum()), random_state=1)
    legit = df[df.Class == 0].sample(n=n - len(fraud), random_state=1)
    combined = pd.concat([fraud, legit]).sample(frac=1, random_state=2).reset_index(drop=True)
    return combined


# ============================================================
# EXPLANATION (real SHAP if available, else simplified KernelSHAP fallback)
# ============================================================
def kernel_shap_explain(predict_fn, x, background, n_coalitions=200, seed=0):
    rng = np.random.RandomState(seed)
    d = len(x)
    baseline_pred = predict_fn(background).mean()
    full_pred = predict_fn(x[None, :])[0]

    Z, weights, fx = [], [], []
    for _ in range(n_coalitions):
        k = rng.randint(1, d)
        z = np.zeros(d)
        z[rng.choice(d, size=k, replace=False)] = 1
        bg_row = background[rng.randint(0, len(background))]
        x_masked = np.where(z == 1, x, bg_row)
        fx.append(predict_fn(x_masked[None, :])[0])
        w_ = 1e6 if k in (0, d) else (d - 1) / (comb(d, k) * k * (d - k))
        Z.append(z); weights.append(w_)

    Z, weights, fx = np.array(Z), np.array(weights), np.array(fx)
    y_ = fx - baseline_pred
    W = np.diag(weights)
    A = Z.T @ W @ Z + 1e-6 * np.eye(d)
    b_ = Z.T @ W @ y_
    phi = np.linalg.solve(A, b_)
    phi += ((full_pred - baseline_pred) - phi.sum()) / d
    return phi, baseline_pred, full_pred


def explain_prediction(pipeline, x_scaled, background):
    if pipeline["explainer"] is not None:
        # Real shap.TreeExplainer path
        shap_values = pipeline["explainer"].shap_values(x_scaled[None, :])
        values = np.array(shap_values)[0]
        base_value = pipeline["explainer"].expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        return values, base_value
    else:
        values, base_value, _ = kernel_shap_explain(
            pipeline["predict_proba"], x_scaled, background, n_coalitions=200
        )
        return values, base_value


def plot_shap_bar(values, base_value, feature_names, top_n=10):
    order = np.argsort(-np.abs(values))[:top_n]
    labels = [feature_names[i] for i in order][::-1]
    vals = [values[i] for i in order][::-1]
    colors = [PINK if v > 0 else MINT for v in vals]

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)

    ax.barh(labels, vals, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.9, zorder=2)

    ax.set_xlabel("SHAP value (impact on fraud probability)", color=TEXT_MUTED, fontsize=10)
    ax.set_title(
        f"Top {top_n} Contributing Features\n(baseline probability = {base_value:.3f})",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=TEXT_MUTED, labelsize=9.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color=BORDER, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ============================================================
# THEME CSS
# ============================================================
def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: radial-gradient(circle at 15% 0%, #250f2e 0%, {BG_PRIMARY} 45%) fixed;
            color: {TEXT_PRIMARY};
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1A0C22 0%, {BG_PRIMARY} 100%);
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY}; }}

        /* ---- hero header ---- */
        .hero {{
            background: linear-gradient(120deg, #3B0F3E 0%, #7A1652 45%, #FF2E88 120%);
            border-radius: 20px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 18px 40px -18px rgba(255, 46, 136, 0.45);
            border: 1px solid rgba(255, 122, 184, 0.25);
        }}
        .hero h1 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 800;
            font-size: 2.05rem;
            margin: 0 0 0.35rem 0;
            color: #FFFFFF;
            letter-spacing: -0.01em;
        }}
        .hero p {{
            margin: 0;
            color: #F3D6E8;
            font-size: 0.98rem;
        }}
        .hero .tag {{
            display: inline-block;
            margin-top: 0.9rem;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            background: rgba(0,0,0,0.25);
            border: 1px solid rgba(255,255,255,0.18);
            font-size: 0.78rem;
            color: #FBD8EA;
            letter-spacing: 0.02em;
        }}

        /* ---- generic card ---- */
        .card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 10px 30px -18px rgba(0,0,0,0.6);
        }}
        .card h3, .card h4 {{
            font-family: 'Poppins', sans-serif;
            color: {TEXT_PRIMARY};
            margin-top: 0;
        }}
        .section-label {{
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: {PINK_SOFT};
            margin-bottom: 0.4rem;
        }}

        /* ---- headings ---- */
        h1, h2, h3, h4 {{ font-family: 'Poppins', sans-serif; color: {TEXT_PRIMARY}; }}
        .stMarkdown p, .stCaption, label, .stMarkdown li {{ color: {TEXT_MUTED} !important; }}

        /* ---- buttons ---- */
        .stButton > button {{
            background: linear-gradient(90deg, {PINK_DEEP}, {PINK});
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.1rem;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 8px 22px -10px rgba(255, 46, 136, 0.55);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 10px 26px -8px rgba(255, 46, 136, 0.7);
            color: #FFFFFF;
        }}
        .stButton > button:active {{ transform: translateY(0px); }}

        /* ---- form submit button (Predict & Explain) ---- */
        [data-testid="stFormSubmitButton"] > button {{
            background: linear-gradient(90deg, {PINK}, #FF6FB0 60%, {AMBER});
            font-size: 1.02rem;
            padding: 0.75rem 1.2rem;
        }}

        /* ---- inputs ---- */
        .stNumberInput input, .stTextInput input, .stSelectbox [data-baseweb="select"] > div {{
            background-color: {BG_INPUT} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
        }}
        .stSelectbox svg {{ fill: {TEXT_MUTED}; }}

        /* ---- form container ---- */
        [data-testid="stForm"] {{
            background: {BG_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.3rem 1.4rem 0.6rem 1.4rem;
        }}

        /* ---- expander ---- */
        [data-testid="stExpander"] {{
            background: {BG_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}
        [data-testid="stExpander"] summary {{ color: {TEXT_PRIMARY} !important; font-weight: 600; }}

        /* ---- metrics ---- */
        [data-testid="stMetric"] {{
            background: {BG_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 0.7rem 0.9rem 0.5rem 0.9rem;
        }}
        [data-testid="stMetricValue"] {{ color: {PINK_SOFT} !important; font-family:'Poppins',sans-serif; }}
        [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

        /* ---- progress bar ---- */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {MINT}, {PINK}) !important;
        }}
        .stProgress > div > div {{ background-color: {BG_INPUT} !important; }}

        /* ---- alert boxes (success / error / warning / info) ---- */
        div[data-testid="stAlertContentSuccess"], .stAlert {{ border-radius: 12px !important; }}
        div[data-baseweb="notification"] {{ border-radius: 12px !important; }}

        /* ---- dataframe ---- */
        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 10px;
            overflow: hidden;
        }}

        /* ---- divider ---- */
        hr {{ border-color: {BORDER} !important; }}

        /* ---- verdict banner ---- */
        .verdict-fraud {{
            background: linear-gradient(120deg, rgba(255,46,136,0.18), rgba(200,30,100,0.05));
            border: 1px solid {PINK};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            text-align: center;
        }}
        .verdict-legit {{
            background: linear-gradient(120deg, rgba(45,212,191,0.16), rgba(45,212,191,0.03));
            border: 1px solid {MINT};
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            text-align: center;
        }}
        .verdict-fraud .verdict-title {{ color: {PINK_SOFT}; }}
        .verdict-legit .verdict-title {{ color: {MINT}; }}
        .verdict-title {{
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            font-size: 1.3rem;
            margin: 0;
        }}

        .footer-note {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            text-align: center;
            padding-top: 0.6rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.set_page_config(page_title="Fraud Detection · SHAP", page_icon="💗", layout="wide")
    inject_theme()

    st.markdown(
        """
        <div class="hero">
            <h1>💗 Explainable Credit Card Fraud Detection</h1>
            <p>Machine learning + SHAP explainability, in real time — every prediction comes with a reason.</p>
            <span class="tag">Alisha Quaser · Major Project 21CSA399A</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️  What is this app? (click to expand)", expanded=False):
        st.markdown(
            "This app scores a single credit card transaction and predicts whether it "
            "looks **fraudulent** or **legitimate**, then explains *why* using SHAP "
            "(SHapley Additive exPlanations) — a technique that shows how much each "
            "input pushed the prediction toward fraud (pink) or legitimate (mint).\n\n"
            "**How to use it:** the fastest way is to pick a real transaction from the "
            "dropdown below and click *Load example into form*, then *Predict & Explain*. "
            "You can also type in your own numbers, though the 28 `V` fields are "
            "anonymized PCA components (not real account details) so they're mainly "
            "useful for demoing edge cases, not something a normal user would fill in by "
            "hand."
        )

    pipeline = load_pipeline()
    model_name = type(pipeline["model"]).__name__.replace("Classifier", "").replace("NP", " (reference)")

    st.sidebar.markdown("### 🩷 Status")
    if "reference" in pipeline["mode"]:
        st.sidebar.warning(
            "⚠️ **Demo mode** — using a lightweight bundled fallback model, not the "
            "trained scikit-learn/XGBoost model. Run notebooks 02-05 first to switch to "
            "real predictions."
        )
    else:
        st.sidebar.success(f"✅ Live model: **{model_name}** (trained on the real Kaggle dataset)")

    perf = load_model_performance()
    if perf is not None:
        st.sidebar.markdown("### 📊 Model Performance")
        st.sidebar.caption("F1-optimal operating point, test set — see README for full comparison")
        row = perf[perf["Model"].str.contains(model_name, case=False, na=False)]
        if not row.empty:
            r = row.iloc[0]
            m1, m2 = st.sidebar.columns(2)
            m1.metric("Precision", f"{r['Precision']:.1%}")
            m2.metric("Recall", f"{r['Recall']:.1%}")
            m1.metric("F1-Score", f"{r['F1-Score']:.3f}")
            m2.metric("AUPRC", f"{r['AUPRC']:.3f}")
            st.sidebar.caption(
                "AUPRC (Area Under the Precision-Recall Curve) is the metric to trust "
                "here — with fraud at 0.17% of transactions, plain accuracy and ROC-AUC "
                "both look artificially high."
            )
        with st.sidebar.expander("Compare all 3 models"):
            st.dataframe(
                perf[["Model", "Precision", "Recall", "F1-Score", "AUPRC"]]
                .style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1-Score": "{:.3f}", "AUPRC": "{:.3f}"}),
                hide_index=True, use_container_width=True,
            )

    background = load_background_sample()
    examples = load_example_transactions()

    st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
    st.markdown("## 📤 Enter a Transaction")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Quick demo:** load a random real transaction")
        st.caption("Recommended — skips typing 28 numbers by hand.")
        if examples is not None:
            idx = st.selectbox(
                "Choose an example transaction",
                options=list(range(len(examples))),
                format_func=lambda i: (
                    f"#{i} — {'FRAUD (label)' if examples.iloc[i]['Class']==1 else 'legit (label)'} "
                    f"— Amount ${examples.iloc[i]['Amount']:.2f}"
                ),
            )
            if st.button("Load example into form", use_container_width=True):
                st.session_state["loaded_example"] = examples.iloc[idx][FEATURE_COLUMNS].to_dict()
        else:
            st.info("No bundled dataset found to sample examples from.")
        st.markdown('</div>', unsafe_allow_html=True)

    defaults = st.session_state.get("loaded_example", {c: 0.0 for c in FEATURE_COLUMNS})

    with col_b:
        with st.form("transaction_form"):
            st.markdown("**Basic details**")
            c1, c2 = st.columns(2)
            with c1:
                time_val = st.number_input(
                    "Time (seconds since first transaction in the dataset)",
                    value=float(defaults.get("Time", 50000.0)),
                    help="Elapsed time, not a clock time — 0 is the first transaction recorded.",
                )
            with c2:
                amount_val = st.number_input(
                    "Amount ($)",
                    value=float(defaults.get("Amount", 100.0)), min_value=0.0,
                    help="The transaction amount in dollars.",
                )

            with st.expander("Advanced: raw anonymized features (V1–V28)", expanded=False):
                st.caption(
                    "These 28 values are PCA components computed from the original "
                    "transaction data, with the real fields (merchant, location, card "
                    "details, etc.) removed to protect cardholder privacy — this is how "
                    "the dataset is published on Kaggle. There's no meaningful way to "
                    "type these in by hand; use *Load example into form* above instead. "
                    "Left at 0.0, they mean \"average/typical transaction.\""
                )
                v_values = {}
                v_cols = st.columns(4)
                for i in range(1, 29):
                    col = v_cols[(i - 1) % 4]
                    v_values[f"V{i}"] = col.number_input(
                        f"V{i}", value=float(defaults.get(f"V{i}", 0.0)), format="%.4f", key=f"v_{i}"
                    )

            submitted = st.form_submit_button("🔍  Predict & Explain", type="primary", use_container_width=True)

    if submitted:
        raw_row = np.array([time_val] + [v_values[f"V{i}"] for i in range(1, 29)] + [amount_val])

        scaler = pipeline["scaler"]
        if hasattr(scaler, "transform_row"):
            x_scaled = scaler.transform_row(raw_row)
        else:
            x_scaled = raw_row.copy()
            idx = [FEATURE_COLUMNS.index(c) for c in ("Time", "Amount")]
            x_scaled[idx] = scaler.transform(raw_row[idx].reshape(1, -1))[0]

        proba = float(pipeline["predict_proba"](x_scaled[None, :])[0])
        verdict = "Fraud" if proba >= 0.5 else "Legitimate"

        st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
        st.markdown("## 🔮 Result")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Prediction")
            verdict_cls = "verdict-fraud" if verdict == "Fraud" else "verdict-legit"
            verdict_icon = "⚠️" if verdict == "Fraud" else "✅"
            st.markdown(
                f'<div class="{verdict_cls}"><p class="verdict-title">{verdict_icon} Likely {verdict}</p></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.metric("Model's fraud probability", f"{proba:.2%}")
            st.progress(min(max(proba, 0.0), 1.0))
            st.caption(
                "This is the model's raw probability score, flagged as Fraud above 50%. "
                "In the Model Performance panel (sidebar), note the deployed model's "
                "actual precision — a 'Likely Fraud' flag is a signal for manual review, "
                "not a certainty."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 🔬 Why? (SHAP explanation)")
            if background is not None:
                values, base_value = explain_prediction(pipeline, x_scaled, background)
                fig = plot_shap_bar(values, base_value, FEATURE_COLUMNS, top_n=10)
                st.pyplot(fig, use_container_width=True)
                st.caption(
                    "🩷 Pink bars pushed this specific prediction toward **Fraud**; "
                    "🟢 mint bars pushed it toward **Legitimate**. Longer bar = bigger "
                    "influence on this transaction's score. This explains *this one* "
                    "prediction, not the model in general — see the sidebar's SHAP global "
                    "importance plot for overall feature rankings."
                )
            else:
                st.warning("No background dataset available to compute an explanation.")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="footer-note">
        ⚠️ Educational / academic project (Major Project 21CSA399A), trained on the real
        ULB/Kaggle Credit Card Fraud Detection dataset. Not a production fraud system —
        see the Final Report for limitations and regulatory considerations.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
