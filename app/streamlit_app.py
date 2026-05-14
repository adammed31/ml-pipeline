import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ML Dashboard", page_icon="🤖", layout="wide")
st.title("ML Dashboard — Supervised Learning")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_URL, disabled=bool(os.getenv("API_URL")))
    try:
        resp = requests.get(f"{api_url}/health", timeout=2)
        if resp.status_code == 200:
            st.success("API connected")
        else:
            st.error("API unreachable")
    except Exception:
        st.error("API unreachable — run `docker-compose up`")

# Tabs
tab_eda, tab_train, tab_predict, tab_model, tab_drift, tab_runs = st.tabs([
    "Analysis", "Training", "Prediction", "Active model", "Drift", "MLflow Runs"
])


# Tab EDA
with tab_eda:
    st.header("Exploratory analysis")
    eda_file = st.file_uploader("Upload CSV", type=["csv"], key="eda")

    if eda_file:
        df = pd.read_csv(eda_file)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isnull().sum().sum()))
        c4.metric("Duplicates", int(df.duplicated().sum()))

        with st.expander("Data preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Types & missing values"):
            dtype_df = pd.DataFrame({
                "Type": df.dtypes.astype(str),
                "Unique values": df.nunique(),
                "Missing": df.isnull().sum(),
                "Missing (%)": (df.isnull().sum() / len(df) * 100).round(1),
            })
            st.dataframe(dtype_df, use_container_width=True)

        with st.expander("Descriptive statistics"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.subheader("Missing values")
            fig, ax = plt.subplots(figsize=(10, max(3, len(missing) * 0.4)))
            pct = (missing / len(df) * 100).round(1)
            bars = ax.barh(missing.index, pct, color="salmon")
            ax.bar_label(bars, fmt="%.1f%%", padding=4)
            ax.set_xlabel("% missing")
            ax.set_xlim(0, 115)
            ax.invert_yaxis()
            ax.set_title("Missing values per column")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("No missing values.")

        st.subheader("Target column analysis")
        target_eda = st.selectbox("Select target column", df.columns.tolist(), key="eda_target")
        col_left, col_right = st.columns(2)
        with col_left:
            fig, ax = plt.subplots(figsize=(6, 4))
            y = df[target_eda]
            if y.dtype == object or y.nunique() <= 20:
                counts = y.value_counts()
                ax.bar(counts.index.astype(str), counts.values, color="steelblue")
                ax.tick_params(axis="x", rotation=30)
            else:
                ax.hist(y.dropna(), bins=30, color="steelblue", edgecolor="white")
            ax.set_title(f"Distribution of '{target_eda}'")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col_right:
            if df[target_eda].dtype == object or df[target_eda].nunique() <= 20:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[target_eda].value_counts().plot.pie(ax=ax, autopct="%1.1f%%", startangle=90, colors=plt.cm.Set2.colors)
                ax.set_ylabel("")
                ax.set_title("Class distribution")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target_eda in num_cols:
            num_cols.remove(target_eda)
        if num_cols:
            st.subheader("Numerical distributions")
            n_cols = 3
            n_rows = (len(num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                axes[i].hist(df[col].dropna(), bins=30, color="teal", alpha=0.8, edgecolor="white")
                axes[i].set_title(col)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] >= 2:
            st.subheader("Correlation matrix")
            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(min(14, corr.shape[1] * 1.2), min(10, corr.shape[0] * 1.0)))
            sns.heatmap(corr, annot=corr.shape[0] <= 15, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5, square=True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_eda in cat_cols:
            cat_cols.remove(target_eda)
        if cat_cols:
            st.subheader("Categorical features")
            selected_cat = st.selectbox("Select a feature", cat_cols, key="cat_col")
            fig, ax = plt.subplots(figsize=(10, 4))
            top = df[selected_cat].value_counts().head(15)
            ax.bar(top.index.astype(str), top.values, color="mediumpurple")
            ax.set_title(f"Top valeurs — {selected_cat}")
            ax.tick_params(axis="x", rotation=35)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# Tab Training
with tab_train:
    st.header("Train a model")
    uploaded = st.file_uploader("Dataset CSV (with target column)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Target column", df.columns.tolist())
        with col2:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        with col3:
            experiment_name = st.text_input("MLflow experiment", "ml_experiment")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isnull().sum().sum()))

        st.subheader("Adversarial Validation")
        st.caption("Checks whether the train/test split is homogeneous before training.")
        if st.button("Run adversarial validation"):
            with st.spinner("Running…"):
                uploaded.seek(0)
                resp = requests.post(
                    f"{api_url}/adversarial_validation",
                    files={"file": ("dataset.csv", uploaded, "text/csv")},
                    params={"target_col": target_col, "test_size": test_size},
                    timeout=120,
                )
                if resp.status_code == 200:
                    av = resp.json()
                    status_color = {"ok": "success", "warning": "warning", "alert": "error"}[av["overall_status"]]
                    getattr(st, status_color)(f"Average AUC: **{av['overall_auc']}** — {av['interpretation']}")

                    col_rf, col_xgb = st.columns(2)
                    for col, model_name in [(col_rf, "RandomForest"), (col_xgb, "XGBoost")]:
                        with col:
                            m = av["models"][model_name]
                            st.metric(f"{model_name} AUC", m["auc"])
                            top_df = pd.DataFrame(m["top_features"])
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.barh(top_df["feature"][::-1], top_df["importance"][::-1], color="steelblue")
                            ax.set_title(f"{model_name} — discriminative features")
                            ax.set_xlabel("Importance")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                else:
                    st.error(resp.json().get("detail"))

        st.divider()
        st.subheader("Preprocessing")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            num_imputer_label = st.selectbox("Numerical imputation", ["median", "mean", "knn", "constant (0)"], help="Median: robust to outliers. Mean: sensitive to outliers. KNN: nearest neighbors (more accurate, slower). Constant: replaces with 0.")
            num_imputer = "constant" if num_imputer_label == "constant (0)" else num_imputer_label
        with col_p2:
            cat_imputer = st.selectbox("Categorical imputation", ["most_frequent", "constant (missing)"], help="Most frequent: dominant value. Constant: creates a 'missing' category.")
            cat_imputer = "constant" if cat_imputer == "constant (missing)" else cat_imputer
        with col_p3:
            cat_encoder = st.selectbox("Categorical encoding", ["onehot", "ordinal"], help="OneHot: one column per category (better for LR). Ordinal: integers (better for RF/XGB/LGBM).")

        st.divider()
        st.subheader("Advanced options")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            use_feature_selection = st.checkbox("Automatic feature selection")
            use_ensemble = st.checkbox("Ensemble models (Voting)")
        with col_b:
            cv_folds = st.slider("Cross-validation (folds)", 0, 10, 0)
            use_tuning = st.checkbox("Hyperparameter tuning")
        with col_c:
            search_strategy = "random"
            n_iter = 20
            if use_tuning:
                search_strategy = st.radio("Strategy", ["random", "grid"])
                if search_strategy == "random":
                    n_iter = st.slider("Combinations to try", 5, 100, 20)

        use_calibration = st.checkbox("Calibrate probabilities (classification only)")
        calibration_method = "sigmoid"
        if use_calibration:
            calibration_method = st.radio("Calibration method", ["sigmoid", "isotonic"], horizontal=True)
            st.caption("**sigmoid** (Platt scaling) — fast, recommended for XGBoost/SVM. **isotonic** — more flexible, needs more data.")

        if use_ensemble or use_tuning:
            st.warning("Estimated duration: **moderate to long** depending on the selected options.")

        st.divider()

        if st.button("Start training", type="primary"):
            with st.spinner("Training…"):
                uploaded.seek(0)
                try:
                    resp = requests.post(
                        f"{api_url}/train",
                        files={"file": ("dataset.csv", uploaded, "text/csv")},
                        params={
                            "target_col": target_col, "test_size": test_size,
                            "experiment_name": experiment_name, "cv_folds": cv_folds,
                            "use_tuning": use_tuning, "search_strategy": search_strategy,
                            "n_iter": n_iter, "use_ensemble": use_ensemble,
                            "use_feature_selection": use_feature_selection,
                            "use_calibration": use_calibration,
                            "calibration_method": calibration_method,
                            "num_imputer": num_imputer,
                            "cat_imputer": cat_imputer,
                            "cat_encoder": cat_encoder,
                        },
                        timeout=600,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Best model: **{data['best_model']}** | Task: **{data['task_type']}**")

                        results_df = pd.DataFrame(data["results"]).set_index("model")
                        cv_cols = [c for c in ["cv_mean", "cv_std", "cv_train_mean"] if c in results_df.columns]
                        other_cols = [c for c in results_df.columns if c not in cv_cols + ["best_params"]]
                        ordered = cv_cols + other_cols + (["best_params"] if "best_params" in results_df.columns else [])
                        st.subheader("Model comparison")
                        st.dataframe(results_df[ordered], use_container_width=True)

                        if "cv_mean" in results_df.columns:
                            st.subheader("Scores CV")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.barh(results_df.index.tolist(), results_df["cv_mean"].tolist(),
                                    xerr=results_df["cv_std"].tolist(), color="steelblue", capsize=5, alpha=0.8)
                            ax.set_xlabel("CV score (mean ± std)")
                            ax.set_xlim(0, 1.05)
                            ax.set_title("Cross-validation — model comparison")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        evaluation = data.get("evaluation", {})
                        task_type_res = data["task_type"]

                        if task_type_res == "classification" and "confusion_matrix" in evaluation:
                            st.subheader(f"Evaluation — {data['best_model']}")
                            cm = np.array(evaluation["confusion_matrix"])
                            classes = evaluation["classes"]
                            col_cm, col_roc = st.columns(2)
                            with col_cm:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("Actual")
                                ax.set_title("Confusion matrix")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            with col_roc:
                                if "roc" in evaluation:
                                    roc = evaluation["roc"]
                                    fig, ax = plt.subplots(figsize=(5, 4))
                                    ax.plot(roc["fpr"], roc["tpr"], color="steelblue", lw=2, label=f"AUC = {roc['auc']:.3f}")
                                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                                    ax.set_xlabel("False positive rate")
                                    ax.set_ylabel("True positive rate")
                                    ax.set_title("ROC curve")
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()

                        elif task_type_res == "regression" and "residuals" in evaluation:
                            st.subheader(f"Evaluation — {data['best_model']}")
                            col_r1, col_r2 = st.columns(2)
                            with col_r1:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                ax.scatter(evaluation["y_test"], evaluation["y_pred"], alpha=0.5, color="steelblue", s=15)
                                mn, mx = min(evaluation["y_test"]), max(evaluation["y_test"])
                                ax.plot([mn, mx], [mn, mx], "r--", lw=1)
                                ax.set_xlabel("Actual")
                                ax.set_ylabel("Predicted")
                                ax.set_title("Actual vs Predicted")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            with col_r2:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                ax.hist(evaluation["residuals"], bins=30, color="salmon", edgecolor="white")
                                ax.axvline(0, color="black", lw=1, linestyle="--")
                                ax.set_title("Residuals distribution")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                    else:
                        st.error(f"Error {resp.status_code}: {resp.json().get('detail')}")
                except requests.ConnectionError:
                    st.error("Could not reach the API.")


# Tab Prediction
with tab_predict:
    st.header("Make predictions")
    pred_file = st.file_uploader("Upload CSV", type=["csv"], key="pred")

    if pred_file:
        df_pred = pd.read_csv(pred_file)
        st.dataframe(df_pred.head(), use_container_width=True)
        st.caption(f"{len(df_pred)} rows · {df_pred.shape[1]} columns")

        pred_target = st.selectbox("Target column to exclude (optional)", ["— none —"] + df_pred.columns.tolist(), key="pred_target")

        def _pred_csv(df):
            if pred_target != "— none —" and pred_target in df.columns:
                df = df.drop(columns=[pred_target])
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            return buf.getvalue().encode()

        col_pred, col_proba = st.columns(2)
        with col_pred:
            if st.button("Predict classes", type="primary"):
                with st.spinner("Predicting…"):
                    resp = requests.post(f"{api_url}/predict", files={"file": ("data.csv", _pred_csv(df_pred), "text/csv")}, timeout=60)
                    if resp.status_code == 200:
                        result = resp.json()
                        df_out = df_pred.copy()
                        df_out["prediction"] = result["predictions"]
                        st.success(f"{result['n_samples']} predictions done.")
                        st.dataframe(df_out, use_container_width=True)
                        st.download_button("Download CSV", df_out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
                    else:
                        st.error(resp.json().get("detail"))
        with col_proba:
            if st.button("Probabilities (classification)"):
                with st.spinner("Computing…"):
                    resp = requests.post(f"{api_url}/predict_proba", files={"file": ("data.csv", _pred_csv(df_pred), "text/csv")}, timeout=60)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.dataframe(pd.DataFrame(result["probabilities"], columns=result.get("classes")), use_container_width=True)
                    else:
                        st.error(resp.json().get("detail"))


# Tab Active model
with tab_model:
    st.header("Active model")

    # Model info and recommendation
    col_info, col_reco = st.columns(2)
    with col_info:
        if st.button("Model info"):
            resp = requests.get(f"{api_url}/model_info", timeout=10)
            if resp.status_code == 200:
                info = resp.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", info["model_type"])
                c2.metric("Task", info["task_type"])
                c3.metric("Features", info["n_features"])
                if info.get("classes"):
                    st.write("**Classes:**", info["classes"])
                with st.expander("Features used"):
                    st.write(info["feature_names"])
            else:
                st.error(resp.json().get("detail"))

    with col_reco:
        if st.button("Auto recommendation"):
            resp = requests.get(f"{api_url}/recommend", timeout=10)
            if resp.status_code == 200:
                reco = resp.json()
                st.success(f"Recommended model: **{reco['recommended_model']}**")
                for r in reco["reasons"]:
                    st.write(f"• {r}")
                if reco.get("overfitting_warning"):
                    st.warning("Overfitting detected on this model.")
                st.write("**All scores:**")
                st.dataframe(pd.DataFrame(reco["all_scores"].items(), columns=["Model", reco["metric"]]), use_container_width=True)
            else:
                st.error(resp.json().get("detail"))

    st.divider()

    # Downloads
    st.subheader("Downloads")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        if st.button("best_model.pkl"):
            resp = requests.get(f"{api_url}/download/model", timeout=30)
            if resp.status_code == 200:
                st.download_button("Click to download", data=resp.content, file_name="best_model.pkl", mime="application/octet-stream")
            else:
                st.error(resp.json().get("detail"))
    with col_dl2:
        if st.button("preprocessor.pkl"):
            resp = requests.get(f"{api_url}/download/preprocessor", timeout=30)
            if resp.status_code == 200:
                st.download_button("Click to download", data=resp.content, file_name="preprocessor.pkl", mime="application/octet-stream")
            else:
                st.error(resp.json().get("detail"))
    with col_dl3:
        if st.button("PDF report"):
            resp = requests.get(f"{api_url}/download/report", timeout=30)
            if resp.status_code == 200:
                st.download_button("Click to download", data=resp.content, file_name="ml_report.pdf", mime="application/pdf")
            else:
                st.error(resp.json().get("detail"))

    st.divider()

    # Explainability
    st.subheader("Explainability")
    st.caption("Feature importance (Gini/coef) + SHAP (RF/XGB/LGBM only) + Permutation importance (all models).")
    expl_file = st.file_uploader("Upload CSV (with target column)", type=["csv"], key="expl")
    if expl_file:
        df_expl = pd.read_csv(expl_file)
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            expl_target = st.selectbox("Target column", df_expl.columns.tolist(), key="expl_target")
        with col_ex2:
            n_repeats = st.slider("Permutation repeats", 3, 30, 10)
        with col_ex3:
            max_samples = st.slider("SHAP samples", 10, 300, 100)

        if st.button("Compute explainability", type="primary"):
            with st.spinner("Computing…"):

                def _plot_bar(ax, features, values, title, color="steelblue", stds=None):
                    colors = ["salmon" if v < 0 else color for v in values[::-1]]
                    ax.barh(features[::-1], values[::-1], xerr=stds[::-1] if stds else None,
                            color=colors, capsize=4 if stds else 0, alpha=0.85)
                    ax.axvline(0, color="black", lw=0.8, linestyle="--")
                    ax.set_title(title, fontsize=10)
                    plt.tight_layout()

                fi_data, pi_data, shap_data = None, None, None
                resp_fi = requests.get(f"{api_url}/feature_importance", timeout=30)
                if resp_fi.status_code == 200:
                    fi_data = resp_fi.json()

                # Permutation importance
                expl_file.seek(0)
                resp_pi = requests.post(
                    f"{api_url}/permutation_importance",
                    files={"file": ("data.csv", expl_file, "text/csv")},
                    params={"target_col": expl_target, "n_repeats": n_repeats},
                    timeout=120,
                )
                if resp_pi.status_code == 200:
                    pi_data = resp_pi.json()

                # SHAP
                expl_file.seek(0)
                resp_shap = requests.post(
                    f"{api_url}/shap",
                    files={"file": ("data.csv", expl_file, "text/csv")},
                    params={"target_col": expl_target, "max_samples": max_samples},
                    timeout=120,
                )
                if resp_shap.status_code == 200:
                    shap_data = resp_shap.json()

                available = [d for d in [fi_data, shap_data, pi_data] if d]
                if not available:
                    st.error("No explainability method available.")
                else:
                    cols = st.columns(len(available))
                    idx = 0

                    if fi_data:
                        with cols[idx]:
                            features = fi_data["feature_names"][:15]
                            values = fi_data["importances"][:15]
                            fig, ax = plt.subplots(figsize=(5, max(4, len(features) * 0.35)))
                            norm = plt.Normalize(min(values), max(values))
                            ax.barh(features[::-1], values[::-1], color=plt.cm.Blues(norm(values[::-1])))
                            ax.set_title(f"Feature Importance\n({fi_data['importance_type']})", fontsize=10)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        idx += 1

                    if shap_data:
                        with cols[idx]:
                            method_label = "SHAP" if shap_data["method"] == "shap" else "Permutation (fallback)"
                            features = shap_data["feature_names"][:15]
                            values = shap_data["importances"][:15]
                            fig, ax = plt.subplots(figsize=(5, max(4, len(features) * 0.35)))
                            _plot_bar(ax, features, values, f"{method_label}\n({shap_data['n_samples']} samples)", color="darkorange")
                            st.pyplot(fig)
                            plt.close()
                        idx += 1

                    if pi_data:
                        with cols[idx]:
                            features = pi_data["feature_names"][:15]
                            means = pi_data["importances_mean"][:15]
                            stds = pi_data["importances_std"][:15]
                            fig, ax = plt.subplots(figsize=(5, max(4, len(features) * 0.35)))
                            _plot_bar(ax, features, means, "Permutation Importance", stds=stds)
                            st.caption("Red = useless or noisy feature.")
                            st.pyplot(fig)
                            plt.close()

    st.divider()

    # Learning curves
    st.subheader("Learning curves")
    lc_file = st.file_uploader("Upload CSV (with target column)", type=["csv"], key="lc")
    if lc_file:
        df_lc = pd.read_csv(lc_file)
        col_lc1, col_lc2 = st.columns(2)
        with col_lc1:
            lc_target = st.selectbox("Target column", df_lc.columns.tolist(), key="lc_target")
        with col_lc2:
            lc_cv = st.slider("Folds CV", 2, 10, 5, key="lc_cv")
        if st.button("Generate learning curves"):
            with st.spinner("Computing…"):
                lc_file.seek(0)
                resp = requests.post(f"{api_url}/learning_curves", files={"file": ("data.csv", lc_file, "text/csv")}, params={"target_col": lc_target, "cv": lc_cv}, timeout=120)
                if resp.status_code == 200:
                    lc = resp.json()
                    sizes = lc["train_sizes"]
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(sizes, lc["train_mean"], "o-", color="steelblue", label="Train")
                    ax.fill_between(sizes,
                        np.array(lc["train_mean"]) - np.array(lc["train_std"]),
                        np.array(lc["train_mean"]) + np.array(lc["train_std"]),
                        alpha=0.15, color="steelblue")
                    ax.plot(sizes, lc["test_mean"], "o-", color="tomato", label="Validation")
                    ax.fill_between(sizes,
                        np.array(lc["test_mean"]) - np.array(lc["test_std"]),
                        np.array(lc["test_mean"]) + np.array(lc["test_std"]),
                        alpha=0.15, color="tomato")
                    ax.set_xlabel("Training set size")
                    ax.set_ylabel(lc["scoring"])
                    ax.set_title("Learning curves")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    gap = abs(lc["train_mean"][-1] - lc["test_mean"][-1])
                    if gap > 0.1:
                        st.warning(f"Train/validation gap of {gap:.3f} — possible overfitting.")
                    elif lc["test_mean"][-1] < 0.7:
                        st.warning("Low validation score — the model might benefit from more data or better features.")
                    else:
                        st.success("Curves converging — the model generalizes well.")
                else:
                    st.error(resp.json().get("detail"))

    st.divider()

    # Calibration curve
    st.subheader("Calibration curve")
    st.caption("Checks whether predicted probabilities are reliable (binary classification only). Enable 'Calibrate probabilities' at training time to see before/after.")
    cal_file = st.file_uploader("Upload CSV (with target column)", type=["csv"], key="cal")
    if cal_file:
        df_cal = pd.read_csv(cal_file)
        cal_target = st.selectbox("Target column", df_cal.columns.tolist(), key="cal_target")
        if st.button("Generate calibration curve"):
            with st.spinner("Computing…"):
                cal_file.seek(0)
                resp = requests.post(f"{api_url}/calibration", files={"file": ("data.csv", cal_file, "text/csv")}, params={"target_col": cal_target}, timeout=60)
                if resp.status_code == 200:
                    cal = resp.json()
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.plot(
                        cal["before"]["mean_predicted_value"],
                        cal["before"]["fraction_of_positives"],
                        "s-", color="steelblue", label="Before calibration"
                    )
                    if cal["calibrated"] and cal["after"]:
                        ax.plot(
                            cal["after"]["mean_predicted_value"],
                            cal["after"]["fraction_of_positives"],
                            "s-", color="darkorange", label="After calibration"
                        )
                    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
                    ax.set_xlabel("Mean predicted probability")
                    ax.set_ylabel("Fraction of positives")
                    ax.set_title("Calibration curve")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    if cal["calibrated"]:
                        st.divider()
                        st.write("**Select the active model for predictions:**")
                        col_orig, col_calib = st.columns(2)
                        with col_orig:
                            if st.button("Use original model"):
                                r = requests.post(f"{api_url}/model/activate", params={"use_calibrated": False})
                                if r.status_code == 200:
                                    st.success("Original model activated.")
                                else:
                                    st.error(r.json().get("detail"))
                        with col_calib:
                            if st.button("Use calibrated model", type="primary"):
                                r = requests.post(f"{api_url}/model/activate", params={"use_calibrated": True})
                                if r.status_code == 200:
                                    st.success("Calibrated model activated.")
                                else:
                                    st.error(r.json().get("detail"))
                    else:
                        st.info("Train with 'Calibrate probabilities' to see the after-calibration curve and pick the best one.")
                    st.caption("If the curve sticks to the diagonal: well-calibrated probabilities. Above: underestimation. Below: overestimation.")
                else:
                    st.error(resp.json().get("detail"))

    st.divider()

    # PDP
    st.subheader("Partial Dependence Plot (PDP)")
    st.caption("Shows how a feature influences predictions, all else being equal.")
    pdp_file = st.file_uploader("Upload CSV", type=["csv"], key="pdp")
    if pdp_file:
        df_pdp = pd.read_csv(pdp_file)
        col_pdp1, col_pdp2 = st.columns(2)
        with col_pdp1:
            pdp_feature = st.selectbox("Feature to analyze", df_pdp.columns.tolist(), key="pdp_feature")
        with col_pdp2:
            pdp_target = st.selectbox("Target column to exclude (optional)", ["— none —"] + df_pdp.columns.tolist(), key="pdp_target")
        if st.button("Generate PDP"):
            with st.spinner("Computing…"):
                if pdp_target != "— none —":
                    df_pdp = df_pdp.drop(columns=[pdp_target])
                buf = io.StringIO()
                df_pdp.to_csv(buf, index=False)
                resp = requests.post(
                    f"{api_url}/pdp",
                    files={"file": ("data.csv", buf.getvalue().encode(), "text/csv")},
                    params={"feature": pdp_feature},
                    timeout=60,
                )
                if resp.status_code == 200:
                    pdp = resp.json()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(pdp["grid_values"], pdp["pdp_values"], color="steelblue", lw=2)
                    ax.fill_between(pdp["grid_values"], pdp["pdp_values"], alpha=0.1, color="steelblue")
                    ax.set_xlabel(pdp["feature"])
                    ax.set_ylabel("Average prediction")
                    ax.set_title(f"PDP - {pdp['feature']}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.error(resp.json().get("detail"))


# Tab Drift
with tab_drift:
    st.header("Data Drift Detection")
    st.caption("Compares new data against training data to detect statistical drift.")

    drift_file = st.file_uploader("Upload CSV (new data)", type=["csv"], key="drift")
    drift_target = st.text_input("Target column (optional, to exclude)", "")

    if drift_file and st.button("Analyze drift", type="primary"):
        with st.spinner("Analyzing…"):
            drift_file.seek(0)
            params = {}
            if drift_target:
                params["target_col"] = drift_target
            resp = requests.post(f"{api_url}/drift", files={"file": ("data.csv", drift_file, "text/csv")}, params=params, timeout=60)

            if resp.status_code == 200:
                result = resp.json()
                summary = result["summary"]

                status_color = {"ok": "success", "warning": "warning", "alert": "error"}[summary["overall_status"]]
                getattr(st, status_color)(f"Overall status: **{summary['overall_status'].upper()}** — {summary['drifted_features']}/{summary['total_features']} drifted features ({summary['drift_rate']*100:.0f}%)")

                features_data = result["features"]
                rows = []
                for feat, info in features_data.items():
                    row = {"feature": feat, "type": info["type"], "drift": "⚠️ Yes" if info["drift_detected"] else "✅ No", "severity": info["severity"]}
                    if info["type"] == "numerical":
                        row.update({"p_value": info["p_value"], "PSI": info["psi"], "ref mean": info.get("ref_mean"), "new mean": info.get("new_mean")})
                    else:
                        row["new categories"] = str(info.get("new_categories", []))
                    rows.append(row)

                df_drift = pd.DataFrame(rows).set_index("feature")
                st.dataframe(df_drift, use_container_width=True)

                drifted = [f for f, i in features_data.items() if i["drift_detected"]]
                if drifted:
                    st.subheader("Drifted features — mean comparison")
                    num_drifted = [f for f in drifted if features_data[f]["type"] == "numerical" and "ref_mean" in features_data[f]][:10]
                    if num_drifted:
                        fig, ax = plt.subplots(figsize=(10, max(3, len(num_drifted) * 0.5)))
                        x = np.arange(len(num_drifted))
                        width = 0.35
                        ref_means = [features_data[f]["ref_mean"] for f in num_drifted]
                        new_means = [features_data[f]["new_mean"] for f in num_drifted]
                        ax.bar(x - width/2, ref_means, width, label="Reference (train)", color="steelblue", alpha=0.8)
                        ax.bar(x + width/2, new_means, width, label="New data", color="tomato", alpha=0.8)
                        ax.set_xticks(x)
                        ax.set_xticklabels(num_drifted, rotation=30, ha="right")
                        ax.legend()
                        ax.set_title("Mean comparison (drifted features)")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            else:
                st.error(resp.json().get("detail"))


# Tab MLflow Runs
with tab_runs:
    st.header("MLflow Runs")

    exp_name = st.text_input("Experiment", "ml_experiment", key="exp_runs")
    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        if st.button("Load runs"):
            resp = requests.get(f"{api_url}/runs", params={"experiment_name": exp_name}, timeout=10)
            if resp.status_code == 200:
                runs = resp.json().get("runs", [])
                if runs:
                    df_runs = pd.DataFrame(runs)
                    st.dataframe(df_runs, use_container_width=True)
                    st.session_state["runs"] = runs
                else:
                    st.info("No runs found.")
            else:
                st.error(resp.json().get("detail"))

    with col_r2:
        if st.button("Recommandation"):
            resp = requests.get(f"{api_url}/recommend", params={"experiment_name": exp_name}, timeout=10)
            if resp.status_code == 200:
                reco = resp.json()
                st.success(f"**{reco['recommended_model']}** ({reco['metric']}={reco['score']:.4f})")
                for r in reco["reasons"]:
                    st.write(f"• {r}")
            else:
                st.error(resp.json().get("detail"))

    st.divider()
    st.subheader("Compare two runs")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        run_id_1 = st.text_input("Run ID 1 (copy from table above)")
    with col_c2:
        run_id_2 = st.text_input("Run ID 2")

    if run_id_1 and run_id_2 and st.button("Compare"):
        resp = requests.get(f"{api_url}/runs/compare", params={"run_id_1": run_id_1, "run_id_2": run_id_2}, timeout=10)
        if resp.status_code == 200:
            cmp = resp.json()
            col_x, col_y = st.columns(2)
            with col_x:
                st.write(f"**Run 1 — {cmp['run_1'].get('model', '?')}**")
                st.json(cmp["run_1"])
            with col_y:
                st.write(f"**Run 2 — {cmp['run_2'].get('model', '?')}**")
                st.json(cmp["run_2"])
        else:
            st.error(resp.json().get("detail"))
