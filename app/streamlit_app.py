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
st.title("ML Dashboard — Apprentissage Supervisé")

with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", value=API_URL, disabled=bool(os.getenv("API_URL")))
    try:
        resp = requests.get(f"{api_url}/health", timeout=2)
        if resp.status_code == 200:
            st.success("API connectée")
        else:
            st.error("API inaccessible")
    except Exception:
        st.error("API inaccessible — lancez `docker-compose up`")

# Tabs
tab_eda, tab_train, tab_predict, tab_model, tab_drift, tab_runs = st.tabs([
    "Analyse", "Entraînement", "Prédiction", "Modèle actif", "Drift", "MLflow Runs"
])


# Tab EDA
with tab_eda:
    st.header("Analyse exploratoire")
    eda_file = st.file_uploader("Upload CSV", type=["csv"], key="eda")

    if eda_file:
        df = pd.read_csv(eda_file)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lignes", df.shape[0])
        c2.metric("Colonnes", df.shape[1])
        c3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))
        c4.metric("Doublons", int(df.duplicated().sum()))

        with st.expander("Aperçu des données", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        with st.expander("Types & valeurs manquantes"):
            dtype_df = pd.DataFrame({
                "Type": df.dtypes.astype(str),
                "Valeurs uniques": df.nunique(),
                "Manquantes": df.isnull().sum(),
                "Manquantes (%)": (df.isnull().sum() / len(df) * 100).round(1),
            })
            st.dataframe(dtype_df, use_container_width=True)

        with st.expander("Statistiques descriptives"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            st.subheader("Valeurs manquantes")
            fig, ax = plt.subplots(figsize=(10, max(3, len(missing) * 0.4)))
            pct = (missing / len(df) * 100).round(1)
            bars = ax.barh(missing.index, pct, color="salmon")
            ax.bar_label(bars, fmt="%.1f%%", padding=4)
            ax.set_xlabel("% manquant")
            ax.set_xlim(0, 115)
            ax.invert_yaxis()
            ax.set_title("Valeurs manquantes par colonne")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("Aucune valeur manquante.")

        st.subheader("Analyse de la colonne cible")
        target_eda = st.selectbox("Choisir la colonne cible", df.columns.tolist(), key="eda_target")
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
            ax.set_title(f"Distribution de « {target_eda} »")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with col_right:
            if df[target_eda].dtype == object or df[target_eda].nunique() <= 20:
                fig, ax = plt.subplots(figsize=(6, 4))
                df[target_eda].value_counts().plot.pie(ax=ax, autopct="%1.1f%%", startangle=90, colors=plt.cm.Set2.colors)
                ax.set_ylabel("")
                ax.set_title("Répartition des classes")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target_eda in num_cols:
            num_cols.remove(target_eda)
        if num_cols:
            st.subheader("Distributions numériques")
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
            st.subheader("Matrice de corrélation")
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
            st.subheader("Variables catégorielles")
            selected_cat = st.selectbox("Choisir une variable", cat_cols, key="cat_col")
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
    st.header("Entraîner un modèle")
    uploaded = st.file_uploader("Dataset CSV (avec colonne cible)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Colonne cible", df.columns.tolist())
        with col2:
            test_size = st.slider("Taille du test set", 0.1, 0.4, 0.2, 0.05)
        with col3:
            experiment_name = st.text_input("Expérience MLflow", "ml_experiment")

        c1, c2, c3 = st.columns(3)
        c1.metric("Lignes", df.shape[0])
        c2.metric("Colonnes", df.shape[1])
        c3.metric("Valeurs manquantes", int(df.isnull().sum().sum()))

        st.subheader("Adversarial Validation")
        st.caption("Vérifie si le train/test split est homogène avant d'entraîner.")
        if st.button("Lancer l'adversarial validation"):
            with st.spinner("Validation en cours…"):
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
                    getattr(st, status_color)(f"AUC moyen : **{av['overall_auc']}** — {av['interpretation']}")

                    col_rf, col_xgb = st.columns(2)
                    for col, model_name in [(col_rf, "RandomForest"), (col_xgb, "XGBoost")]:
                        with col:
                            m = av["models"][model_name]
                            st.metric(f"{model_name} AUC", m["auc"])
                            top_df = pd.DataFrame(m["top_features"])
                            fig, ax = plt.subplots(figsize=(5, 4))
                            ax.barh(top_df["feature"][::-1], top_df["importance"][::-1], color="steelblue")
                            ax.set_title(f"{model_name} — features discriminantes")
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
            num_imputer_label = st.selectbox("Imputation numérique", ["median", "mean", "knn", "constant (0)"], help="Médiane : robuste aux outliers. Moyenne : sensible aux outliers. KNN : voisins les plus proches (plus précis, plus lent). Constante : remplace par 0.")
            num_imputer = "constant" if num_imputer_label == "constant (0)" else num_imputer_label
        with col_p2:
            cat_imputer = st.selectbox("Imputation catégorielle", ["most_frequent", "constant (missing)"], help="Plus fréquente : valeur dominante. Constante : crée une catégorie 'missing'.")
            cat_imputer = "constant" if cat_imputer == "constant (missing)" else cat_imputer
        with col_p3:
            cat_encoder = st.selectbox("Encodage catégoriel", ["onehot", "ordinal"], help="OneHot : une colonne par modalité (mieux pour LR). Ordinal : entiers (mieux pour RF/XGB/LGBM).")

        st.divider()
        st.subheader("Options avancées")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            use_feature_selection = st.checkbox("Sélection automatique de features")
            use_ensemble = st.checkbox("Modèles ensemblistes (Voting)")
        with col_b:
            cv_folds = st.slider("Cross-validation (folds)", 0, 10, 0)
            use_tuning = st.checkbox("Fine-tuning des hyperparamètres")
        with col_c:
            search_strategy = "random"
            n_iter = 20
            if use_tuning:
                search_strategy = st.radio("Stratégie", ["random", "grid"])
                if search_strategy == "random":
                    n_iter = st.slider("Combinaisons testées", 5, 100, 20)

        use_calibration = st.checkbox("Calibrer les probabilités (classification uniquement)")
        calibration_method = "sigmoid"
        if use_calibration:
            calibration_method = st.radio("Méthode de calibration", ["sigmoid", "isotonic"], horizontal=True)
            st.caption("**sigmoid** (Platt scaling) — rapide, recommandé pour XGBoost/SVM. **isotonic** — plus flexible, nécessite plus de données.")

        if use_ensemble or use_tuning:
            st.warning("Durée estimée : **modérée à longue** selon les options choisies.")

        st.divider()

        if st.button("Lancer l'entraînement", type="primary"):
            with st.spinner("Entraînement en cours…"):
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
                        st.success(f"Meilleur modèle : **{data['best_model']}** | Tâche : **{data['task_type']}**")

                        results_df = pd.DataFrame(data["results"]).set_index("model")
                        cv_cols = [c for c in ["cv_mean", "cv_std", "cv_train_mean"] if c in results_df.columns]
                        other_cols = [c for c in results_df.columns if c not in cv_cols + ["best_params"]]
                        ordered = cv_cols + other_cols + (["best_params"] if "best_params" in results_df.columns else [])
                        st.subheader("Comparaison des modèles")
                        st.dataframe(results_df[ordered], use_container_width=True)

                        if "cv_mean" in results_df.columns:
                            st.subheader("Scores CV")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.barh(results_df.index.tolist(), results_df["cv_mean"].tolist(),
                                    xerr=results_df["cv_std"].tolist(), color="steelblue", capsize=5, alpha=0.8)
                            ax.set_xlabel("Score CV (moyenne ± écart-type)")
                            ax.set_xlim(0, 1.05)
                            ax.set_title("Cross-validation — comparaison des modèles")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        evaluation = data.get("evaluation", {})
                        task_type_res = data["task_type"]

                        if task_type_res == "classification" and "confusion_matrix" in evaluation:
                            st.subheader(f"Évaluation — {data['best_model']}")
                            cm = np.array(evaluation["confusion_matrix"])
                            classes = evaluation["classes"]
                            col_cm, col_roc = st.columns(2)
                            with col_cm:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
                                ax.set_xlabel("Prédit")
                                ax.set_ylabel("Réel")
                                ax.set_title("Matrice de confusion")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            with col_roc:
                                if "roc" in evaluation:
                                    roc = evaluation["roc"]
                                    fig, ax = plt.subplots(figsize=(5, 4))
                                    ax.plot(roc["fpr"], roc["tpr"], color="steelblue", lw=2, label=f"AUC = {roc['auc']:.3f}")
                                    ax.plot([0, 1], [0, 1], "k--", lw=1)
                                    ax.set_xlabel("Faux positifs")
                                    ax.set_ylabel("Vrais positifs")
                                    ax.set_title("Courbe ROC")
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()

                        elif task_type_res == "regression" and "residuals" in evaluation:
                            st.subheader(f"Évaluation — {data['best_model']}")
                            col_r1, col_r2 = st.columns(2)
                            with col_r1:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                ax.scatter(evaluation["y_test"], evaluation["y_pred"], alpha=0.5, color="steelblue", s=15)
                                mn, mx = min(evaluation["y_test"]), max(evaluation["y_test"])
                                ax.plot([mn, mx], [mn, mx], "r--", lw=1)
                                ax.set_xlabel("Réel")
                                ax.set_ylabel("Prédit")
                                ax.set_title("Réel vs Prédit")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            with col_r2:
                                fig, ax = plt.subplots(figsize=(5, 4))
                                ax.hist(evaluation["residuals"], bins=30, color="salmon", edgecolor="white")
                                ax.axvline(0, color="black", lw=1, linestyle="--")
                                ax.set_title("Distribution des résidus")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                    else:
                        st.error(f"Erreur {resp.status_code}: {resp.json().get('detail')}")
                except requests.ConnectionError:
                    st.error("Impossible de joindre l'API.")


# Tab Prediction
with tab_predict:
    st.header("Faire des prédictions")
    pred_file = st.file_uploader("Upload CSV", type=["csv"], key="pred")

    if pred_file:
        df_pred = pd.read_csv(pred_file)
        st.dataframe(df_pred.head(), use_container_width=True)
        st.caption(f"{len(df_pred)} lignes · {df_pred.shape[1]} colonnes")

        pred_target = st.selectbox("Colonne cible à exclure (optionnel)", ["— aucune —"] + df_pred.columns.tolist(), key="pred_target")

        def _pred_csv(df):
            if pred_target != "— aucune —" and pred_target in df.columns:
                df = df.drop(columns=[pred_target])
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            return buf.getvalue().encode()

        col_pred, col_proba = st.columns(2)
        with col_pred:
            if st.button("Prédire les classes", type="primary"):
                with st.spinner("Prédiction…"):
                    resp = requests.post(f"{api_url}/predict", files={"file": ("data.csv", _pred_csv(df_pred), "text/csv")}, timeout=60)
                    if resp.status_code == 200:
                        result = resp.json()
                        df_out = df_pred.copy()
                        df_out["prediction"] = result["predictions"]
                        st.success(f"{result['n_samples']} prédictions effectuées.")
                        st.dataframe(df_out, use_container_width=True)
                        st.download_button("Télécharger CSV", df_out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
                    else:
                        st.error(resp.json().get("detail"))
        with col_proba:
            if st.button("Probabilités (classification)"):
                with st.spinner("Calcul…"):
                    resp = requests.post(f"{api_url}/predict_proba", files={"file": ("data.csv", _pred_csv(df_pred), "text/csv")}, timeout=60)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.dataframe(pd.DataFrame(result["probabilities"], columns=result.get("classes")), use_container_width=True)
                    else:
                        st.error(resp.json().get("detail"))


# Tab Modèle actif
with tab_model:
    st.header("Modèle actif")

    # Model info and recommendation
    col_info, col_reco = st.columns(2)
    with col_info:
        if st.button("Infos du modèle"):
            resp = requests.get(f"{api_url}/model_info", timeout=10)
            if resp.status_code == 200:
                info = resp.json()
                c1, c2, c3 = st.columns(3)
                c1.metric("Modèle", info["model_type"])
                c2.metric("Tâche", info["task_type"])
                c3.metric("Features", info["n_features"])
                if info.get("classes"):
                    st.write("**Classes :**", info["classes"])
                with st.expander("Features utilisées"):
                    st.write(info["feature_names"])
            else:
                st.error(resp.json().get("detail"))

    with col_reco:
        if st.button("Recommandation automatique"):
            resp = requests.get(f"{api_url}/recommend", timeout=10)
            if resp.status_code == 200:
                reco = resp.json()
                st.success(f"Modèle recommandé : **{reco['recommended_model']}**")
                for r in reco["reasons"]:
                    st.write(f"• {r}")
                if reco.get("overfitting_warning"):
                    st.warning("Surapprentissage détecté sur ce modèle.")
                st.write("**Tous les scores :**")
                st.dataframe(pd.DataFrame(reco["all_scores"].items(), columns=["Modèle", reco["metric"]]), use_container_width=True)
            else:
                st.error(resp.json().get("detail"))

    st.divider()

    # Downloads
    st.subheader("Télécharger")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        if st.button("best_model.pkl"):
            resp = requests.get(f"{api_url}/download/model", timeout=30)
            if resp.status_code == 200:
                st.download_button("Cliquer pour télécharger", data=resp.content, file_name="best_model.pkl", mime="application/octet-stream")
            else:
                st.error(resp.json().get("detail"))
    with col_dl2:
        if st.button("preprocessor.pkl"):
            resp = requests.get(f"{api_url}/download/preprocessor", timeout=30)
            if resp.status_code == 200:
                st.download_button("Cliquer pour télécharger", data=resp.content, file_name="preprocessor.pkl", mime="application/octet-stream")
            else:
                st.error(resp.json().get("detail"))
    with col_dl3:
        if st.button("Rapport PDF"):
            resp = requests.get(f"{api_url}/download/report", timeout=30)
            if resp.status_code == 200:
                st.download_button("Cliquer pour télécharger", data=resp.content, file_name="rapport_ml.pdf", mime="application/pdf")
            else:
                st.error(resp.json().get("detail"))

    st.divider()

    # Explainability
    st.subheader("Explicabilité")
    st.caption("Feature importance (Gini/coef) + SHAP (RF/XGB/LGBM uniquement) + Permutation importance (tous les modèles).")
    expl_file = st.file_uploader("Upload CSV (avec colonne cible)", type=["csv"], key="expl")
    if expl_file:
        df_expl = pd.read_csv(expl_file)
        col_ex1, col_ex2, col_ex3 = st.columns(3)
        with col_ex1:
            expl_target = st.selectbox("Colonne cible", df_expl.columns.tolist(), key="expl_target")
        with col_ex2:
            n_repeats = st.slider("Répétitions permutation", 3, 30, 10)
        with col_ex3:
            max_samples = st.slider("Échantillons SHAP", 10, 300, 100)

        if st.button("Calculer l'explicabilité", type="primary"):
            with st.spinner("Calcul en cours…"):

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
                    st.error("Aucune méthode disponible.")
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
                            _plot_bar(ax, features, values, f"{method_label}\n({shap_data['n_samples']} échantillons)", color="darkorange")
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
                            st.caption("Rouge = feature inutile ou bruitée.")
                            st.pyplot(fig)
                            plt.close()

    st.divider()

    # Learning curves
    st.subheader("Courbes d'apprentissage")
    lc_file = st.file_uploader("Upload CSV (avec colonne cible)", type=["csv"], key="lc")
    if lc_file:
        df_lc = pd.read_csv(lc_file)
        col_lc1, col_lc2 = st.columns(2)
        with col_lc1:
            lc_target = st.selectbox("Colonne cible", df_lc.columns.tolist(), key="lc_target")
        with col_lc2:
            lc_cv = st.slider("Folds CV", 2, 10, 5, key="lc_cv")
        if st.button("Générer les courbes d'apprentissage"):
            with st.spinner("Calcul…"):
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
                    ax.set_xlabel("Taille du dataset d'entraînement")
                    ax.set_ylabel(lc["scoring"])
                    ax.set_title("Courbes d'apprentissage")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    gap = abs(lc["train_mean"][-1] - lc["test_mean"][-1])
                    if gap > 0.1:
                        st.warning(f"Écart train/validation de {gap:.3f} — possible surapprentissage.")
                    elif lc["test_mean"][-1] < 0.7:
                        st.warning("Score de validation faible — le modèle gagnerait à avoir plus de données ou de meilleures features.")
                    else:
                        st.success("Courbes convergentes — le modèle généralise bien.")
                else:
                    st.error(resp.json().get("detail"))

    st.divider()

    # Calibration curve
    st.subheader("Courbe de calibration")
    st.caption("Vérifie si les probabilités prédites sont fiables (classification binaire uniquement). Active 'Calibrer les probabilités' à l'entraînement pour voir avant/après.")
    cal_file = st.file_uploader("Upload CSV (avec colonne cible)", type=["csv"], key="cal")
    if cal_file:
        df_cal = pd.read_csv(cal_file)
        cal_target = st.selectbox("Colonne cible", df_cal.columns.tolist(), key="cal_target")
        if st.button("Générer la courbe de calibration"):
            with st.spinner("Calcul…"):
                cal_file.seek(0)
                resp = requests.post(f"{api_url}/calibration", files={"file": ("data.csv", cal_file, "text/csv")}, params={"target_col": cal_target}, timeout=60)
                if resp.status_code == 200:
                    cal = resp.json()
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.plot(
                        cal["before"]["mean_predicted_value"],
                        cal["before"]["fraction_of_positives"],
                        "s-", color="steelblue", label="Avant calibration"
                    )
                    if cal["calibrated"] and cal["after"]:
                        ax.plot(
                            cal["after"]["mean_predicted_value"],
                            cal["after"]["fraction_of_positives"],
                            "s-", color="darkorange", label="Après calibration"
                        )
                    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibration parfaite")
                    ax.set_xlabel("Probabilité prédite moyenne")
                    ax.set_ylabel("Fraction de positifs réels")
                    ax.set_title("Courbe de calibration")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    if cal["calibrated"]:
                        st.divider()
                        st.write("**Choisir le modèle actif pour les prédictions :**")
                        col_orig, col_calib = st.columns(2)
                        with col_orig:
                            if st.button("Utiliser le modèle original"):
                                r = requests.post(f"{api_url}/model/activate", params={"use_calibrated": False})
                                if r.status_code == 200:
                                    st.success("Modèle original activé.")
                                else:
                                    st.error(r.json().get("detail"))
                        with col_calib:
                            if st.button("Utiliser le modèle calibré", type="primary"):
                                r = requests.post(f"{api_url}/model/activate", params={"use_calibrated": True})
                                if r.status_code == 200:
                                    st.success("Modèle calibré activé.")
                                else:
                                    st.error(r.json().get("detail"))
                    else:
                        st.info("Entraîne avec 'Calibrer les probabilités' pour voir la courbe après calibration et choisir le meilleur.")
                    st.caption("Si la courbe colle à la diagonale : probabilités bien calibrées. Au-dessus : sous-estimation. En dessous : surestimation.")
                else:
                    st.error(resp.json().get("detail"))

    st.divider()

    # PDP
    st.subheader("Partial Dependence Plot (PDP)")
    st.caption("Montre comment une feature influence les prédictions, toutes choses égales par ailleurs.")
    pdp_file = st.file_uploader("Upload CSV", type=["csv"], key="pdp")
    if pdp_file:
        df_pdp = pd.read_csv(pdp_file)
        col_pdp1, col_pdp2 = st.columns(2)
        with col_pdp1:
            pdp_feature = st.selectbox("Feature à analyser", df_pdp.columns.tolist(), key="pdp_feature")
        with col_pdp2:
            pdp_target = st.selectbox("Colonne cible à exclure (optionnel)", ["— aucune —"] + df_pdp.columns.tolist(), key="pdp_target")
        if st.button("Générer le PDP"):
            with st.spinner("Calcul…"):
                if pdp_target != "— aucune —":
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
                    ax.set_ylabel("Prédiction moyenne")
                    ax.set_title(f"PDP - {pdp['feature']}")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.error(resp.json().get("detail"))


# Tab Drift
with tab_drift:
    st.header("Détection de Data Drift")
    st.caption("Compare de nouvelles données avec les données d'entraînement pour détecter des dérives statistiques.")

    drift_file = st.file_uploader("Upload CSV (nouvelles données)", type=["csv"], key="drift")
    drift_target = st.text_input("Colonne cible (optionnel, pour l'exclure)", "")

    if drift_file and st.button("Analyser le drift", type="primary"):
        with st.spinner("Analyse en cours…"):
            drift_file.seek(0)
            params = {}
            if drift_target:
                params["target_col"] = drift_target
            resp = requests.post(f"{api_url}/drift", files={"file": ("data.csv", drift_file, "text/csv")}, params=params, timeout=60)

            if resp.status_code == 200:
                result = resp.json()
                summary = result["summary"]

                status_color = {"ok": "success", "warning": "warning", "alert": "error"}[summary["overall_status"]]
                getattr(st, status_color)(f"Statut global : **{summary['overall_status'].upper()}** — {summary['drifted_features']}/{summary['total_features']} features en drift ({summary['drift_rate']*100:.0f}%)")

                features_data = result["features"]
                rows = []
                for feat, info in features_data.items():
                    row = {"feature": feat, "type": info["type"], "drift": "⚠️ Oui" if info["drift_detected"] else "✅ Non", "sévérité": info["severity"]}
                    if info["type"] == "numerical":
                        row.update({"p_value": info["p_value"], "PSI": info["psi"], "moy. ref": info.get("ref_mean"), "moy. new": info.get("new_mean")})
                    else:
                        row["nouvelles catégories"] = str(info.get("new_categories", []))
                    rows.append(row)

                df_drift = pd.DataFrame(rows).set_index("feature")
                st.dataframe(df_drift, use_container_width=True)

                drifted = [f for f, i in features_data.items() if i["drift_detected"]]
                if drifted:
                    st.subheader("Features en drift — comparaison moyenne")
                    num_drifted = [f for f in drifted if features_data[f]["type"] == "numerical" and "ref_mean" in features_data[f]][:10]
                    if num_drifted:
                        fig, ax = plt.subplots(figsize=(10, max(3, len(num_drifted) * 0.5)))
                        x = np.arange(len(num_drifted))
                        width = 0.35
                        ref_means = [features_data[f]["ref_mean"] for f in num_drifted]
                        new_means = [features_data[f]["new_mean"] for f in num_drifted]
                        ax.bar(x - width/2, ref_means, width, label="Référence (train)", color="steelblue", alpha=0.8)
                        ax.bar(x + width/2, new_means, width, label="Nouvelles données", color="tomato", alpha=0.8)
                        ax.set_xticks(x)
                        ax.set_xticklabels(num_drifted, rotation=30, ha="right")
                        ax.legend()
                        ax.set_title("Comparaison des moyennes (features en drift)")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            else:
                st.error(resp.json().get("detail"))


# Tab MLflow Runs
with tab_runs:
    st.header("MLflow Runs")

    exp_name = st.text_input("Expérience", "ml_experiment", key="exp_runs")
    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        if st.button("Charger les runs"):
            resp = requests.get(f"{api_url}/runs", params={"experiment_name": exp_name}, timeout=10)
            if resp.status_code == 200:
                runs = resp.json().get("runs", [])
                if runs:
                    df_runs = pd.DataFrame(runs)
                    st.dataframe(df_runs, use_container_width=True)
                    st.session_state["runs"] = runs
                else:
                    st.info("Aucun run trouvé.")
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
    st.subheader("Comparer deux runs")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        run_id_1 = st.text_input("Run ID 1 (copier depuis le tableau ci-dessus)")
    with col_c2:
        run_id_2 = st.text_input("Run ID 2")

    if run_id_1 and run_id_2 and st.button("Comparer"):
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
