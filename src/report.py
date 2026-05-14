import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fpdf import FPDF

matplotlib.use("Agg")


class _MLReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(30, 90, 160)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "  ML Pipeline - Training Report", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(235, 242, 255)
        self.cell(0, 9, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def kv(self, key: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(55, 7, f"{key} :", new_x="RIGHT", new_y="LAST")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_pdf_report(
    dataset_info: dict,
    results_df: pd.DataFrame,
    best_model_name: str,
    task_type: str,
    evaluation: dict,
    output_path: str = "models/report.pdf",
) -> str:
    pdf = _MLReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(5)
    pdf.cell(0, 14, "ML Training Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.section_title("Dataset info")
    for k, v in dataset_info.items():
        pdf.kv(k, v)
    pdf.ln(4)

    pdf.section_title("Results")
    pdf.kv("Task", task_type)
    pdf.kv("Best model", best_model_name)
    pdf.kv("Models evaluated", str(len(results_df)))
    pdf.ln(4)

    pdf.section_title("Model comparison")
    cols_to_show = [c for c in results_df.columns if c not in ("best_params",)]
    col_w = min(28, (pdf.w - 20) / (len(cols_to_show) + 1))

    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(200, 215, 245)
    pdf.cell(35, 7, "Model", border=1, fill=True)
    for c in cols_to_show:
        pdf.cell(col_w, 7, c[:10], border=1, fill=True, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for _, row in results_df.iterrows():
        fill = row.name == best_model_name
        pdf.set_fill_color(230, 255, 230) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(35, 6, str(row.name)[:16], border=1, fill=fill)
        for c in cols_to_show:
            val = row.get(c, "")
            pdf.cell(col_w, 6, f"{val:.4f}" if isinstance(val, float) else str(val)[:8], border=1, fill=fill, align="C")
        pdf.ln()
    pdf.ln(5)

    tmp_files = []

    if task_type == "classification" and "confusion_matrix" in evaluation:
        cm = np.array(evaluation["confusion_matrix"])
        classes = evaluation.get("classes", list(range(len(cm))))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=axes[0])
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title("Confusion matrix")

        if "roc" in evaluation:
            roc = evaluation["roc"]
            axes[1].plot(roc["fpr"], roc["tpr"], color="steelblue", lw=2, label=f"AUC = {roc['auc']:.3f}")
            axes[1].plot([0, 1], [0, 1], "k--", lw=1)
            axes[1].set_xlabel("False positive rate")
            axes[1].set_ylabel("True positive rate")
            axes[1].set_title("ROC curve")
            axes[1].legend()
        else:
            axes[1].set_visible(False)

        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        tmp_files.append(tmp.name)
        plt.close()

        pdf.add_page()
        pdf.section_title("Evaluation - Confusion Matrix & ROC Curve")
        pdf.image(tmp.name, x=10, w=190)

    elif task_type == "regression" and "residuals" in evaluation:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(evaluation["y_test"], evaluation["y_pred"], alpha=0.4, color="steelblue", s=12)
        mn, mx = min(evaluation["y_test"]), max(evaluation["y_test"])
        axes[0].plot([mn, mx], [mn, mx], "r--", lw=1)
        axes[0].set_xlabel("Actual")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Actual vs Predicted")

        axes[1].hist(evaluation["residuals"], bins=30, color="salmon", edgecolor="white")
        axes[1].axvline(0, color="black", lw=1, linestyle="--")
        axes[1].set_title("Residuals distribution")

        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        tmp_files.append(tmp.name)
        plt.close()

        pdf.add_page()
        pdf.section_title("Evaluation - Actual vs Predicted & Residuals")
        pdf.image(tmp.name, x=10, w=190)

    for f in tmp_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    Path(output_path).parent.mkdir(exist_ok=True)
    pdf.output(output_path)
    return output_path
