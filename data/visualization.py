import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

metrics = {
    "XGB": {
        "MSE": 0.479,
        "MAE": 0.519,
        "R2": 0.561,
        "CV_MSE": 0.478,
        "CV_MAE": 0.519,
        "CV_R2": 0.563,
        "Precision@10": 0.108,
        "Recall@10": 0.754,
        "MAP": 0.750,
        "train_start": "2025-08-29 20:58:30",
        "train_end": "2025-08-29 21:00:10",
        "cv_end": "2025-08-29 21:25:59"
    },
    "RF": {
        "MSE": 0.841,
        "MAE": 0.700,
        "R2": 0.230,
        "CV_MSE": 0.844,
        "CV_MAE": 0.702,
        "CV_R2": 0.227,
        "Precision@10": 0.108,
        "Recall@10": 0.753,
        "MAP": 0.711,
        "train_start": "2025-08-29 21:55:28",
        "train_end": "2025-08-29 22:15:07",
        "cv_end": "2025-08-30 02:52:08"
    }
}
output_path = "data/model_comparison.png"

def parse_time(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

for model, m in metrics.items():
    train_time = (parse_time(m["train_end"]) - parse_time(m["train_start"])).total_seconds() / 60
    cv_time = (parse_time(m["cv_end"]) - parse_time(m["train_end"])).total_seconds() / 60
    m["Train Time (min)"] = train_time
    m["CV Time (min)"] = cv_time

def plot_grouped_bar(ax, metric_names, title, higher_better=True):
    values_xgb = [metrics["XGB"][m] for m in metric_names]
    values_rf = [metrics["RF"][m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    ax.bar(x - width/2, values_xgb, width, label="XGB")
    ax.bar(x + width/2, values_rf, width, label="RF")

    ax.set_title(f"{title}\n({'Higher' if higher_better else 'Lower'} is better)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=25, ha="right")
    ax.legend()

def compute_overall_winner():
    scores = {"XGB": 0, "RF": 0}
    comparisons = [
        ("MSE", False), ("MAE", False), ("R2", True),
        ("CV_MSE", False), ("CV_MAE", False), ("CV_R2", True),
        ("Precision@10", True), ("Recall@10", True), ("MAP", True),
        ("Train Time (min)", False), ("CV Time (min)", False)
    ]
    for metric, higher_better in comparisons:
        v1, v2 = metrics["XGB"][metric], metrics["RF"][metric]
        if v1 == v2:
            continue
        winner = "XGB" if ((v1 > v2) == higher_better) else "RF"
        scores[winner] += 1
    return scores

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 4, figsize=(16, 10))

plot_grouped_bar(axes[0,0], ["MSE", "MAE"], "Training Errors", higher_better=False)
plot_grouped_bar(axes[0,1], ["R2"], "Training R²", higher_better=True)

plot_grouped_bar(axes[0,2], ["CV_MSE", "CV_MAE"], "CV Errors", higher_better=False)
plot_grouped_bar(axes[0,3], ["CV_R2"], "CV R²", higher_better=True)

plot_grouped_bar(axes[1,0], ["Precision@10", "Recall@10", "MAP"], "Ranking Metrics", higher_better=True)

plot_grouped_bar(axes[1,1], ["Train Time (min)"], "Training Time", higher_better=False)
plot_grouped_bar(axes[1,2], ["CV Time (min)"], "CV Time", higher_better=False)

axes[1,3].axis('off')
legend_lines = [
    "",
    "",
    "Regression Models",
    "XGB: Extreme Gradient Boosting",
    "RF: Random Forest",
    "",
    "Regression Metrics",
    "MSE: Mean Squared Error",
    "MAE: Mean Absolute Error",
    "R²: Coefficient of Determination",
    "CV: Cross-Validation",
    "Precision@10: Precision at 10",
    "Recall@10: Recall at 10",
    "MAP: Mean Average Precision"
]
legend_text = '\n'.join(legend_lines)
axes[1,3].text(
    0.02, 0.98,
    legend_text,
    fontsize=14,
    va='top',
    ha='left',
    linespacing=1.5,
    bbox=dict(boxstyle='round,pad=0.7', facecolor='#f7f7f7', edgecolor='#cccccc', alpha=0.95),
    transform=axes[1,3].transAxes,
)
axes[1,3].text(
    0.48, 0.98,
    "Legend",
    fontsize=16,
    va='top',
    ha='center',
    fontweight='bold',
    transform=axes[1,3].transAxes
)

plt.tight_layout(h_pad=6, w_pad=4)
plt.savefig(output_path, dpi=200, bbox_inches='tight')
plt.show()
plt.close()
