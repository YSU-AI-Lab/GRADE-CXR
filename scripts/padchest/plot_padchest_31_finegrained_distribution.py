from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


AUDIT_DIR = Path(
    "/home/hyr/clip/biomedclip_encoder_generalization/"
    "motivation_analysis/padchest_193_label_audit"
)
POOL_CSV = AUDIT_DIR / "padchest_evaluable_label_pool.csv"
FINE31_CSV = AUDIT_DIR / "padchest_finegrained_31_label_set.csv"


ABBREV = {
    "chronic changes": "Chr. changes",
    "COPD signs": "COPD",
    "apical pleural thickening": "Apical pl. thick.",
    "aortic elongation": "Ao. elong.",
    "unchanged": "Unchanged",
    "laminar atelectasis": "Lam. atel.",
    "interstitial pattern": "Interstitial",
    "alveolar pattern": "Alveolar",
    "costophrenic angle blunting": "CP angle blunt.",
    "vertebral degenerative changes": "Vert. degen.",
    "infiltrates": "Infiltrates",
    "aortic atheromatosis": "Ao. athero.",
    "scoliosis": "Scoliosis",
    "fibrotic band": "Fibrotic band",
    "callus rib fracture": "Callus rib fx",
    "nodule": "Nodule",
    "kyphosis": "Kyphosis",
    "sternotomy": "Sternotomy",
    "air trapping": "Air trapping",
    "vascular hilar enlargement": "Vasc. hilar enl.",
    "increased density": "Incr. density",
    "volume loss": "Volume loss",
    "heart insufficiency": "Heart insuff.",
    "hilar congestion": "Hilar congest.",
    "bronchiectasis": "Bronchiect.",
    "vertebral anterior compression": "Vert. ant. compr.",
    "pseudonodule": "Pseudonodule",
    "hemidiaphragm elevation": "Hemidiaph. elev.",
    "suboptimal study": "Suboptimal",
    "pulmonary mass": "Pulm. mass",
    "bronchovascular markings": "Bronchovasc.",
}


def main():
    source_csv = FINE31_CSV if FINE31_CSV.exists() else POOL_CSV
    df = pd.read_csv(source_csv)
    if source_csv == FINE31_CSV:
        fine = df.copy()
    else:
        needed = [
            "label",
            "matched_positive_in_current_split",
            "prevalence_in_current_matched_split",
            "prevalence_group",
            "exact_chexpert14_duplicate",
            "is_no_finding",
            "is_device_related",
            "selection_mapping_type",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {POOL_CSV}: {missing}")
        for c in ["exact_chexpert14_duplicate", "is_no_finding", "is_device_related"]:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
        fine = df[
            ~df["exact_chexpert14_duplicate"]
            & ~df["is_no_finding"]
            & ~df["is_device_related"]
        ].copy()
    fine["matched_positive_in_current_split"] = pd.to_numeric(
        fine["matched_positive_in_current_split"], errors="coerce"
    ).fillna(0).astype(int)
    fine["prevalence_in_current_matched_split"] = pd.to_numeric(
        fine["prevalence_in_current_matched_split"], errors="coerce"
    )
    if "finegrained_prevalence_group" in fine.columns:
        fine["plot_group"] = fine["finegrained_prevalence_group"].fillna("Unknown")
    else:
        fine["plot_group"] = fine["prevalence_group"].fillna("unknown")
    fine["plot_group"] = fine["plot_group"].astype(str)
    fine = fine.sort_values(
        ["matched_positive_in_current_split", "label"], ascending=[False, True]
    ).reset_index(drop=True)

    plot_df = fine[
        [
            "label",
            "matched_positive_in_current_split",
            "prevalence_in_current_matched_split",
            "plot_group",
            "selection_mapping_type",
        ]
    ].rename(
        columns={
            "matched_positive_in_current_split": "positive_count",
            "prevalence_in_current_matched_split": "prevalence",
            "plot_group": "prevalence_group",
        }
    )
    plot_df["display_label"] = plot_df["label"].map(lambda x: ABBREV.get(x, x))
    plot_df.to_csv(AUDIT_DIR / "padchest_31_finegrained_label_distribution_data.csv", index=False)

    print(f"Input: {source_csv}")
    print(f"Fine-grained labels after exclusions: {len(plot_df)}")
    print(plot_df["prevalence_group"].value_counts().to_string())
    print(plot_df["selection_mapping_type"].value_counts().to_string())

    # Restrained palette: muted clinical blue, warm sand, and soft coral.
    palette = {
        "Frequent": "#5B8FB9",
        "Intermediate": "#E8C16A",
        "Rare": "#D96B5F",
        "Unknown": "#B8B8B8",
    }
    group_labels = {
        "Frequent": "Frequent (>5%)",
        "Intermediate": "Intermediate (1%-5%)",
        "Rare": "Rare (<1%)",
        "Unknown": "Unknown",
    }
    colors = [palette.get(g, palette["Unknown"]) for g in plot_df["prevalence_group"]]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.titlesize": 9.3,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.45, 3.45))
    x = range(len(plot_df))
    bars = ax.bar(
        x,
        plot_df["positive_count"],
        width=0.90,
        color=colors,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_yscale("log")
    ax.set_ylim(10, plot_df["positive_count"].max() * 2.55)
    ax.set_ylabel("Counts (log)")
    ax.set_title(
        "Fine-grained PadChest physician-label193 findings after deterministic exclusions",
        pad=9,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["display_label"], rotation=78, ha="right", rotation_mode="anchor")
    ax.margins(x=0.003)
    ax.grid(axis="y", which="major", color="#E7EAED", linewidth=0.6)
    ax.grid(axis="y", which="minor", color="#F2F4F5", linewidth=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9E9E9E")
    ax.spines["bottom"].set_color("#9E9E9E")
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.tick_params(axis="y", width=0.65, colors="#303030")

    for bar, count in zip(bars, plot_df["positive_count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count * 1.08,
            f"{int(count)}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=5.2,
            color="#2F2F2F",
        )

    groups = [g for g in ["Frequent", "Intermediate", "Rare"] if g in set(plot_df["prevalence_group"])]
    handles = [Patch(facecolor=palette[g], edgecolor="none", label=group_labels[g]) for g in groups]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.05,
        handletextpad=0.45,
    )
    leg.get_frame().set_linewidth(0.45)
    leg.get_frame().set_edgecolor("#D7D7D7")
    leg.get_frame().set_facecolor("white")
    ax.text(-0.06, 1.04, "a", transform=ax.transAxes, fontsize=12.5, fontweight="bold")

    fig.tight_layout(pad=0.32)
    png = AUDIT_DIR / "padchest_31_finegrained_label_distribution.png"
    pdf = AUDIT_DIR / "padchest_31_finegrained_label_distribution.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print("Outputs:")
    print(png)
    print(pdf)
    print(AUDIT_DIR / "padchest_31_finegrained_label_distribution_data.csv")


if __name__ == "__main__":
    main()
