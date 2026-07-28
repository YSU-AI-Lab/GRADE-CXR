import os
from pathlib import Path

import numpy as np
import pandas as pd


LABEL193_CSV = Path("/home/hyr/clip/UniChest-main/A1_DATA/Physician_label193_all.csv")
SPLIT_ROOT = Path(
    "/home/hyr/clip/MedCLIP-SAMv2-main/data/loso_splits_two_text/"
    "chexpert14_csv_by_holdout/holdout_sid2_PadChest_full_from_biomedclip_jsonl"
)
OUT_DIR = Path(
    "/home/hyr/clip/biomedclip_encoder_generalization/"
    "motivation_analysis/padchest_193_label_audit"
)

META_COLS = ["img_path", "Labels", "labelCUIS"]
CHEXPERT14 = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("_", " ").replace("-", " ")


def map_to_chexpert14(label: str):
    n = normalize_name(label)
    exact = {
        "normal": "No Finding",
        "cardiomegaly": "Cardiomegaly",
        "pneumonia": "Pneumonia",
        "atypical pneumonia": "Pneumonia",
        "consolidation": "Consolidation",
        "atelectasis": "Atelectasis",
        "pneumothorax": "Pneumothorax",
        "pleural effusion": "Pleural Effusion",
        "fracture": "Fracture",
    }
    if n in exact:
        return exact[n], "exact_or_direct_synonym"

    rules = [
        (
            "Support Devices",
            [
                "tube",
                "catheter",
                "pacemaker",
                "device",
                "prosthesis",
                "valve",
                "drain",
                "stent",
                "endoprosthesis",
                "reservoir",
                "metal",
                "osteosynthesis",
                "suture material",
                "bone cement",
                "nephrostomy",
                "gastrostomy",
            ],
        ),
        (
            "Fracture",
            ["fracture", "compression"],
        ),
        (
            "Pleural Effusion",
            [
                "pleural effusion",
                "loculated fissural effusion",
                "hydropneumothorax",
                "costophrenic angle blunting",
            ],
        ),
        (
            "Pleural Other",
            [
                "pleural thickening",
                "pleural plaques",
                "pleural mass",
                "asbestosis",
            ],
        ),
        (
            "Atelectasis",
            [
                "atelectasis",
                "hypoexpansion",
                "volume loss",
                "flattened diaphragm",
            ],
        ),
        (
            "Edema",
            [
                "pulmonary edema",
                "heart insufficiency",
                "venous hypertension",
                "kerley",
                "vascular redistribution",
                "hilar congestion",
            ],
        ),
        (
            "Lung Lesion",
            [
                "mass",
                "nodule",
                "granuloma",
                "cavitation",
                "abscess",
                "metastasis",
                "cyst",
                "adenocarcinoma",
                "pseudonodule",
            ],
        ),
        (
            "Lung Opacity",
            [
                "infiltrates",
                "increased density",
                "ground glass",
                "interstitial pattern",
                "alveolar pattern",
                "miliary opacities",
                "reticular",
                "reticulonodular",
                "air bronchogram",
                "bronchovascular markings",
            ],
        ),
        (
            "Enlarged Cardiomediastinum",
            [
                "mediastinal enlargement",
                "superior mediastinal enlargement",
                "aortic button enlargement",
            ],
        ),
        (
            "Cardiomegaly",
            ["cardiomegaly"],
        ),
        (
            "Consolidation",
            ["consolidation"],
        ),
        (
            "Pneumonia",
            ["pneumonia"],
        ),
        (
            "Pneumothorax",
            ["pneumothorax"],
        ),
    ]
    for target, keywords in rules:
        if any(k in n for k in keywords):
            return target, "keyword_rule"
    return "", "padchest_specific_or_not_mapped"


def prevalence_group(prevalence: float) -> str:
    if prevalence > 0.10:
        return "high"
    if prevalence >= 0.01:
        return "medium"
    return "low"


def read_split(split: str) -> pd.DataFrame:
    path = SPLIT_ROOT / f"{split}.csv"
    df = pd.read_csv(path)
    df = df.copy()
    df["basename"] = df["Image Index"].map(lambda x: os.path.basename(str(x)))
    df["split"] = split
    return df[["basename", "split", "Image Index"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_df = pd.read_csv(LABEL193_CSV)
    label_cols = [c for c in label_df.columns if c not in META_COLS]
    if len(label_cols) != 193:
        raise RuntimeError(f"Expected 193 label columns, found {len(label_cols)}")

    label_df = label_df.copy()
    label_df["basename"] = label_df["img_path"].map(lambda x: os.path.basename(str(x)))
    duplicated = int(label_df["basename"].duplicated().sum())

    for c in label_cols:
        label_df[c] = pd.to_numeric(label_df[c], errors="coerce").fillna(0).clip(0, 1)

    splits = pd.concat([read_split(s) for s in ["train", "val", "test"]], ignore_index=True)
    split_sizes = splits.groupby("split").size().to_dict()
    merged = splits.merge(label_df[["basename"] + label_cols], on="basename", how="left", indicator=True)
    merged["_matched193"] = merged["_merge"].eq("both")

    matched = merged[merged["_matched193"]].copy()
    matched_sizes = matched.groupby("split").size().reindex(["train", "val", "test"], fill_value=0).to_dict()

    records = []
    n_all = len(label_df)
    n_matched = len(matched)
    for label in label_cols:
        total_pos = int(label_df[label].sum())
        matched_pos = int(matched[label].sum())
        split_counts = (
            matched.groupby("split")[label]
            .sum()
            .reindex(["train", "val", "test"], fill_value=0)
            .astype(int)
        )
        split_prevalence = {
            f"{split}_prevalence_matched": (
                split_counts[split] / matched_sizes[split] if matched_sizes[split] else np.nan
            )
            for split in ["train", "val", "test"]
        }
        prevalence_all = total_pos / n_all if n_all else np.nan
        prevalence_matched = matched_pos / n_matched if n_matched else np.nan
        mapped, mapping_rule = map_to_chexpert14(label)
        records.append(
            {
                "label": label,
                "total_positive_in_physician193": total_pos,
                "prevalence_in_physician193": prevalence_all,
                "matched_positive_in_current_split": matched_pos,
                "prevalence_in_current_matched_split": prevalence_matched,
                "train_positive": int(split_counts["train"]),
                "val_positive": int(split_counts["val"]),
                "test_positive": int(split_counts["test"]),
                **split_prevalence,
                "prevalence_group": prevalence_group(prevalence_matched),
                "passes_min_count_filter": bool(
                    int(split_counts["train"]) >= 50 and int(split_counts["test"]) >= 20
                ),
                "chexpert14_mapping": mapped,
                "mapping_rule": mapping_rule,
                "is_padchest_specific": mapped == "",
            }
        )

    counts = pd.DataFrame(records).sort_values(
        ["passes_min_count_filter", "prevalence_in_current_matched_split", "label"],
        ascending=[False, False, True],
    )
    filtered = counts[counts["passes_min_count_filter"]].copy()
    groups = counts[["label", "prevalence_in_current_matched_split", "prevalence_group"]].copy()
    mapping = counts[
        ["label", "chexpert14_mapping", "mapping_rule", "is_padchest_specific"]
    ].copy()

    counts.to_csv(OUT_DIR / "padchest_193_label_counts.csv", index=False)
    filtered.to_csv(OUT_DIR / "padchest_filtered_labels.csv", index=False)
    groups.to_csv(OUT_DIR / "padchest_prevalence_groups.csv", index=False)
    mapping.to_csv(OUT_DIR / "padchest_mapping_to_chexpert14.csv", index=False)

    group_counts_all = counts["prevalence_group"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    group_counts_filtered = (
        filtered["prevalence_group"].value_counts().reindex(["high", "medium", "low"], fill_value=0)
    )
    mapped_counts = counts["chexpert14_mapping"].replace("", "PadChest-specific").value_counts()
    filtered_mapped_counts = filtered["chexpert14_mapping"].replace("", "PadChest-specific").value_counts()

    audit_md = f"""# PadChest 193-label Audit

## Inputs

- Physician 193-label file: `{LABEL193_CSV}`
- Held-out PadChest split root: `{SPLIT_ROOT}`

## Basic Checks

- Label table rows: {n_all}
- Label columns detected: {len(label_cols)}
- Duplicate image basenames in label table: {duplicated}
- Current PadChest split rows: train={split_sizes.get('train', 0)}, val={split_sizes.get('val', 0)}, test={split_sizes.get('test', 0)}
- Matched rows with 193-label table: train={matched_sizes.get('train', 0)}, val={matched_sizes.get('val', 0)}, test={matched_sizes.get('test', 0)}, total={n_matched}
- Match coverage: train={matched_sizes.get('train', 0) / split_sizes.get('train', 1):.2%}, val={matched_sizes.get('val', 0) / split_sizes.get('val', 1):.2%}, test={matched_sizes.get('test', 0) / split_sizes.get('test', 1):.2%}

## Prevalence Groups

Groups are assigned using prevalence within the current matched held-out PadChest split subset.

| Group | Rule | All 193 labels | Filtered usable labels |
|---|---:|---:|---:|
| High | >10% | {int(group_counts_all['high'])} | {int(group_counts_filtered['high'])} |
| Medium | 1%-10% | {int(group_counts_all['medium'])} | {int(group_counts_filtered['medium'])} |
| Low | <1% | {int(group_counts_all['low'])} | {int(group_counts_filtered['low'])} |

## Filtering Rule

A disease is marked usable for follow-up long-tail transfer analysis if:

- train positive count >= 50
- test positive count >= 20

Usable labels: {len(filtered)} / {len(counts)}

## CheXpert-style Mapping

Mapping is conservative and keyword-based. Labels without a clear CheXpert-style 14-class equivalent are marked as PadChest-specific.

- All mapped/non-specific counts:

{mapped_counts.to_string()}

- Filtered usable mapped/non-specific counts:

{filtered_mapped_counts.to_string()}

## Interpretation

The 193-label physician file is valid and contains exactly 193 binary disease/finding labels. However, it covers only a subset of the current held-out PadChest split. Therefore, long-tail disease migration analysis is feasible only on the matched subset, not the full PadChest split currently used for 14-class experiments.

The filtered label list provides the safest candidate set for later experiments. Low-prevalence labels that fail the train/test count threshold should not be used for stable AUC/AP reporting unless they are grouped or evaluated with caution.
"""
    (OUT_DIR / "padchest_label_audit.md").write_text(audit_md)

    print(f"Output directory: {OUT_DIR}")
    print(f"Detected label columns: {len(label_cols)}")
    print(f"Matched rows: {n_matched} / split total {len(splits)}")
    print("All prevalence groups:", group_counts_all.to_dict())
    print("Filtered prevalence groups:", group_counts_filtered.to_dict())
    print(f"Filtered usable labels: {len(filtered)}")
    print("Wrote:")
    for name in [
        "padchest_label_audit.md",
        "padchest_193_label_counts.csv",
        "padchest_filtered_labels.csv",
        "padchest_prevalence_groups.csv",
        "padchest_mapping_to_chexpert14.csv",
    ]:
        print(f"- {OUT_DIR / name}")


if __name__ == "__main__":
    main()
