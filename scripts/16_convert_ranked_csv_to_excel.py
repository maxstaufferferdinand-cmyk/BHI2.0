from pathlib import Path
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = Path("data_processed_crossdomain/llm_ranked_crossdomain_bridges.csv")

OUT_DIR = Path("data_processed_crossdomain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = OUT_DIR / "llm_ranked_crossdomain_bridges_excel.xlsx"
OUTPUT_CSV_SEMICOLON = OUT_DIR / "llm_ranked_crossdomain_bridges_excel_friendly_semicolon.csv"

OUTPUT_TOP_XLSX = OUT_DIR / "llm_ranked_crossdomain_bridges_top_excel.xlsx"


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("CONVERT LLM RANKED CSV TO EXCEL-FRIENDLY FILES")
print("=" * 80)

print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print(f"Rows loaded: {len(df)}")
print("Columns:")
print(df.columns.tolist())


# ============================================================
# NUMERIC COLUMNS
# ============================================================

numeric_cols = [
    "cosine_similarity",
    "surgery_occurrence",
    "arxiv_occurrence",
    "mechanistic_bridge_score",
    "creative_distance_score",
    "surgical_relevance_score",
    "triviality_score",
    "final_score",
    "token_overlap_fraction",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# SORT
# ============================================================

if "final_score" in df.columns:
    df = df.sort_values(
        ["final_score", "mechanistic_bridge_score", "creative_distance_score", "surgical_relevance_score"],
        ascending=[False, False, False, False],
        na_position="last"
    )


# ============================================================
# COLUMN ORDER
# ============================================================

preferred_cols = [
    "pair_id",
    "decision",
    "final_score",
    "mechanistic_bridge_score",
    "creative_distance_score",
    "surgical_relevance_score",
    "triviality_score",
    "surgery_concept",
    "arxiv_concept",
    "bridge_principle",
    "hypothesis",
    "experimental_hint",
    "short_reason",
    "cosine_similarity",
    "distance_band",
    "surgery_occurrence",
    "surgery_n_pmids",
    "surgery_first_year",
    "surgery_last_year",
    "surgery_query_names",
    "surgery_journals",
    "surgery_example_titles",
    "arxiv_occurrence",
    "arxiv_n_ids",
    "arxiv_first_year",
    "arxiv_last_year",
    "arxiv_concept_types",
    "arxiv_modules",
    "arxiv_example_titles",
    "token_overlap_fraction",
]

ordered_cols = [c for c in preferred_cols if c in df.columns]
remaining_cols = [c for c in df.columns if c not in ordered_cols]
df = df[ordered_cols + remaining_cols].copy()


# ============================================================
# EXPORT SEMICOLON CSV
# ============================================================

df.to_csv(
    OUTPUT_CSV_SEMICOLON,
    index=False,
    sep=";",
    encoding="utf-8-sig"
)

print(f"Saved semicolon CSV: {OUTPUT_CSV_SEMICOLON}")


# ============================================================
# EXPORT XLSX
# ============================================================

with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="ranked_bridges", index=False)

    workbook = writer.book
    worksheet = writer.sheets["ranked_bridges"]

    # Formats
    header_fmt = workbook.add_format({
        "bold": True,
        "text_wrap": True,
        "valign": "top",
        "border": 1
    })

    wrap_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top"
    })

    num_fmt = workbook.add_format({
        "num_format": "0.00",
        "valign": "top"
    })

    int_fmt = workbook.add_format({
        "num_format": "0",
        "valign": "top"
    })

    # Header
    for col_num, col_name in enumerate(df.columns):
        worksheet.write(0, col_num, col_name, header_fmt)

    # Freeze header row + autofilter
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

    # Column widths
    for idx, col in enumerate(df.columns):
        if col in [
            "bridge_principle",
            "hypothesis",
            "experimental_hint",
            "short_reason",
            "surgery_example_titles",
            "arxiv_example_titles",
        ]:
            worksheet.set_column(idx, idx, 55, wrap_fmt)
        elif col in ["surgery_concept", "arxiv_concept"]:
            worksheet.set_column(idx, idx, 32, wrap_fmt)
        elif col in [
            "final_score",
            "mechanistic_bridge_score",
            "creative_distance_score",
            "surgical_relevance_score",
            "triviality_score",
            "cosine_similarity",
            "token_overlap_fraction",
        ]:
            worksheet.set_column(idx, idx, 14, num_fmt)
        elif col in ["pair_id", "decision", "distance_band"]:
            worksheet.set_column(idx, idx, 14, wrap_fmt)
        else:
            worksheet.set_column(idx, idx, 22, wrap_fmt)

    # Row height
    worksheet.set_default_row(45)

print(f"Saved Excel file: {OUTPUT_XLSX}")


# ============================================================
# EXPORT TOP XLSX
# ============================================================

top_df = df.head(1000).copy()

with pd.ExcelWriter(OUTPUT_TOP_XLSX, engine="xlsxwriter") as writer:
    top_df.to_excel(writer, sheet_name="top_ranked_bridges", index=False)

    workbook = writer.book
    worksheet = writer.sheets["top_ranked_bridges"]

    header_fmt = workbook.add_format({
        "bold": True,
        "text_wrap": True,
        "valign": "top",
        "border": 1
    })

    wrap_fmt = workbook.add_format({
        "text_wrap": True,
        "valign": "top"
    })

    num_fmt = workbook.add_format({
        "num_format": "0.00",
        "valign": "top"
    })

    for col_num, col_name in enumerate(top_df.columns):
        worksheet.write(0, col_num, col_name, header_fmt)

    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, len(top_df), len(top_df.columns) - 1)

    for idx, col in enumerate(top_df.columns):
        if col in [
            "bridge_principle",
            "hypothesis",
            "experimental_hint",
            "short_reason",
            "surgery_example_titles",
            "arxiv_example_titles",
        ]:
            worksheet.set_column(idx, idx, 55, wrap_fmt)
        elif col in ["surgery_concept", "arxiv_concept"]:
            worksheet.set_column(idx, idx, 32, wrap_fmt)
        elif col in [
            "final_score",
            "mechanistic_bridge_score",
            "creative_distance_score",
            "surgical_relevance_score",
            "triviality_score",
            "cosine_similarity",
            "token_overlap_fraction",
        ]:
            worksheet.set_column(idx, idx, 14, num_fmt)
        else:
            worksheet.set_column(idx, idx, 22, wrap_fmt)

    worksheet.set_default_row(45)

print(f"Saved top Excel file: {OUTPUT_TOP_XLSX}")

print("\nDone.")