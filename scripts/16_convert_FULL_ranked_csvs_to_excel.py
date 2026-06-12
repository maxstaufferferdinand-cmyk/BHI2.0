from pathlib import Path
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

OUT_DIR = Path("data_processed_crossdomain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    {
        "input": OUT_DIR / "llm_ranked_crossdomain_bridges_FULL.csv",
        "xlsx": OUT_DIR / "llm_ranked_crossdomain_bridges_FULL_excel.xlsx",
        "semicolon_csv": OUT_DIR / "llm_ranked_crossdomain_bridges_FULL_excel_friendly_semicolon.csv",
        "sheet": "full_ranked_bridges",
    },
    {
        "input": OUT_DIR / "llm_ranked_crossdomain_bridges_FULL_top.csv",
        "xlsx": OUT_DIR / "llm_ranked_crossdomain_bridges_FULL_top_excel.xlsx",
        "semicolon_csv": OUT_DIR / "llm_ranked_crossdomain_bridges_FULL_top_excel_friendly_semicolon.csv",
        "sheet": "top_ranked_bridges",
    },
]


# ============================================================
# HELPERS
# ============================================================

def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=";", dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(path, sep=";", dtype=str).fillna("")

    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
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

    if "final_score" in df.columns:
        sort_cols = [
            c for c in [
                "final_score",
                "mechanistic_bridge_score",
                "creative_distance_score",
                "surgical_relevance_score",
            ]
            if c in df.columns
        ]

        df = df.sort_values(
            sort_cols,
            ascending=[False] * len(sort_cols),
            na_position="last",
        )

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

    return df[ordered_cols + remaining_cols].copy()


def export_excel(df: pd.DataFrame, output_xlsx: Path, sheet_name: str):
    with pd.ExcelWriter(output_xlsx, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_fmt = workbook.add_format({
            "bold": True,
            "text_wrap": True,
            "valign": "top",
            "border": 1,
        })

        wrap_fmt = workbook.add_format({
            "text_wrap": True,
            "valign": "top",
        })

        num_fmt = workbook.add_format({
            "num_format": "0.00",
            "valign": "top",
        })

        int_fmt = workbook.add_format({
            "num_format": "0",
            "valign": "top",
        })

        for col_num, col_name in enumerate(df.columns):
            worksheet.write(0, col_num, col_name, header_fmt)

        worksheet.freeze_panes(1, 0)
        worksheet.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))

        for idx, col in enumerate(df.columns):
            if col in [
                "bridge_principle",
                "hypothesis",
                "experimental_hint",
                "short_reason",
                "surgery_example_titles",
                "arxiv_example_titles",
            ]:
                worksheet.set_column(idx, idx, 60, wrap_fmt)

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

            elif col in [
                "surgery_occurrence",
                "surgery_n_pmids",
                "arxiv_occurrence",
                "arxiv_n_ids",
                "surgery_first_year",
                "surgery_last_year",
                "arxiv_first_year",
                "arxiv_last_year",
            ]:
                worksheet.set_column(idx, idx, 14, int_fmt)

            elif col in ["pair_id", "decision", "distance_band"]:
                worksheet.set_column(idx, idx, 14, wrap_fmt)

            else:
                worksheet.set_column(idx, idx, 24, wrap_fmt)

        worksheet.set_default_row(45)


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("CONVERT FULL LLM RANKED CSV FILES TO EXCEL")
print("=" * 80)

for spec in INPUT_FILES:
    input_file = spec["input"]
    output_xlsx = spec["xlsx"]
    output_semicolon = spec["semicolon_csv"]
    sheet_name = spec["sheet"]

    print("\n" + "-" * 80)
    print(f"Loading: {input_file}")

    df = read_csv_flexible(input_file)
    print(f"Rows loaded: {len(df)}")
    print("Columns:", df.columns.tolist())

    df = prepare_dataframe(df)

    df.to_csv(
        output_semicolon,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    print(f"Saved semicolon CSV: {output_semicolon}")

    export_excel(df, output_xlsx, sheet_name)
    print(f"Saved Excel file: {output_xlsx}")

print("\nDone.")