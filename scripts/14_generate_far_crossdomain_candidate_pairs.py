from pathlib import Path
import csv
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# INPUT FILES
# ============================================================

SURGERY_CLEAN_FILE = Path("data_processed_crossdomain/surgery_concepts_full_clean.csv")
SURGERY_EMBEDDING_FILE = Path("data_processed_crossdomain/surgery_embeddings_full_clean.csv")

ARXIV_CLEAN_FILE = Path("data_processed_crossdomain/arxiv_concepts_clean.csv")
ARXIV_EMBEDDING_FILE = Path("data_processed_crossdomain/arxiv_embeddings_clean.csv")


# ============================================================
# OUTPUT FILES
# ============================================================

OUT_DIR = Path("data_processed_crossdomain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "surgery_arxiv_far_candidate_pairs.csv"
SUMMARY_FILE = OUT_DIR / "surgery_arxiv_far_candidate_pairs_summary.txt"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TARGET_PAIRS = 50000

# bewusst eher fern
COSINE_MIN = 0.10
COSINE_MAX = 0.30

MAX_PAIRS_PER_SURGERY_CONCEPT = 5

# Surgery wird blockweise geladen und berechnet
SURGERY_BLOCK_SIZE = 50

MIN_SURGERY_OCCURRENCE = 1
MIN_ARXIV_OCCURRENCE = 1

MAX_TOKEN_OVERLAP_FRACTION = 0.25

CHECKPOINT_EVERY_PAIRS = 5000


# ============================================================
# HELPERS
# ============================================================

def normalize_text(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"[^a-z0-9α-ωβγδκλμσπφχψω\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def parse_embedding(x):
    return np.array([float(v) for v in str(x).split(",")], dtype=np.float32)


def l2_normalize_vec(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def l2_normalize_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def token_overlap_fraction(a, b):
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 1.0

    overlap = len(a_tokens & b_tokens)
    denom = min(len(a_tokens), len(b_tokens))

    if denom == 0:
        return 1.0

    return overlap / denom


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

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
        .str.lower()
    )

    return df


def standardize_metadata(df, domain):
    df = df.copy()

    if "concept" not in df.columns:
        raise ValueError(f"{domain} metadata missing concept column. Columns: {df.columns.tolist()}")

    df["concept"] = df["concept"].apply(normalize_text)

    if "occurrence" in df.columns:
        df["occurrence"] = pd.to_numeric(df["occurrence"], errors="coerce").fillna(0).astype(int)
    else:
        df["occurrence"] = 0

    for col in ["first_year", "last_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df


def stream_embeddings(path: Path, keep_set: set):
    """
    Memory-safe reader for files with columns:
    concept,embedding
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        fieldnames = [x.replace("\ufeff", "").strip().lower() for x in reader.fieldnames]

        concept_col = None
        embedding_col = None

        for original, lower in zip(reader.fieldnames, fieldnames):
            if lower == "concept":
                concept_col = original
            if lower == "embedding":
                embedding_col = original

        if concept_col is None or embedding_col is None:
            raise ValueError(
                f"{path} must have concept and embedding columns. Found: {reader.fieldnames}"
            )

        for row in reader:
            concept = normalize_text(row.get(concept_col, ""))

            if concept not in keep_set:
                continue

            emb_str = row.get(embedding_col, "")
            if not emb_str:
                continue

            try:
                emb = parse_embedding(emb_str)
            except Exception:
                continue

            yield concept, emb


def save_candidates(rows):
    pd.DataFrame(rows).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# LOAD METADATA ONLY WITH PANDAS
# ============================================================

print("=" * 80)
print("GENERATE FAR CROSS-DOMAIN SURGERY × ARXIV CANDIDATE PAIRS - MEMORY SAFE")
print("=" * 80)

print("Loading surgery clean metadata...")
surg_meta = standardize_metadata(read_csv_flexible(SURGERY_CLEAN_FILE), "surgery")

print("Loading arXiv clean metadata...")
arxiv_meta = standardize_metadata(read_csv_flexible(ARXIV_CLEAN_FILE), "arxiv")

surg_meta = surg_meta[surg_meta["occurrence"] >= MIN_SURGERY_OCCURRENCE].copy()
arxiv_meta = arxiv_meta[arxiv_meta["occurrence"] >= MIN_ARXIV_OCCURRENCE].copy()

print("\nMetadata sizes:")
print(f"Surgery clean concepts: {len(surg_meta)}")
print(f"arXiv clean concepts: {len(arxiv_meta)}")

surg_meta["concept"] = surg_meta["concept"].apply(normalize_text)
arxiv_meta["concept"] = arxiv_meta["concept"].apply(normalize_text)

surg_keep = set(surg_meta["concept"])
arxiv_keep = set(arxiv_meta["concept"])

surg_meta_dict = surg_meta.drop_duplicates("concept").set_index("concept").to_dict("index")
arxiv_meta_dict = arxiv_meta.drop_duplicates("concept").set_index("concept").to_dict("index")


# ============================================================
# LOAD ARXIV EMBEDDINGS INTO MATRIX
# ============================================================

print("\nLoading arXiv embeddings into matrix...")

arxiv_concepts = []
arxiv_vectors = []

for concept, emb in tqdm(
    stream_embeddings(ARXIV_EMBEDDING_FILE, arxiv_keep),
    desc="Streaming arXiv embeddings"
):
    arxiv_concepts.append(concept)
    arxiv_vectors.append(emb)

if not arxiv_vectors:
    raise RuntimeError("No arXiv embeddings loaded.")

arxiv_matrix = np.vstack(arxiv_vectors).astype(np.float32)
arxiv_matrix = l2_normalize_matrix(arxiv_matrix)

print(f"arXiv embeddings loaded: {len(arxiv_concepts)}")
print(f"arXiv matrix shape: {arxiv_matrix.shape}")


# ============================================================
# STREAM SURGERY EMBEDDINGS BLOCKWISE
# ============================================================

candidate_rows = []
seen_pairs = set()

surgery_stream = stream_embeddings(SURGERY_EMBEDDING_FILE, surg_keep)

block_concepts = []
block_vectors = []

processed_surgery = 0

print("\nGenerating candidate pairs...")
print(f"Cosine window: {COSINE_MIN}–{COSINE_MAX}")
print(f"Target pairs: {TARGET_PAIRS}")

for concept, emb in tqdm(surgery_stream, desc="Streaming surgery embeddings"):
    block_concepts.append(concept)
    block_vectors.append(emb)
    processed_surgery += 1

    if len(block_concepts) < SURGERY_BLOCK_SIZE:
        continue

    if len(candidate_rows) >= TARGET_PAIRS:
        break

    block_matrix = np.vstack(block_vectors).astype(np.float32)
    block_matrix = l2_normalize_matrix(block_matrix)

    sims = block_matrix @ arxiv_matrix.T

    for local_i, sconcept in enumerate(block_concepts):
        if len(candidate_rows) >= TARGET_PAIRS:
            break

        srec = surg_meta_dict.get(sconcept, {})
        row_sims = sims[local_i, :]

        valid_idx = np.where(
            (row_sims >= COSINE_MIN) &
            (row_sims <= COSINE_MAX)
        )[0]

        if len(valid_idx) == 0:
            continue

        valid_idx = list(valid_idx)
        random.shuffle(valid_idx)

        selected_count = 0

        for ai in valid_idx:
            if selected_count >= MAX_PAIRS_PER_SURGERY_CONCEPT:
                break

            aconcept = arxiv_concepts[int(ai)]

            if token_overlap_fraction(sconcept, aconcept) > MAX_TOKEN_OVERLAP_FRACTION:
                continue

            pair_key = (sconcept, aconcept)
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            selected_count += 1

            arec = arxiv_meta_dict.get(aconcept, {})
            sim = float(row_sims[ai])

            candidate_rows.append({
                "surgery_concept": sconcept,
                "arxiv_concept": aconcept,
                "cosine_similarity": sim,
                "distance_band": "far_defined",
                "surgery_occurrence": srec.get("occurrence", ""),
                "surgery_n_pmids": srec.get("n_pmids", ""),
                "surgery_first_year": srec.get("first_year", ""),
                "surgery_last_year": srec.get("last_year", ""),
                "surgery_query_names": srec.get("query_names", ""),
                "surgery_journals": srec.get("journals", ""),
                "surgery_example_titles": srec.get("example_titles", ""),
                "arxiv_occurrence": arec.get("occurrence", ""),
                "arxiv_n_ids": arec.get("n_arxiv_ids", ""),
                "arxiv_first_year": arec.get("first_year", ""),
                "arxiv_last_year": arec.get("last_year", ""),
                "arxiv_concept_types": arec.get("concept_types", ""),
                "arxiv_modules": arec.get("modules", ""),
                "arxiv_example_titles": arec.get("example_titles", ""),
                "token_overlap_fraction": token_overlap_fraction(sconcept, aconcept),
            })

    if len(candidate_rows) > 0 and len(candidate_rows) % CHECKPOINT_EVERY_PAIRS < SURGERY_BLOCK_SIZE:
        save_candidates(candidate_rows)
        print(f"\nCheckpoint saved: {len(candidate_rows)} pairs", flush=True)

    block_concepts = []
    block_vectors = []


# ============================================================
# PROCESS FINAL PARTIAL BLOCK
# ============================================================

if block_concepts and len(candidate_rows) < TARGET_PAIRS:
    block_matrix = np.vstack(block_vectors).astype(np.float32)
    block_matrix = l2_normalize_matrix(block_matrix)

    sims = block_matrix @ arxiv_matrix.T

    for local_i, sconcept in enumerate(block_concepts):
        if len(candidate_rows) >= TARGET_PAIRS:
            break

        srec = surg_meta_dict.get(sconcept, {})
        row_sims = sims[local_i, :]

        valid_idx = np.where(
            (row_sims >= COSINE_MIN) &
            (row_sims <= COSINE_MAX)
        )[0]

        if len(valid_idx) == 0:
            continue

        valid_idx = list(valid_idx)
        random.shuffle(valid_idx)

        selected_count = 0

        for ai in valid_idx:
            if selected_count >= MAX_PAIRS_PER_SURGERY_CONCEPT:
                break

            aconcept = arxiv_concepts[int(ai)]

            if token_overlap_fraction(sconcept, aconcept) > MAX_TOKEN_OVERLAP_FRACTION:
                continue

            pair_key = (sconcept, aconcept)
            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)
            selected_count += 1

            arec = arxiv_meta_dict.get(aconcept, {})
            sim = float(row_sims[ai])

            candidate_rows.append({
                "surgery_concept": sconcept,
                "arxiv_concept": aconcept,
                "cosine_similarity": sim,
                "distance_band": "far_defined",
                "surgery_occurrence": srec.get("occurrence", ""),
                "surgery_n_pmids": srec.get("n_pmids", ""),
                "surgery_first_year": srec.get("first_year", ""),
                "surgery_last_year": srec.get("last_year", ""),
                "surgery_query_names": srec.get("query_names", ""),
                "surgery_journals": srec.get("journals", ""),
                "surgery_example_titles": srec.get("example_titles", ""),
                "arxiv_occurrence": arec.get("occurrence", ""),
                "arxiv_n_ids": arec.get("n_arxiv_ids", ""),
                "arxiv_first_year": arec.get("first_year", ""),
                "arxiv_last_year": arec.get("last_year", ""),
                "arxiv_concept_types": arec.get("concept_types", ""),
                "arxiv_modules": arec.get("modules", ""),
                "arxiv_example_titles": arec.get("example_titles", ""),
                "token_overlap_fraction": token_overlap_fraction(sconcept, aconcept),
            })


# ============================================================
# SAVE FINAL
# ============================================================

cand = pd.DataFrame(candidate_rows)

if cand.empty:
    raise RuntimeError(
        "No candidate pairs generated. Try widening cosine window, e.g. 0.05–0.35."
    )

cand = cand.drop_duplicates(subset=["surgery_concept", "arxiv_concept"])
cand = cand.sort_values(
    ["cosine_similarity", "surgery_occurrence", "arxiv_occurrence"],
    ascending=[True, False, False]
)

if len(cand) > TARGET_PAIRS:
    cand = cand.head(TARGET_PAIRS).copy()

cand.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

summary_lines = []
summary_lines.append("SURGERY × ARXIV FAR CANDIDATE PAIRS SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append(f"Surgery metadata concepts: {len(surg_meta)}")
summary_lines.append(f"arXiv metadata concepts: {len(arxiv_meta)}")
summary_lines.append(f"arXiv embeddings loaded: {len(arxiv_concepts)}")
summary_lines.append(f"Surgery embeddings processed: {processed_surgery}")
summary_lines.append(f"Target pairs: {TARGET_PAIRS}")
summary_lines.append(f"Generated pairs: {len(cand)}")
summary_lines.append(f"Cosine min: {COSINE_MIN}")
summary_lines.append(f"Cosine max: {COSINE_MAX}")
summary_lines.append(f"Max pairs per surgery concept: {MAX_PAIRS_PER_SURGERY_CONCEPT}")
summary_lines.append("")
summary_lines.append("Cosine distribution:")
summary_lines.append(str(cand["cosine_similarity"].describe()))
summary_lines.append("")
summary_lines.append("Top arXiv concept types:")
summary_lines.append(str(cand["arxiv_concept_types"].value_counts().head(20)))
summary_lines.append("")
summary_lines.append("Example candidate pairs:")
summary_lines.append(str(cand[[
    "surgery_concept",
    "arxiv_concept",
    "cosine_similarity",
    "arxiv_concept_types",
    "arxiv_modules",
]].head(50)))

SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved candidate pairs: {OUTPUT_FILE}")
print(f"Saved summary: {SUMMARY_FILE}")
print(f"Generated pairs: {len(cand)}")
print("\nCosine distribution:")
print(cand["cosine_similarity"].describe())
print("\nExample pairs:")
print(cand[[
    "surgery_concept",
    "arxiv_concept",
    "cosine_similarity",
    "arxiv_concept_types",
    "arxiv_modules",
]].head(20))