from pathlib import Path
import os
import re
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, AuthenticationError


# ============================================================
# INPUT FILES
# ============================================================

SURGERY_CONCEPT_ROWS_FILE = Path("data_processed/concepts_2_per_abstract.csv")
EXISTING_SURGERY_EMBEDDING_FILE = Path("data_processed/concept_embeddings.csv")


# ============================================================
# OUTPUT FILES
# ============================================================

OUT_DIR = Path("data_processed_crossdomain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_SURGERY_NODES_FILE = OUT_DIR / "surgery_concepts_full_nodes.csv"
FULL_SURGERY_CLEAN_FILE = OUT_DIR / "surgery_concepts_full_clean.csv"
FULL_SURGERY_REMOVED_FILE = OUT_DIR / "removed_surgery_full_concepts_review.csv"
FULL_SURGERY_EMBEDDING_FILE = OUT_DIR / "surgery_embeddings_full_clean.csv"
SUMMARY_FILE = OUT_DIR / "surgery_full_vocab_summary.txt"


# ============================================================
# SETTINGS
# ============================================================

MIN_SURGERY_OCCURRENCE = 1
MIN_WORDS = 2
MAX_WORDS = 8

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.2
SAVE_EVERY_BATCHES = 10

# Erstmal Test möglich, aber für jetzt Full sinnvoll.
TEST_MODE = False
TEST_N_CONCEPTS = None


# ============================================================
# API KEY
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. In PowerShell setzen mit:\n"
        '$env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=api_key)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"[^a-z0-9α-ωβγδκλμσπφχψω\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def word_count(x: str) -> int:
    return len(normalize_text(x).split())


def contains_any(x: str, terms) -> bool:
    x = normalize_text(x)
    return any(t in x for t in terms)


def exact_any(x: str, terms) -> bool:
    x = normalize_text(x)
    return x in terms


def ends_with_any(x: str, terms) -> bool:
    x = normalize_text(x)
    words = x.split()
    if not words:
        return True
    return words[-1] in terms


def standardize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .str.lower()
    )
    return df


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=";", dtype=str).fillna("")
        return standardize_columns(df)
    except Exception:
        return standardize_columns(pd.read_csv(path, sep=";", dtype=str).fillna(""))


# ============================================================
# FILTER TERMS
# ============================================================

UNIVERSAL_GENERIC_EXACT = {
    "study",
    "model",
    "models",
    "method",
    "methods",
    "result",
    "results",
    "framework",
    "system",
    "systems",
    "approach",
    "analysis",
    "simulation",
    "simulations",
    "experiment",
    "experiments",
    "performance",
    "algorithm",
    "algorithms",
    "data analysis",
    "statistical analysis",
    "numerical analysis",
    "computational model",
    "mathematical model",
    "experimental results",
    "proposed method",
    "proposed approach",
    "new method",
    "novel method",
    "efficient algorithm",
}

UNIVERSAL_BAD_ENDINGS = {
    "model",
    "models",
    "method",
    "methods",
    "approach",
    "framework",
    "system",
    "systems",
    "analysis",
    "simulation",
    "simulations",
    "performance",
    "algorithm",
    "algorithms",
    "result",
    "results",
    "study",
    "studies",
    "effect",
    "effects",
    "impact",
    "evaluation",
    "assessment",
    "comparison",
}

SURGERY_GENERIC_EXACT = {
    "surgery",
    "surgical procedure",
    "surgical procedures",
    "operative procedure",
    "operative procedures",
    "abdominal surgery",
    "general surgery",
    "visceral surgery",
    "laparoscopic surgery",
    "robotic surgery",
    "minimally invasive surgery",
    "open surgery",
    "treatment",
    "therapy",
    "patient",
    "patients",
    "cohort",
    "clinical outcome",
    "clinical outcomes",
    "surgical outcome",
    "surgical outcomes",
    "postoperative outcome",
    "postoperative outcomes",
    "perioperative outcome",
    "perioperative outcomes",
    "short term outcome",
    "long term outcome",
    "risk factor",
    "risk factors",
    "predictive factor",
    "predictive factors",
    "prognostic factor",
    "prognostic factors",
    "mortality",
    "morbidity",
    "survival",
    "overall survival",
    "disease free survival",
    "recurrence free survival",
    "length of stay",
    "hospital stay",
    "readmission",
    "complication",
    "complications",
    "postoperative complication",
    "postoperative complications",
    "major complication",
    "major complications",
    "minor complication",
    "minor complications",
    "cancer",
    "tumor",
    "neoplasm",
    "oncologic outcome",
    "oncological outcome",
    "quality of life",
    "health related quality of life",
}

SURGERY_GENERIC_SUBSTRINGS = [
    "outcome",
    "outcomes",
    "complication rate",
    "complication rates",
    "mortality rate",
    "morbidity rate",
    "risk factor",
    "risk factors",
    "survival rate",
    "learning curve",
    "clinical impact",
    "treatment effect",
    "treatment outcome",
    "therapy outcome",
    "patient reported",
    "quality of life",
    "cost effectiveness",
    "meta analysis",
    "systematic review",
    "retrospective study",
    "prospective study",
    "randomized trial",
    "cohort study",
]

SURGERY_BAD_ENDINGS = UNIVERSAL_BAD_ENDINGS | {
    "outcome",
    "outcomes",
    "rate",
    "rates",
    "risk",
    "risks",
    "factor",
    "factors",
    "mortality",
    "morbidity",
    "survival",
    "complication",
    "complications",
    "incidence",
    "prevalence",
    "diagnosis",
    "management",
    "treatment",
    "therapy",
}

SURGERY_PROTECT_SUBSTRINGS = [
    "anastomotic leakage",
    "anastomotic leak",
    "pancreatic fistula",
    "postoperative pancreatic fistula",
    "bile leak",
    "biliary leak",
    "surgical site infection",
    "mesh infection",
    "wound dehiscence",
    "intra abdominal adhesion",
    "peritoneal adhesion",
    "ischemia reperfusion",
    "postoperative ileus",
    "drain amylase",
    "biliary stricture",
    "pancreaticojejunostomy",
    "hepaticojejunostomy",
    "low anterior resection syndrome",
    "clavien dindo",
    "comprehensive complication index",
    "failure to rescue",
    "peritoneal metastasis",
    "liver regeneration",
    "organ perfusion",
    "tissue perfusion",
    "intraoperative fluorescence",
    "indocyanine green",
    "microbiome",
    "biofilm",
    "fibrosis",
    "anastomotic perfusion",
    "colorectal anastomosis",
    "pancreatic anastomosis",
    "tumor budding",
    "circulating tumor dna",
    "cell free dna",
    "sarcopenia",
    "frailty",
    "neoadjuvant immunotherapy",
    "conversion surgery",
    "parenchymal sparing hepatectomy",
    "associating liver partition",
    "portal vein embolization",
    "liver hypertrophy",
    "post hepatectomy liver failure",
    "pancreatic remnant",
    "soft pancreatic texture",
    "fistula risk score",
    "drain fluid",
    "bile acid",
    "bacterial translocation",
    "anastomotic ischemia",
    "bowel perfusion",
    "peritoneal lavage",
    "intra abdominal infection",
    "abdominal sepsis",
]

BROAD_DISEASE_ONLY = {
    "colorectal cancer",
    "pancreatic cancer",
    "gastric cancer",
    "esophageal cancer",
    "liver cancer",
    "rectal cancer",
    "colon cancer",
    "hepatocellular carcinoma",
    "cholangiocarcinoma",
    "pancreatic adenocarcinoma",
}


# ============================================================
# FILTER
# ============================================================

def base_reject_reason(concept: str):
    c = normalize_text(concept)
    wc = word_count(c)

    if not c:
        return "empty"

    if wc < MIN_WORDS:
        return "too_few_words"

    if wc > MAX_WORDS:
        return "too_many_words"

    if len(c) < 6:
        return "too_short"

    if re.fullmatch(r"[0-9\s]+", c):
        return "numeric_only"

    if exact_any(c, UNIVERSAL_GENERIC_EXACT):
        return "universal_generic_exact"

    if ends_with_any(c, UNIVERSAL_BAD_ENDINGS):
        return "universal_bad_ending"

    return ""


def surgery_reject_reason(concept: str):
    c = normalize_text(concept)

    if contains_any(c, SURGERY_PROTECT_SUBSTRINGS):
        return ""

    base = base_reject_reason(c)
    if base:
        return base

    if exact_any(c, SURGERY_GENERIC_EXACT):
        return "surgery_generic_exact"

    if contains_any(c, SURGERY_GENERIC_SUBSTRINGS):
        return "surgery_generic_substring"

    if ends_with_any(c, SURGERY_BAD_ENDINGS):
        return "surgery_bad_ending"

    if c in BROAD_DISEASE_ONLY:
        return "broad_disease_only"

    return ""


# ============================================================
# LOAD CONCEPT ROWS AND BUILD FULL NODES
# ============================================================

print("=" * 80)
print("REBUILD FULL SURGERY VOCABULARY FROM CONCEPT ROWS")
print("=" * 80)

print(f"Loading surgery concept rows: {SURGERY_CONCEPT_ROWS_FILE}", flush=True)
df = read_csv_flexible(SURGERY_CONCEPT_ROWS_FILE)

print("Columns:", df.columns.tolist(), flush=True)
print(f"Rows loaded: {len(df)}", flush=True)

required = ["concept"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in surgery concept rows: {missing}")

df["concept"] = df["concept"].apply(normalize_text)

if "year" in df.columns:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 2020) & (df["year"] <= 2026)].copy()
else:
    df["year"] = ""

if "pmid" not in df.columns:
    df["pmid"] = ""

if "title" not in df.columns:
    df["title"] = ""

if "query_name" not in df.columns:
    df["query_name"] = ""

if "journal" not in df.columns:
    df["journal"] = ""

df = df[df["concept"].str.len() > 0].copy()

print(f"Rows after basic cleaning/year filter: {len(df)}", flush=True)
print(f"Unique concepts before generic filter: {df['concept'].nunique()}", flush=True)

node_summary = (
    df.groupby("concept", dropna=False)
    .agg(
        occurrence=("concept", "count"),
        n_pmids=("pmid", lambda x: len(set(map(str, x))) if "pmid" in df.columns else 0),
        first_year=("year", "min"),
        last_year=("year", "max"),
        query_names=("query_name", lambda x: "; ".join(sorted(set(map(str, x)))[:10])),
        journals=("journal", lambda x: "; ".join(sorted(set(map(str, x)))[:10])),
        example_titles=("title", lambda x: " || ".join(list(dict.fromkeys(map(str, x)))[:3])),
    )
    .reset_index()
    .sort_values(["occurrence", "concept"], ascending=[False, True])
)

node_summary.to_csv(FULL_SURGERY_NODES_FILE, index=False, encoding="utf-8-sig")
print(f"Saved full surgery nodes: {FULL_SURGERY_NODES_FILE}", flush=True)
print(f"Full surgery unique concepts: {len(node_summary)}", flush=True)


# ============================================================
# CLEAN FULL NODES
# ============================================================

node_summary["reject_reason"] = node_summary["concept"].apply(surgery_reject_reason)
node_summary["keep"] = node_summary["reject_reason"].eq("")
node_summary["keep"] = node_summary["keep"] & (node_summary["occurrence"] >= MIN_SURGERY_OCCURRENCE)

surgery_clean = node_summary[node_summary["keep"]].copy()
surgery_removed = node_summary[~node_summary["keep"]].copy()

surgery_clean = surgery_clean.sort_values(["occurrence", "concept"], ascending=[False, True])
surgery_removed = surgery_removed.sort_values(["reject_reason", "occurrence", "concept"], ascending=[True, False, True])

surgery_clean.to_csv(FULL_SURGERY_CLEAN_FILE, index=False, encoding="utf-8-sig")
surgery_removed.to_csv(FULL_SURGERY_REMOVED_FILE, index=False, encoding="utf-8-sig")

print(f"Saved full clean surgery concepts: {FULL_SURGERY_CLEAN_FILE}", flush=True)
print(f"Saved removed surgery concepts: {FULL_SURGERY_REMOVED_FILE}", flush=True)
print(f"Kept surgery concepts: {len(surgery_clean)}", flush=True)
print(f"Removed surgery concepts: {len(surgery_removed)}", flush=True)


# ============================================================
# LOAD EXISTING EMBEDDINGS AND EMBED MISSING
# ============================================================

print(f"Loading existing surgery embeddings: {EXISTING_SURGERY_EMBEDDING_FILE}", flush=True)

if EXISTING_SURGERY_EMBEDDING_FILE.exists():
    existing_emb = read_csv_flexible(EXISTING_SURGERY_EMBEDDING_FILE)
else:
    existing_emb = pd.DataFrame(columns=["concept", "embedding"])

if "concept" not in existing_emb.columns or "embedding" not in existing_emb.columns:
    raise ValueError(f"Existing embedding file must have concept and embedding columns. Columns: {existing_emb.columns.tolist()}")

existing_emb["concept"] = existing_emb["concept"].apply(normalize_text)
existing_emb = existing_emb.drop_duplicates(subset=["concept"])

keep_concepts = surgery_clean["concept"].dropna().astype(str).tolist()
keep_set = set(keep_concepts)

results = existing_emb[existing_emb["concept"].isin(keep_set)].to_dict("records")
done = set(r["concept"] for r in results)

concepts_todo = [c for c in keep_concepts if c not in done]

if TEST_MODE:
    concepts_todo = concepts_todo[:TEST_N_CONCEPTS]
    print(f"TEST MODE ACTIVE: embedding only first {len(concepts_todo)} missing concepts", flush=True)

print(f"Existing matching embeddings: {len(done)}", flush=True)
print(f"Missing embeddings to create: {len(concepts_todo)}", flush=True)

batches = [
    concepts_todo[i:i + BATCH_SIZE]
    for i in range(0, len(concepts_todo), BATCH_SIZE)
]

for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding missing full surgery concepts")):
    try:
        response = client.embeddings.create(
            model=MODEL,
            input=batch,
        )

        for concept, emb in zip(batch, response.data):
            results.append({
                "concept": concept,
                "embedding": ",".join(map(str, emb.embedding)),
            })

    except AuthenticationError as e:
        print("Authentication failed: invalid OpenAI API key. Saving and stopping.", flush=True)
        pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
            FULL_SURGERY_EMBEDDING_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...", flush=True)
        pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
            FULL_SURGERY_EMBEDDING_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except Exception as e:
        print(f"Error in embedding batch {batch_idx + 1}: {repr(e)}", flush=True)
        time.sleep(5)

    if (batch_idx + 1) % SAVE_EVERY_BATCHES == 0:
        pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
            FULL_SURGERY_EMBEDDING_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        print(
            f"Saved checkpoint: {FULL_SURGERY_EMBEDDING_FILE} | rows: {len(pd.DataFrame(results).drop_duplicates(subset=['concept']))}",
            flush=True
        )

    time.sleep(SLEEP_BETWEEN_BATCHES)

emb_out = pd.DataFrame(results).drop_duplicates(subset=["concept"])
emb_out = emb_out[emb_out["concept"].isin(keep_set)].copy()

emb_out.to_csv(FULL_SURGERY_EMBEDDING_FILE, index=False, encoding="utf-8-sig")

print(f"Saved full surgery embeddings: {FULL_SURGERY_EMBEDDING_FILE}", flush=True)
print(f"Final full surgery embedding rows: {len(emb_out)}", flush=True)


# ============================================================
# SUMMARY
# ============================================================

summary_lines = []
summary_lines.append("FULL SURGERY VOCABULARY SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Input concept rows: {len(df)}")
summary_lines.append(f"Unique concepts before filtering: {len(node_summary)}")
summary_lines.append(f"Kept concepts after generic filter: {len(surgery_clean)}")
summary_lines.append(f"Removed concepts: {len(surgery_removed)}")
summary_lines.append(f"Existing embeddings reused: {len(done)}")
summary_lines.append(f"New embeddings requested: {len(concepts_todo)}")
summary_lines.append(f"Final embedding rows: {len(emb_out)}")
summary_lines.append("")
summary_lines.append("Top removed reasons:")
summary_lines.append(str(surgery_removed["reject_reason"].value_counts().head(20)))
summary_lines.append("")
summary_lines.append("Top kept concepts:")
summary_lines.append(str(surgery_clean[["concept", "occurrence"]].head(50)))
summary_lines.append("")
summary_lines.append("Top singleton kept concepts:")
summary_lines.append(str(surgery_clean[surgery_clean["occurrence"] == 1][["concept", "occurrence"]].head(50)))

SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved: {FULL_SURGERY_NODES_FILE}")
print(f"Saved: {FULL_SURGERY_CLEAN_FILE}")
print(f"Saved: {FULL_SURGERY_REMOVED_FILE}")
print(f"Saved: {FULL_SURGERY_EMBEDDING_FILE}")
print(f"Saved: {SUMMARY_FILE}")
print("\n".join(summary_lines[:10]))