from pathlib import Path
import os
import re
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, AuthenticationError


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("data_processed_arxiv_external/arxiv_external_concepts_3_per_abstract.csv")

OUTPUT_DIR = Path("data_processed_arxiv_external")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_SUMMARY_FILE = OUTPUT_DIR / "arxiv_external_concept_nodes_summary.csv"
EMBEDDING_FILE = OUTPUT_DIR / "arxiv_external_concept_embeddings.csv"

MODEL = "text-embedding-3-small"

BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 0.2
SAVE_EVERY_BATCHES = 10

# Für Testlauf:
TEST_MODE = False
TEST_N_CONCEPTS = None

# Für Full Run:
# TEST_MODE = False
# TEST_N_CONCEPTS = None


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
# GENERIC / LOW-VALUE CONCEPT FILTER
# ============================================================

GENERIC_EXACT = {
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
    "optimization",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural network",
    "neural networks",
    "robotics",
    "materials",
    "dynamics",
    "phase transition",
    "pattern formation",
    "numerical analysis",
    "data analysis",
    "statistical analysis",
    "computational model",
    "mathematical model",
    "physical model",
    "theoretical model",
    "experimental results",
    "proposed method",
    "proposed approach",
    "novel method",
    "new method",
    "efficient algorithm",
    "algorithm",
    "algorithms",
    "learning algorithm",
    "control system",
    "complex system",
    "complex systems",
    "active matter",
    "soft matter",
}

GENERIC_SUBSTRINGS = [
    "this study",
    "our study",
    "proposed method",
    "proposed model",
    "experimental result",
    "numerical result",
    "simulation result",
    "machine learning model",
    "deep learning model",
    "state of the art",
    "high performance",
    "real world",
    "large scale",
]

BAD_ENDINGS = {
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
    "optimization",
    "algorithm",
    "algorithms",
    "result",
    "results",
}

# Diese Typen sind tendenziell nützlich für Cross-Domain-Bridge-Ideen.
PREFERRED_TYPES = {
    "physical_mechanism",
    "control_principle",
    "computational_principle",
    "material_principle",
    "sensing_principle",
    "collective_behavior",
    "transport_principle",
    "failure_resilience",
    "interface_phenomenon",
    "optimization_design",
    "dynamical_system",
    "robotic_principle",
    "other",
}


def normalize_concept(x: str) -> str:
    x = str(x).lower().strip()
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"[^a-z0-9\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def is_generic_or_low_value(concept: str) -> bool:
    c = normalize_concept(concept)

    if not c:
        return True

    words = c.split()

    if len(words) < 2:
        return True

    if len(words) > 8:
        return True

    if c in GENERIC_EXACT:
        return True

    if any(s in c for s in GENERIC_SUBSTRINGS):
        return True

    if words[-1] in BAD_ENDINGS:
        return True

    # zu kurze technische Floskeln
    if len(c) < 6:
        return True

    # reine Zahlen / komische Artefakte
    if re.fullmatch(r"[0-9\s]+", c):
        return True

    return False


# ============================================================
# LOAD CONCEPTS
# ============================================================

print("SCRIPT STARTED", flush=True)
print(f"Loading concepts: {INPUT_FILE}", flush=True)

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print(f"Concept rows loaded: {len(df)}", flush=True)
print("Columns:", df.columns.tolist(), flush=True)

required_cols = ["arxiv_id", "concept", "concept_type", "module", "year", "title"]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input file: {missing}")

df["concept"] = df["concept"].apply(normalize_concept)
df["concept_type"] = df["concept_type"].astype(str).str.strip()
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)
df = df[(df["year"] >= 2020) & (df["year"] <= 2026)].copy()

df = df[df["concept_type"].isin(PREFERRED_TYPES)].copy()

print(f"Rows after year/type filter: {len(df)}", flush=True)

df["is_generic"] = df["concept"].apply(is_generic_or_low_value)

print("Generic/low-value rows:", int(df["is_generic"].sum()), flush=True)

df_clean = df[~df["is_generic"]].copy()

print(f"Rows after generic filter: {len(df_clean)}", flush=True)
print(f"Unique concepts after generic filter: {df_clean['concept'].nunique()}", flush=True)


# ============================================================
# NODE SUMMARY
# ============================================================

node_summary = (
    df_clean
    .groupby("concept", dropna=False)
    .agg(
        occurrence=("arxiv_id", "count"),
        n_arxiv_ids=("arxiv_id", "nunique"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        concept_types=("concept_type", lambda x: "; ".join(sorted(set(map(str, x))))),
        modules=("module", lambda x: "; ".join(sorted(set(map(str, x))))),
        example_titles=("title", lambda x: " || ".join(list(dict.fromkeys(map(str, x)))[:3])),
    )
    .reset_index()
)

node_summary = node_summary.sort_values(
    ["occurrence", "concept"],
    ascending=[False, True]
)

node_summary.to_csv(
    NODE_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Saved node summary: {NODE_SUMMARY_FILE}", flush=True)
print(f"Node summary rows: {len(node_summary)}", flush=True)


# ============================================================
# RESUME EMBEDDINGS
# ============================================================

concepts = node_summary["concept"].dropna().astype(str).tolist()

if TEST_MODE:
    concepts = concepts[:TEST_N_CONCEPTS]
    print(f"TEST MODE ACTIVE: embedding first {len(concepts)} concepts", flush=True)

if EMBEDDING_FILE.exists() and EMBEDDING_FILE.stat().st_size > 0:
    existing = pd.read_csv(EMBEDDING_FILE, dtype=str).fillna("")

    if "concept" in existing.columns:
        done = set(existing["concept"].astype(str))
        results = existing.to_dict("records")
        print(f"Resuming embeddings. Already done: {len(done)}", flush=True)
    else:
        done = set()
        results = []
else:
    done = set()
    results = []

concepts_todo = [c for c in concepts if c not in done]

print(f"Total target concepts: {len(concepts)}", flush=True)
print(f"Remaining concepts to embed: {len(concepts_todo)}", flush=True)


# ============================================================
# EMBEDDING LOOP
# ============================================================

batches = [
    concepts_todo[i:i + BATCH_SIZE]
    for i in range(0, len(concepts_todo), BATCH_SIZE)
]

print(f"Total embedding batches: {len(batches)}", flush=True)

for batch_idx, batch in enumerate(tqdm(batches, desc="Embedding arXiv concepts")):
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
        print("Authentication failed: invalid OpenAI API key. Stopping script.", flush=True)
        print(repr(e), flush=True)
        pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
            EMBEDDING_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...", flush=True)
        pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
            EMBEDDING_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except Exception as e:
        print(f"Error in embedding batch {batch_idx + 1}: {repr(e)}", flush=True)
        time.sleep(5)

    if (batch_idx + 1) % SAVE_EVERY_BATCHES == 0:
        pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
            EMBEDDING_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        print(
            f"Saved checkpoint: {EMBEDDING_FILE} | rows: {len(pd.DataFrame(results).drop_duplicates(subset=['concept']))}",
            flush=True
        )

    time.sleep(SLEEP_BETWEEN_BATCHES)


pd.DataFrame(results).drop_duplicates(subset=["concept"]).to_csv(
    EMBEDDING_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Done. Saved embeddings: {EMBEDDING_FILE}", flush=True)
print(f"Final embedding rows: {len(pd.DataFrame(results).drop_duplicates(subset=['concept']))}", flush=True)