from pathlib import Path
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, AuthenticationError


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = Path("data_processed_crossdomain/surgery_arxiv_far_candidate_pairs_filtered.csv")

OUT_DIR = Path("data_processed_crossdomain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "llm_ranked_crossdomain_bridges_FULL.csv"
OUTPUT_TOP_FILE = OUT_DIR / "llm_ranked_crossdomain_bridges_FULL_top.csv"


# ============================================================
# SETTINGS
# ============================================================

MODEL = "gpt-4o-mini"

# Full run
TEST_MODE = False
TEST_N = None

# BATCH_SIZE 10 ist meist effizienter als 5.
# Falls JSON-Fehler auftreten, wieder auf 5 setzen.
BATCH_SIZE = 10

SAVE_EVERY_BATCHES = 5
SLEEP_BETWEEN_BATCHES = 0.5
TIMEOUT_SECONDS = 180

RANDOM_SAMPLE_TEST = False
RANDOM_SEED = 42

TOP_N_EXPORT = 1000


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
# LOAD INPUT
# ============================================================

print("=" * 80)
print("LLM RANK CROSS-DOMAIN BRIDGES - FULL RUN")
print("=" * 80)
print(f"Loading input: {INPUT_FILE}", flush=True)

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print(f"Loaded pairs: {len(df)}", flush=True)
print("Columns:", df.columns.tolist(), flush=True)

required = [
    "surgery_concept",
    "arxiv_concept",
    "cosine_similarity",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["cosine_similarity"] = pd.to_numeric(df["cosine_similarity"], errors="coerce")
df = df.dropna(subset=["cosine_similarity"]).copy()

df = df.reset_index(drop=True)
df["pair_id"] = df.index.astype(str)

if TEST_MODE:
    if RANDOM_SAMPLE_TEST:
        df = df.sample(n=min(TEST_N, len(df)), random_state=RANDOM_SEED).copy()
    else:
        df = df.head(TEST_N).copy()

    print(f"TEST MODE ACTIVE: processing {len(df)} pairs", flush=True)


# ============================================================
# RESUME
# ============================================================

if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
    existing = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")

    if "pair_id" in existing.columns:
        done_ids = set(existing["pair_id"].astype(str))
        results = existing.to_dict("records")
        print(f"Resuming. Already processed pair_ids: {len(done_ids)}", flush=True)
    else:
        done_ids = set()
        results = []
else:
    done_ids = set()
    results = []

df_todo = df[~df["pair_id"].astype(str).isin(done_ids)].copy()

print(f"Remaining pairs to rank: {len(df_todo)}", flush=True)


# ============================================================
# HELPERS
# ============================================================

def safe_str(x, max_len=700):
    x = str(x)
    x = " ".join(x.split())
    return x[:max_len]


def clamp_score(x, min_val=0, max_val=5):
    try:
        val = int(round(float(x)))
    except Exception:
        val = 0

    return max(min_val, min(max_val, val))


def compute_final_score(mechanistic, creative, surgical, triviality):
    return (
        float(mechanistic)
        + float(creative)
        + float(surgical)
        - float(triviality)
    )


def extract_batch(batch_df: pd.DataFrame):
    items = []

    for _, row in batch_df.iterrows():
        items.append({
            "pair_id": str(row["pair_id"]),
            "surgery_concept": safe_str(row.get("surgery_concept", ""), 300),
            "arxiv_concept": safe_str(row.get("arxiv_concept", ""), 300),
            "cosine_similarity": float(row.get("cosine_similarity", 0)),
            "distance_band": safe_str(row.get("distance_band", ""), 100),
            "surgery_occurrence": safe_str(row.get("surgery_occurrence", ""), 50),
            "surgery_example_titles": safe_str(row.get("surgery_example_titles", ""), 900),
            "arxiv_occurrence": safe_str(row.get("arxiv_occurrence", ""), 50),
            "arxiv_concept_types": safe_str(row.get("arxiv_concept_types", ""), 200),
            "arxiv_modules": safe_str(row.get("arxiv_modules", ""), 300),
            "arxiv_example_titles": safe_str(row.get("arxiv_example_titles", ""), 900),
        })

    prompt = f"""
You are ranking cross-domain bridge candidates for scientific hypothesis generation.

Project:
We extracted surgical concepts from biomedical/surgical abstracts and technical concepts from non-PubMed arXiv abstracts.
The goal is NOT to find obvious semantic matches.
The goal is to identify surprising but mechanistically bridgeable combinations that could inspire new surgical research hypotheses.

Task:
For each candidate pair, evaluate whether the arXiv concept can provide a useful external principle, mechanism, design analogy, sensing logic, control logic, material principle, interface principle, failure-resilience principle, transport principle, or dynamical-systems principle for the surgical concept.

Important:
Do NOT reward obvious semantic similarity.
Do NOT rank a pair highly just because both concepts are medical, imaging-related, AI-related, prediction-related, or oncology-related.
Do NOT only rely on your own knowledge. Consider creative and NEW solutions.
Prefer pairs that are non-obvious, creative, and still mechanistically or experimentally testable.
A pair can be far in semantic embedding space and still be valuable if a concrete bridge principle can be stated.

Scores:
- mechanistic_bridge_score: 0 to 5
  0 = no mechanism; 5 = clear transferable mechanism/principle.
- creative_distance_score: 0 to 5
  0 = obvious/trivial or no useful distance; 5 = surprising cross-domain jump with a meaningful analogy.
- surgical_relevance_score: 0 to 5
  0 = not surgically relevant; 5 = directly relevant to an important surgical problem, complication, device, technique, or perioperative mechanism.
- triviality_score: 0 to 5
  0 = not trivial; 10 = obvious/semantic/standard/boring combination.

Final score:
final_score = mechanistic_bridge_score + creative_distance_score + surgical_relevance_score - triviality_score

Decision:
- "keep" only if final_score is high enough AND the bridge is not merely metaphorical.
- "reject" if the pair is semantically obvious, clinically irrelevant, purely AI/prediction without a new surgical mechanism, or impossible to operationalize.

For each kept pair:
- bridge_principle should be a short mechanistic principle, not a vague metaphor.
- hypothesis should be a testable surgical hypothesis in one sentence.
- experimental_hint should name a feasible experiment, simulation, bench test, animal model, device prototype, retrospective analysis, or ex vivo setup.

For rejected pairs:
- You may leave bridge_principle/hypothesis short, but explain the rejection briefly.

Return valid JSON only.

Return exactly this JSON structure:
{{
  "items": [
    {{
      "pair_id": "0",
      "decision": "keep",
      "mechanistic_bridge_score": 4,
      "creative_distance_score": 5,
      "surgical_relevance_score": 4,
      "triviality_score": 1,
      "final_score": 12,
      "bridge_principle": "entropy-based quantification of irregular pressure-flow signatures",
      "hypothesis": "Entropy features from intraoperative insufflation or dye-flow signals can improve detection of subclinical anastomotic leaks.",
      "experimental_hint": "Bench-test intestinal anastomosis phantoms with controlled microleaks and compare entropy-based signal features against standard pressure decay testing.",
      "short_reason": "Non-obvious information-theoretic principle with plausible leak-testing application."
    }}
  ]
}}

Candidate pairs:
{json.dumps(items, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        timeout=TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous but creative cross-domain scientific hypothesis judge. "
                    "You reward non-obvious mechanistic bridges, not generic semantic similarity. "
                    "You return strict JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
    )

    text = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("JSON parse failed. Raw output:", flush=True)
        print(text[:2000], flush=True)
        return []

    return parsed.get("items", [])


# ============================================================
# MAIN LOOP
# ============================================================

batches = [
    df_todo.iloc[i:i + BATCH_SIZE]
    for i in range(0, len(df_todo), BATCH_SIZE)
]

print(f"Total batches: {len(batches)}", flush=True)

meta_cols = [
    "surgery_concept",
    "arxiv_concept",
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

existing_meta_cols = [c for c in meta_cols if c in df.columns]

for batch_idx, batch_df in enumerate(tqdm(batches, desc="LLM ranking FULL")):
    try:
        extracted = extract_batch(batch_df)

        batch_meta = (
            batch_df
            .set_index("pair_id", drop=False)[existing_meta_cols + ["pair_id"]]
            .to_dict("index")
        )

        for item in extracted:
            pair_id = str(item.get("pair_id", "")).strip()

            if pair_id not in batch_meta:
                continue

            mechanistic = clamp_score(item.get("mechanistic_bridge_score", 0))
            creative = clamp_score(item.get("creative_distance_score", 0))
            surgical = clamp_score(item.get("surgical_relevance_score", 0))
            triviality = clamp_score(item.get("triviality_score", 0))

            final_score = compute_final_score(
                mechanistic,
                creative,
                surgical,
                triviality
            )

            decision = str(item.get("decision", "reject")).strip().lower()
            if decision not in {"keep", "reject"}:
                decision = "reject"

            row_out = dict(batch_meta[pair_id])

            row_out.update({
                "decision": decision,
                "mechanistic_bridge_score": mechanistic,
                "creative_distance_score": creative,
                "surgical_relevance_score": surgical,
                "triviality_score": triviality,
                "final_score": final_score,
                "bridge_principle": safe_str(item.get("bridge_principle", ""), 1000),
                "hypothesis": safe_str(item.get("hypothesis", ""), 1200),
                "experimental_hint": safe_str(item.get("experimental_hint", ""), 1200),
                "short_reason": safe_str(item.get("short_reason", ""), 1200),
            })

            results.append(row_out)

        print(
            f"Batch {batch_idx + 1}/{len(batches)} done. Total ranked rows: {len(results)}",
            flush=True
        )

    except AuthenticationError as e:
        print("Authentication failed: invalid OpenAI API key. Saving and stopping.", flush=True)
        pd.DataFrame(results).drop_duplicates(subset=["pair_id"]).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...", flush=True)
        pd.DataFrame(results).drop_duplicates(subset=["pair_id"]).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except Exception as e:
        print(f"Error in batch {batch_idx + 1}: {repr(e)}", flush=True)

    if (batch_idx + 1) % SAVE_EVERY_BATCHES == 0:
        out_df = pd.DataFrame(results)

        if not out_df.empty and "pair_id" in out_df.columns:
            out_df = out_df.drop_duplicates(subset=["pair_id"])

        out_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"Saved checkpoint: {OUTPUT_FILE} | rows: {len(out_df)}", flush=True)

    time.sleep(SLEEP_BETWEEN_BATCHES)


# ============================================================
# FINAL SAVE
# ============================================================

out_df = pd.DataFrame(results)

if not out_df.empty and "pair_id" in out_df.columns:
    out_df = out_df.drop_duplicates(subset=["pair_id"])

if not out_df.empty:
    out_df["final_score"] = pd.to_numeric(out_df["final_score"], errors="coerce")
    out_df["mechanistic_bridge_score"] = pd.to_numeric(out_df["mechanistic_bridge_score"], errors="coerce")
    out_df["creative_distance_score"] = pd.to_numeric(out_df["creative_distance_score"], errors="coerce")
    out_df["surgical_relevance_score"] = pd.to_numeric(out_df["surgical_relevance_score"], errors="coerce")
    out_df["triviality_score"] = pd.to_numeric(out_df["triviality_score"], errors="coerce")

    out_df = out_df.sort_values(
        ["final_score", "mechanistic_bridge_score", "creative_distance_score", "surgical_relevance_score"],
        ascending=[False, False, False, False]
    )

out_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

top_df = out_df.head(TOP_N_EXPORT).copy()
top_df.to_csv(
    OUTPUT_TOP_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved full ranked file: {OUTPUT_FILE}")
print(f"Saved top ranked file: {OUTPUT_TOP_FILE}")
print(f"Rows ranked: {len(out_df)}")

if not out_df.empty:
    print("\nTop 20:")
    cols = [
        "surgery_concept",
        "arxiv_concept",
        "final_score",
        "mechanistic_bridge_score",
        "creative_distance_score",
        "surgical_relevance_score",
        "triviality_score",
        "bridge_principle",
        "hypothesis",
    ]
    cols = [c for c in cols if c in out_df.columns]
    print(out_df[cols].head(20))