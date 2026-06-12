from pathlib import Path
import os
import time
import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path("data_raw_arxiv_external/all_modules_combined_PROGRESS.csv")
OUTPUT_FILE = Path("data_processed_arxiv_external/arxiv_external_concepts_3_per_abstract.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"

# Kleine Batchgröße ist stabiler und leichter zu debuggen.
BATCH_SIZE = 5

SAVE_EVERY_BATCHES = 5
SLEEP_BETWEEN_BATCHES = 0.5
TIMEOUT_SECONDS = 90

# Für Testlauf:
TEST_MODE = False
TEST_N = None

# Für echten Lauf später:
# TEST_MODE = False
# TEST_N = None


# ============================================================
# API KEY
# ============================================================

# Key in PowerShell setzen:
# $env:OPENAI_API_KEY="sk-..."

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Setze ihn in PowerShell mit:\n"
        '$env:OPENAI_API_KEY="sk-..."'
    )

client = OpenAI(api_key=api_key)


# ============================================================
# LOAD INPUT
# ============================================================

print("SCRIPT STARTED", flush=True)
print(f"Loading input file: {INPUT_FILE}", flush=True)

df = pd.read_csv(INPUT_FILE, sep=";", dtype=str).fillna("")

print(f"Loaded rows: {len(df)}", flush=True)
print("Columns:", df.columns.tolist(), flush=True)

required_cols = [
    "module",
    "arxiv_id",
    "year",
    "title",
    "abstract",
    "primary_category",
    "categories",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in input file: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

df = df[(df["year"] >= 2020) & (df["year"] <= 2026)].copy()
df = df.dropna(subset=["arxiv_id", "title", "abstract"])

df["arxiv_id"] = df["arxiv_id"].astype(str)
df["title"] = df["title"].astype(str)
df["abstract"] = df["abstract"].astype(str)

df = df[df["abstract"].str.len() > 50].copy()

print(f"Usable abstracts 2020-2026: {len(df)}", flush=True)

if TEST_MODE:
    df = df.head(TEST_N).copy()
    print(f"TEST MODE ACTIVE: first {len(df)} abstracts only", flush=True)


# ============================================================
# RESUME
# ============================================================

if OUTPUT_FILE.exists():
    existing = pd.read_csv(OUTPUT_FILE, dtype=str).fillna("")
    if "arxiv_id" in existing.columns:
        done_ids = set(existing["arxiv_id"].astype(str).unique())
        results = existing.to_dict("records")
        print(f"Resuming. Already processed arXiv IDs: {len(done_ids)}", flush=True)
    else:
        done_ids = set()
        results = []
else:
    done_ids = set()
    results = []

df_todo = df[~df["arxiv_id"].astype(str).isin(done_ids)].copy()

print(f"Remaining abstracts: {len(df_todo)}", flush=True)


# ============================================================
# HELPERS
# ============================================================

def clean_text(x):
    x = str(x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_concept(x):
    x = str(x).lower().strip()
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def extract_batch(batch_df: pd.DataFrame):
    items = []

    for _, row in batch_df.iterrows():
        items.append({
            "arxiv_id": str(row["arxiv_id"]),
            "module": str(row["module"]),
            "primary_category": str(row["primary_category"]),
            "categories": str(row["categories"])[:500],
            "title": clean_text(row["title"])[:800],
            "abstract": clean_text(row["abstract"])[:4500],
        })

    prompt = f"""
You are extracting reusable scientific concepts from non-biomedical arXiv abstracts.

Project goal:
We are building an external, non-PubMed concept library for cross-domain scientific hypothesis generation.
The concepts will later be recombined with surgical problems, but you must NOT force any surgical or biomedical interpretation now.

Task:
For each abstract, extract EXACTLY 3 high-value concepts.

What counts as a good concept:
- A transferable technical, physical, computational, mathematical, material, robotic, control, sensing, dynamical, or systems principle.
- Concepts that could become nodes in a cross-domain hypothesis graph.
- Concepts that capture mechanisms, structures, design principles, phenomena, modeling frameworks, or control rules.
- Prefer specific concepts over field labels.

Good examples:
- velocity dependent alignment
- active crystal clusters
- nonequilibrium collective dynamics
- phase field reaction diffusion duality
- attracting manifold embedding
- sharp interface velocity law
- topological edge transport
- defect tolerant architecture
- field mediated actuation
- rare event precursor detection
- morphological computation
- percolation driven transport
- adaptive sampling strategy
- nonreciprocal wave propagation
- jamming transition control

Bad concepts:
- study
- model
- method
- results
- framework
- system
- dynamics
- optimization
- machine learning
- robotics
- materials
- phase transition
- pattern formation
- artificial intelligence
- simulation
- numerical analysis
- experiment
- performance

Rules:
- Return EXACTLY 3 concepts per abstract.
- Each concept must be 2 to 7 words.
- Use lowercase.
- Normalize singular/plural where obvious.
- Do not use full sentences.
- Avoid abbreviations unless standard and meaningful.
- Avoid generic field names.
- Do not extract biomedical or surgical interpretations.
- Concepts should be understandable without reading the full abstract.
- Prefer concepts that are specific enough to be useful later.

Also assign each concept one concept_type from this controlled list:
- physical_mechanism
- control_principle
- computational_principle
- material_principle
- sensing_principle
- collective_behavior
- transport_principle
- failure_resilience
- interface_phenomenon
- optimization_design
- dynamical_system
- robotic_principle
- other

Return valid JSON only.

Return exactly this JSON structure:
{{
  "items": [
    {{
      "arxiv_id": "1234.56789",
      "concepts": [
        {{
          "concept": "concept one",
          "concept_type": "physical_mechanism"
        }},
        {{
          "concept": "concept two",
          "concept_type": "collective_behavior"
        }},
        {{
          "concept": "concept three",
          "concept_type": "dynamical_system"
        }}
      ]
    }}
  ]
}}

Abstracts:
{json.dumps(items, ensure_ascii=False)}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        timeout=TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You extract normalized cross-domain scientific concepts from non-biomedical arXiv abstracts and return strict JSON only."
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

allowed_types = {
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

for batch_idx, batch_df in enumerate(tqdm(batches, desc="Extracting concepts")):
    try:
        extracted = extract_batch(batch_df)

        meta_cols = [
            "module",
            "year",
            "title",
            "abstract",
            "primary_category",
            "categories",
            "doi",
            "journal_ref",
            "abs_url",
            "pdf_url",
        ]

        existing_meta_cols = [c for c in meta_cols if c in batch_df.columns]
        meta = batch_df.set_index("arxiv_id")[existing_meta_cols].to_dict("index")

        for item in extracted:
            arxiv_id = str(item.get("arxiv_id", "")).strip()
            concepts = item.get("concepts", [])

            if arxiv_id not in meta:
                continue

            if not isinstance(concepts, list):
                continue

            for rank, cobj in enumerate(concepts[:3], start=1):
                if isinstance(cobj, dict):
                    concept = cobj.get("concept", "")
                    concept_type = cobj.get("concept_type", "other")
                else:
                    concept = cobj
                    concept_type = "other"

                concept = normalize_concept(concept)
                concept_type = str(concept_type).strip()

                if concept_type not in allowed_types:
                    concept_type = "other"

                n_words = len(concept.split())

                if n_words < 2:
                    continue
                if n_words > 8:
                    continue

                results.append({
                    "arxiv_id": arxiv_id,
                    "concept_rank": rank,
                    "concept": concept,
                    "concept_type": concept_type,
                    "module": meta[arxiv_id].get("module", ""),
                    "year": meta[arxiv_id].get("year", ""),
                    "primary_category": meta[arxiv_id].get("primary_category", ""),
                    "categories": meta[arxiv_id].get("categories", ""),
                    "title": meta[arxiv_id].get("title", ""),
                    "doi": meta[arxiv_id].get("doi", ""),
                    "journal_ref": meta[arxiv_id].get("journal_ref", ""),
                    "abs_url": meta[arxiv_id].get("abs_url", ""),
                    "pdf_url": meta[arxiv_id].get("pdf_url", ""),
                })

        print(
            f"Batch {batch_idx + 1}/{len(batches)} done. Total concept rows: {len(results)}",
            flush=True
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...", flush=True)
        pd.DataFrame(results).drop_duplicates().to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        raise

    except Exception as e:
        print(f"Error in batch {batch_idx + 1}: {repr(e)}", flush=True)

    if (batch_idx + 1) % SAVE_EVERY_BATCHES == 0:
        pd.DataFrame(results).drop_duplicates().to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )
        print(f"Saved checkpoint: {OUTPUT_FILE} | rows: {len(results)}", flush=True)

    time.sleep(SLEEP_BETWEEN_BATCHES)


pd.DataFrame(results).drop_duplicates().to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Done. Saved: {OUTPUT_FILE}", flush=True)
print(f"Final concept rows: {len(pd.DataFrame(results).drop_duplicates())}", flush=True)