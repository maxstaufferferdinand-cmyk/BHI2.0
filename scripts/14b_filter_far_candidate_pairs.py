from pathlib import Path
import re
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = Path("data_processed_crossdomain/surgery_arxiv_far_candidate_pairs.csv")

OUT_DIR = Path("data_processed_crossdomain")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUT_DIR / "surgery_arxiv_far_candidate_pairs_filtered.csv"
REJECTED_FILE = OUT_DIR / "surgery_arxiv_far_candidate_pairs_rejected_review.csv"
SUMMARY_FILE = OUT_DIR / "surgery_arxiv_far_candidate_pairs_filter_summary.txt"


# ============================================================
# SETTINGS
# ============================================================

# Hauptfenster bleibt "fern", aber wir entfernen extreme Grenzfälle optional nicht.
COSINE_MIN_KEEP = 0.10
COSINE_MAX_KEEP = 0.30

# Behalte maximal so viele Paare für LLM-Stufe 1.
# None = alle nach Filter behalten.
MAX_OUTPUT_PAIRS = 30000

# Pro chirurgischem Concept begrenzen, damit nicht ein paar Begriffe dominieren.
MAX_PAIRS_PER_SURGERY_CONCEPT = 3

# Pro arXiv Concept begrenzen, damit keine arXiv-Hubs dominieren.
MAX_PAIRS_PER_ARXIV_CONCEPT = 10

# Sortierung: niedrigere Cosine zuerst = ferner/kreativer.
SORT_BY_FARNESS = True


# ============================================================
# NORMALIZATION
# ============================================================

def norm(x):
    x = str(x).lower().strip()
    x = re.sub(r"[-_/]", " ", x)
    x = re.sub(r"[^a-z0-9α-ωβγδκλμσπφχψω\s]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def word_count(x):
    return len(norm(x).split())


def contains_any(x, terms):
    x = norm(x)
    return any(t in x for t in terms)


def exact_any(x, terms):
    return norm(x) in terms


def token_overlap_fraction(a, b):
    a_tokens = set(norm(a).split())
    b_tokens = set(norm(b).split())
    if not a_tokens or not b_tokens:
        return 1.0
    return len(a_tokens & b_tokens) / min(len(a_tokens), len(b_tokens))


# ============================================================
# FILTER LISTS
# ============================================================

SURGERY_BAD_EXACT = {
    "postoperative follow up",
    "follow up",
    "clinical follow up",
    "long term follow up",
    "short term follow up",
    "survival analysis",
    "overall survival",
    "disease free survival",
    "recurrence free survival",
    "progression free survival",
    "mortality risk",
    "morbidity risk",
    "risk prediction",
    "risk stratification",
    "risk assessment",
    "risk factors",
    "prognostic factors",
    "predictive factors",
    "treatment outcomes",
    "clinical outcomes",
    "surgical outcomes",
    "oncologic outcomes",
    "postoperative outcomes",
    "perioperative outcomes",
    "quality of life",
    "health related quality of life",
}

SURGERY_BAD_SUBSTRINGS = [
    "prognosis",
    "prognostic",
    "efficacy",
    "guidance",
    "prediction",
    "predictive",
    "risk factor",
    "risk factors",
    "survival",
    "mortality",
    "morbidity",
    "outcome",
    "outcomes",
    "follow up",
    "quality of life",
    "cost effectiveness",
    "meta analysis",
    "systematic review",
    "retrospective",
    "prospective cohort",
    "randomized trial",
    "clinical trial",
    "nomogram",
    "score validation",
    "treatment selection",
    "chemotherapy guidance",
    "adjuvant chemotherapy",
    "neoadjuvant chemotherapy",
]

# Aber diese substrings dürfen bleiben, weil chirurgisch/mechanistisch relevant.
SURGERY_PROTECT_SUBSTRINGS = [
    "postoperative pancreatic fistula",
    "pancreatic fistula",
    "bile leak",
    "biliary leak",
    "anastomotic leak",
    "anastomotic leakage",
    "anastomotic perfusion",
    "anastomotic ischemia",
    "wound dehiscence",
    "surgical site infection",
    "mesh infection",
    "biofilm",
    "postoperative ileus",
    "intra abdominal infection",
    "abdominal sepsis",
    "peritoneal adhesion",
    "intra abdominal adhesion",
    "ischemia reperfusion",
    "post hepatectomy liver failure",
    "liver regeneration",
    "liver hypertrophy",
    "drain amylase",
    "drain fluid",
    "soft pancreatic texture",
    "fistula risk score",
    "biliary stricture",
    "pancreaticojejunostomy",
    "hepaticojejunostomy",
    "colorectal anastomosis",
    "pancreatic anastomosis",
    "bowel perfusion",
    "organ perfusion",
    "tissue perfusion",
    "intraoperative fluorescence",
    "indocyanine green",
    "tumor budding",
    "circulating tumor dna",
    "cell free dna",
    "sarcopenia",
    "frailty",
    "microbiome",
    "bacterial translocation",
    "peritoneal metastasis",
]

ARXIV_BAD_EXACT = {
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "neural network",
    "neural networks",
    "optimization",
    "classification",
    "prediction",
    "estimation",
    "detection",
    "simulation",
    "model",
    "system",
    "framework",
    "algorithm",
    "active matter",
    "soft matter",
    "complex system",
    "complex systems",
    "dynamics",
    "pattern formation",
}

ARXIV_BAD_SUBSTRINGS = [
    "benchmark",
    "state of the art",
    "machine learning model",
    "deep learning model",
    "neural network model",
    "proposed algorithm",
    "proposed framework",
    "numerical result",
    "simulation result",
    "experimental result",
]

# Diese arXiv-Begriffe sind für Bridge-Ideen ausdrücklich gut.
ARXIV_PROTECT_SUBSTRINGS = [
    "topological",
    "nonreciprocal",
    "self organization",
    "self assembly",
    "collective motion",
    "collective dynamics",
    "active crystal",
    "velocity dependent alignment",
    "jamming transition",
    "granular jamming",
    "percolation",
    "reaction diffusion",
    "turing pattern",
    "morphological computation",
    "reservoir computing",
    "field mediated",
    "remote actuation",
    "magnetic actuation",
    "defect tolerance",
    "damage tolerance",
    "graceful degradation",
    "crack arrest",
    "failure resilience",
    "phase field",
    "sharp interface",
    "attracting manifold",
    "rare event",
    "precursor detection",
    "critical transition",
    "tipping point",
    "adaptive sampling",
    "closed loop discovery",
    "boundary effect",
    "wetting transition",
    "contact line",
    "capillary interaction",
    "interfacial instability",
    "programmable stiffness",
    "variable stiffness",
    "tunable compliance",
]

PREFERRED_ARXIV_TYPE_SUBSTRINGS = [
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
]


# ============================================================
# REJECTION LOGIC
# ============================================================

def surgery_reject_reason(x):
    x_norm = norm(x)

    if contains_any(x_norm, SURGERY_PROTECT_SUBSTRINGS):
        return ""

    if word_count(x_norm) < 2:
        return "surgery_too_short"

    if word_count(x_norm) > 8:
        return "surgery_too_long"

    if exact_any(x_norm, SURGERY_BAD_EXACT):
        return "surgery_bad_exact"

    if contains_any(x_norm, SURGERY_BAD_SUBSTRINGS):
        return "surgery_bad_substring"

    # Begriffe, die nur klinische Entscheidung/Outcome andeuten
    bad_endings = {
        "prognosis", "prediction", "guidance", "efficacy", "outcome", "outcomes",
        "survival", "mortality", "morbidity", "rate", "rates", "risk", "risks",
        "factor", "factors", "management", "therapy", "treatment"
    }
    words = x_norm.split()
    if words and words[-1] in bad_endings:
        return "surgery_bad_ending"

    return ""


def arxiv_reject_reason(x, concept_types=""):
    x_norm = norm(x)

    if contains_any(x_norm, ARXIV_PROTECT_SUBSTRINGS):
        return ""

    if word_count(x_norm) < 2:
        return "arxiv_too_short"

    if word_count(x_norm) > 8:
        return "arxiv_too_long"

    if exact_any(x_norm, ARXIV_BAD_EXACT):
        return "arxiv_bad_exact"

    if contains_any(x_norm, ARXIV_BAD_SUBSTRINGS):
        return "arxiv_bad_substring"

    bad_endings = {
        "model", "models", "method", "methods", "system", "systems", "framework",
        "algorithm", "algorithms", "analysis", "simulation", "performance",
        "classification", "prediction", "estimation", "detection"
    }
    words = x_norm.split()
    if words and words[-1] in bad_endings:
        return "arxiv_bad_ending"

    # Optional: arXiv type should not be empty/irrelevant.
    ctype = str(concept_types)
    if ctype and not any(t in ctype for t in PREFERRED_ARXIV_TYPE_SUBSTRINGS):
        # nicht hart rejecten, weil "other" manchmal gut sein kann
        pass

    return ""


def pair_reject_reason(row):
    s = row["surgery_concept"]
    a = row["arxiv_concept"]

    s_reason = surgery_reject_reason(s)
    if s_reason:
        return s_reason

    a_reason = arxiv_reject_reason(a, row.get("arxiv_concept_types", ""))
    if a_reason:
        return a_reason

    try:
        sim = float(row["cosine_similarity"])
    except Exception:
        return "missing_cosine"

    if sim < COSINE_MIN_KEEP:
        return "cosine_too_low"

    if sim > COSINE_MAX_KEEP:
        return "cosine_too_high"

    overlap = token_overlap_fraction(s, a)
    if overlap > 0.25:
        return "token_overlap_too_high"

    return ""


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("FILTER FAR CROSS-DOMAIN CANDIDATE PAIRS")
print("=" * 80)

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print(f"Loaded candidate pairs: {len(df)}")
print("Columns:", df.columns.tolist())

required = ["surgery_concept", "arxiv_concept", "cosine_similarity"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["cosine_similarity"] = pd.to_numeric(df["cosine_similarity"], errors="coerce")
df["surgery_concept_norm"] = df["surgery_concept"].apply(norm)
df["arxiv_concept_norm"] = df["arxiv_concept"].apply(norm)

df["filter_reject_reason"] = df.apply(pair_reject_reason, axis=1)
df["keep"] = df["filter_reject_reason"].eq("")

kept = df[df["keep"]].copy()
rejected = df[~df["keep"]].copy()

print(f"Kept after rule filters: {len(kept)}")
print(f"Rejected after rule filters: {len(rejected)}")

# Dominance control: max pairs per surgery concept
if len(kept) > 0:
    if SORT_BY_FARNESS:
        kept = kept.sort_values(
            ["cosine_similarity", "surgery_occurrence", "arxiv_occurrence"],
            ascending=[True, False, False]
        )
    else:
        kept = kept.sample(frac=1, random_state=42)

    kept["rank_within_surgery"] = kept.groupby("surgery_concept_norm").cumcount() + 1
    kept = kept[kept["rank_within_surgery"] <= MAX_PAIRS_PER_SURGERY_CONCEPT].copy()

    # Dominance control: max pairs per arXiv concept
    kept = kept.sort_values(
        ["cosine_similarity", "surgery_occurrence", "arxiv_occurrence"],
        ascending=[True, False, False]
    )
    kept["rank_within_arxiv"] = kept.groupby("arxiv_concept_norm").cumcount() + 1
    kept = kept[kept["rank_within_arxiv"] <= MAX_PAIRS_PER_ARXIV_CONCEPT].copy()

    kept = kept.drop(columns=["rank_within_surgery", "rank_within_arxiv"], errors="ignore")

if MAX_OUTPUT_PAIRS is not None and len(kept) > MAX_OUTPUT_PAIRS:
    kept = kept.head(MAX_OUTPUT_PAIRS).copy()

# Remove helper keep column from output but keep reject file informative
kept_out = kept.drop(columns=["keep"], errors="ignore")
rejected_out = rejected.drop(columns=["keep"], errors="ignore")

kept_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
rejected_out.to_csv(REJECTED_FILE, index=False, encoding="utf-8-sig")

summary_lines = []
summary_lines.append("FILTER FAR CROSS-DOMAIN CANDIDATE PAIRS SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append(f"Input pairs: {len(df)}")
summary_lines.append(f"Kept pairs: {len(kept_out)}")
summary_lines.append(f"Rejected pairs: {len(rejected_out)}")
summary_lines.append(f"Cosine keep window: {COSINE_MIN_KEEP}-{COSINE_MAX_KEEP}")
summary_lines.append(f"Max output pairs: {MAX_OUTPUT_PAIRS}")
summary_lines.append(f"Max pairs per surgery concept: {MAX_PAIRS_PER_SURGERY_CONCEPT}")
summary_lines.append(f"Max pairs per arXiv concept: {MAX_PAIRS_PER_ARXIV_CONCEPT}")
summary_lines.append("")
summary_lines.append("Reject reasons:")
summary_lines.append(str(rejected_out["filter_reject_reason"].value_counts().head(30)))
summary_lines.append("")
summary_lines.append("Kept cosine distribution:")
summary_lines.append(str(kept_out["cosine_similarity"].describe()))
summary_lines.append("")
summary_lines.append("Example kept pairs:")
example_cols = [
    "surgery_concept",
    "arxiv_concept",
    "cosine_similarity",
    "arxiv_concept_types",
    "arxiv_modules",
]
example_cols = [c for c in example_cols if c in kept_out.columns]
summary_lines.append(str(kept_out[example_cols].head(50)))

SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print(f"Saved filtered pairs: {OUTPUT_FILE}")
print(f"Saved rejected review: {REJECTED_FILE}")
print(f"Saved summary: {SUMMARY_FILE}")
print(f"Kept pairs: {len(kept_out)}")
print("\nReject reasons:")
print(rejected_out["filter_reject_reason"].value_counts().head(20))
print("\nExample kept pairs:")
print(kept_out[example_cols].head(20))