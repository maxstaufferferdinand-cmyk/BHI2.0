# BHI2.0
BHI2.0 is a Python pipeline for cross-domain surgical hypothesis generation. It mines arXiv concepts, links them with surgical PubMed concepts, calculates semantic similarity, and ranks bridge hypotheses using LLM-based prioritization.

# BHI2.0

**Bridge-Based Hypothesis Intelligence 2.0** is a Python-based research pipeline for cross-domain hypothesis generation in surgery. The project explores whether technical, physical, computational, robotic, material-science, and systems-theory concepts from external scientific literature can be linked to surgical concepts in order to generate mechanistically plausible research hypotheses.

The repository contains the **arXiv-side external concept mining pipeline** and the **cross-domain bridge ranking pipeline** used to generate the first Top-1000 ranked hypothesis table based on a defined cosine-similarity window. It is intended as a methodological feasibility framework for literature-based discovery and concept-driven biomedical innovation.

Importantly, this repository does **not** contain the original PubMed mining code used to create the surgical concept vocabulary. The PubMed-derived surgical concept files are treated as pre-existing inputs. This repository starts from those existing surgical concept tables and focuses on mining external arXiv concepts, embedding concepts, constructing cross-domain candidate pairs, filtering candidates, and ranking bridge hypotheses with an LLM.

---

## Project rationale

Biomedical innovation often emerges when concepts from one field are transferred into another. In surgery, many major advances have historically depended on engineering, imaging, materials science, robotics, sensors, optics, and computational modeling. However, the search space for such cross-domain connections is enormous.

BHI2.0 addresses this problem by reducing large literature corpora into structured concept vocabularies. These vocabularies are embedded into a semantic vector space and then compared across domains. The aim is not to use a language model to invent hypotheses from scratch. Instead, the pipeline first creates candidate bridges through structured computational steps and then uses an LLM as a prioritization layer.

The core idea is:

```text
surgical concept + external technical concept → candidate bridge hypothesis
```

The first main run in this repository focuses on a **defined FAR** semantic-distance window. This was the first Top-1000 ranking generated before later ULTRANEAR, ULTRAFAR, or isolated-year experimental variants were explored.

---

## What this repository contains

This repository contains the code and documentation for the external arXiv and cross-domain ranking part of the project.

Included components:

* arXiv abstract mining
* extraction of three transferable external concepts per abstract
* concept normalization and filtering
* embedding of arXiv concepts
* cleaning and harmonization of surgical and arXiv vocabularies
* reconstruction of a full surgical vocabulary from pre-existing surgical concept rows
* generation of cross-domain candidate pairs
* filtering of candidate pairs into a defined cosine-similarity window
* LLM-based ranking of candidate bridge hypotheses
* optional export of ranked results to Excel

Excluded components:

* original PubMed mining scripts
* original PubMed abstract retrieval pipeline
* raw PubMed datasets
* raw arXiv datasets
* large processed CSV files
* embedding matrices
* generated output rankings
* API keys and environment files

The repository is therefore not a full end-to-end PubMed-to-hypothesis pipeline. It is the arXiv-to-cross-domain-ranking component that depends on precomputed surgical concept inputs.

---

## Repository structure

```text
BHI2.0/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── pipeline_overview.md
│   └── first_top1000_defined_far_run.md
│
├── scripts/
│   ├── 01_arxiv_external_mining.py
│   ├── 02_extract_arxiv_external_concepts.py
│   ├── 12_embed_arxiv_concepts.py
│   ├── 13_clean_surg_arxiv.py
│   ├── 13a_rebuild_full_surgery_vocab.py
│   ├── 14_generate_far_crossdomain_candidate_pairs.py
│   ├── 14b_filter_far_candidate_pairs.py
│   ├── 15_llm_rank_crossdomain_bridges.py
│   ├── 16_convert_FULL_ranked_csvs_to_excel.py
│   └── 16_convert_ranked_csv_to_excel.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── processed_arxiv_external/
│   └── processed_crossdomain/
│
├── outputs/
│   ├── rankings/
│   ├── figures/
│   └── excel_exports/
│
└── archive_excluded/
```

The `data/` and `outputs/` folders are intentionally ignored by Git except for placeholder files. This prevents large research files, embeddings, generated rankings, and intermediate CSV files from being committed accidentally.

---

## Core pipeline

The first defined-FAR Top-1000 ranking was generated with the following core script sequence:

```text
01 → 02 → 12 → 13 → 13a → 14 → 14b → 15 → optional 16
```

### Step 01: arXiv abstract mining

```text
scripts/01_arxiv_external_mining.py
```

Downloads external scientific abstracts from arXiv using predefined thematic modules. These modules target non-biomedical scientific areas that may contain transferable mechanisms, such as self-organization, topological transport, active matter, morphological computation, failure tolerance, rare-event prediction, and dynamical systems.

Typical outputs:

```text
data_raw_arxiv_external/all_modules_combined_PROGRESS.csv
data_raw_arxiv_external/all_modules_combined.csv
```

### Step 02: external concept extraction

```text
scripts/02_extract_arxiv_external_concepts.py
```

Uses an OpenAI language model to extract exactly three reusable scientific concepts from each arXiv abstract. The focus is on transferable mechanisms, structures, control principles, material principles, sensing principles, computational principles, and dynamical-system concepts.

Typical output:

```text
data_processed_arxiv_external/arxiv_external_concepts_3_per_abstract.csv
```

### Step 12: arXiv concept embedding

```text
scripts/12_embed_arxiv_concepts.py
```

Cleans the extracted arXiv concepts, removes generic or low-value concepts, summarizes unique concept nodes, and generates semantic embeddings using an OpenAI embedding model.

Typical outputs:

```text
data_processed_arxiv_external/arxiv_external_concept_nodes_summary.csv
data_processed_arxiv_external/arxiv_external_concept_embeddings.csv
```

### Step 13: cross-domain vocabulary cleaning

```text
scripts/13_clean_surg_arxiv.py
```

Loads pre-existing surgical concept nodes and embeddings together with arXiv concept nodes and embeddings. It applies domain-specific cleaning rules to remove overly generic, low-value, or non-informative terms. It then produces cleaned surgical and arXiv vocabularies for cross-domain comparison.

Typical inputs:

```text
data_processed/concept_nodes_summary.csv
data_processed/concept_embeddings.csv
data_processed_arxiv_external/arxiv_external_concept_nodes_summary.csv
data_processed_arxiv_external/arxiv_external_concept_embeddings.csv
```

Typical outputs:

```text
data_processed_crossdomain/surgery_concepts_clean.csv
data_processed_crossdomain/arxiv_concepts_clean.csv
data_processed_crossdomain/surgery_embeddings_clean.csv
data_processed_crossdomain/arxiv_embeddings_clean.csv
```

### Step 13a: full surgical vocabulary reconstruction

```text
scripts/13a_rebuild_full_surgery_vocab.py
```

Rebuilds a more complete surgical vocabulary from pre-existing surgical concept rows. This step is needed because the cross-domain bridge generation depends on a sufficiently broad surgical concept space.

Typical input:

```text
data_processed/concepts_2_per_abstract.csv
data_processed/concept_embeddings.csv
```

Typical outputs:

```text
data_processed_crossdomain/surgery_concepts_full_nodes.csv
data_processed_crossdomain/surgery_concepts_full_clean.csv
data_processed_crossdomain/surgery_embeddings_full_clean.csv
```

### Step 14: defined-FAR candidate generation

```text
scripts/14_generate_far_crossdomain_candidate_pairs.py
```

Generates cross-domain candidate pairs between cleaned surgical concepts and cleaned external arXiv concepts. Candidate pairs are selected according to semantic distance and cosine similarity. This step creates the candidate universe for the first defined-FAR hypothesis ranking.

Typical outputs include cross-domain candidate pair tables in the processed cross-domain folder.

### Step 14b: candidate pair filtering

```text
scripts/14b_filter_far_candidate_pairs.py
```

Filters the generated candidate pairs to obtain a more focused set for LLM ranking. The intended first main run uses a defined cosine-similarity window rather than later ULTRANEAR or ULTRAFAR variants.

### Step 15: LLM ranking of bridge hypotheses

```text
scripts/15_llm_rank_crossdomain_bridges.py
```

Ranks candidate bridge hypotheses with an LLM. The model evaluates whether a surgical concept and an external technical concept form a plausible and potentially useful cross-domain research hypothesis.

The LLM is used as a ranking and prioritization component, not as the sole generator of ideas.

Typical output:

```text
llm_ranked_crossdomain_bridges_FULL_top.csv
```

### Step 16: optional Excel export

```text
scripts/16_convert_FULL_ranked_csvs_to_excel.py
scripts/16_convert_ranked_csv_to_excel.py
```

Converts ranked CSV outputs into Excel files for manual inspection, sharing, and downstream scientific review.

---

## Data requirements

This repository expects several input files that are not included in GitHub because they are large, generated, or derived from prior workflows.

Expected pre-existing surgical inputs include:

```text
data_processed/concept_nodes_summary.csv
data_processed/concept_embeddings.csv
data_processed/concepts_2_per_abstract.csv
```

These files originate from a separate PubMed-based surgical concept extraction workflow. That PubMed mining workflow is not part of this repository.

Expected arXiv-side generated files include:

```text
data_raw_arxiv_external/all_modules_combined_PROGRESS.csv
data_processed_arxiv_external/arxiv_external_concepts_3_per_abstract.csv
data_processed_arxiv_external/arxiv_external_concept_nodes_summary.csv
data_processed_arxiv_external/arxiv_external_concept_embeddings.csv
```

Generated cross-domain outputs are written to folders such as:

```text
data_processed_crossdomain/
outputs/
```

These outputs are ignored by Git.

---

## Environment variables

The concept extraction, embedding generation, and LLM ranking scripts require access to the OpenAI API.

Set the API key locally as an environment variable.

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

On macOS/Linux:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Never commit API keys to GitHub. `.env` files and key files are ignored by `.gitignore`.

---

## Installation

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

A minimal `requirements.txt` should include packages such as:

```text
pandas
numpy
tqdm
requests
openai
scikit-learn
openpyxl
```

Depending on future visualization or graph extensions, additional packages may be required.

---

## Reproducibility notes

This repository documents the computational logic of the first defined-FAR Top-1000 bridge ranking. Full reproducibility requires access to the same input concept tables, embeddings, and OpenAI model settings used during the original run.

Potential sources of variation include:

* updated arXiv search results
* changes in external API behavior
* differences in OpenAI model versions
* non-identical input concept vocabularies
* changes in filtering thresholds
* missing intermediate CSV files

For this reason, the repository should be understood as a transparent methodological codebase rather than a frozen data archive.

---

## What BHI2.0 is not

BHI2.0 is not a clinical decision support system.

It does not provide medical recommendations.

It does not validate whether a generated hypothesis is true.

It does not replace expert review, systematic literature search, experimental testing, or clinical validation.

It is a research pipeline for hypothesis prioritization and cross-domain exploration.

---

## Intended use

BHI2.0 may be useful for:

* computational hypothesis generation
* literature-based discovery
* surgical innovation research
* identifying translational engineering concepts
* prioritizing unconventional research directions
* exploring concept-level links between biomedical and technical literature
* preparing candidate ideas for expert review

The final interpretation of any generated bridge hypothesis requires domain expertise.

---

## License

No license has currently been selected. Until a license is added, all rights are reserved by the repository owner by default. Users should not assume permission to reuse, redistribute, or modify the code outside the repository without explicit permission.

---

## Citation

A formal citation will be added if this project is published or archived with a DOI.

For now, please cite the repository URL if referencing this codebase informally.

---

## Status

This repository is under active development.

Current focus:

```text
defined-FAR Top-1000 cross-domain bridge ranking
```

Later experimental branches such as ULTRANEAR, ULTRAFAR, isolated-year ranking, and graph visualization are not part of the initial core pipeline documented here.

