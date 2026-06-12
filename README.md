# BHI2.0

BHI2.0 is a research codebase for cross-domain hypothesis generation in surgery. The central idea of the project is to connect surgical problems with technical, physical, computational, robotic, material-science and systems-theory concepts from external scientific literature. The aim is not to generate medical claims or clinical recommendations, but to create a structured pipeline that can help identify unusual and potentially productive conceptual bridges for future research.

The project was developed as part of a broader attempt to make surgical hypothesis generation more systematic. In many areas of surgery, innovation depends on ideas that originally come from outside medicine: imaging physics, sensor technology, materials science, robotics, fluid mechanics, control theory, machine learning, optics, or engineering design. Searching for such links manually is difficult because the possible combinations between surgical concepts and external scientific principles grow extremely quickly. BHI2.0 tries to reduce this search space by turning scientific abstracts into reusable concepts, embedding those concepts into a semantic vector space, and then ranking possible cross-domain bridges.

This repository contains the arXiv-side external concept mining workflow and the cross-domain ranking workflow used for the first Top-1000 bridge-hypothesis table based on a defined cosine-similarity window. This first ranking is referred to as the defined-FAR run. It should be distinguished from later experimental variants such as ULTRANEAR, ULTRAFAR, isolated-year runs, or additional graph-visualization workflows. Those later experiments are not the core focus of this repository.

A key point is that the original PubMed mining pipeline is not included here. The surgical concept vocabulary was generated in a separate workflow from PubMed-derived surgical abstracts. In BHI2.0, these surgical concept files are treated as existing input files. Therefore, this repository does not represent a complete PubMed-to-hypothesis pipeline. It starts after the surgical concept vocabulary has already been created and focuses on mining external arXiv concepts, embedding them, harmonizing them with the surgical vocabulary, generating candidate pairs, filtering the candidate space, and ranking the resulting bridge hypotheses.

The computational workflow begins with arXiv abstract mining. The script `01_arxiv_external_mining.py` searches arXiv for abstracts from selected non-biomedical scientific areas that may contain transferable ideas. These areas include, for example, self-organization, complex systems, topological transport, active matter, morphological computation, failure tolerance, rare-event prediction and related fields. The purpose of this step is to build an external scientific concept source that is not already limited to surgery or biomedicine.

The second step, implemented in `02_extract_arxiv_external_concepts.py`, uses a language model to extract three reusable scientific concepts from each arXiv abstract. These concepts are intended to describe mechanisms, design principles, material properties, control strategies, sensing principles, dynamical-system ideas or computational structures. The script is not meant to reinterpret arXiv papers medically. It extracts external concepts first, and the cross-domain connection to surgery is only introduced later.

The script `12_embed_arxiv_concepts.py` then cleans and summarizes these extracted concepts and creates semantic embeddings. These embeddings allow concepts to be compared computationally. In parallel, pre-existing surgical concept nodes and surgical concept embeddings are loaded from the earlier PubMed-derived workflow. The cleaning and harmonization step is handled by `13_clean_surg_arxiv.py`, which removes overly generic or low-value terms and prepares both vocabularies for cross-domain comparison.

Because the surgical vocabulary needed to be sufficiently broad for bridge generation, `13a_rebuild_full_surgery_vocab.py` reconstructs a fuller surgical concept vocabulary from the pre-existing surgical concept rows. This is still based on already generated PubMed-derived surgical concept data; the PubMed mining code itself is not part of this repository.

The bridge-generation part of the workflow is then performed by `14_generate_far_crossdomain_candidate_pairs.py` and `14b_filter_far_candidate_pairs.py`. These scripts create and filter candidate pairs between surgical concepts and external arXiv concepts. The first main ranking run used a defined cosine-similarity window. This means the pipeline did not simply choose the closest concepts, nor did it only focus on extremely distant pairs. Instead, it aimed to identify concept pairs that were semantically far enough to be interesting but not so far that a plausible transfer became unlikely.

The resulting candidate bridges are ranked in `15_llm_rank_crossdomain_bridges.py`. In this step, the language model acts as a prioritization layer. It evaluates whether a candidate bridge appears novel, plausible, mechanistically meaningful and relevant enough to deserve manual scientific review. The model is therefore not the sole generator of the hypotheses. The candidate space is first constructed by earlier computational steps, and the LLM is then used to help rank and interpret that space.

The optional conversion scripts, including `16_convert_FULL_ranked_csvs_to_excel.py` and `16_convert_ranked_csv_to_excel.py`, convert ranked CSV outputs into Excel files for manual inspection and downstream review.

The core script sequence for the first defined-FAR Top-1000 run is:

```text
01 → 02 → 12 → 13 → 13a → 14 → 14b → 15 → optional 16
```

The repository is organized around code and documentation rather than raw data. Large CSV files, embeddings, raw abstract exports and generated ranking outputs are intentionally not tracked in Git. This keeps the repository usable and prevents accidental upload of large intermediate files or API-derived outputs. The expected data folders are present only as structure placeholders. Users who want to reproduce the full workflow need to provide the corresponding input files locally.

The most important expected pre-existing surgical inputs are files such as `concept_nodes_summary.csv`, `concept_embeddings.csv` and `concepts_2_per_abstract.csv`. These originate from the separate PubMed-based surgical concept extraction workflow. On the arXiv side, the pipeline creates files such as `all_modules_combined_PROGRESS.csv`, `arxiv_external_concepts_3_per_abstract.csv`, `arxiv_external_concept_nodes_summary.csv` and `arxiv_external_concept_embeddings.csv`. Cross-domain outputs are written into processed output folders and are ignored by Git by default.

Several scripts require access to the OpenAI API for concept extraction, embedding generation or LLM-based ranking. The API key should be provided locally as an environment variable and must never be committed to GitHub. On Windows PowerShell, this can be done with:

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

A minimal Python environment should include packages such as `pandas`, `numpy`, `tqdm`, `requests`, `openai`, `scikit-learn` and `openpyxl`. Additional packages may be needed for later visualization or graph-based extensions.

BHI2.0 should be understood as a methodological and exploratory research pipeline. It is not a clinical decision-support system, does not validate the truth of generated hypotheses, and does not provide medical recommendations. Any hypothesis generated by the workflow requires expert review, literature checking, experimental validation and, where relevant, clinical testing.

The long-term purpose of the project is to investigate whether concept-based literature mining can help make surgical innovation more systematic. Instead of relying only on chance encounters between disciplines, BHI2.0 tries to create a computational framework that proposes candidate bridges between surgery and external scientific domains. These candidate bridges can then be reviewed by researchers and potentially developed into retrospective studies, systematic reviews, translational experiments, device concepts or interdisciplinary collaborations.

## Citation: citation will be added when available, ## Licence as well

## AI disclosure

Parts of this repository were developed with the assistance of AI tools. AI was used to support code adaptation, code refactoring, documentation drafting, pipeline explanation and organization of the repository structure. The scientific direction, project framing, selection of the surgical research problem, interpretation of outputs and final responsibility for the repository remain with the repository owner. AI-generated or AI-assisted code and documentation should be reviewed carefully before reuse, extension or publication.


