import requests
import csv
import time
import re
from pathlib import Path
from xml.etree import ElementTree


# ============================================================
# SETTINGS
# ============================================================

OUTPUT_DIR = Path("data_raw_arxiv_external_TEST")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://export.arxiv.org/api/query"

# arXiv: langsam abfragen, sonst Fehler/Block.
REQUEST_DELAY = 3.2

# Testlauf: 100 Abstracts aus dem ersten Modul
BATCH_SIZE = 25
TEST_MODE = True
TEST_MAX_RESULTS_PER_MODULE = 100

DATE_START = "202001010000"
DATE_END = "202612312359"

SORT_BY = "submittedDate"
SORT_ORDER = "descending"

COMBINED_PROGRESS_FILE = OUTPUT_DIR / "all_modules_combined_PROGRESS.csv"
COMBINED_FINAL_FILE = OUTPUT_DIR / "all_modules_combined.csv"


# ============================================================
# TEST MODULES
# ============================================================

MODULES = {
    "self_organization_emergence_complex_systems": {
        "terms": [
            "self organization",
            "emergent behavior",
            "complex adaptive systems",
            "collective dynamics",
            "pattern formation",
            "self organized criticality",
            "autonomous organization",
            "adaptive self organization",
        ],
        "categories": [
            "nlin.AO",
            "cond-mat.soft",
            "physics.comp-ph",
            "cs.MA",
            "eess.SY",
        ],
    },

    "topological_physics_robust_transport": {
        "terms": [
            "topological protection",
            "topological transport",
            "topological defects",
            "topological metamaterials",
            "topological mechanics",
            "protected transport",
            "robust transport",
        ],
        "categories": [
            "cond-mat.mes-hall",
            "cond-mat.mtrl-sci",
            "cond-mat.soft",
            "physics.app-ph",
            "physics.optics",
            "nlin.PS",
        ],
    },

    "morphological_computation_embodied_intelligence": {
        "terms": [
            "morphological computation",
            "embodied intelligence",
            "physical reservoir computing",
            "material computation",
            "mechanical computation",
            "in materio computing",
            "physical intelligence",
        ],
        "categories": [
            "cs.RO",
            "cs.AI",
            "cs.ET",
            "eess.SY",
            "cond-mat.soft",
            "physics.app-ph",
        ],
    },

    "programmable_matter_active_matter": {
        "terms": [
            "programmable matter",
            "active matter",
            "active particles",
            "active colloids",
            "self propelled particles",
            "collective motion",
            "active fluids",
            "robotic matter",
        ],
        "categories": [
            "cond-mat.soft",
            "physics.bio-ph",
            "physics.flu-dyn",
            "cs.RO",
            "nlin.AO",
        ],
    },

    "defect_engineering_failure_tolerance": {
        "terms": [
            "defect engineering",
            "defect tolerance",
            "damage tolerance",
            "fault tolerance",
            "graceful degradation",
            "self repair",
            "self healing systems",
            "failure resilience",
            "adaptive repair",
            "damage localization",
            "crack arrest",
            "fracture control",
        ],
        "categories": [
            "cond-mat.mtrl-sci",
            "cond-mat.soft",
            "physics.app-ph",
            "cs.ET",
            "eess.SY",
            "cs.RO",
        ],
    },

    "rare_event_prediction_extreme_value_systems": {
        "terms": [
            "rare event prediction",
            "extreme event prediction",
            "extreme value theory",
            "catastrophic failure",
            "failure precursors",
            "early warning signals",
            "critical transition",
            "tipping point",
            "regime shift",
            "precursor detection",
            "tail risk",
        ],
        "categories": [
            "physics.data-an",
            "cs.LG",
            "eess.SP",
            "nlin.AO",
            "stat.ML",
            "eess.SY",
        ],
    },
}

# Erstmal nur EIN Modul testen.
# Danach kannst du mehrere hinzufügen.
TEST_MODULES_ONLY = [
    "self_organization_emergence_complex_systems",
]


# ============================================================
# XML NAMESPACES
# ============================================================

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


# ============================================================
# HELPERS
# ============================================================

def normalize_space(x):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def phrase_to_arxiv_all(term):
    """
    Robust arXiv all-field phrase query.
    Hyphens are replaced because arXiv query parser can be fragile.
    """
    term = normalize_space(term)
    term = term.replace("-", " ")
    return f'all:"{term}"'


def build_query(module_spec):
    """
    Builds robust arXiv query:
    ((all:"term1" OR all:"term2") AND (cat:x OR cat:y) AND submittedDate:[... TO ...])
    """
    term_query = " OR ".join(
        phrase_to_arxiv_all(t) for t in module_spec["terms"]
    )

    cat_query = " OR ".join(
        f"cat:{c}" for c in module_spec["categories"]
    )

    date_filter = f"submittedDate:[{DATE_START} TO {DATE_END}]"

    return f"(({term_query}) AND ({cat_query}) AND {date_filter})"


def extract_arxiv_id(entry_id):
    if not entry_id:
        return ""
    return entry_id.rstrip("/").split("/")[-1]


def strip_version(arxiv_id):
    return re.sub(r"v\d+$", "", arxiv_id)


def get_text(parent, tag):
    el = parent.find(tag)
    if el is None or el.text is None:
        return ""
    return normalize_space(el.text)


def parse_entry(entry, module_name):
    entry_id = get_text(entry, ATOM_NS + "id")
    arxiv_id_versioned = extract_arxiv_id(entry_id)
    arxiv_id = strip_version(arxiv_id_versioned)

    title = get_text(entry, ATOM_NS + "title")
    abstract = get_text(entry, ATOM_NS + "summary")
    published = get_text(entry, ATOM_NS + "published")
    updated = get_text(entry, ATOM_NS + "updated")

    year = published[:4] if published else ""

    authors = []
    for author in entry.findall(ATOM_NS + "author"):
        name = get_text(author, ATOM_NS + "name")
        if name:
            authors.append(name)

    primary_category = ""
    primary_el = entry.find(ARXIV_NS + "primary_category")
    if primary_el is not None:
        primary_category = primary_el.attrib.get("term", "")

    categories = []
    for cat in entry.findall(ATOM_NS + "category"):
        term = cat.attrib.get("term", "")
        if term:
            categories.append(term)

    doi = get_text(entry, ARXIV_NS + "doi")
    journal_ref = get_text(entry, ARXIV_NS + "journal_ref")
    comment = get_text(entry, ARXIV_NS + "comment")

    abs_url = ""
    pdf_url = ""

    for link in entry.findall(ATOM_NS + "link"):
        href = link.attrib.get("href", "")
        rel = link.attrib.get("rel", "")
        title_attr = link.attrib.get("title", "")
        link_type = link.attrib.get("type", "")

        if rel == "alternate":
            abs_url = href
        if title_attr == "pdf" or link_type == "application/pdf":
            pdf_url = href

    return {
        "module": module_name,
        "arxiv_id": arxiv_id,
        "arxiv_id_versioned": arxiv_id_versioned,
        "entry_id": entry_id,
        "published": published,
        "updated": updated,
        "year": year,
        "title": title,
        "abstract": abstract,
        "authors": "; ".join(authors),
        "n_authors": len(authors),
        "primary_category": primary_category,
        "categories": "; ".join(sorted(set(categories))),
        "doi": doi,
        "journal_ref": journal_ref,
        "comment": comment,
        "abs_url": abs_url,
        "pdf_url": pdf_url,
    }


def safe_request(params, retries=5):
    for attempt in range(1, retries + 1):
        time.sleep(REQUEST_DELAY)

        try:
            response = requests.get(BASE_URL, params=params, timeout=90)

            if response.status_code in [429, 500, 502, 503, 504]:
                wait = REQUEST_DELAY * attempt * 2
                print(
                    f"  Warning: HTTP {response.status_code}, retry {attempt}/{retries}, sleeping {wait:.1f}s",
                    flush=True
                )
                print("  URL was:", response.url[:1000], flush=True)
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.content

        except Exception as e:
            wait = REQUEST_DELAY * attempt * 2
            print(
                f"  Warning: request failed attempt {attempt}/{retries}: {repr(e)} | sleeping {wait:.1f}s",
                flush=True
            )
            time.sleep(wait)

    raise RuntimeError("Request failed after multiple retries.")


def get_total_results(query):
    """
    max_results=1 instead of 0, because 0 can be fragile.
    """
    params = {
        "search_query": query,
        "start": 0,
        "max_results": 1,
        "sortBy": SORT_BY,
        "sortOrder": SORT_ORDER,
    }

    xml_bytes = safe_request(params)
    root = ElementTree.fromstring(xml_bytes)

    total_el = root.find(OPENSEARCH_NS + "totalResults")
    if total_el is None or total_el.text is None:
        return 0

    return int(total_el.text)


def fetch_batch(query, start, max_results):
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": SORT_BY,
        "sortOrder": SORT_ORDER,
    }

    xml_bytes = safe_request(params)
    root = ElementTree.fromstring(xml_bytes)

    entries = root.findall(ATOM_NS + "entry")
    return entries


def save_csv(rows, filepath):
    fieldnames = [
        "module",
        "arxiv_id",
        "arxiv_id_versioned",
        "entry_id",
        "published",
        "updated",
        "year",
        "title",
        "abstract",
        "authors",
        "n_authors",
        "primary_category",
        "categories",
        "doi",
        "journal_ref",
        "comment",
        "abs_url",
        "pdf_url",
    ]

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {filepath} | rows: {len(rows)}", flush=True)


# ============================================================
# MAIN
# ============================================================

def main():
    all_rows = []
    seen_arxiv_ids = set()

    print("=" * 80)
    print("arXiv external abstract mining TEST RUN")
    print("=" * 80)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Date range submittedDate: {DATE_START} to {DATE_END}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Request delay: {REQUEST_DELAY}s")
    print(f"Test mode: {TEST_MODE}")
    print(f"Test max results per module: {TEST_MAX_RESULTS_PER_MODULE}")
    print(f"Test modules only: {TEST_MODULES_ONLY}")
    print("=" * 80)

    for module_name, module_spec in MODULES.items():

        if TEST_MODE and module_name not in TEST_MODULES_ONLY:
            continue

        print("\n" + "=" * 80)
        print(f"MODULE: {module_name}")
        print("=" * 80)

        query = build_query(module_spec)
        print("Query preview:")
        print(query[:1500])
        print("-" * 80)

        try:
            total = get_total_results(query)
        except Exception as e:
            print(f"Could not get total results for {module_name}: {repr(e)}")
            continue

        target = min(total, TEST_MAX_RESULTS_PER_MODULE)

        print(f"Total hits according to arXiv: {total}")
        print(f"Target download for module: {target}")

        if target == 0:
            print("No hits for this module.")
            continue

        module_rows = []

        for start in range(0, target, BATCH_SIZE):
            current_batch_size = min(BATCH_SIZE, target - start)

            print(
                f"  Fetching {start + 1}-{start + current_batch_size} / {target}",
                flush=True
            )

            try:
                entries = fetch_batch(
                    query,
                    start=start,
                    max_results=current_batch_size
                )
            except Exception as e:
                print(f"  Batch failed at start={start}: {repr(e)}")
                continue

            if not entries:
                print("  No entries returned. Stopping this module.")
                break

            for entry in entries:
                row = parse_entry(entry, module_name)

                if not row["title"] or not row["abstract"]:
                    continue

                module_rows.append(row)

                if row["arxiv_id"] not in seen_arxiv_ids:
                    all_rows.append(row)
                    seen_arxiv_ids.add(row["arxiv_id"])

        module_file = OUTPUT_DIR / f"{module_name}.csv"
        save_csv(module_rows, module_file)

        save_csv(all_rows, COMBINED_PROGRESS_FILE)

    print("\n" + "=" * 80)
    print("FINAL COMBINED FILE")
    print("=" * 80)

    save_csv(all_rows, COMBINED_FINAL_FILE)

    print("\nDone.")
    print(f"Unique arXiv abstracts total: {len(all_rows)}")
    print(f"Final file: {COMBINED_FINAL_FILE}")


if __name__ == "__main__":
    main()