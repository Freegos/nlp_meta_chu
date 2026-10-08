"""
load_corpus.py

Chargement du corpus PubMed France (un JSONL par annee) en un seul appel :

    from scrapper.load_corpus import load_corpus
    records = load_corpus()                    # list[dict], toutes annees
    records = load_corpus(2000, 2010)          # bornes d'annees (incluses)
"""

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "pubmed_france"


def load_corpus(from_year=None, to_year=None, corpus_dir=CORPUS_DIR):
    """Retourne la liste de toutes les notices des fichiers AAAA.jsonl
    (et avant_1950.jsonl), sans doublon de PMID. Lignes corrompues ignorees."""
    records, seen = [], set()
    for path in sorted(Path(corpus_dir).glob("*.jsonl")):
        stem = path.stem
        year = int(stem) if stem.isdigit() else 0   # "avant_1950" -> 0
        if (from_year and year < from_year) or (to_year and year > to_year):
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec["pmid"] not in seen:
                    seen.add(rec["pmid"])
                    records.append(rec)
    return records
