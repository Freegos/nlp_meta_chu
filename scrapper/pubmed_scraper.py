"""
pubmed_scraper.py

Scrape toutes les notices PubMed dont au moins une affiliation contient
"France", depuis le debut de PubMed, via l'API NCBI E-utilities
(esearch + efetch). Sauvegarde incrementale (un JSONL par annee) et reprise
automatique en cas d'interruption : il suffit de relancer le script.

Installation :
    pip install biopython

Configuration : fichier .env a la racine du projet (voir .env.example)
    NCBI_EMAIL     email (obligatoire, exige par les conditions d'utilisation NCBI)
    NCBI_API_KEY   cle API NCBI (optionnelle, ~9 req/s au lieu de ~3 req/s)
Les variables deja definies dans l'environnement ont la priorite.

Usage :
    python pubmed_scraper.py                       # tout
    python pubmed_scraper.py --from-year 1985 --to-year 1985
    python pubmed_scraper.py --dry-run             # compte seulement, sans telecharger
"""

import argparse
import calendar
import json
import os
import time
from datetime import date
from pathlib import Path

from Bio import Entrez

# ============================== CONFIG ==============================

def load_env(path):
    """Lit un fichier .env (lignes CLE=valeur) sans ecraser l'environnement."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env(Path(__file__).resolve().parent.parent / ".env")

# "or None" : une valeur vide dans .env equivaut a "non definie"
Entrez.email = os.environ.get("NCBI_EMAIL") or None
Entrez.api_key = os.environ.get("NCBI_API_KEY") or None
Entrez.tool = "pubmed-france-nlp-project"

# NCBI autorise 3 requetes/s sans cle API, 10 requetes/s avec une cle.
REQUEST_DELAY = 0.34 if Entrez.api_key is None else 0.11

AFFIL_QUERY = '"France"[Affiliation]'

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "pubmed_france"
PMID_DIR = DATA_DIR / "pmids"

FIRST_YEAR = 1950          # tout ce qui precede est regroupe dans une seule fenetre
EARLIEST_DATE = date(1700, 1, 1)
ESEARCH_LIMIT = 9999       # esearch ne renvoie jamais plus de 10 000 IDs
BATCH_SIZE = 200
MAX_RETRIES = 5

# ============================== RECHERCHE ==============================

def _esearch(start, end, retmax):
    """esearch sur la fenetre [start, end] (dates) selon la date de publication."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            handle = Entrez.esearch(
                db="pubmed", term=AFFIL_QUERY, datetype="pdat",
                mindate=start.strftime("%Y/%m/%d"),
                maxdate=end.strftime("%Y/%m/%d"),
                retmax=retmax,
            )
            record = Entrez.read(handle)
            handle.close()
            time.sleep(REQUEST_DELAY)
            return record
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"[!] esearch erreur ({e}), nouvel essai dans {wait}s ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


def _subwindows(start, end):
    """Decoupe une fenetre en annees, mois ou jours selon sa taille."""
    if (end - start).days > 366:
        return [(date(y, 1, 1), date(y, 12, 31))
                for y in range(start.year, end.year + 1)]
    if (end - start).days > 31:
        out = []
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            last = calendar.monthrange(y, m)[1]
            out.append((date(y, m, 1), date(y, m, last)))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return out
    return [(date.fromordinal(o), date.fromordinal(o))
            for o in range(start.toordinal(), end.toordinal() + 1)]


def search_window(start, end):
    """Retourne tous les PMIDs de la fenetre, en la sous-decoupant
    (annee -> mois -> jour) si elle depasse la limite d'esearch."""
    count = int(_esearch(start, end, retmax=0)["Count"])
    if count == 0:
        return []
    if count <= ESEARCH_LIMIT:
        return _esearch(start, end, retmax=ESEARCH_LIMIT + 1)["IdList"]
    if start == end:
        print(f"[!] {start} : {count} notices > limite, seules {ESEARCH_LIMIT + 1} recuperees.")
        return _esearch(start, end, retmax=ESEARCH_LIMIT + 1)["IdList"]
    print(f"[i] Fenetre {start}..{end} : {count} notices > limite, sous-decoupage...")
    pmids = {}
    for s, e in _subwindows(start, end):
        for p in search_window(s, e):
            pmids[p] = None
    return list(pmids)


def get_pmids(label, start, end, cacheable=True):
    """PMIDs d'une fenetre, avec cache local (data/pmids/<label>.txt)."""
    cache = PMID_DIR / f"{label}.txt"
    if cacheable and cache.exists():
        return cache.read_text().split()
    pmids = search_window(start, end)
    if cacheable:
        PMID_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text("\n".join(pmids))
        tmp.replace(cache)   # ecriture atomique
    return pmids

# ============================== TELECHARGEMENT ==============================

def already_done_pmids(output_file):
    """Relit le fichier de sortie existant pour connaitre les PMIDs deja
    traites (permet la reprise apres interruption)."""
    done = set()
    if output_file.exists():
        with output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(str(json.loads(line)["pmid"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def fetch_batch(pmid_batch):
    """Recupere un lot de notices completes en XML, avec retry exponentiel
    en cas d'erreur reseau."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmid_batch),
                                   rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            return records.get("PubmedArticle", [])
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[x] Echec definitif pour ce lot apres {MAX_RETRIES} tentatives ({e}).")
                return []
            wait = 2 ** attempt
            print(f"[!] Erreur ({e}), nouvelle tentative dans {wait}s ({attempt}/{MAX_RETRIES})...")
            time.sleep(wait)


def _doi(article, art):
    for e in art.get("ELocationID", []):
        if e.attributes.get("EIdType") == "doi":
            return str(e)
    ids = article.get("PubmedData", {}).get("ArticleIdList", [])
    for i in ids:
        if i.attributes.get("IdType") == "doi":
            return str(i)
    return ""


def parse_article(article):
    """Extrait les champs utiles d'une notice PubMed (structure retournee
    par Entrez.read en mode XML)."""
    medline = article["MedlineCitation"]
    pmid = str(medline["PMID"])
    art = medline["Article"]

    title = str(art.get("ArticleTitle", ""))

    abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
    abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else ""

    journal = str(art.get("Journal", {}).get("Title", ""))

    pub_date = {}
    try:
        pub_date = art["Journal"]["JournalIssue"]["PubDate"]
    except KeyError:
        pass
    year = str(pub_date.get("Year") or pub_date.get("MedlineDate", ""))
    full_date = "-".join(str(pub_date[k]) for k in ("Year", "Month", "Day") if k in pub_date) \
        or str(pub_date.get("MedlineDate", ""))

    mesh_terms = []
    for mh in medline.get("MeshHeadingList", []):
        mesh_terms.append(str(mh["DescriptorName"]))

    authors = []
    for au in art.get("AuthorList", []):
        last = str(au.get("LastName", ""))
        fore = str(au.get("ForeName", ""))
        collective = str(au.get("CollectiveName", ""))
        affiliations = [
            str(aff.get("Affiliation", ""))
            for aff in au.get("AffiliationInfo", [])
        ]
        if last or fore or collective:
            authors.append({
                "last_name": last,
                "fore_name": fore,
                "collective_name": collective,
                "affiliations": affiliations,
            })

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "pub_date": full_date,
        "language": [str(l) for l in art.get("Language", [])],
        "pub_types": [str(t) for t in art.get("PublicationTypeList", [])],
        "doi": _doi(article, art),
        "mesh_terms": mesh_terms,
        "authors": authors,
        "n_authors": len(authors),
    }


def scrape_window(label, pmids):
    """Telecharge les PMIDs manquants d'une fenetre dans <label>.jsonl."""
    output_file = OUTPUT_DIR / f"{label}.jsonl"
    done = already_done_pmids(output_file)
    todo = [p for p in pmids if p not in done]
    print(f"[i] {label} : total {len(pmids)} | deja recuperes {len(done)} | restant {len(todo)}")
    if not todo:
        return

    n_batches = (len(todo) - 1) // BATCH_SIZE + 1
    with output_file.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            print(f"[i] {label} lot {i // BATCH_SIZE + 1}/{n_batches} ({len(batch)} notices)...")
            for article in fetch_batch(batch):
                try:
                    record = parse_article(article)
                except Exception as e:
                    print(f"[!] Erreur de parsing, notice ignoree : {e}")
                    continue
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # si le script coupe, ce lot est deja sauvegarde
            time.sleep(REQUEST_DELAY)

# ============================== MAIN ==============================

def build_windows(from_year, to_year):
    """(label, start, end) : une fenetre 'avant_1950', puis une par annee
    jusqu'a l'annee suivante (certaines notices ont une pdat future)."""
    windows = []
    if from_year < FIRST_YEAR:
        windows.append((f"avant_{FIRST_YEAR}", EARLIEST_DATE, date(FIRST_YEAR - 1, 12, 31)))
    for y in range(max(from_year, FIRST_YEAR), to_year + 1):
        windows.append((str(y), date(y, 1, 1), date(y, 12, 31)))
    return windows


def main():
    this_year = date.today().year
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=EARLIEST_DATE.year)
    ap.add_argument("--to-year", type=int, default=this_year + 1)
    ap.add_argument("--dry-run", action="store_true", help="compter seulement, sans telecharger")
    args = ap.parse_args()

    if not Entrez.email:
        print("[!] NCBI_EMAIL non defini (exige par NCBI). Definissez-le avant de lancer.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[i] Requete : {AFFIL_QUERY} | cle API : {'oui' if Entrez.api_key else 'non'}")

    total = 0
    for label, start, end in build_windows(args.from_year, args.to_year):
        # les annees en cours / futures changent : jamais mises en cache
        cacheable = end.year < this_year
        pmids = get_pmids(label, start, end, cacheable=cacheable)
        total += len(pmids)
        if args.dry_run:
            print(f"[dry-run] {label} : {len(pmids)} notices")
            continue
        scrape_window(label, pmids)

    print(f"[OK] Termine. {total} notices au total.")


if __name__ == "__main__":
    main()
