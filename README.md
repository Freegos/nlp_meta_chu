# nlp_meta_chu

Corpus PubMed des publications françaises (au moins une affiliation « France »), construit pour des analyses NLP / bibliométriques sur les CHU. La feuille de route du projet est dans [`docs/feuille_de_route.md`](docs/feuille_de_route.md).

## Installation

```bash
uv sync                # ou : pip install biopython
cp .env.example .env   # puis renseigner NCBI_EMAIL (obligatoire) et NCBI_API_KEY (optionnelle)
```

La clé API se génère sur https://account.ncbi.nlm.nih.gov/ (Account settings → API Key Management). Avec clé : ~9 requêtes/s, sans : ~3.

## Récupérer les données

```bash
python scrapper/pubmed_scraper.py                                # tout PubMed France
python scrapper/pubmed_scraper.py --from-year 1985 --to-year 1985
python scrapper/pubmed_scraper.py --dry-run                      # compte sans télécharger
```

- Recherche découpée par année (puis mois / jour si > 9 999 résultats, limite d'esearch).
- Sortie : un fichier par année `data/pubmed_france/AAAA.jsonl` (+ `avant_1950.jsonl`). Les PMIDs sont mis en cache dans `data/pmids/`.
- **Reprise** : si le script est interrompu, il suffit de le relancer ; seules les notices manquantes sont téléchargées (par lots de 200).

Champs d'une notice : `pmid`, `title`, `abstract`, `journal`, `year`, `pub_date`, `language`, `pub_types`, `doi`, `mesh_terms`, `authors` (nom, prénom, affiliations), `n_authors`.

## Charger le corpus

```python
from scrapper.load_corpus import load_corpus

records = load_corpus()            # toutes les années
records = load_corpus(2000, 2010) # bornes incluses
```

## Notes

- `data/` et `.env` ne sont pas versionnés.
- `scrapper/split_corpus.py` (échantillons / splits train-val-test) lit encore un seul JSONL et n'a pas été adapté au nouveau format par année.
