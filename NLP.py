# %% [markdown]
# # PubMed CHU de Rennes : évolution dans le temps + auteurs (genre, position)
# Version allégée : pas de spaCy, pas d'embeddings, pas de LDA/UMAP/HDBSCAN/RAG.
# Tourne en quelques dizaines de secondes.
#
# Installation :  uv add pandas scikit-learn matplotlib gender-guesser
# Lancer avec  :  uv run python NLP_simple.py     (graphiques dans figures/)
#
# Contenu :
#   1. Chargement / nettoyage / année
#   2. Volume de publications et taille des équipes par année
#   3. Thèmes (TF-IDF + NMF) et évolution dans le temps
#   4. Auteurs : prénom -> genre, place dans la liste (premier / milieu / dernier)
#   5. Évolution de la part des femmes par année et par place, et par thème

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # évite des plantages silencieux sur Mac

import faulthandler
import re
import warnings
from pathlib import Path

faulthandler.enable()                                    # si Python plante, affiche où
print("Démarrage du script...", flush=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = Path("data")        # dossier des .jsonl (un par an)
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

COL_ABSTRACT = "abstract"
COL_TITLE = "title"
COL_AUTHORS = None             # None = détection automatique (authors, author, ...)
COL_DATE = None                # None = détection automatique

N_TOPICS = 8
ANNEE_MIN, ANNEE_MAX = 1955, 2026
MIN_ART_ANNEE = 30             # années avec moins d'articles : ignorées dans les courbes
MIN_AUT_POINT = 30             # un point de courbe genre exige au moins ce nb d'auteurs identifiés
RANDOM_STATE = 0


def show(nom):
    chemin = FIG_DIR / f"{nom}.png"
    plt.savefig(chemin, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"  -> {chemin}")


# ----------------------------------------------------------------------------
# 1. Chargement et nettoyage
# ----------------------------------------------------------------------------
print("Dossier de travail :", Path.cwd(), flush=True)
fichiers = sorted(DATA_DIR.glob("*.jsonl"))
print(f"{len(fichiers)} fichier(s) .jsonl trouvé(s) dans {DATA_DIR.resolve()}", flush=True)
if not fichiers:
    raise FileNotFoundError(f"Aucun .jsonl dans {DATA_DIR.resolve()}")

morceaux = []
for f in fichiers:
    d = pd.read_json(f, lines=True)
    m = re.search(r"(?:19|20)\d{2}", f.stem)
    d["annee_fichier"] = int(m.group()) if m else np.nan
    morceaux.append(d)
    print(f"  {f.name} : {len(d)} articles", flush=True)
df = pd.concat(morceaux, ignore_index=True)
print(df.shape, df.columns.tolist())


def find_col(candidats):
    low = {c.lower(): c for c in df.columns}
    return next((low[c] for c in candidats if c in low), None)


COL_AUTHORS = COL_AUTHORS or find_col(["authors", "author", "author_list", "authorlist", "auteurs"])
COL_DATE = COL_DATE or find_col(["year", "pub_year", "publication_year", "pubdate", "publication_date", "date"])
print("Colonnes :", COL_ABSTRACT, COL_TITLE, COL_AUTHORS, COL_DATE)

df[COL_ABSTRACT] = df[COL_ABSTRACT].apply(lambda x: " ".join(map(str, x)) if isinstance(x, list) else x)
df = df.dropna(subset=[COL_ABSTRACT])
df = df[df[COL_ABSTRACT].str.strip().str.len() > 50]
df = df.drop_duplicates(subset=[COL_ABSTRACT]).reset_index(drop=True)

SECTIONS = (r"\b(?:BACKGROUND|OBJECTIVES?|AIMS?|PURPOSE|INTRODUCTION|DESIGN|SETTING|PATIENTS|"
            r"MATERIALS? AND METHODS|METHODS?|RESULTS?|CONCLUSIONS?|DISCUSSION)\s*:")
abstract_clean = df[COL_ABSTRACT].str.replace(SECTIONS, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
df["text"] = (df[COL_TITLE].fillna("").astype(str) + ". " + abstract_clean) if COL_TITLE else abstract_clean

# Année : colonne date si dispo, sinon année lue dans le nom du fichier
if COL_DATE:
    df["year"] = pd.to_numeric(df[COL_DATE].astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
else:
    df["year"] = np.nan
df["year"] = df["year"].fillna(df["annee_fichier"])

df = df[df["year"].between(ANNEE_MIN, ANNEE_MAX)].reset_index(drop=True)
effectifs = df["year"].value_counts()
annees_ok = sorted(effectifs[effectifs >= MIN_ART_ANNEE].index)
print(f"{len(df)} documents ; années retenues pour les courbes : {[int(a) for a in annees_ok]}")

# ----------------------------------------------------------------------------
# 4 (préparation). Parsing des auteurs -> liste (nom, prénom) dans l'ordre
# ----------------------------------------------------------------------------
def parse_auteurs(x):
    """Gère : liste de dicts (ForeName/LastName...), liste de chaînes, ou chaîne séparée par ';'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        x = [a for a in re.split(r";|\|", x) if a.strip()]
    if not isinstance(x, list):
        return []
    out = []
    for a in x:
        if isinstance(a, dict):
            low = {str(k).lower(): v for k, v in a.items()}
            pren = next((low[k] for k in ("forename", "fore_name", "first_name", "firstname",
                                          "given", "given_names", "given_name") if low.get(k)), "")
            nom = next((low[k] for k in ("lastname", "last_name", "surname", "family", "name") if low.get(k)), "")
            out.append((str(nom), str(pren)))
        else:
            s = str(a).strip()
            if "," in s:                                   # « Dupont, Jean »
                nom, pren = [p.strip() for p in s.split(",", 1)]
            else:                                          # format PubMed « Dupont Jean » ou « Dupont J »
                p = s.split(" ", 1)
                nom, pren = p[0], (p[1] if len(p) > 1 else "")
            out.append((nom, pren))
    return out


if COL_AUTHORS:
    df["auteurs"] = df[COL_AUTHORS].apply(parse_auteurs)
    exemple = df["auteurs"][df["auteurs"].str.len() > 0].head(3).tolist()
    print("Exemple d'auteurs parsés (À VÉRIFIER) :", exemple)
else:
    df["auteurs"] = [[] for _ in range(len(df))]
    print("!! Aucune colonne d'auteurs trouvée : renseigner COL_AUTHORS")

# ----------------------------------------------------------------------------
# 2. Volume de publications et taille des équipes
# ----------------------------------------------------------------------------
d = df[df["year"].isin(annees_ok)]
d.groupby("year").size().plot(marker="o", figsize=(8, 4), title="Nombre de publications par année")
plt.ylabel("articles")
show("01_publications_par_annee")

df["n_auteurs"] = df["auteurs"].str.len()
d = df[df["year"].isin(annees_ok) & (df["n_auteurs"] > 0)]
if len(d):
    d.groupby("year")["n_auteurs"].agg(["mean", "median"]).plot(marker="o", figsize=(8, 4),
                                                                title="Nombre d'auteurs par article")
    show("02_taille_equipes")

# ----------------------------------------------------------------------------
# 3. Thèmes : TF-IDF + NMF, puis évolution dans le temps
# ----------------------------------------------------------------------------
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DOMAIN_STOP = {
    "patient", "patients", "study", "studies", "result", "results", "conclusion", "conclusions",
    "method", "methods", "background", "objective", "aim", "use", "used", "using", "included",
    "associated", "compared", "group", "groups", "analysis", "showed", "found", "high", "low",
    "year", "years", "case", "cases", "significant", "significantly", "data", "ci", "vs",
}
STOP = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP)

tfidf = TfidfVectorizer(stop_words=STOP, sublinear_tf=True, min_df=5, max_df=0.5, ngram_range=(1, 2))
X = tfidf.fit_transform(df["text"].str.lower())
mots = tfidf.get_feature_names_out()

nmf = NMF(n_components=N_TOPICS, init="nndsvd", random_state=RANDOM_STATE, max_iter=400)
W = nmf.fit_transform(X)
H = nmf.components_
NOMS = {k: f"T{k}: " + ", ".join(mots[H[k].argsort()[::-1][:3]]) for k in range(N_TOPICS)}
df["topic"] = W.argmax(axis=1)
df["topic_nom"] = df["topic"].map(NOMS)

print("\nThèmes (8 mots les plus forts) :")
for k in range(N_TOPICS):
    print(f"{k:>2} | " + ", ".join(mots[H[k].argsort()[::-1][:8]]))

d = df[df["year"].isin(annees_ok)]
part = pd.crosstab(d["year"], d["topic_nom"], normalize="index").rolling(3, min_periods=1, center=True).mean()
ax = part.plot.area(figsize=(12, 6), colormap="tab10", alpha=0.9)
ax.set_title("Part de chaque thème par année (lissé sur 3 ans)")
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
show("03_themes_part_par_annee")

pd.crosstab(d["year"], d["topic_nom"]).plot(figsize=(12, 5), title="Nombre d'articles par thème et par année")
plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
show("04_themes_volumes")

recents = d["year"] >= d["year"].max() - 2
evol = (d[recents]["topic_nom"].value_counts(normalize=True)
        .sub(d[~recents]["topic_nom"].value_counts(normalize=True), fill_value=0)
        .sort_values(ascending=False))
print("\nThèmes en hausse/baisse (part 3 dernières années - avant) :\n", evol.round(3))

# ----------------------------------------------------------------------------
# 4. Auteurs : prénom -> genre, place dans la liste
# ----------------------------------------------------------------------------
import gender_guesser.detector as gd

detecteur = gd.Detector()


def premier_prenom(p):
    """Premier prénom utilisable ; None si seulement des initiales (« J », « JP », « J. P. »)."""
    mots_ = re.sub(r"[^\w\-\s]", " ", str(p)).split()
    if not mots_:
        return None
    t = mots_[0]
    if len(t) < 2 or (t.isupper() and len(t) <= 3):
        return None
    return t


def genre(prenom):
    if not prenom:
        return "inconnu"
    for essai in dict.fromkeys([prenom, prenom.split("-")[0]]):    # « Jean-Pierre » -> « Jean »
        g = detecteur.get_gender(essai.capitalize())
        if g in ("male", "mostly_male"):
            return "H"
        if g in ("female", "mostly_female"):
            return "F"
    return "inconnu"                                               # androgyne, inconnu, initiales


lignes = []
for i, (aut, an, top) in enumerate(zip(df["auteurs"], df["year"], df["topic_nom"])):
    n = len(aut)
    for pos, (nom, pren) in enumerate(aut, start=1):
        lignes.append((i, an, top, pos, n, premier_prenom(pren)))
A = pd.DataFrame(lignes, columns=["doc", "year", "topic_nom", "pos", "n", "prenom"])

if A.empty:
    print("Pas d'auteurs exploitables : fin du script.")
    raise SystemExit

A["place"] = np.select([A["n"] == 1, A["pos"] == 1, A["pos"] == A["n"]], ["unique", "premier", "dernier"], "milieu")
genres = {p: genre(p) for p in A["prenom"].dropna().unique()}      # une seule recherche par prénom distinct
A["genre"] = A["prenom"].map(genres).fillna("inconnu")

taux_id = (A["genre"] != "inconnu").mean()
print(f"\n{len(A)} signatures ; genre identifié pour {taux_id:.0%} d'entre elles")
print(A.groupby("year")["genre"].apply(lambda s: (s != "inconnu").mean()).round(2).rename("part identifiée").to_string())
print(A["genre"].value_counts())

A.to_csv("auteurs_genre_position.csv", index=False)
Ai = A[(A["genre"] != "inconnu") & A["year"].isin(annees_ok)].copy()
Ai["femme"] = (Ai["genre"] == "F").astype(int)

# ----------------------------------------------------------------------------
# 5. Évolution de la part des femmes
# ----------------------------------------------------------------------------
def part_femmes(sous, par):
    g = sous.groupby(par)["femme"].agg(["mean", "size"])
    g.loc[g["size"] < MIN_AUT_POINT, "mean"] = np.nan              # trop peu d'auteurs : point non fiable
    return g["mean"]


# 5a. Tous auteurs confondus
tout = part_femmes(Ai, "year")
tout.plot(marker="o", figsize=(8, 4), title="Part de femmes parmi les auteurs (genre identifié)")
plt.ylim(0, 1)
plt.axhline(0.5, color="grey", ls="--", lw=0.8)
show("05_femmes_global")

# 5b. Selon la place : premier / milieu / dernier
fig, ax = plt.subplots(figsize=(9, 5))
for place, couleur in [("premier", "#d1495b"), ("milieu", "#7a7a7a"), ("dernier", "#00798c")]:
    s = part_femmes(Ai[Ai["place"] == place], "year")
    s.rolling(3, min_periods=1, center=True).mean().plot(ax=ax, marker="o", label=place, color=couleur)
ax.axhline(0.5, color="grey", ls="--", lw=0.8)
ax.set_ylim(0, 1)
ax.set_ylabel("part de femmes")
ax.set_title("Part de femmes selon la place dans la liste d'auteurs (lissé 3 ans)")
ax.legend()
show("06_femmes_par_place")

# Tableau année x place (valeurs brutes)
tab = Ai.pivot_table(index="year", columns="place", values="femme", aggfunc="mean").round(3)
tab.to_csv("femmes_par_annee_et_place.csv")
print("\nPart de femmes par année et par place :\n", tab.to_string())

# 5c. Combinaison premier/dernier auteur par article (articles avec >= 2 auteurs)
pd_ = Ai[Ai["place"].isin(["premier", "dernier"])].pivot_table(
    index="doc", columns="place", values="genre", aggfunc="first").dropna()
if len(pd_):
    pd_["year"] = df.loc[pd_.index, "year"].values
    pd_["combo"] = pd_["premier"] + " / " + pd_["dernier"]         # premier / dernier
    combo = pd.crosstab(pd_["year"], pd_["combo"], normalize="index").rolling(3, min_periods=1, center=True).mean()
    combo.plot.area(figsize=(9, 5), colormap="Set2", title="Genre du premier / dernier auteur (lissé 3 ans)")
    plt.legend(title="premier / dernier", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    show("07_combinaisons_premier_dernier")

# 5d. Par thème : où les femmes publient-elles en premier / dernier auteur ?
par_theme = Ai[Ai["place"].isin(["premier", "dernier"])].pivot_table(
    index="topic_nom", columns="place", values="femme", aggfunc="mean")
if not par_theme.empty:
    par_theme.sort_values("premier").plot.barh(figsize=(8, 5), title="Part de femmes par thème et par place")
    plt.axvline(0.5, color="grey", ls="--", lw=0.8)
    plt.tight_layout()
    show("08_femmes_par_theme")

# 5e. Taille d'équipe selon le genre du premier auteur (optionnel, rapide)
prem = Ai[Ai["place"] == "premier"].groupby(["year", "genre"])["n"].mean().unstack()
if not prem.empty:
    prem.plot(marker="o", figsize=(8, 4), title="Nb d'auteurs de l'article selon le genre du premier auteur")
    show("09_taille_equipe_par_genre")

# ----------------------------------------------------------------------------
# Sauvegarde
# ----------------------------------------------------------------------------
cols = [c for c in [COL_TITLE, "year", "topic_nom", "n_auteurs"] if c and c in df.columns]
df[cols].to_csv("pubmed_resultats_simple.csv", index=False)
print("\n*** TERMINÉ ***  Graphiques dans :", FIG_DIR.resolve())