# %% [markdown]
# # NLP appliqué aux publications PubMed du CHU de Rennes
# Plan (suit le cours) :
# 1. Chargement et nettoyage
# 2. Tokenisation / lemmatisation
# 3. TF-IDF : mots-clés, recherche, doublons
# 4. Analyse de sentiment (VADER)
# 5. Topic modeling : NMF, LDA
# 6. Reconnaissance d'entités nommées (NER)
# 7. Embeddings de phrases et recherche sémantique
# 8. BM25 + recherche hybride (RRF)
# 9. Clustering d'embeddings (UMAP + HDBSCAN)
# 10. Mini-RAG : construire le prompt pour un LLM

# %% 0. Configuration
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "/pubmed_chu_rennes.jsonl"

# Noms des colonnes (None = détection automatique, voir df.columns)
COL_ABSTRACT = "abstract"
COL_TITLE = "title"
COL_DATE = None
COL_JOURNAL = None

# Interrupteurs : mettre False pour sauter les parties lourdes
RUN_SPACY = True          # lemmatisation + NER (modèle spaCy requis)
RUN_EMBEDDINGS = True     # sentence-transformers (plus lent, GPU conseillé)
RUN_CLUSTERING = True     # UMAP + HDBSCAN (nécessite RUN_EMBEDDINGS)
RUN_SENT_TRANSFORMER = False  # comparer VADER à un modèle DistilBERT

N_TOPICS = 10
RANDOM_STATE = 0

# %% 1. Chargement et nettoyage
df = pd.read_json(PATH, lines=True)
print(df.shape)
print(df.columns.tolist())


def find_col(candidates):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in low:
            return low[c]
    return None


COL_ABSTRACT = COL_ABSTRACT if COL_ABSTRACT in df.columns else find_col(["abstract", "abstract_text"])
COL_TITLE = COL_TITLE if COL_TITLE in df.columns else find_col(["title", "article_title"])
COL_DATE = COL_DATE or find_col(["year", "pub_year", "publication_year", "pubdate", "publication_date", "date"])
COL_JOURNAL = COL_JOURNAL or find_col(["journal", "journal_title", "source"])
print("Colonnes utilisées :", COL_ABSTRACT, COL_TITLE, COL_DATE, COL_JOURNAL)

# Certains exports PubMed stockent l'abstract comme une liste de sections
df[COL_ABSTRACT] = df[COL_ABSTRACT].apply(lambda x: " ".join(map(str, x)) if isinstance(x, list) else x)

df = df.dropna(subset=[COL_ABSTRACT])
df = df[df[COL_ABSTRACT].str.strip().str.len() > 50]
df = df.drop_duplicates(subset=[COL_ABSTRACT]).reset_index(drop=True)

# Retirer les intitulés de sections d'abstracts structurés (BACKGROUND:, METHODS:...)
SECTIONS = r"\b(?:BACKGROUND|OBJECTIVES?|AIMS?|PURPOSE|INTRODUCTION|DESIGN|SETTING|PATIENTS|MATERIALS? AND METHODS|METHODS?|RESULTS?|CONCLUSIONS?|DISCUSSION)\s*:"
df["abstract_clean"] = df[COL_ABSTRACT].str.replace(SECTIONS, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

# Texte analysé = titre + abstract (le titre est très informatif)
if COL_TITLE:
    df["text"] = df[COL_TITLE].fillna("").astype(str) + ". " + df["abstract_clean"]
else:
    df["text"] = df["abstract_clean"]

# Année de publication (extraction robuste d'une année à 4 chiffres)
if COL_DATE:
    df["year"] = pd.to_numeric(df[COL_DATE].astype(str).str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
else:
    df["year"] = np.nan

print(f"{len(df)} documents après nettoyage")
df[["text"]].head()

# %% 2. Tokenisation et lemmatisation
# Installation du modèle avec uv :
#   uv add "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
nlp = None
if RUN_SPACY:
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    docs = nlp.pipe(df["text"], batch_size=64)
    df["lemmas"] = [
        " ".join(t.lemma_.lower() for t in d if t.is_alpha and not t.is_stop and len(t) > 2)
        for d in docs
    ]
else:
    df["lemmas"] = df["text"].str.lower()

# Les LLM facturent en tokens : combien font nos abstracts ?
try:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    df["n_tokens"] = df["text"].map(lambda t: len(enc.encode(t)))
    print(df["n_tokens"].describe().round(0))
    print(f"Corpus total : {df['n_tokens'].sum():,} tokens")
except Exception as e:  # tiktoken absent ou pas de réseau pour le vocabulaire
    print("tiktoken indisponible :", e)

# %% 3. TF-IDF : vectorisation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stopwords du domaine : mots omniprésents dans les abstracts, sans information
DOMAIN_STOP = {
    "patient", "patients", "study", "studies", "result", "results", "conclusion", "conclusions",
    "method", "methods", "background", "objective", "aim", "use", "used", "using", "include",
    "included", "associate", "associated", "compare", "compared", "group", "groups", "analysis",
    "show", "showed", "find", "found", "high", "low", "year", "years", "case", "cases",
    "significant", "significantly", "datum", "data", "n", "p", "ci",
}
STOP = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP)

tfidf = TfidfVectorizer(
    stop_words=STOP,
    sublinear_tf=True,   # 1 + log(tf)
    min_df=5,            # ignore fautes de frappe, identifiants
    max_df=0.5,          # ignore les mots dans plus de 50 % des documents
    ngram_range=(1, 2),  # bigrammes : "heart failure", "deep brain"
)
X = tfidf.fit_transform(df["lemmas"])
mots = tfidf.get_feature_names_out()
print("Matrice TF-IDF :", X.shape)

# %% 3b. Mots-clés par document et pour le corpus
def mots_cles(i, k=5):
    ligne = X[i]  # ligne creuse : on ne regarde que les mots présents
    idx = ligne.indices[np.argsort(ligne.data)[::-1][:k]]
    return list(mots[idx])


df["mots_cles"] = [", ".join(mots_cles(i)) for i in range(len(df))]
display_cols = [c for c in [COL_TITLE, "mots_cles"] if c]
print(df[display_cols].head(10).to_string())

# Mots les plus importants sur l'ensemble du corpus
scores_corpus = np.asarray(X.mean(axis=0)).ravel()
top_corpus = pd.Series(scores_corpus, index=mots).nlargest(25)
top_corpus.sort_values().plot.barh(figsize=(6, 7), title="Termes TF-IDF moyens les plus forts")
plt.tight_layout()
plt.show()

# %% 3c. Recherche lexicale et doublons
def recherche_tfidf(requete, k=5):
    q = tfidf.transform([requete.lower()])
    scores = cosine_similarity(q, X).ravel()
    top = np.argsort(scores)[::-1][:k]
    return pd.DataFrame({"score": scores[top].round(3), "texte": df["text"].iloc[top].str[:90].values}, index=top)


print(recherche_tfidf("kidney transplantation rejection"))


def doublons(X, seuil=0.9, bloc=1000):
    """Paires de documents quasi identiques, calculées par blocs pour économiser la mémoire."""
    paires = []
    for s in range(0, X.shape[0], bloc):
        S = cosine_similarity(X[s:s + bloc], X)
        for i, j in zip(*np.where(S >= seuil)):
            if s + i < j:
                paires.append((s + i, j, round(float(S[i, j]), 3)))
    return pd.DataFrame(paires, columns=["i", "j", "sim"])


dup = doublons(X)
print(f"{len(dup)} paires de quasi-doublons (cosinus >= 0.9)")

# %% 4. Analyse de sentiment : VADER (sur le texte BRUT, sans nettoyage)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
df["compound"] = df["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
df["sentiment"] = np.select([df["compound"] >= 0.05, df["compound"] <= -0.05], ["positif", "négatif"], default="neutre")

print(df["sentiment"].value_counts(normalize=True).round(3))
df["compound"].plot.hist(bins=40, title="Distribution du score VADER (compound)")
plt.show()

# Limite vue en cours : un lexique généraliste s'adapte mal au vocabulaire médical
# (« tumor », « failure », « death » sont négatifs sans que l'article le soit).
# À lire comme un indicateur de ton, pas comme une opinion.
if df["year"].notna().any():
    (df[df["year"].between(2000, 2030)].groupby("year")["compound"].mean()
     .plot(marker="o", title="Score VADER moyen par année"))
    plt.show()

# %% 4b. (Optionnel) Comparer à un transformer fine-tuné
if RUN_SENT_TRANSFORMER:
    from transformers import pipeline

    sent = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    echantillon = df.sample(min(300, len(df)), random_state=RANDOM_STATE)
    preds = sent(echantillon["text"].tolist(), batch_size=32, truncation=True, max_length=512)
    echantillon = echantillon.assign(bert=[p["label"] for p in preds])
    print(pd.crosstab(echantillon["sentiment"], echantillon["bert"]))

# %% 5. Topic modeling : NMF sur TF-IDF
from sklearn.decomposition import NMF, LatentDirichletAllocation

# 5a. Choisir k : l'erreur de reconstruction baisse toujours avec k, on cherche un coude
erreurs = {}
for k in range(4, 31, 4):
    erreurs[k] = NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=300).fit(X).reconstruction_err_
pd.Series(erreurs).plot(marker="o", title="NMF : erreur de reconstruction selon k")
plt.xlabel("k (nombre de thèmes)")
plt.show()

# %% 5b. NMF final
nmf = NMF(n_components=N_TOPICS, init="nndsvd", random_state=RANDOM_STATE, max_iter=500)
W = nmf.fit_transform(X)   # (n_docs, k) : poids des thèmes par document
H = nmf.components_        # (k, V)      : poids des mots par thème


def afficher_themes(H, mots, n=8):
    for k, ligne in enumerate(H):
        print(f"{k:>2} | " + ", ".join(mots[ligne.argsort()[::-1][:n]]))


afficher_themes(H, mots)

# C'est à vous de NOMMER les thèmes après lecture. Noms provisoires = 3 premiers mots :
NOMS = {k: f"T{k}: " + ", ".join(mots[H[k].argsort()[::-1][:3]]) for k in range(N_TOPICS)}
df["topic"] = W.argmax(axis=1)
df["topic_nom"] = df["topic"].map(NOMS)
df["topic_force"] = W.max(axis=1) / (W.sum(axis=1) + 1e-12)   # part du thème dominant
print(df["topic_nom"].value_counts())

# %% 5c. Thèmes dans le temps (« hot topics ») et sentiment par thème
if df["year"].notna().any():
    d = df[df["year"].between(2000, 2030)]
    part = pd.crosstab(d["year"], d["topic_nom"], normalize="index")
    part.plot(figsize=(10, 5), title="Part de chaque thème par année")
    plt.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Thèmes en croissance : part sur les 3 dernières années vs. avant
    recents = d["year"] >= d["year"].max() - 2
    evolution = (d[recents]["topic_nom"].value_counts(normalize=True)
                 .sub(d[~recents]["topic_nom"].value_counts(normalize=True), fill_value=0)
                 .sort_values(ascending=False))
    print("Évolution de la part (récent - ancien) :\n", evolution.round(3))

print(df.groupby("topic_nom")["compound"].mean().sort_values().round(3))

if COL_JOURNAL:
    top_j = df[COL_JOURNAL].value_counts().head(8).index
    print(pd.crosstab(df[df[COL_JOURNAL].isin(top_j)][COL_JOURNAL], df["topic"]))

# %% 5d. LDA : le modèle probabiliste (comptages bruts, pas TF-IDF)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words=STOP, min_df=5, max_df=0.5)
counts = cv.fit_transform(df["lemmas"])
lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="online", random_state=RANDOM_STATE)
theta = lda.fit_transform(counts)   # chaque ligne somme à 1 : P(thème | document)
mots_cv = cv.get_feature_names_out()
for k, ligne in enumerate(lda.components_):
    print(f"{k:>2} | " + ", ".join(mots_cv[ligne.argsort()[::-1][:8]]))
df["topic_lda"] = theta.argmax(axis=1)
print("Perplexité :", round(lda.perplexity(counts), 1))

# %% 6. Reconnaissance d'entités nommées (NER) avec spaCy
# Le modèle généraliste repère ORG, GPE (pays/villes), PERSON...
# Pour du biomédical : scispaCy (en_ner_bc5cdr_md : DISEASE / CHEMICAL).
if RUN_SPACY:
    nlp_ner = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    lignes = []
    for i, d in enumerate(nlp_ner.pipe(df["text"], batch_size=64)):
        for e in d.ents:
            if e.label_ in {"ORG", "GPE", "PERSON", "NORP"}:
                lignes.append((i, e.text.strip(), e.label_))
    ents = pd.DataFrame(lignes, columns=["doc", "entite", "type"])
    for typ in ["ORG", "GPE"]:
        print(f"\n--- {typ} les plus cités ---")
        print(ents[ents["type"] == typ]["entite"].value_counts().head(15))

    # Piège du cours : variantes d'une même entité (« CHU Rennes », « Rennes University Hospital »...)
    # -> normalisation minimale avant comptage
    ents["entite_norm"] = ents["entite"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()

# %% 7. Embeddings de phrases et recherche sémantique
E = None
model = None
if RUN_EMBEDDINGS:
    from pathlib import Path
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")   # 50 langues, 384 dim
    model.max_seq_length = 256   # au-delà, l'abstract est tronqué
    cache = Path("embeddings.npy")
    if cache.exists() and np.load(cache).shape[0] == len(df):
        E = np.load(cache)
    else:
        E = model.encode(df["text"].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        np.save(cache, E)   # on calcule une seule fois et on stocke
    print("Embeddings :", E.shape)


def recherche_semantique(requete, k=5):
    q = model.encode([requete], normalize_embeddings=True)
    scores = (E @ q.T).ravel()   # cosinus = produit scalaire (vecteurs normalisés)
    top = scores.argsort()[::-1][:k]
    return pd.DataFrame({"score": scores[top].round(3), "texte": df["text"].iloc[top].str[:90].values}, index=top)


if E is not None:
    print(recherche_semantique("rejection after kidney graft"))   # trouve aussi sans mots communs

# %% 8. BM25 + recherche hybride (Reciprocal Rank Fusion)
from rank_bm25 import BM25Okapi

corpus_tok = [t.split() for t in df["lemmas"]]
bm25 = BM25Okapi(corpus_tok, k1=1.5, b=0.75)


def tok_requete(q):
    if nlp is not None:
        return [t.lemma_.lower() for t in nlp(q) if t.is_alpha and not t.is_stop]
    return [m for m in q.lower().split() if m not in STOP]


def rrf(listes, k=60):
    fusion = {}
    for classement in listes:
        for rang, doc in enumerate(classement, start=1):
            fusion[doc] = fusion.get(doc, 0) + 1 / (k + rang)
    return sorted(fusion, key=fusion.get, reverse=True)


def recherche_hybride(requete, k=5, n_candidats=100):
    top_bm25 = bm25.get_scores(tok_requete(requete)).argsort()[::-1][:n_candidats]
    listes = [top_bm25]
    if E is not None:
        q = model.encode([requete], normalize_embeddings=True)
        listes.append((E @ q.T).ravel().argsort()[::-1][:n_candidats])
    return rrf(listes)[:k]


requete = "antibiotic resistance in intensive care"
res = recherche_hybride(requete)
for i in res:
    print(i, "|", df["text"].iloc[i][:100])

# %% 9. Clustering d'embeddings : UMAP + HDBSCAN
if RUN_EMBEDDINGS and RUN_CLUSTERING:
    import umap
    import hdbscan

    E5 = umap.UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine",
                   random_state=RANDOM_STATE).fit_transform(E)
    df["cluster"] = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(E5)   # -1 = bruit
    print(df["cluster"].value_counts().head(15))

    # Nommer chaque cluster par ses mots TF-IDF les plus forts (idée du c-TF-IDF de BERTopic)
    for c in sorted(df["cluster"].unique()):
        if c == -1:
            continue
        moyenne = np.asarray(X[(df["cluster"] == c).values].mean(axis=0)).ravel()
        print(f"cluster {c:>3} ({(df['cluster'] == c).sum():>4} docs) :", ", ".join(mots[moyenne.argsort()[::-1][:6]]))

    # Carte 2D du corpus, colorée par thème NMF
    E2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=RANDOM_STATE).fit_transform(E)
    plt.figure(figsize=(9, 7))
    plt.scatter(E2[:, 0], E2[:, 1], c=df["topic"], cmap="tab10", s=6)
    plt.title("Carte UMAP des publications (couleur = thème NMF)")
    plt.show()

    # Version tout-en-un (optionnelle) : uv add bertopic
    # from bertopic import BERTopic
    # topic_model = BERTopic(embedding_model=model, min_topic_size=15)
    # topics, probs = topic_model.fit_transform(df["text"].tolist(), E)
    # topic_model.get_topic_info().head()

# %% 10. Mini-RAG : retrouver des passages puis construire le prompt (cours NLP-1, slide RAG)
def build_rag_prompt(question, k=5):
    idx = recherche_hybride(question, k=k)
    contexte = "\n---------------------\n".join(
        f"[{n}] {df['text'].iloc[i][:1200]}" for n, i in enumerate(idx, start=1)
    )
    return (
        "Context information is below.\n---------------------\n"
        f"{contexte}\n---------------------\n"
        "Given the context information and not prior knowledge, answer the query. "
        "Cite the passages you use as [n]. If the context does not contain the answer, say so. "
        "Let's think step by step.\n"
        "This is very important to my career.\n"
        f"Query: {question}\nAnswer:"
    )


prompt = build_rag_prompt("Which topics are studied about kidney transplantation?")
print(prompt[:1500])

# Envoi à un LLM (optionnel) : uv add anthropic, clé dans ANTHROPIC_API_KEY
# import anthropic
# client = anthropic.Anthropic()
# rep = client.messages.create(model="claude-sonnet-5-5", max_tokens=800, temperature=0,
#                              messages=[{"role": "user", "content": prompt}])
# print(rep.content[0].text)
# temperature=0 : réponses cohérentes ; plus haut = plus de créativité (cours, slide « température »)

# %% 11. Sauvegarde des résultats
cols = [c for c in [COL_TITLE, "year", COL_JOURNAL, "mots_cles", "topic_nom", "topic_lda",
                    "compound", "sentiment", "cluster"] if c and c in df.columns]
df[cols].to_csv("pubmed_resultats_nlp.csv", index=False)
print("Résultats enregistrés :", cols)