### Pour lancer le code : uv run python -c 'from pathlib import Path; p = Path("analyse_llm.py"); p.write_text(p.read_text(encoding="utf-8").replace("gemini-2.5-flash", "gemini-flash-lite-latest"), encoding="utf-8")' && uv run analyse_llm.py 3
## Adapter le 3 de la fin en focus sur le thème souhaité (1 à 5). Le script génère un fichier HTML et un fichier Markdown


import html
import json
import os
import re
import sys
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

# Choix du thème à explorer (ex. : uv run analyse_llm.py 3)
TOPIC_CIBLE = int(sys.argv[1]) if len(sys.argv) > 1 else 3

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-flash-lite-latest"

load_dotenv(override=True)
client = OpenAI(base_url=ENDPOINT, api_key=os.environ["GEMINI_API_KEY"])

# 1. Chargement du JSON du thème choisi
fichier_theme = Path("top5_themes") / f"top5_T{TOPIC_CIBLE}.json"
if not fichier_theme.exists():
    raise FileNotFoundError(f"Fichier introuvable : {fichier_theme}. Lance d'abord NLP.py.")

with open(fichier_theme, "r", encoding="utf-8") as f:
    donnees_theme = json.load(f)

topic_nom = donnees_theme["topic_nom"]
mots_cles_liste = donnees_theme["mots_cles"]
articles_source = donnees_theme["articles"]

articles_texte = ""
for art in articles_source:
    articles_texte += (
        f"\n--- Article {art['rang']} ({art['year']}) ---\n"
        f"Titre : {art.get('title', 'Sans titre')}\n"
        f"Poids NMF : {art['poids_nmf']} (Part du thème : {art['part_theme']:.0%})\n"
        f"Résumé : {art['abstract']}\n"
    )

# 2. Appel au LLM (sortie JSON structurée pour garantir les champs)
prompt_systeme = (
    "Tu es un expert en biostatistique et analyse bibliométrique médicale. "
    "On a appliqué une NMF sur des articles PubMed français (1955-2026). "
    "Réponds UNIQUEMENT par un objet JSON valide en français avec les clés : "
    '"nom_propose" (string, 3 à 6 mots pour nommer ce thème), '
    '"synthese_globale" (string, ce qui relie scientifiquement ces 5 articles), '
    '"methodes_ou_sujets_cles" (liste de strings), '
    '"analyse_par_article" (liste de 5 objets dans le même ordre avec les clés "titre", "annee", "resume_apport").'
)

prompt_utilisateur = (
    f"Thème NMF : {topic_nom}\n"
    f"Top 8 mots-clés TF-IDF : {', '.join(mots_cles_liste)}\n\n"
    f"Voici les 5 articles les plus représentatifs de ce thème :\n{articles_texte}"
)

print(f"Interrogation de {MODEL} sur {topic_nom}...")
try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt_systeme},
            {"role": "user", "content": prompt_utilisateur},
        ],
        temperature=0,
    )
except APIStatusError as e:
    print(f"Erreur API ({e.status_code}) :", e.body)
    raise

brut = response.choices[0].message.content.strip()
brut = re.sub(r"^```(?:json)?\s*|\s*```$", "", brut)
resultat = json.loads(brut)

# 3. Génération du rapport Markdown (prêt pour ton rapport / Quarto)
lignes_md = [
    f"# {topic_nom} — {resultat['nom_propose']}",
    f"**Mots-clés TF-IDF :** {', '.join(mots_cles_liste)}\n",
    "## Synthèse globale",
    f"{resultat['synthese_globale']}\n",
    "**Axes & méthodes identifiés :** " + ", ".join(resultat["methodes_ou_sujets_cles"]) + "\n",
    "## Top 5 des articles les plus significatifs\n",
]

for idx, art_llm in enumerate(resultat["analyse_par_article"]):
    meta = articles_source[idx] if idx < len(articles_source) else {}
    poids_info = f" | Poids NMF : `{meta.get('poids_nmf', '-')}` ({meta.get('part_theme', 0):.0%})" if meta else ""
    lignes_md.append(f"### {idx+1}. {art_llm['titre']} ({art_llm['annee']}{poids_info})")
    lignes_md.append(f"{art_llm['resume_apport']}\n")

fichier_md = Path(f"analyse_T{TOPIC_CIBLE}.md")
fichier_md.write_text("\n".join(lignes_md), encoding="utf-8")

# 4. Génération de la page HTML visuelle
badges_tfidf = "".join(f'<span class="badge tfidf">{html.escape(m)}</span>' for m in mots_cles_liste)
badges_methodes = "".join(
    f'<span class="badge methode">{html.escape(m)}</span>' for m in resultat["methodes_ou_sujets_cles"]
)

cartes_articles = ""
for idx, art_llm in enumerate(resultat["analyse_par_article"]):
    meta = articles_source[idx] if idx < len(articles_source) else {}
    poids = meta.get("poids_nmf", "-")
    part = f"{meta.get('part_theme', 0):.0%}" if meta else "-"
    cartes_articles += f"""
    <div class="card">
        <div class="card-header">
            <span class="rank">#{idx+1}</span>
            <span class="year">{html.escape(str(art_llm['annee']))}</span>
            <span class="score">Poids NMF : <strong>{poids}</strong> &bull; Pureté : <strong>{part}</strong></span>
        </div>
        <h3>{html.escape(art_llm['titre'])}</h3>
        <p>{html.escape(art_llm['resume_apport'])}</p>
    </div>
    """

template_html = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Analyse NMF - __TOPIC_NOM__</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f6f9; color: #1e293b; max-width: 920px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
  .header { background: #0f172a; color: white; padding: 28px 32px; border-radius: 14px; margin-bottom: 24px; }
  .header .tag { text-transform: uppercase; font-size: 12px; letter-spacing: 1px; color: #38bdf8; font-weight: 700; }
  .header h1 { margin: 6px 0 14px 0; font-size: 26px; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .badge { padding: 4px 10px; border-radius: 999px; font-size: 13px; font-weight: 500; }
  .badge.tfidf { background: rgba(255,255,255,0.12); color: #e2e8f0; }
  .badge.methode { background: #e0f2fe; color: #0369a1; }
  .synthese { background: white; padding: 24px 28px; border-radius: 12px; border-left: 5px solid #0284c7; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 28px; }
  .synthese h2 { margin-top: 0; font-size: 18px; color: #0f172a; }
  .card { background: white; padding: 20px 24px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 16px; border: 1px solid #e2e8f0; }
  .card-header { display: flex; align-items: center; gap: 10px; font-size: 13px; color: #64748b; margin-bottom: 8px; }
  .rank { background: #0f172a; color: white; font-weight: 700; padding: 2px 8px; border-radius: 6px; }
  .year { background: #f1f5f9; color: #334155; font-weight: 600; padding: 2px 8px; border-radius: 6px; }
  .score { margin-left: auto; }
  .card h3 { margin: 4px 0 10px 0; font-size: 17px; color: #0f172a; }
  .card p { margin: 0; color: #334155; }
</style>
</head>
<body>
  <div class="header">
    <div class="tag">__TOPIC_NOM__ &bull; PubMed France (1955-2026)</div>
    <h1>__NOM_PROPOSE__</h1>
    <div class="badges">__BADGES_TFIDF__</div>
  </div>
  <div class="synthese">
    <h2>Synthèse scientifique du thème</h2>
    <p>__SYNTHESE__</p>
    <div class="badges">__BADGES_METHODES__</div>
  </div>
  <h2>Top 5 des articles les plus significatifs</h2>
  __CARTES__
</body>
</html>"""

page_html = (
    template_html
    .replace("__TOPIC_NOM__", html.escape(topic_nom))
    .replace("__NOM_PROPOSE__", html.escape(resultat["nom_propose"]))
    .replace("__BADGES_TFIDF__", badges_tfidf)
    .replace("__SYNTHESE__", html.escape(resultat["synthese_globale"]))
    .replace("__BADGES_METHODES__", badges_methodes)
    .replace("__CARTES__", cartes_articles)
)

fichier_html = Path(f"analyse_T{TOPIC_CIBLE}.html")
fichier_html.write_text(page_html, encoding="utf-8")

print(f"Synthèse générée :\n  -> {fichier_html}\n  -> {fichier_md}")
webbrowser.open(fichier_html.resolve().as_uri())