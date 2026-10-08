Voici une proposition de feuille de route en cinq étapes. Les étapes 1 et 2 sont indispensables, la 5 est à garder si le temps le permet.

1. Extraction PubMed pour toute la France

Adapter votre script : remplacer la requête « CHU Rennes » par un dictionnaire de variantes de nom pour chacun des ~32 CHU (nom de ville, nom des hôpitaux, « CHU », « University Hospital »). Le cas de Paris est le plus lourd : AP-HP et des dizaines de noms d'hôpitaux.
Volume à prévoir : plusieurs centaines de milliers de notices. L'API E-utilities ne renvoie que 10 000 résultats par requête, il faut donc découper par année, voire par mois. Avec une clé API NCBI, comptez 10 requêtes par seconde.
Pour chaque article, conserver en XML : l'affiliation de chaque auteur, le prénom et le nom des premier et dernier auteurs, les MeSH, le type de publication et la langue.
Attribuer chaque article à un CHU et à une région (les 13 régions actuelles). Contrôler à la main 100 articles tirés au hasard. Pour Rennes, comparer les volumes annuels avec les chiffres SIGAPS de la DRCI.

2. Description des publications par CHU et dans le temps

Volume annuel par CHU, proportions par type de publication et par langue.
Analyse principale : premier auteur affilié au CHU, sur toute la période.
Analyse de sensibilité sur 2014-2025 : premier auteur seulement comparé à « au moins un auteur » du CHU. L'écart chiffre ce que la règle d'avant 2014 laisse de côté, et peut être comparé d'un CHU à l'autre.

3. Genre au cours du temps

Genre du premier auteur à partir de 2002 seulement, car PubMed ne contient que des initiales avant. Pour le dernier auteur, se limiter à 2014 et après, puisque son affiliation n'est connue qu'à partir de là.
Pièges propres aux prénoms français : prénoms mixtes (Dominique, Claude, Camille) et prénoms composés comme Jean-Marie, qui contient « Marie » mais est masculin. Classer ces cas en « indéterminé » et indiquer la proportion.

4. Genre et thèmes

Ajuster un seul modèle sur tout le corpus : MeSH, ou STM avec le genre et l'année comme covariables. Cela permet de tester si certains thèmes sont plus souvent portés par des femmes, et si cet écart se réduit au fil du temps (interaction genre × année).
Attention à l'interprétation : un thème plus « féminin » peut simplement refléter la proportion de femmes dans la spécialité concernée (pédiatrie, gynécologie…). Il faut le dire, ou ajuster sur la spécialité si les MeSH permettent de l'approcher.

5. Stratification par CHU puis par région

Mêmes indicateurs, stratifiés. Les petits CHU ont peu d'articles par an : regrouper par périodes de 5 ans, ou utiliser un modèle hiérarchique avec un effet aléatoire CHU, pour éviter des estimations instables.

Pour tenir en quelques heures, je conseille de faire 1 et 2 pour toute la France, et 3 et 4 d'abord sur Rennes plus deux ou trois CHU de taille comparable (Nantes, Brest, Angers par exemple). La généralisation viendra ensuite si le pipeline fonctionne.

Si vous partagez votre script actuel, je peux l'adapter à toute la France en commençant par le dictionnaire d'affiliations des CHU.