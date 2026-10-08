"""
split_corpus.py

Utilitaire pour decouper le corpus PubMed JSONL selon plusieurs strategies :
  - Extraire un petit sous-ensemble pour tests rapides (ex: 10 articles)
  - Creer des splits train/val/test
  - Stratifier par decennie ou autre critere

Usage:
    # Mini-corpus de 10 articles pour test rapide
    python split_corpus.py --mode mini --n_samples 10 --output data/mini/

    # Split 70/15/15 (train/val/test)
    python split_corpus.py --mode split --train 0.7 --val 0.15 --output data/splits/

    # Split stratifie par decennie
    python split_corpus.py --mode split --train 0.7 --stratify decade --output data/splits/

    # Split stratifie par MeSH principal
    python split_corpus.py --mode split --train 0.7 --stratify mesh --output data/splits/
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

def load_jsonl(path):
    """Charge un fichier JSONL et retourne une liste de dicts."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"[!] Erreur JSON a la ligne {i+1} : {e}")
                continue
    return data


def save_jsonl(data, path):
    """Sauvegarde une liste de dicts en JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"✅ {len(data)} records -> {path}")


def extract_stratum(record, stratify_by):
    """Extrait la valeur de stratification pour un record."""
    if stratify_by == 'decade':
        year = record.get('year')
        if year:
            try:
                year = float(year)
                return int(year // 10 * 10)
            except (ValueError, TypeError):
                return None
        return None
    
    elif stratify_by == 'mesh':
        # Prendre le premier MeSH term comme "classe principale"
        mesh = record.get('mesh_terms', [])
        if isinstance(mesh, list) and len(mesh) > 0:
            return mesh[0]
        return 'unknown'
    
    elif stratify_by == 'journal':
        return record.get('journal', 'unknown')
    
    else:
        return None


def cmd_mini(args):
    """Cree un mini-corpus pour tests rapides."""
    print(f"\n📦 MODE : Mini-corpus")
    print(f"   Source : {args.input}")
    print(f"   Taille demandee : {args.n_samples} articles")
    
    # Charger les donnees
    data = load_jsonl(args.input)
    print(f"   Total dans le corpus : {len(data)}")
    
    if len(data) < args.n_samples:
        print(f"   [!] Attention : corpus plus petit que demande ({len(data)} < {args.n_samples})")
        mini = data
    else:
        # Prendre les premiers N (deterministe, ou aleatoire ?)
        if args.random_seed is not None:
            np.random.seed(args.random_seed)
            indices = np.random.choice(len(data), size=args.n_samples, replace=False)
            mini = [data[i] for i in sorted(indices)]
        else:
            mini = data[:args.n_samples]
    
    # Sauvegarder
    output_path = Path(args.output) / "mini.jsonl"
    save_jsonl(mini, output_path)
    
    print(f"\n📊 Aperçu du mini-corpus :")
    print(f"   Premiere notice : PMID {mini[0]['pmid']}")
    print(f"   Derniere notice : PMID {mini[-1]['pmid']}")


def cmd_split(args):
    """Cree des splits train/val/test."""
    print(f"\n📦 MODE : Train/Val/Test split")
    print(f"   Source : {args.input}")
    print(f"   Ratios : train={args.train}, val={args.val}, test={args.test}")
    
    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        print(f"   [!] Erreur : ratios ne somment pas a 1.0")
        return
    
    # Charger les donnees
    data = load_jsonl(args.input)
    print(f"   Total dans le corpus : {len(data)}")
    
    # Stratification ?
    if args.stratify:
        print(f"   Stratification par : {args.stratify}")
        # Creer les strates
        strata = {}
        for record in data:
            stratum = extract_stratum(record, args.stratify)
            if stratum not in strata:
                strata[stratum] = []
            strata[stratum].append(record)
        
        print(f"   Nombre de strates : {len(strata)}")
        
        # Afficher distribution
        stratum_sizes = {s: len(records) for s, records in strata.items()}
        for stratum in sorted(stratum_sizes.keys()):
            n = stratum_sizes[stratum]
            pct = 100 * n / len(data)
            print(f"     - {str(stratum):30s} : {n:4d} ({pct:5.1f}%)")
        
        # Split dans chaque strate
        train_data = []
        val_data = []
        test_data = []
        
        for stratum, records in strata.items():
            n = len(records)
            n_train = int(n * args.train)
            n_val = int(n * args.val)
            # n_test = n - n_train - n_val
            
            # Melanger
            if args.random_seed is not None:
                np.random.seed(args.random_seed)
            indices = np.random.permutation(n)
            
            train_data.extend([records[i] for i in indices[:n_train]])
            val_data.extend([records[i] for i in indices[n_train:n_train+n_val]])
            test_data.extend([records[i] for i in indices[n_train+n_val:]])
    
    else:
        print(f"   Sans stratification (aleatoire)")
        
        # Melanger et splitter
        if args.random_seed is not None:
            np.random.seed(args.random_seed)
        indices = np.random.permutation(len(data))
        
        n_train = int(len(data) * args.train)
        n_val = int(len(data) * args.val)
        
        train_data = [data[i] for i in indices[:n_train]]
        val_data = [data[i] for i in indices[n_train:n_train+n_val]]
        test_data = [data[i] for i in indices[n_train+n_val:]]
    
    # Sauvegarder
    output_dir = Path(args.output)
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(val_data, output_dir / "val.jsonl")
    save_jsonl(test_data, output_dir / "test.jsonl")
    
    print(f"\n📊 Resultats du split :")
    print(f"   Train : {len(train_data)} ({100*len(train_data)/len(data):.1f}%)")
    print(f"   Val   : {len(val_data)} ({100*len(val_data)/len(data):.1f}%)")
    print(f"   Test  : {len(test_data)} ({100*len(test_data)/len(data):.1f}%)")


def cmd_info(args):
    """Affiche des infos sur le corpus."""
    print(f"\n📊 MODE : Info")
    print(f"   Source : {args.input}")
    
    data = load_jsonl(args.input)
    print(f"   Total : {len(data)} notices")
    
    # Extraire les années
    years = [record.get('year') for record in data if record.get('year')]
    try:
        years = [float(y) for y in years]
        print(f"   Années : {min(years):.0f} - {max(years):.0f}")
    except:
        pass
    
    # Extraire les journaux
    journals = [record.get('journal') for record in data if record.get('journal')]
    print(f"   Journaux distincts : {len(set(journals))})")
    
    # Abstracts
    n_with_abstract = sum(1 for r in data if r.get('abstract') and len(r.get('abstract', '')) > 0)
    print(f"   Avec abstract : {n_with_abstract} ({100*n_with_abstract/len(data):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/pubmed_chu_rennes.jsonl',
        help='Fichier JSONL source (defaut: data/pubmed_chu_rennes.jsonl)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/processed/',
        help='Repertoire de sortie (defaut: data/processed/)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['mini', 'split', 'info'],
        default='info',
        help='Mode d\'operation (defaut: info)'
    )
    
    # Arguments pour mode 'mini'
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10,
        help='Nombre d\'echantillons pour le mini-corpus (defaut: 10)'
    )
    
    # Arguments pour mode 'split'
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Proportion train (defaut: 0.7)'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='Proportion validation (defaut: 0.15)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Proportion test (defaut: 0.15, auto-calculee si omis)'
    )
    
    parser.add_argument(
        '--stratify',
        type=str,
        choices=['decade', 'mesh', 'journal', None],
        default=None,
        help='Stratification du split (defaut: aucune)'
    )
    
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Graine aleatoire pour reproductibilite (defaut: 42)'
    )
    
    args = parser.parse_args()
    
    # Ajuster test si necessaire
    if args.test is None or args.test == 0.15:
        args.test = 1.0 - args.train - args.val
    
    print(f"\n╔════════════════════════════════════════╗")
    print(f"║       PubMed Corpus Split Tool         ║")
    print(f"╚════════════════════════════════════════╝")
    
    if args.mode == 'mini':
        cmd_mini(args)
    elif args.mode == 'split':
        cmd_split(args)
    elif args.mode == 'info':
        cmd_info(args)
    
    print("\n✅ Done!\n")


if __name__ == '__main__':
    main()