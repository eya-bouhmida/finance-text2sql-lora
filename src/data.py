"""Chargement, filtrage du domaine finance, nettoyage et préparation des prompts.

Dataset : gretelai/synthetic_text_to_sql (Apache 2.0)
Colonnes : id, domain, domain_description, sql_complexity, sql_complexity_description,
           sql_task_type, sql_task_type_description, sql_prompt, sql_context, sql, sql_explanation
"""

import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

DATASET_NAME = "gretelai/synthetic_text_to_sql"

# Mots-clés utilisés pour repérer les domaines financiers dans la colonne `domain`.
# La liste des domaines retenus est affichée dans le notebook pour vérification manuelle.
FINANCE_KEYWORDS = [
    "financ", "bank", "fintech", "invest", "venture capital", "insurance",
    "credit", "loan", "trading", "stock", "fraud", "accounting", "wealth",
]

# On garde uniquement les tâches de lecture (SELECT) : c'est le cas d'usage d'un analyste
# qui interroge une base. Les tâches qui modifient la base (INSERT, UPDATE, CREATE...) sont exclues.
READ_ONLY_TASKS = ["analytics and reporting", "data retrieval"]


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_gretel():
    """Charge le dataset Gretel depuis Hugging Face et renvoie (train_df, test_df)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME)
    return ds["train"].to_pandas(), ds["test"].to_pandas()


# ---------------------------------------------------------------------------
# Filtrage du domaine finance
# ---------------------------------------------------------------------------

def find_finance_domains(domains, keywords=FINANCE_KEYWORDS):
    """Renvoie la liste triée des domaines dont le nom contient un mot-clé financier."""
    found = {d for d in domains if any(k in str(d).lower() for k in keywords)}
    return sorted(found)


def filter_domains(df, domains):
    """Garde uniquement les lignes appartenant aux domaines donnés."""
    return df[df["domain"].isin(domains)].copy()


# ---------------------------------------------------------------------------
# Nettoyage
# ---------------------------------------------------------------------------

def is_select_query(sql):
    """Vrai si la requête est une lecture (commence par SELECT ou WITH)."""
    return bool(re.match(r"^\s*(select|with)\b", str(sql), flags=re.IGNORECASE))


def run_query(context, sql, max_steps=1_000_000):
    """Crée une base SQLite en mémoire à partir de `context`, exécute `sql`.

    Renvoie (ok, résultat ou message d'erreur).
    Le progress handler interrompt les requêtes trop longues (boucles, produits cartésiens).
    """
    conn = sqlite3.connect(":memory:")
    steps = {"n": 0}

    def _guard():
        steps["n"] += 1
        return 1 if steps["n"] > max_steps // 1000 else 0

    conn.set_progress_handler(_guard, 1000)
    try:
        conn.executescript(context)
        rows = conn.execute(sql).fetchall()
        return True, rows
    except Exception as e:  # syntaxe non SQLite, table manquante, etc.
        return False, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def clean_dataset(df, require_non_empty=False):
    """Applique toutes les étapes de nettoyage et renvoie (df_propre, rapport).

    Étapes :
      1. tâches en lecture seule uniquement
      2. requêtes SELECT / WITH uniquement
      3. suppression des doublons (même question + même contexte)
      4. suppression des lignes vides
      5. vérification que le contexte et la requête s'exécutent dans SQLite
      6. (option) suppression des requêtes qui renvoient un résultat vide
    """
    report = {"départ": len(df)}

    df = df[df["sql_task_type"].isin(READ_ONLY_TASKS)]
    report["après filtre lecture seule"] = len(df)

    df = df[df["sql"].apply(is_select_query)]
    report["après filtre SELECT"] = len(df)

    df = df.drop_duplicates(subset=["sql_prompt", "sql_context"])
    report["après suppression doublons"] = len(df)

    df = df.dropna(subset=["sql_prompt", "sql_context", "sql"])
    df = df[df["sql_prompt"].str.strip().astype(bool) & df["sql"].str.strip().astype(bool)]
    report["après suppression vides"] = len(df)

    results = df.apply(lambda r: run_query(r["sql_context"], r["sql"]), axis=1)
    df = df.assign(
        executable=[ok for ok, _ in results],
        n_result_rows=[len(res) if ok else -1 for ok, res in results],
        error=[None if ok else res for ok, res in results],
    )
    errors = df.loc[~df["executable"], "error"]
    df = df[df["executable"]]
    report["après vérification exécution SQLite"] = len(df)

    if require_non_empty:
        df = df[df["n_result_rows"] > 0]
        report["après suppression résultats vides"] = len(df)

    df = df.drop(columns=["error"]).reset_index(drop=True)
    return df, report, errors


# ---------------------------------------------------------------------------
# Préparation des prompts
# ---------------------------------------------------------------------------

def schema_only(context):
    """Garde uniquement les instructions CREATE (TABLE / VIEW) du contexte.

    Le modèle ne voit que le schéma, jamais les données (INSERT) :
    c'est le fonctionnement réel d'un assistant Text-to-SQL en entreprise.
    """
    statements = [s.strip() for s in str(context).split(";") if s.strip()]
    creates = [s for s in statements if re.match(r"^create\b", s, flags=re.IGNORECASE)]
    return ";\n".join(creates) + ";" if creates else ""


SYSTEM_PROMPT = (
    "You are an expert SQL assistant for financial databases. "
    "Given a database schema and a question, write a single SQLite query that answers it. "
    "Return only the SQL query."
)


def build_messages(context, question, sql=None):
    """Construit la conversation (format chat) pour un exemple.

    Si `sql` est fourni, la réponse attendue est ajoutée (exemple d'entraînement).
    """
    user = f"### Schema:\n{schema_only(context)}\n\n### Question:\n{question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql.strip()})
    return messages


def to_records(df, with_answer=True):
    """Convertit un DataFrame en liste d'exemples prêts pour l'entraînement / l'évaluation."""
    records = []
    for _, r in df.iterrows():
        records.append({
            "id": int(r["id"]),
            "domain": r["domain"],
            "sql_complexity": r["sql_complexity"],
            "question": r["sql_prompt"],
            "context": r["sql_context"],      # complet (CREATE + INSERT) : sert à exécuter
            "gold_sql": r["sql"],
            "messages": build_messages(r["sql_context"], r["sql_prompt"],
                                       r["sql"] if with_answer else None),
        })
    return records


def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]