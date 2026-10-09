"""Exécution des requêtes SQL et calcul de l'execution accuracy.

Convention d'évaluation (identique à BIRD) : une requête générée est correcte si,
exécutée sur la même base, elle renvoie le même ensemble de lignes que la requête de référence.
"""

import json
import re
from pathlib import Path

import pandas as pd

from src.data import run_query


# ---------------------------------------------------------------------------
# Nettoyage de la sortie du modèle
# ---------------------------------------------------------------------------

def extract_sql(text):
    """Extrait la requête SQL de la réponse brute du modèle.

    Gère les blocs ```sql ... ```, le texte autour, et garde la première instruction.
    """
    text = str(text).strip()
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if block:
        text = block.group(1).strip()
    else:
        start = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
        if start:
            text = text[start.start():]
    first = text.split(";")[0].strip()
    return first + ";" if first else ""


def normalize_sql(sql):
    """Normalisation simple pour l'exact match (minuscules, espaces, point-virgule final)."""
    sql = re.sub(r"\s+", " ", str(sql)).strip().rstrip(";").strip()
    return sql.lower()


# ---------------------------------------------------------------------------
# Comparaison des résultats
# ---------------------------------------------------------------------------

def _normalize_value(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def results_match(rows_a, rows_b):
    """Vrai si les deux résultats contiennent le même ensemble de lignes (convention BIRD)."""
    a = {tuple(_normalize_value(v) for v in row) for row in rows_a}
    b = {tuple(_normalize_value(v) for v in row) for row in rows_b}
    return a == b


def evaluate_one(context, gold_sql, pred_sql):
    """Évalue une prédiction. Renvoie un dict avec les indicateurs."""
    gold_ok, gold_rows = run_query(context, gold_sql)
    pred_ok, pred_rows = run_query(context, pred_sql) if pred_sql else (False, "requête vide")
    return {
        "executable": pred_ok,
        "execution_match": bool(gold_ok and pred_ok and results_match(gold_rows, pred_rows)),
        "exact_match": normalize_sql(gold_sql) == normalize_sql(pred_sql),
        "pred_error": None if pred_ok else str(pred_rows),
    }


def evaluate_predictions(records, raw_outputs):
    """Évalue une liste de prédictions brutes. Renvoie un DataFrame (une ligne par question)."""
    rows = []
    for rec, raw in zip(records, raw_outputs):
        pred = extract_sql(raw)
        res = evaluate_one(rec["context"], rec["gold_sql"], pred)
        rows.append({
            "id": rec["id"],
            "domain": rec["domain"],
            "sql_complexity": rec["sql_complexity"],
            "question": rec["question"],
            "gold_sql": rec["gold_sql"],
            "raw_output": raw,
            "pred_sql": pred,
            **res,
        })
    return pd.DataFrame(rows)


def summarize(df):
    """Résumé global des métriques (en %)."""
    return {
        "n": int(len(df)),
        "execution_accuracy": round(float(100 * df["execution_match"].mean()), 2),
        "valid_sql_rate": round(float(100 * df["executable"].mean()), 2),
        "exact_match": round(float(100 * df["exact_match"].mean()), 2),
    }


def summarize_by(df, column="sql_complexity"):
    """Execution accuracy par catégorie (complexité, domaine...)."""
    return (df.groupby(column)
              .agg(n=("execution_match", "size"),
                   execution_accuracy=("execution_match", lambda s: round(100 * s.mean(), 1)),
                   valid_sql_rate=("executable", lambda s: round(100 * s.mean(), 1)))
              .sort_values("n", ascending=False))


# ---------------------------------------------------------------------------
# Statistiques pour la comparaison baseline / fine-tuné
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    """Intervalle de confiance à 95 % (Wilson) d'une proportion k/n, en %."""
    import math
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (round(100 * (center - half), 1), round(100 * (center + half), 1))


def mcnemar_test(correct_a, correct_b):
    """Test exact de McNemar sur des prédictions appariées (mêmes questions).

    Ne regarde que les questions où les deux modèles diffèrent :
      b = A juste et B faux, c = A faux et B juste.
    Renvoie (b, c, p_value). p < 0.05 : la différence n'est pas due au hasard.
    """
    import math
    b = int(sum(1 for x, y in zip(correct_a, correct_b) if x and not y))
    c = int(sum(1 for x, y in zip(correct_a, correct_b) if not x and y))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, p)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)