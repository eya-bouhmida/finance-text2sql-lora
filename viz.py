"""Style visuel commun du projet : couleurs, graphiques, tableaux et cartes de chiffres clés.

Couleurs issues d'une palette validée pour le daltonisme (contraste et séparation vérifiés).
"""

import html

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Markdown, display

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLORS = {
    "finetuned": "#2a78d6",   # bleu : le modèle fine-tuné (la série mise en avant)
    "baseline": "#b4b3ab",    # gris neutre : la référence
    "train": "#2a78d6",
    "val": "#eb6834",
    "good": "#0ca30c",        # correct
    "serious": "#ec835a",     # SQL valide mais faux
    "critical": "#d03b3b",    # SQL invalide
    "text": "#0b0b0b",
    "text_muted": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#fcfcfb",
}


def setup_style():
    """Style matplotlib sobre : axes discrets, grille horizontale légère, typographie lisible."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["axis"],
        "axes.labelcolor": COLORS["text_muted"],
        "axes.titlecolor": COLORS["text"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["text_muted"],
        "ytick.color": COLORS["text_muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Graphiques
# ---------------------------------------------------------------------------

def plot_learning_curves(train_loss, eval_loss, path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_loss["step"], train_loss["loss"], color=COLORS["train"], linewidth=2)
    ax.plot(eval_loss["step"], eval_loss["eval_loss"], color=COLORS["val"], linewidth=2,
            marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5)
    # Étiquettes directes en bout de courbe (plus lisibles qu'une légende)
    ax.annotate("Entraînement", (train_loss["step"].iloc[-1], train_loss["loss"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center",
                color=COLORS["text_muted"], fontsize=9)
    ax.annotate("Validation", (eval_loss["step"].iloc[-1], eval_loss["eval_loss"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center",
                color=COLORS["text_muted"], fontsize=9)
    ax.set_title("Courbes d'apprentissage")
    ax.set_xlabel("Étape d'entraînement")
    ax.set_ylabel("Loss")
    ax.set_xlim(right=train_loss["step"].max() * 1.12)
    if path:
        fig.savefig(path)
    plt.show()


def plot_by_complexity(by_cx, path=None):
    """Barres groupées baseline / fine-tuné par complexité, avec valeurs affichées."""
    import numpy as np

    labels = [f"{cx}\nn = {k}" for cx, k in zip(by_cx.index, by_cx["n"])]
    x = np.arange(len(by_cx))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar(x - w / 2, by_cx["Baseline (%)"], w, color=COLORS["baseline"],
                label="Baseline", edgecolor="white", linewidth=2)
    b2 = ax.bar(x + w / 2, by_cx["Fine-tuné (%)"], w, color=COLORS["finetuned"],
                label="Fine-tuné QLoRA", edgecolor="white", linewidth=2)
    ax.bar_label(b1, fmt="%.0f", padding=3, fontsize=9, color=COLORS["text_muted"])
    ax.bar_label(b2, fmt="%.0f", padding=3, fontsize=9, color=COLORS["text"], fontweight="bold")
    # Les catégories trop petites sont signalées en grisé
    for i, k in enumerate(by_cx["n"]):
        if k < 10:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", zorder=0)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_title("Execution accuracy par complexité de requête", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    ax.text(1.0, -0.2, "Zone grisée : moins de 10 questions, résultat peu fiable",
            transform=ax.transAxes, ha="right", fontsize=8, color=COLORS["text_muted"])
    if path:
        fig.savefig(path)
    plt.show()


def plot_outcomes(outcomes, path=None):
    """Barres empilées horizontales : correct / valide mais faux / invalide."""
    pct = 100 * outcomes / outcomes.sum()
    colors = [COLORS["good"], COLORS["serious"], COLORS["critical"]]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.grid(False)
    rows = list(pct.columns)[::-1]          # fine-tuné en haut
    for y, model in enumerate(rows):
        left = 0
        for (cat, color) in zip(pct.index, colors):
            v = pct.loc[cat, model]
            ax.barh(y, v, left=left, color=color, edgecolor="white", linewidth=2, height=0.6)
            if v >= 6:
                ax.text(left + v / 2, y, f"{v:.0f} %", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            left += v
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% des questions de test")
    ax.set_title("Répartition des résultats")
    for s in ["left", "bottom"]:
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, ["✓ Correct", "≈ SQL valide mais faux", "✗ SQL invalide"],
              loc="upper center", bbox_to_anchor=(0.5, -0.38), ncols=3)
    if path:
        fig.savefig(path)
    plt.show()


# ---------------------------------------------------------------------------
# Tableaux et cartes
# ---------------------------------------------------------------------------

TABLE_STYLES = [
    {"selector": "th", "props": [("background-color", "#f0efec"), ("color", "#0b0b0b"),
                                 ("font-weight", "600"), ("text-align", "left"),
                                 ("padding", "6px 12px")]},
    {"selector": "td", "props": [("padding", "6px 12px")]},
    {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"),
                                      ("font-weight", "600"), ("font-size", "13px"),
                                      ("color", "#0b0b0b"), ("padding-bottom", "6px")]},
]


def styled(df, caption=None, precision=1):
    """Tableau pandas mis en forme (en-têtes, espacement, nombres arrondis)."""
    s = df.style.set_table_styles(TABLE_STYLES).format(precision=precision, thousands=" ")
    if caption:
        s = s.set_caption(caption)
    return s


def kpi_cards(items):
    """Affiche une rangée de cartes de chiffres clés.

    items : liste de dict {"label", "value", "sub" (optionnel), "accent" (optionnel)}
    """
    cards = []
    for it in items:
        accent = it.get("accent", COLORS["finetuned"])
        sub = f'<div style="font-size:12px;color:#52514e;margin-top:4px">{html.escape(it["sub"])}</div>' if it.get("sub") else ""
        cards.append(
            f'<div style="flex:1;min-width:150px;background:#fcfcfb;border:1px solid #e1e0d9;'
            f'border-top:4px solid {accent};border-radius:8px;padding:14px 16px">'
            f'<div style="font-size:12px;color:#52514e;text-transform:uppercase;letter-spacing:.04em">{html.escape(it["label"])}</div>'
            f'<div style="font-size:26px;font-weight:700;color:#0b0b0b;margin-top:4px">{html.escape(str(it["value"]))}</div>'
            f'{sub}</div>'
        )
    display(HTML('<div style="display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 4px">' + "".join(cards) + "</div>"))


def show_messages(messages):
    """Affiche une conversation (système / utilisateur / assistant) de manière lisible."""
    icons = {"system": "⚙️ Système", "user": "👤 Utilisateur", "assistant": "🤖 Assistant (réponse attendue)"}
    parts = []
    for m in messages:
        lang = "sql" if m["role"] == "assistant" else ""
        parts.append(f"**{icons.get(m['role'], m['role'])}**\n```{lang}\n{m['content']}\n```")
    display(Markdown("\n\n".join(parts)))


def show_examples(df, title, k=3):
    """Affiche des exemples question / référence / baseline / fine-tuné avec coloration SQL."""
    out = [f"#### {title} — {len(df)} questions"]
    for _, r in df.head(k).iterrows():
        b_icon = "✅" if r["baseline_ok"] else "❌"
        f_icon = "✅" if r["finetuned_ok"] else "❌"
        out.append(
            f"**❓ {r['question']}**\n\n"
            f"Référence\n```sql\n{r['gold_sql']}\n```\n"
            f"{b_icon} Baseline\n```sql\n{r['baseline_sql']}\n```\n"
            f"{f_icon} Fine-tuné\n```sql\n{r['finetuned_sql']}\n```\n---"
        )
    display(Markdown("\n\n".join(out)))


def fr(x, decimals=1):
    """Nombre au format français (virgule décimale)."""
    return f"{x:.{decimals}f}".replace(".", ",")
