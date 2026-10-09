# Text-to-SQL financier avec QLoRA

**Fine-tuning efficace d'un petit LLM open source pour interroger des bases financières en langage naturel**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Modèle](https://img.shields.io/badge/Mod%C3%A8le-Qwen2.5--Coder--1.5B-2a78d6)
![Méthode](https://img.shields.io/badge/M%C3%A9thode-QLoRA%20(r%3D16)-eb6834)
![GPU](https://img.shields.io/badge/GPU-T4%20gratuit-76B900?logo=nvidia&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-Apache%202.0-52514e)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eya-bouhmida/finance-text2sql-lora/blob/main/projet_final.ipynb)

| Résultat | Gain | Significativité | Coût |
|:---:|:---:|:---:|:---:|
| **71,5 %** d'execution accuracy | **+12,7 points** vs modèle de base | McNemar **p = 0,0002** | **1,2 %** des paramètres · **18,5 min** · **85 Mo** |

---

## 1. Problématique

Dans une institution financière, les analystes, contrôleurs et managers ont besoin d'interroger des bases SQL mais ne savent pas écrire de SQL : ils dépendent de l'équipe data, ce qui crée des délais. Les grands LLM propriétaires savent traduire une question en SQL, mais une banque ne peut pas toujours leur envoyer ses schémas et ses questions, pour des raisons de confidentialité, de conformité et de coût.

> **Question du projet :** un petit modèle open source, fine-tuné avec LoRA sur un seul GPU, peut-il générer des requêtes SQL fiables sur des bases financières, tout en restant déployable en interne ?

```
Question en langage naturel ---+
                               +--> LLM fine-tuné --> Requête SQL --> Exécution --> Résultat
Schéma de la base (CREATE) ----+
```

Le modèle ne voit que le **schéma** de la base, **jamais les données** : c'est le code qui exécute la requête générée.

## 2. État de l'art

Plusieurs travaux ont étudié la génération de requêtes SQL dans la finance. **BIRD** (2023) montre que les LLM peinent sur des bases réelles, où seuls les très gros modèles atteignent environ 80 %. **BookSQL** (2024) confirme que même GPT-4 échoue souvent en comptabilité. **FinSQL** (2024) montre que LoRA permet d'adapter un LLM à la finance à faible coût. Ce projet se place dans un cadre plus contraint : un modèle de 1,5 milliard de paramètres et un seul GPU gratuit.

## 3. Dataset

[Gretel `synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache 2.0) : environ 105 000 exemples synthétiques sur 100 domaines, **filtrés sur les domaines financiers**. Chaque exemple contient une question, le schéma et les données (`CREATE TABLE` + `INSERT`) et la requête de référence, ce qui permet d'**exécuter** les requêtes pour les évaluer.

## 4. Nettoyage et préparation

- Requêtes en **lecture seule** (`SELECT`) uniquement : un analyste interroge la base, il ne la modifie pas
- Suppression des doublons
- Vérification que chaque requête **s'exécute dans SQLite**
- Test : suppression des requêtes à résultat vide, et retrait du train des questions présentes dans le test
- Format conversation : système (consigne), utilisateur (schéma + question), assistant (requête SQL)

## 5. Baseline

Le modèle **[Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)**, sans entraînement, obtient **58,9 %** d'execution accuracy. Il maîtrise la syntaxe SQL (92,4 % de requêtes valides), mais un tiers de ses réponses sont des requêtes valides qui renvoient un mauvais résultat.

## 6. Entraînement avec QLoRA

Le modèle de base est chargé en **4 bits** et gelé ; seuls de petits adaptateurs **LoRA** (ΔW = B·A) sont entraînés.

| Hyperparamètre | Valeur |
|---|---|
| Rang `r` / `lora_alpha` | 16 / 32 |
| Couches ciblées | toutes les couches linéaires |
| Learning rate | 2e-4 (cosine) |
| Epochs / batch effectif | 2 / 16 |
| Loss | sur la requête SQL uniquement |

| Paramètres entraînés | Durée | Adaptateur |
|:---:|:---:|:---:|
| ≈ 18,5 M (≈ 1,2 % du modèle) | 18,5 min sur un T4 | 85 Mo (contre ≈ 3 Go pour le modèle) |

![Courbes d'apprentissage](results/learning_curves.png)

Pas de sous-apprentissage ; au début de la deuxième epoch, la loss d'entraînement chute alors que la validation se stabilise (début de mémorisation, sans surapprentissage nuisible). Le meilleur checkpoint de validation est conservé.

## 7. Résultats

Évaluation sur **158 questions de test**, avec le même chargement (float16) et le même décodage (glouton, déterministe) pour les deux modèles.

| Modèle | Execution accuracy | IC 95 % | SQL valide |
|---|:---:|:---:|:---:|
| Baseline (Qwen2.5-Coder-1.5B) | 58,9 % | [51,1 ; 66,2] | 92,4 % |
| **Fine-tuné QLoRA** | **71,5 %** | **[64,0 ; 78,0]** | **98,7 %** |

## 8. Benchmarking

- **Significativité :** le fine-tuning corrige **24 questions** et n'en casse que **4** (test de McNemar exact, **p = 0,0002**).
- **Types d'erreurs :** les requêtes invalides passent de 12 à 2 ; les requêtes valides mais fausses de 53 à 43.
- **Par complexité :** gains les plus nets sur les agrégations (+19 points) et les requêtes simples (+11 points).

![Execution accuracy par complexité](results/accuracy_by_complexity.png)

![Répartition des résultats](results/outcomes.png)

## 9. Limitations

- **Données synthétiques** : schémas petits et propres, loin des bases bancaires réelles
- **Jeu de test réduit** (158 questions, incertitude d'environ ±7 points) ; certaines catégories comptent moins de 10 questions
- **Même distribution** pour le train et le test : une partie du gain peut refléter l'adaptation au style du dataset
- **Une seule configuration** (r = 16, une seule graine), pas d'étude d'ablation
- SQLite et questions en anglais uniquement

**Perspectives :** évaluation sur la base bancaire réelle `financial` de BIRD, comparaison de plusieurs rangs LoRA, comparaison avec un grand modèle propriétaire, questions en français, intégration dans un agent conversationnel.

## 10. Conclusion

Un petit modèle open source de 1,5 milliard de paramètres, fine-tuné avec QLoRA sur un seul GPU gratuit, passe de **58,9 %** à **71,5 %** d'execution accuracy sur des requêtes SQL financières (**+12,7 points**, p = 0,0002), en n'entraînant qu'environ **1,2 % de ses paramètres** en **18,5 minutes**, avec un adaptateur de **85 Mo**. Il reste déployable en interne, sans envoyer de données à une API externe. Le principal point faible reste la logique des requêtes ; ces résultats sont à confirmer sur des bases bancaires réelles.

---

## Structure du projet

```
finance-text2sql-lora/
├── projet_final.ipynb          ← notebook complet (les 11 étapes)
├── notebooks/
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_entrainement_lora.ipynb
│   └── 04_resultats_benchmarking_conclusion.ipynb
├── src/
│   ├── data.py       ← chargement, filtrage finance, nettoyage, prompts
│   ├── model.py      ← chargement du modèle et génération
│   ├── train.py      ← configuration QLoRA et entraînement
│   ├── evaluate.py   ← execution accuracy, intervalles de confiance, McNemar
│   └── viz.py        ← style des graphiques et tableaux
├── results/          ← métriques et figures
└── requirements.txt
```

## Reproduire les résultats

1. Ouvrir les notebooks dans **Google Colab** avec un **GPU T4** (gratuit)
2. Exécuter dans l'ordre : `01` (données) → `02` (baseline) → `03` (entraînement, ≈ 20 min) → `04` (évaluation)
3. Les données, checkpoints et l'adaptateur sont sauvegardés sur Google Drive (`MyDrive/text2sql-finance-qlora/`)

## Références

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
2. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
3. Li et al., *Can LLM Already Serve as a Database Interface? (BIRD)*, NeurIPS 2023. [arXiv:2305.03111](https://arxiv.org/abs/2305.03111)
4. Kumar et al., *BookSQL: A Large Scale Text-to-SQL Dataset for Accounting Domain*, NAACL 2024. [ACL Anthology](https://aclanthology.org/2024.naacl-long.28)
5. Zhang et al., *FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial Analysis*, SIGMOD 2024. [arXiv:2401.10506](https://arxiv.org/abs/2401.10506)

---

**Remarque :** ce projet a été réalisé avec l'assistance de Claude (Claude Opus 5.5, Anthropic), utilisé pour comprendre les concepts (LoRA, QLoRA, Text-to-SQL), structurer le code, corriger des erreurs et rédiger la documentation. Les choix du projet, l'exécution des expériences et l'analyse des résultats ont été réalisés et vérifiés par l'auteure.

**Auteure :** Eya Bouhmida · [GitHub](https://github.com/eya-bouhmida)
