# Automotive Sentiment Analysis — DistilBERT Fine-Tuning

A follow-up to my earlier [TF-IDF vs Word2Vec automotive sentiment project](#), where I fine-tune a transformer model (DistilBERT) on the same task to see whether contextual understanding can break past a recall ceiling that classical ML couldn't solve.

## Background

In the original project, I built a 3-class (Negative / Neutral / Positive) sentiment classifier on 4,716 car reviews using TF-IDF + Logistic Regression. The best classical model reached:

- **Macro F1: 0.67** | Accuracy: 69% | **Neutral recall: 48%**

Every classical variant I tried — bigrams, Word2Vec, Random Forest, XGBoost, tighter label boundaries — hit the same wall: **Neutral-class recall stuck at 43–48%**, regardless of model or vectorizer. This looked like a label-ambiguity ceiling (a 2-star and a 3.9-star review both get labeled "Neutral," but don't read the same way in the text), not something classical bag-of-words models could fix.

My mentor's suggestion: try a transformer model — its contextual attention mechanism might succeed where bag-of-words representations couldn't.

## What I did

### 1. Fine-tuned DistilBERT on the same 4,716-row dataset (apples-to-apples)

To keep the comparison fair, I first fine-tuned `distilbert-base-uncased` on the **exact same dataset** used for the TF-IDF baseline, with the same train/val/test logic.

| Version | Config | Macro F1 | Accuracy | Neutral Recall |
|---|---|---|---|---|
| TF-IDF + bigram + LogReg (baseline) | — | 0.67 | 69% | 48% |
| DistilBERT v1 | 4 epochs, LR 2e-5 | 0.65 | 68% | 51% |
| DistilBERT v2 | 3 epochs, LR 1e-5 | 0.63 | 67% | 51% |
| DistilBERT v3 | 6 epochs, LR 2e-5, best@epoch 3 | 0.66 | 68% | 53% |

**Honest takeaway:** on identical data, DistilBERT did **not** clearly beat the TF-IDF baseline on overall macro F1 — it came close (0.66 vs 0.67) and gave a small Neutral-recall bump (53% vs 48%), but this supported rather than refuted my original hypothesis: the problem looked more like label ambiguity than something a better model alone could fix.

### 2. Expanded the dataset

Rather than only tuning hyperparameters further, I went after the data itself. I combined all 49 brand files from the [Edmunds car reviews Kaggle dataset](https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews), applied the same rating→label logic (≥4 = Positive, 2–4 = Neutral, <2 = Negative), deduplicated against the original dataset, and merged:

- **Original:** 4,716 rows (heavily Positive-skewed)
- **Expanded:** 27,155 rows — Positive 9,982 / Neutral 9,529 / Negative 7,644 (far more balanced)

### 3. Re-trained DistilBERT on the expanded dataset

| Version | Config | Macro F1 | Accuracy | Neutral Recall |
|---|---|---|---|---|
| DistilBERT v4 | 27,155 rows, 3 epochs, LR 2e-5 | **0.71** | **72%** | **61%** |

This is the first version to clearly beat the TF-IDF baseline on every metric, including a **13-point jump in Neutral recall** (48% → 61%).

**Caveat, stated honestly:** this isn't a perfectly clean same-data comparison — the TF-IDF baseline's 0.67 was measured on a test split of the original 4,716-row dataset, while DistilBERT v4's 0.71 was measured on a test split of the expanded 27,155-row dataset. Both more data *and* a better architecture likely contributed to the gain; this experiment doesn't isolate which mattered more.

## What I learned

- A better model isn't automatically a better result — on identical data, DistilBERT only barely matched TF-IDF, not decisively beat it.
- Training curve smoothness isn't a reliable predictor of final performance — a lower learning rate (v2) gave the smoothest training curve but the worst test result.
- Scaling up data (with better class balance) moved the needle far more than any single hyperparameter change did.
- Subword tokenization and attention genuinely help with harder, more context-dependent classes (Neutral, Negative) — the gain wasn't uniform across classes.

## Next steps

- Explore cloud deployment (AWS/GCP) instead of a static client-side site, since inference now requires a full model forward pass rather than simple TF-IDF weight math.

## Tech stack

Python, pandas, scikit-learn, Hugging Face `transformers` & `datasets`, PyTorch, Google Colab (T4 GPU)
