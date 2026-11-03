# Sentiment Model Robustness Testing

Behavioral, metamorphic, and bias testing of a pretrained NLP sentiment classification model, applying software testing techniques to a system with no reliable test oracle.

## Overview

This project tests [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english), an open-source, binary (POSITIVE/NEGATIVE), DistilBERT-based sentiment classifier using:

- **Metamorphic testing** — checking whether the model's predictions stay consistent (or change predictably) under systematic input transformations, without requiring labeled ground truth
- **Perturbation-based robustness testing** — measuring sensitivity to character-level typo noise
- **Dataset-level accuracy evaluation** — measuring real accuracy degradation on labeled IMDB reviews under noise
- **Bias/fairness testing** — checking whether swapping only a person's name changes the predicted sentiment

Full methodology and findings are documented in the project's accompanying report.

## Project structure

```text
sentiment-model-robustness-testing/
|-- tests/
|   |-- baseline_test.py
|   |-- metamorphic_test.py
|   |-- robustness_test.py
|   `-- bias_test.py
|-- charts/
|   |-- metamorphic_chart_gen.py
|   |-- robustness_chart_gen.py
|   `-- bias_chart_gen.py
|-- results/
|   |-- data/
|   `-- charts/
|-- requirements.txt
`-- README.md
```
