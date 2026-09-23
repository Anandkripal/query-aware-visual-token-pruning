# Query-Aware Visual Token Pruning

An empirical study of **query-aware visual token pruning** in multimodal large language models (MLLMs), using pruning as a diagnostic probe of whether attention identifies visual evidence that is useful for answering a given query.

The project evaluates how different visual-token selection strategies affect the accuracy–efficiency trade-off in **LLaVA-1.5-7B** and **Qwen2.5-VL-3B-Instruct**.

## Overview

Multimodal language models can represent an image as hundreds of visual tokens. Processing all of these tokens increases decoder-side computation, even though many tokens may be irrelevant to a particular question.

This project asks:

> Can a model's own query-conditioned attention identify the visual tokens that matter most for answering a question?

Rather than treating token pruning only as an acceleration technique, the experiments use pruning as a controlled intervention for studying multimodal attention.

## Methods

The experiments compare:

- **No compression** — all visual tokens are retained.
- **Random retention** — a random subset of visual tokens is kept.
- **Uniform spatial retention** — tokens are sampled across the visual sequence to preserve broad spatial coverage.
- **Attention-based retention** — tokens with the highest query-conditioned attention scores are retained.
- **Inverse-attention retention** — the lowest-attention tokens are retained as a causal sanity check.

Attention is evaluated at multiple transformer depths to test whether visual relevance changes across layers.

## Models

- **LLaVA-1.5-7B**
- **Qwen2.5-VL-3B-Instruct**

The two models use different token-reduction implementations, providing a useful cross-architecture comparison.

## Evaluation

Primary evaluation is performed on a controlled subset of **GQA testdev-balanced**, with an additional **MMBench** sanity check.

Experiments include:

- retention-rate sweeps
- early-, middle-, and late-layer attention comparisons
- random and uniform baselines
- inverse-attention causal probes
- question-type analysis
- analytical decoder FLOPs estimates
- accuracy–efficiency Pareto analysis

## Key Findings

The study finds that visual-token relevance is strongly **layer dependent**.

Middle-layer attention provides a substantially more useful pruning signal than early-layer attention, particularly under aggressive token reduction. Attention-based retention is most reliable for queries whose evidence is compact or localized, while spatial and relational questions are more fragile when visual context is heavily reduced.

The inverse-attention experiments also show that retaining low-attention tokens performs much worse than retaining high-attention tokens at the same token budget, indicating that the attention ranking contains useful information about visual evidence.

At the same time, random and uniform baselines remain competitive at higher retention rates, suggesting that some pruning robustness comes from redundancy in the visual-token sequence rather than perfect query-conditioned grounding.

## Repository Structure

```text
query-aware-visual-token-pruning/
├── README.md
├── requirements.txt
├── .gitignore
├── paper/
│   └── query_aware_visual_token_pruning.pdf
└── notebooks/
    └── query_aware_visual_token_pruning.ipynb
```

## Running the Notebook

The notebook was developed for **Google Colab** and uses Hugging Face models and datasets.

A CUDA-capable GPU is strongly recommended.

### Install dependencies

```bash
pip install -r requirements.txt
```

You will also need access to the relevant Hugging Face model weights and datasets. If required, configure your Hugging Face token in the notebook environment rather than committing credentials to GitHub.

## Main Experiment Configuration

The notebook includes experiments across retention rates such as:

```python
RETENTION_RATES = [0.9, 0.7, 0.5, 0.25, 0.1]
```

and candidate attention layers:

```python
LAYER_CANDIDATES = [2, 8, 15, 22]
```

It also includes analysis utilities for:

- overall accuracy
- structural question types
- semantic categories
- layer-depth ablations
- top-attention vs inverse-attention comparisons
- FLOPs estimation
- efficiency–accuracy visualisation

## Paper

The accompanying manuscript is available here:

**[Query-Aware Visual Token Pruning as a Diagnostic Probe of Multimodal Attention](paper/query_aware_visual_token_pruning.pdf)**

## Notes

The reported FLOPs values are analytical compute proxies rather than direct hardware latency measurements. The notebook also contains model-specific implementation details, so results should be interpreted in the context of the experimental setup described in the paper.

## Future Work

Potential extensions include:

- larger and more diverse multimodal benchmarks
- OCR- and counting-heavy tasks
- adaptive retention rates based on question type
- gradient-based or rollout-based attribution
- explicit cross-attention architectures
- region-level causal interventions
- measured wall-clock and memory benchmarks

## Acknowledgements

This repository contains work originally developed as part of a university research project.
