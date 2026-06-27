# ZeroRel: Relational Reasoning via Graph-guided Large Language Models

This repository contains the code for the paper **"ZeroRel: Relational Reasoning via Graph-guided Large Language Models"**.

<p align="center">
  <img src="2.jpg" alt="RelZero Framework" width="90%">
</p>

---

## Overview

Relational databases (RDBs) are essential in many real-world applications, including e-commerce, social media, healthcare, and industrial systems. With the rapid progress of large language models (LLMs), leveraging LLMs for reasoning over relational data has become an increasingly important research direction.

However, existing approaches still face two major limitations:

1. **Text-based serialization of RDBs** often leads to excessive context length and loss of structural information.
2. **Graph-based relational modeling** usually depends on supervised learning with large amounts of task-specific labels, which limits scalability.

To address these issues, we propose **RelZero**, a **self-supervised** framework for relational reasoning over RDBs. RelZero treats context sparsity as a controllable curriculum variable and uses it to drive a progressive transition from **semantic-dominant inference** to **structure-aware relational reasoning**.

Our framework consists of two key modules:

- **Graph-guided Prompt Alignment (GrPA)**: encodes multi-table relational structures with a heterogeneous GNN and projects the resulting structural representations into the semantic space of LLMs.
- **Progressive Sparsity-based Context Refinement (PSCR)**: gradually reduces visible attribute context and acts as an information bottleneck, encouraging the model to internalize cross-table dependencies instead of relying on superficial semantic shortcuts.

Extensive experiments on **7 datasets and 12 downstream tasks** demonstrate the effectiveness of RelZero. Notably, **RelZero trained without any task-specific labels achieves an average improvement of 6.24% over models trained with supervised labels**.

---

## Key Features

- **Label-free relational reasoning** through self-supervised learning
- **Graph-guided structural prompting** for multi-table databases
- **Progressive sparsity curriculum** to encourage robust relational inference
- **Compatible with RelBench** benchmarks for relational learning
- **LLM-based framework** that bridges structural modeling and semantic reasoning

---

To facilitate quick reproduction, we publicly release the trained checkpoints for all tasks across the three datasets. The checkpoints can be downloaded from:  [10.5281/zenodo.20251716](https://zenodo.org/records/20251716)

After downloading the checkpoints, you can directly run testing with:

```bash
python main.py  --dataset_source=rel-avito --task_source=user-clicks --testing  --best_model_path=source_best_model_clicks.pt
```




---





