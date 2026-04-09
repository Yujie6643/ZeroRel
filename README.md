# ZeroRel: Relational Reasoning via Graph-guided Large Language Models

This repository contains the code for the paper **"ZeroRel: Relational Reasoning via Graph-guided Large Language Models"**, submitted to the **32nd SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2026), Research Track**.

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



## 📦 Installation

Install dependencies at once:
```bash
conda env create -f environment.yml
conda activate llm 

## Don’t pin pyg-lib / torch-scatter / torch-sparse / torch-cluster / torch-spline-conv in YAML. 
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

Alternatively, manually install packages in turn:
```bash
conda create -n ZeroRel python=3.11 && conda activate ZeroRel
pip install torch==2.8.0  --index-url https://download.pytorch.org/whl/cu124
pip install wandb pandas pillow pyarrow pooch
pip install relbench
pip install torch-frame 
pip install -U sentence-transformers   # for Glove 
pip install transformers peft
```

To enable modeling features via RelBench:
```bash
pip install relbench[full]
pip install pytorch_frame[full]  
```

Here, `Llama-3.1` is leveraged. Please log in to Huggingface for downloading the model weights directly. 

## 🗞️ Examples

### rel-avito (user-clicks)

```bash
python main.py  --dataset_source=rel-avito --task_source=user-clicks --pretrain --lr=0.001 --dropout=0.4  --text_embedder=mpnet  --loss_class_weight 0.8 0.2 --debug  
```

## 📚 Datasets

ZeroRel is evaluated on **7 real-world relational datasets** from [RelBench](https://relbench.stanford.edu).

These datasets span a wide range of multi-table relational scenarios, including user behavior modeling, event participation, advertising, e-commerce, question answering communities, retail forecasting, motorsport analytics, and clinical trial prediction.

- 🏟 **`rel-event`**: social event participation, repeat attendance, and user churn prediction  
- 🛍 **`rel-amazon`**: e-commerce user behavior, product interaction, and item lifespan prediction  
- 💬 **`rel-stack`**: question-answering community engagement, reputation, and badge-related prediction  
- 🧾 **`rel-avito`**: advertisement visits, user clicks, and click-through behavior prediction  
- 🏎 **`rel-f1`**: Formula 1 racing analytics, including driver performance and race outcome prediction  
- 🛒 **`rel-hm`**: retail transaction modeling and H\&M sales forecasting  
- 🧪 **`rel-trial`**: clinical trial outcome and adverse event prediction  

Please refer to the official RelBench benchmark for detailed dataset construction, schema information, and task definitions.



---





