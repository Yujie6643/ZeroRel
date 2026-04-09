from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import re
import random
from typing import Any, Dict, List
import contextlib
import random
import copy
import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict, Sigmoid, Sequential, Linear, Dropout
from transformers import AutoTokenizer, AutoModel
import torch_frame.data
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from relbench.base import TaskType
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from torch_geometric.nn import HeteroConv, GCNConv, GraphConv, GATConv
from torch_frame.utils.infer_stype import infer_series_stype
from torch_frame import stype
from utils import question_dict, description_dict, initialize_weights
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from torch import nn
import numpy as np
import math


# llama model type: https://huggingface.co/meta-llama
# encode special tokens for Llama 3.2: https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
BOS = '<|begin_of_text|>'
EOS_USER = '<|eot_id|>'  # end of the message in a turn
EOS = '<|end_of_text|>'
IGNORE_INDEX = -100  # default = -100 in Pytorch CrossEntropyLoss, https://github.com/huggingface/transformers/issues/29819
accept_stypes = [stype.numerical, stype.categorical, stype.text_tokenized, stype.multicategorical, stype.text_embedded]   # no timestamp



def _to_list(x):
        try:
            return list(x)
        except Exception:
            return []

def _is_timestamp_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["time", "date", "timestamp", "ts", "dt"])

def _is_text_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["text", "desc", "description", "content", "message", "comment"])

def _is_categorical_like(name: str) -> bool:
    n = name.lower()
    return n.endswith("type") or n.endswith("category") or "flag" in n or "status" in n

def _is_numeric_like(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ["count","num","total","amount","price","score","rank","value","size","len","width","height","weight"])

def _is_pk_name(table: str, col: str) -> bool:
    t = table.lower()
    c = col.lower()
    return c in ("id", f"{t}id", f"{t}_id") or c.endswith("_pk")

def _fk_patterns(table_names: List[str]) -> List[re.Pattern]:
    pats = []
    for t in table_names:
        t_low = t.lower()
        pats.append(re.compile(fr"^{t_low}id$"))
        pats.append(re.compile(fr"^{t_low}_id$"))
    pats.append(re.compile(r".+id$"))
    pats.append(re.compile(r".+_id$"))
    return pats
def _get_max_len_from_model(model) -> int:
        cfg = getattr(model, "config", None)
        return getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "max_sequence_length", 4096) or 4096

def _pack_causal_example(user_text: str, tgt_text: str, tokenizer, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must define bos_token_id and eos_token_id.")

    u_ids = tokenizer(user_text, add_special_tokens=False).input_ids
    a_ids = tokenizer(tgt_text,  add_special_tokens=False).input_ids

    need = 1 + len(u_ids) + len(a_ids) + 1
    if need > max_len:
        room_user = max(0, max_len - 2 - len(a_ids))
        u_ids = u_ids[:room_user]

    input_ids = [bos_id] + u_ids + a_ids + [eos_id]
    labels    = [IGNORE_INDEX] * (1 + len(u_ids)) + a_ids + [eos_id]
    attn_mask = [1] * len(input_ids)

    ids  = torch.tensor(input_ids, dtype=torch.long)
    labs = torch.tensor(labels,    dtype=torch.long)
    mask = torch.tensor(attn_mask, dtype=torch.long)
    return ids, labs, mask

def infer_schema_facts(db, batch, main_table: str, max_cols: int = 64) -> Dict:

        all_tables = sorted(_to_list(db.table_dict.keys()))
        facts = {
            "main_table": main_table,
            "tables": [],
            "relations": [],
            "has_time": False
        }

        # 先收集各表列
        name_to_cols = {}
        for t in all_tables:
            try:
                df = getattr(batch[t], "df", None)
                cols = [str(c) for c in _to_list(getattr(df, "columns", []))]
            except Exception:
                cols = []
            cols = cols[:max_cols]
            name_to_cols[t] = cols

            # 列类型粗分类计数（基于列名）
            ts_cols = [c for c in cols if _is_timestamp_like(c)]
            txt_cols = [c for c in cols if _is_text_like(c)]
            cat_cols = [c for c in cols if _is_categorical_like(c)]
            num_cols = [c for c in cols if _is_numeric_like(c)]

            facts["tables"].append({
                "name": t,
                "columns": cols,
                "n_columns": len(cols),
                "n_time_like": len(ts_cols),
                "n_text_like": len(txt_cols),
                "n_categorical_like": len(cat_cols),
                "n_numeric_like": len(num_cols),
                "pk_candidates": [c for c in cols if _is_pk_name(t, c)],
                "has_time_like": len(ts_cols) > 0
            })
            facts["has_time"] = facts["has_time"] or (len(ts_cols) > 0)


        table_names = [t for t in all_tables]
        fk_pats = _fk_patterns(table_names)
        for t, cols in name_to_cols.items():
            for c in cols:
                c_low = c.lower()
                if any(p.match(c_low) for p in fk_pats):

                    dst = None
                    for cand in table_names:
                        if c_low in (f"{cand.lower()}id", f"{cand.lower()}_id"):
                            dst = cand
                            break
                    facts["relations"].append({
                        "src_table": t,
                        "src_col": c,
                        "dst_table_guess": dst
                    })

        return facts

def build_schema_prompt_generic(facts: Dict, max_tables: int = 64) -> str:
    lines: List[str] = []
    lines.append("You are given a relational database schema.")
    lines.append(f"Main table: {facts['main_table']}")
    lines.append("Tables and columns:")

    for t in sorted(facts["tables"], key=lambda x: (x["name"] != facts["main_table"], x["name"]))[:max_tables]:
        cols = t["columns"]
        cols_str = ", ".join(cols) if cols else "[unknown]"
        lines.append(f"- {t['name']} ({t['n_columns']} cols): {cols_str}")

    if facts["relations"]:
        lines.append("Potential foreign-key-like columns (heuristic, name-based):")
        for r in facts["relations"][:max_tables]:
            tail = f" -> {r['dst_table_guess']}" if r["dst_table_guess"] else ""
            lines.append(f"  - {r['src_table']}.{r['src_col']}{tail}")


    lines.append("Hint: Some tables may contain timestamp/date columns." if facts["has_time"] else
                "Hint: No obvious timestamp/date columns were detected by name heuristics.")


    lines.append(
        "Summarize the database at a high level: key entities, event/interaction tables, "
        "reference tables, likely foreign-key relations, and analytics this schema enables."
    )
    return "\n".join(lines)

def build_schema_target_generic(facts: Dict) -> str:

    n_tables = len(facts["tables"])
    n_rel = len(facts["relations"])

    # 时间备注
    time_note = (
        "Some tables appear to include temporal fields."
        if facts["has_time"]
        else "Temporal fields are not obvious from column names."
    )


    table_names = ", ".join([t["name"] for t in facts["tables"]])


    main_table = facts["main_table"]
    main_table_info = next(t for t in facts["tables"] if t["name"] == main_table)
    main_cols_preview = ", ".join(main_table_info["columns"][:6])

    other_tables_desc = " ; ".join(
        [
            f"{t['name']}: " + ", ".join(t["columns"][:6])
            for t in facts["tables"]
            if t["name"] != main_table
        ]
    )


    fk_preview = ", ".join(
        [f"{r['src_table']}.{r['src_col']}" for r in facts["relations"][:6]]
    )

    return (
        f"This database contains approximately {n_tables} tables, including {table_names}. "
        f"The main table is '{main_table}', with columns such as {main_cols_preview}. "

        f"Other tables include: {other_tables_desc}. "

        f"Heuristic inspection identifies about {n_rel} foreign-key-like columns, "
        f"such as {fk_preview}. {time_note} "

        "These structures suggest connections between entity tables, event records, and user interactions, "
        "enabling multi-table joins, temporal feature construction, and relational analytics. "
        "Such a schema can support tasks like user-centric predictions, event behavior modeling, "
        "relationship inference, and sequence-based learning based on time-aware event logs."
    )


def _pad_batch(tensors: List[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_val)
        for i, t in enumerate(tensors):
            out[i, : t.size(0)] = t
        return out

class RowMLPAggregator(nn.Module):
    def __init__(self, hidden_dim, neighbor_k=2):
        super().__init__()
        self.neighbor_k = neighbor_k

        # seed + 上下邻居 (2k+1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * (2 * neighbor_k + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_rows, seed_indices):

        B, D = seed_indices.size(0), x_rows.size(1)
        chunks = []

        for offset in range(-self.neighbor_k, self.neighbor_k + 1):
            idx = seed_indices + offset
            idx = torch.clamp(idx, 0, x_rows.size(0) - 1)
            chunks.append(x_rows[idx])  # [B, D]

        x_cat = torch.cat(chunks, dim=-1)  # [B, (2k+1)D]
        return self.mlp(x_cat)



class Model(torch.nn.Module):

    def __init__(self, data: HeteroData, col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]], num_layers: int, channels: int, out_channels: int, aggr: str,
                 norm: str = "batch_norm", dropout=0.0, shallow_list: List[NodeType] = [],  # List of node types to add shallow embeddings to input
                 id_awareness: bool = False, model_type: str = "meta-llama/Llama-3.2-1B", max_new_tokens=1, llm_frozen=False, output_mlp=False, output_probs=True, num_demo=4,
                 dataset=None, task=None, db = None, gamma=2.0, alpha=[1.0, 1.0], mask_ratio=0.5,  pretrain_random_table=False, pretrain_mask_cell=False):
        super().__init__()
        self.encoder = HeteroEncoder(channels=channels, node_to_col_names_dict={node_type: data[node_type].tf.col_names_dict for node_type in data.node_types},
                                     node_to_col_stats=col_stats_dict, )
        self.temporal_encoder = HeteroTemporalEncoder(node_types=[node_type for node_type in data.node_types if "time" in data[node_type]], channels=channels, )

        self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types, channels=channels, aggr=aggr, num_layers=2)



        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout)
        self.mlp_l = MLP(out_channels, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout)
        self.mlp_s = MLP(out_channels, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout)

        self.embedding_dict = ModuleDict({node: Embedding(data.num_nodes_dict[node], channels) for node in shallow_list})

        self.id_awareness_emb = Embedding(1, channels) if id_awareness else None
        self.output_mlp = output_mlp
        self.output_probs = output_probs
        self.gamma = gamma
        self.alpha = alpha

        self.best_test_value = None
        self.best_test_epoch = None
        self.best_test_patience = 0
        self.progress_locked = False
        self.progress_lock_epoch = None
        self.best_test_state = None


        self.row_gate_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.table_fuse_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * channels, channels),
            torch.nn.ReLU(),
            torch.nn.Linear(channels, channels)
        )


        self.pretrain_mask_cell = pretrain_mask_cell
        self.pretrain_random_table = pretrain_random_table
        self.mask_ratio = mask_ratio
        self.mask_embed = Embedding(1, channels)
        self.column_keep = {}
        self.contrastive_weight = 0.1
        self.contrastive_temperature = 0.1

        # https://huggingface.co/meta-llama/Llama-3.2-1B
        if model_type == 'gnn':
            self.model = None
            print('Using default GNNs without LLMs')
        elif model_type == './huggingface_cache/Llama-3.2-1B' or model_type == "./huggingface_cache/Llama-3.2-3B-Instruct":
            print('Loading LLAMA')
            self.num_demo = num_demo
            self.dataset = dataset
            self.task = task
            self.max_new_tokens = max_new_tokens # TODO: how many is the optimal?
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False, padding_side="left")  # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
            self.tokenizer.pad_token = self.tokenizer.eos_token  # for padding, https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36
            self.tokenizer.add_special_tokens({'mask_token': '<MASK>'})  # add masked token
            model = AutoModelForCausalLM.from_pretrained(model_type, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 1})  # 16 instead of 32 with less memory!

            model.resize_token_embeddings(len(self.tokenizer))  # expand vocab due to '<MASK>', https://huggingface.co/docs/transformers/en/main_classes/tokenizer
            if llm_frozen:
                print("Freezing LLAMA!")
                for name, param in model.named_parameters():
                    param.requires_grad = False
            else:
                print("Training LLAMA with LORA!")  # TODO: use_dora=True
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)
                lora_r: int = 8
                lora_alpha: int = 16
                lora_dropout: float = 0.05
                lora_target_modules = ["q_proj", "v_proj", ]
                config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(model, config)

            self.model = model
            self.word_embedding = self.model.model.get_input_embeddings()
            if model_type == "Qwen/Qwen2.5-7B-Instruct":
                out_dim = 3584
            elif model_type == './huggingface_cache/Llama-3.2-1B':
                out_dim = 2048
            elif model_type == "./huggingface_cache/Llama-3.2-3B-Instruct":
                out_dim = 3072
            self.out_dim = out_dim
            self.projector = Sequential(Linear(channels, 1024), Sigmoid(), Dropout(dropout), Linear(1024, out_dim), Dropout(dropout)).to(self.model.device)
            self.lm_head = MLP(out_dim, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout) if self.output_mlp else None
            self.table_start_embed = nn.Parameter(torch.randn(1, out_dim) * 0.02)
            self.table_end_embed   = nn.Parameter(torch.randn(1, out_dim) * 0.02)


            self.llama_projector = torch.nn.Sequential(
                torch.nn.Linear(out_dim, out_dim),  
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, out_dim)
            )
            # cached token embeddings
            self.bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
            self.pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        elif model_type == './huggingface_cache/bert-base-uncased':
            print('Loading BERT')
            self.dataset = dataset
            self.task = task
            self.tokenizer = AutoTokenizer.from_pretrained("./huggingface_cache/bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("./huggingface_cache/bert-base-uncased")
            self.projector = Sequential(Linear(channels, 1024), Sigmoid(), Dropout(dropout), Linear(1024, 768),
                                        Dropout(dropout)).to(self.bert_model.device)
        elif model_type == './huggingface_cache/all-roberta-large-v1':
            print('Loading RoBERTa')
            self.dataset = dataset
            self.task = task
            #self.bert_model = SentenceTransformer('./huggingface_cache/all-roberta-large-v1')
            self.tokenizer = AutoTokenizer.from_pretrained('./huggingface_cache/all-roberta-large-v1')
            self.bert_model = AutoModel.from_pretrained('./huggingface_cache/all-roberta-large-v1')
            self.projector = Sequential(Linear(channels, 1024), Sigmoid(), Dropout(dropout), Linear(1024, 1024),
                                        Dropout(dropout)).to(self.bert_model.device)
    
    


    def regression_value_to_token(self, value_float):

        key = round(float(value_float), 6)

        if not hasattr(self, "regression_value_vocab"):
            self.regression_value_vocab = {}
            self.next_regression_token_id = 50000

        vocab = self.regression_value_vocab

        if key not in vocab:
            vocab[key] = self.next_regression_token_id
            self.next_regression_token_id += 1

        return vocab[key]

    def is_easy_column(
    self, table_name, column_name, batch=None,
    max_unique=10,
    max_avg_len=12,
):


        import re
        import pandas as pd

        MANUAL_BAN_COLUMNS = {
            "driverId",
            "driverRef",
            "code",
            "user_id",
            "session_id",
            "item_id",
            "product_id",
        }

        if column_name in MANUAL_BAN_COLUMNS:
            return False

        if batch is None:
            return False
        if table_name not in batch:
            return False
        
        df = batch[table_name].df
        if column_name not in df.columns:
            return False

        col = df[column_name]


        values = col[col != '\\N'].astype(str)
        if len(values) == 0:
            return False

        missing_ratio = (col == '\\N').mean()
        if missing_ratio > 0.3:
            return False


        uniq = values.nunique(dropna=True)

        if uniq > 50:
            return False
        uniq_simple = uniq <= max_unique
        avg_len = values.str.len().mean()
        len_simple = avg_len <= max_avg_len

        is_numeric = False
        try:
            pd.to_numeric(values)
            is_numeric = True
        except:
            is_numeric = False
        if is_numeric and uniq <= max_unique:
            return True

        if uniq_simple and len_simple:
            return True

        return False



    def class_value_to_id(self, table, col, value):
        if table not in self.class_value_map:
            self.class_value_map[table] = {}
        if col not in self.class_value_map[table]:
            self.class_value_map[table][col] = {}

        mp = self.class_value_map[table][col]

        if value not in mp:
            mp[value] = self.next_class_id
            self.next_class_id += 1

        return mp[value]


    def build_column_types(self, db):
        self.numeric_columns = {}
        self.categorical_columns = {}

        for table_name, table in db.table_dict.items():
            df = table.df

            numeric_cols = []
            categorical_cols = []

            for col in df.columns:

                if col in ["id", "ID"] or df[col].dtype == "datetime64[ns]":
                    continue


                try:
                    valid_ratio = df[col].astype(float).notna().mean()
                    if valid_ratio > 0.7:
                        numeric_cols.append(col)
                    else:
                        categorical_cols.append(col)
                except:
                    categorical_cols.append(col)

            self.numeric_columns[table_name] = numeric_cols
            self.categorical_columns[table_name] = categorical_cols

    def build_value_vocab(self, db):

        self.value_vocab = {}
        base = 100000

        for table_name, table in db.table_dict.items():
            df = table.df
            self.value_vocab[table_name] = {}

            for col in df.columns:
                if col in self.categorical_columns.get(table_name, []):
                    unique_vals = df[col].dropna().unique().tolist()
                    self.value_vocab[table_name][col] = {
                        val: base + i for i, val in enumerate(unique_vals)
                    }
                    base += len(unique_vals) + 10

    

    def vocab_value_to_token(self, table_name, col_name, value):


        if table_name not in self.value_vocab:
            self.value_vocab[table_name] = {}

        if col_name not in self.value_vocab[table_name]:
            self.value_vocab[table_name][col_name] = {}

        table_vocab = self.value_vocab[table_name][col_name]

        if value not in table_vocab:
            table_vocab[value] = self.next_vocab_token_id
            self.next_vocab_token_id += 1

        return table_vocab[value]


    def nt_xent_loss(self, z1: Tensor, z2: Tensor, temperature: float = 0.1):
        """
        Simple NT-Xent / InfoNCE between two views z1 and z2.
        z1, z2: (N, D)
        returns scalar loss
        """
        assert z1.size(0) == z2.size(0)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        logits = torch.matmul(z1, z2.t()) / temperature  # (N, N)
        labels = torch.arange(z1.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def encode_r(self, texts):

        return self.bert_model.encode(texts, convert_to_tensor=True)

    def encode_with_bert(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.bert_model(**enc).last_hidden_state  # (B, L, D)
        emb = outputs.mean(dim=1)  # mean pooling -> (B, D)

        return emb

    def get_neighbor_embedding_row(self, neighbor_ids, x_dict, entity_table):

        embeds = []
        for nid in neighbor_ids:
            embeds.append(x_dict[entity_table][nid].unsqueeze(0))
        if embeds:
            return torch.cat(embeds, dim=0)
        else:
            return None


    def update_col_stats(self, new_col_stats: Dict[str, Dict[str, Dict[StatType, Any]]], temporary=False):

        if temporary:
            self._col_stats_backup_current = self.encoder.node_to_col_stats
        self.encoder.node_to_col_stats = new_col_stats
        if hasattr(self.encoder, "_build_encoders"):
            self.encoder._build_encoders()

        if temporary:
            return _ColStatsContext(self)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        #self.gnn1.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()
        if self.mlp_l is not None:
            self.mlp_l.reset_parameters()
        if self.mlp_s is not None:
            self.mlp_s.reset_parameters()
        if self.bert_model is not None:
            self.projector.apply(initialize_weights)
        if self.lm_head is not None:
            self.lm_head.reset_parameters()

    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def eos_user_id_list(self):
        return self.tokenizer(EOS_USER, add_special_tokens=False).input_ids

    @property
    def eos_id_list(self):
        return self.tokenizer(EOS, add_special_tokens=False).input_ids  # LLAMA tokenizer does not add an eos_token_id at the end of inputs

    @property
    def false_id(self):
        return self.tokenizer('No', add_special_tokens=False).input_ids[0]

    @property
    def true_id(self):
        return self.tokenizer('Yes', add_special_tokens=False).input_ids[0]

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast; if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def encode(self, batch, entity_table, add_noise: bool = False, sigma_ratio: float = 0.5, phase="test", is_last_batch=False):
        seed_time = batch[entity_table].seed_time  # seed time indicates at which time the target is to be predicted, filtering future data.
        batch_size = len(seed_time)
        x_dict = self.encoder(batch.tf_dict, add_noise=add_noise, sigma_ratio=sigma_ratio,phase=phase, is_last_batch=is_last_batch)  # encode interactions within each table (tensor_frame)
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)  # add time embedding to time-dependent node features
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():  # id embedding
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        return x_dict, batch_size

    def encode_with_roberta(self, texts):
        # tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        ).to(self.bert_model.device)

        # forward
        outputs = self.bert_model(**encoded)

        # mean pooling
        last_hidden = outputs.last_hidden_state  # (B, L, 1024)
        attention_mask = encoded['attention_mask'].unsqueeze(-1)  # (B, L, 1)

        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        length = attention_mask.sum(dim=1)

        emb = sum_hidden / length  # (B, 1024)

        return emb


    def column_filter(self, df, df_name, refresh=False):
        if refresh or df_name not in self.column_keep:
            self.column_keep[df_name] = [
                col for col in df.columns if infer_series_stype(df[col]) in accept_stypes
            ]
        return self.column_keep[df_name]

    def noise(self, batch, entity_table, phase="test", is_last_batch=False):
        batch_size = len(batch[entity_table].seed_time)
        num_tokens_to_mask = int(batch_size * self.mask_ratio)  # Number of tokens to mask
        x_dict, _ = self.encode(batch, entity_table, phase=phase, is_last_batch=is_last_batch)

    







     
    def _ensure_mask_token(self) -> Tuple[str, int]:
        if hasattr(self, "mask_token_text") and hasattr(self, "mask_token_id"):
            return self.mask_token_text, self.mask_token_id

        def find_atomic_mask_token(tokenizer, max_scan=50000):
            vocab_size = getattr(tokenizer, "vocab_size", None) or max_scan
            limit = min(vocab_size, max_scan)
            for tok_id in range(limit):
                text = tokenizer.decode([tok_id])
                if not text.strip() or len(text) > 5:
                    continue
                ids = tokenizer(text, add_special_tokens=False).input_ids
                if len(ids) != 1 or ids[0] != tok_id:
                    continue
                test_sentence = f"Column is {text}."
                test_ids = tokenizer(test_sentence, add_special_tokens=False).input_ids
                if tok_id in test_ids:
                    return text, tok_id
            raise RuntimeError("No atomic mask token found in vocab.")

        self.mask_token_text, self.mask_token_id = find_atomic_mask_token(self.tokenizer)
        return self.mask_token_text, self.mask_token_id

    def _build_llm_batch(
        self,
        samples: List[Dict[str, Any]],
        node_embeds: torch.Tensor,
        q_embed: torch.Tensor,
        q_len: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not samples:
            return None, None, None

        batch_embeds: List[torch.Tensor] = []
        batch_labels: List[List[int]] = []
        batch_mask: List[List[int]] = []
        seq_lens: List[int] = []

        for t in samples:
            pid = t["prompt_ids"]
            aid = t["answer_ids"]
            node = t["node"]
            if len(aid) == 0:
                continue

            tok = self.word_embedding(torch.tensor(pid + aid, device=device))
            ne = node_embeds[node].unsqueeze(0)
            x = torch.cat([self.bos_embeds, ne, q_embed, tok], dim=0)

            plen = len(pid)
            prefix = 1 + 1 + q_len + plen
            labels = [IGNORE_INDEX] * prefix + aid

            batch_embeds.append(x)
            batch_labels.append(labels)
            seq_lens.append(x.size(0))

        if len(batch_embeds) == 0:
            return None, None, None

        max_len = max(seq_lens)
        for i, (x, lbl) in enumerate(zip(batch_embeds, batch_labels)):
            cur = x.size(0)
            pad = max_len - cur
            if pad > 0:
                x = torch.cat([self.pad_embeds.repeat(pad, 1), x], dim=0)
                lbl = [IGNORE_INDEX] * pad + lbl
            batch_embeds[i] = x
            batch_labels[i] = lbl
            batch_mask.append([0] * (max_len - x.size(0)) + [1] * x.size(0))  # 保守

        embeds = torch.stack(batch_embeds, 0).to(device)
        attn = torch.tensor(batch_mask, device=device)
        labels = torch.tensor(batch_labels, device=device)
        return embeds, attn, labels





    def _make_row_tokens(self, row_vec: torch.Tensor) -> torch.Tensor:
        """
        row_vec: [d] 或 [1, d]
        return: [3, d] = <table_start>, <tabular_node>, <table_end>
        """
        if row_vec.dim() == 1:
            row_vec = row_vec.unsqueeze(0)  # [1, d]
        return torch.cat([self.table_start_embed, row_vec, self.table_end_embed], dim=0)  # [3, d]


    # ---------- Row-only ----------
    def pretrain_row(
    self,
    batch: Any,
    entity_table: str,
    *,
    add_noise: bool = False,
    epoch: int = 0,
    b: float = 0.5,
    mask_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:


        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[entity_table].seed_time)

        CELL_RATIO = 1.0

        if mask_indices is None:
            num_mask = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
            mask_row = torch.randperm(batch_size, device=device)[:num_mask]
        else:
            mask_row = mask_indices


        progress = min(1.0, epoch / 30.0)
        #epochs = math.ceil(epoch/1000) * 1000
        #progress = min(1.0, epochs / 50000.0) 


        raw_keep = 1.0 - float(b) * progress
        ROW_EASY_COL_PROB = float(max(0.25, min(1.0, raw_keep)))  # 至少保留 25%




        batch_row = copy.deepcopy(batch)
        x_row, _ = self.encode(batch_row, select_table, add_noise=add_noise)
        x_row[select_table][mask_row] = self.mask_embed.weight
        x_row = self.gnn(x_row, batch_row.edge_index_dict)
        node_embed_row = self.projector(x_row[select_table][:batch_size])


        df = batch[select_table].df
        df_row = df.iloc[batch[select_table].n_id[mask_row].cpu()]
        avail_cols = df_row.columns.tolist()

        row_samples = []

        for b_idx, (_, row) in zip(mask_row.tolist(), df_row.iterrows()):
            row_dict = row.to_dict()

            cols = [c for c, v in row_dict.items() if v not in ['\\N', None]]
            if not cols:
                continue

            easy_cols = [c for c in cols if self.is_easy_column(select_table, c, batch=batch)]
            hard_cols = [c for c in cols if c not in easy_cols]


            prob = torch.rand(1).item()
            tgt = (easy_cols if prob < ROW_EASY_COL_PROB and easy_cols else hard_cols)[
                torch.randint(len(easy_cols if prob < ROW_EASY_COL_PROB and easy_cols else hard_cols), (1,)).item()
            ]

            true = row_dict[tgt]
            ctx_cols = sorted(cols)

            parts = []
            for col_name in ctx_cols:
                val = row_dict[col_name]
                if col_name == tgt:
                    parts.append(f"{col_name} is {mask_token_text}.")
                else:
                    parts.append(f"{col_name} is {val}.")
            context_text = " ".join(parts)

            prompt_text = (
                context_text
                + f" Please output the correct value of [{tgt}]."
            )

            row_samples.append({
                "node": b_idx,
                "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                "answer_ids": self.tokenizer(str(true), add_special_tokens=False).input_ids,
            })

        if len(row_samples) == 0:
            return torch.tensor(0.0, device=device)

        q_ids = self.tokenizer(
            "Question: predict the masked column value.",
            add_special_tokens=False, return_tensors='pt'
        ).input_ids[0].to(device)
        q_embed = self.word_embedding(q_ids)

        embeds, attn, labels = self._build_llm_batch(
            row_samples, node_embed_row, q_embed, q_ids.size(0), device
        )
        if embeds is None:
            return torch.tensor(0.0, device=device)

        # 🔥 去掉 AMP nondeterminism
        out = self.model(inputs_embeds=embeds, attention_mask=attn, labels=labels, return_dict=True)
        return out.loss, out.loss, out.loss

    # ---------- Cell-only ----------
    def pretrain_cell(
    self,
    batch: Any,
    entity_table: str,
    *,
    add_noise: bool = False,
    epoch: int = 0,
    b: float = 0.5,
    mask_indices: Optional[torch.Tensor] = None,
    max_target_cols: int = 3,
) -> torch.Tensor:


        import copy
        import math
        import numpy as np
        import torch

        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[select_table].seed_time)


        CELL_RATIO = 1.0

        if mask_indices is None:
            num_mask = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
            mask_cell = torch.randperm(batch_size, device=device)[:num_mask]
        else:
            mask_cell = mask_indices

        progress = min(1.0, epoch / 30.0)

        #
        MIN_KEEP = 0.15
        MAX_KEEP = 1.0

        raw_keep = MIN_KEEP + float(b) * progress

        CELL_CONTEXT_KEEP_RATIO = float(
            max(MIN_KEEP, min(MAX_KEEP, raw_keep))
        )



        batch_cell = copy.deepcopy(batch)
        batch_cell[select_table].tf = copy.copy(batch[select_table].tf)
        batch_cell[select_table].tf.feat_dict = {
            k: v.clone() for k, v in batch[select_table].tf.feat_dict.items()
        }
        tf = batch_cell[select_table].tf

        # ---------------------------------------------------------
        # Helper: 统一清洗/格式化单元格值
        # ---------------------------------------------------------
        def normalize_val(v):
            if isinstance(v, dict):

                if len(v) == 0:
                    return None
                v = next(iter(v.values()))

            if isinstance(v, (np.generic,)):
                v = np.asarray(v).item()

            if isinstance(v, (list, tuple, set)):
                if len(v) == 0:
                    return None

                v = ", ".join(map(str, v))

            if v in ["\\N", None, "nan", "NaN"]:
                return None

            try:
                if isinstance(v, float) and math.isnan(v):
                    return None
            except Exception:
                pass

            return str(v)


        df_full = batch_cell[select_table].df

        valid_columns: List[str] = []
        for col, stype_idx in tf._col_to_stype_idx.items():
            # stype_idx: (stype_name, col_idx)
            if not getattr(stype_idx, "__getitem__", None):
                continue
            stype_name, _ = stype_idx


            if str(stype_name) == "timestamp":
                continue

            col_series = df_full[col]
            non_missing_ratio = (col_series != "\\N").mean()
            if non_missing_ratio <= 0.6:
                continue

            valid_columns.append(col)

        cell_target_cols: List[str] = []
        if len(valid_columns) > 0:
            num_cell_cols = min(max_target_cols, len(valid_columns))
            perm_cols = torch.randperm(len(valid_columns))[:num_cell_cols].tolist()
            cell_target_cols = [valid_columns[i] for i in perm_cols]

            for col_name in cell_target_cols:
                stype_name, col_idx = tf._col_to_stype_idx[col_name]
                feat = batch_cell[select_table].tf.feat_dict[stype_name]

                if hasattr(feat, "values") and hasattr(feat, "offset"):
                    mask_values = feat.values.clone()
                    off = feat.offset
                    mask_values[mask_cell, off[col_idx]:off[col_idx + 1]] = 0
                    batch_cell[select_table].tf.feat_dict[stype_name].values = mask_values
                else:
                    batch_cell[select_table].tf.feat_dict[stype_name][mask_cell] = 0

        x_cell, _ = self.encode(batch_cell, select_table, add_noise=add_noise)
        x_cell = self.gnn(x_cell, batch_cell.edge_index_dict)
        node_embed_cell = self.projector(x_cell[select_table][:batch_size])

        df = batch[select_table].df

        df_cell = df.iloc[batch[select_table].n_id[mask_cell].cpu().numpy()]


        filtered_cell = df_cell[self.column_filter(df_cell, select_table, refresh=True)]

        cell_samples: List[Dict[str, Any]] = []

        if len(cell_target_cols) > 0:

            target_cols = [c for c in cell_target_cols if c in filtered_cell.columns]


            if len(target_cols) == 0:
                return torch.tensor(0.0, device=device),torch.tensor(0.0, device=device),torch.tensor(0.0, device=device)

            for b_idx, (row_index, row_view) in zip(mask_cell.tolist(), filtered_cell.iterrows()):

                full = df_cell.loc[row_index]
                row_dict_full = full.to_dict()

                cols_all = []
                for c in filtered_cell.columns:
                    v = normalize_val(row_dict_full.get(c, None))
                    if v is not None:
                        cols_all.append(c)

                if len(cols_all) == 0:
                    continue

                for tgt in target_cols:
                    true_raw = row_dict_full.get(tgt, None)
                    true = normalize_val(true_raw)
                    if true is None:
                        continue

                    k = max(2, int(len(cols_all) * CELL_CONTEXT_KEEP_RATIO))
                    k = min(k, len(cols_all))

                    idx_perm = torch.randperm(len(cols_all))[:k].tolist()
                    context_cols = [cols_all[i] for i in idx_perm]

                    if tgt not in context_cols:
                        if tgt in cols_all:
                            context_cols[0] = tgt
                        else:
                            context_cols[-1] = tgt

                    parts = []
                    for col_name in context_cols:
                        v = normalize_val(row_dict_full.get(col_name, None))
                        if v is None:
                            continue
                        if col_name == tgt:
                            parts.append(f"{col_name} is {mask_token_text}.")
                        else:
                            parts.append(f"{col_name} is {v}.")

                    if not parts:
                        continue

                    context_text = " ".join(parts)

                    prompt_text = (
                        context_text
                        + f" Please output the correct value of [{tgt}]."
                    )

                    cell_samples.append({
                        "node": b_idx,
                        "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                        "answer_ids": self.tokenizer(str(true), add_special_tokens=False).input_ids,
                    })

        if len(cell_samples) == 0:
            return torch.tensor(0.0, device=device),torch.tensor(0.0, device=device),torch.tensor(0.0, device=device)

        q_ids = self.tokenizer(
            "Question: predict the value of the masked cell.",
            add_special_tokens=False, return_tensors="pt",
        ).input_ids[0].to(device)
        q_embed = self.word_embedding(q_ids)

        embeds, attn, labels = self._build_llm_batch(
            cell_samples, node_embed_cell, q_embed, q_ids.size(0), device
        )
        if embeds is None:
            return torch.tensor(0.0, device=device),torch.tensor(0.0, device=device),torch.tensor(0.0, device=device)

        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )
        return out.loss, out.loss, out.loss


    


    def label_tokenize(self, batch, entity_table):
        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            label = ['Yes' if i else 'No' for i in batch[entity_table].y.bool().tolist()]  # convert 0/1 to true/false
        elif self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            label = [str(i) for i in batch[entity_table].y.int().tolist()]  # int number
        elif self.task.task_type == TaskType.REGRESSION:
            label = [str(i) for i in batch[entity_table].y.float().tolist()]
        labels = self.tokenizer(label, add_special_tokens=False)
        return labels

    def get_demo_info(self, demo_batch, entity_table):
        x_dict, demo_batch_size = self.encode(demo_batch, entity_table)
        assert self.num_demo <= demo_batch_size, 'Too large demo numbers!'
        x_dict = self.gnn(x_dict, demo_batch.edge_index_dict)
        demo_node_embeds = self.projector(x_dict[entity_table][:demo_batch_size])
        demo_labels = self.label_tokenize(demo_batch, entity_table).input_ids
        demo_labels = torch.tensor(demo_labels, device=self.device)
        return demo_node_embeds, demo_labels

    def recursive_sample(self, batch_data: HeteroData, node_type: str, target_nodes: torch.Tensor, num_hops: int = 2):
        """
        Recursively samples neighbors from a batch heterogeneous graph while ensuring previously sampled node types are excluded.
        Args:
            batch_data (HeteroData): A batched heterogeneous graph from PyG's NeighborLoader.
            target_nodes (torch.Tensor): The indices of the target nodes in the `node_type`.
            node_type (str): The node type of the target nodes (e.g., "entity_table").
            num_hops (int): Number of recursive hops to sample.
        """
        sampled_nodes = [node_type]  # Track sampled node types to avoid re-sampling
        neighbor_dict = {node_type: {node: {} for node in target_nodes.tolist()}}  # Initialize nested dictionary

        def sample_neighbors(current_nodes, current_node_type, depth, tmp_dict):
            """Recursively sample neighbors up to num_hops while avoiding duplicate node types."""
            if depth == num_hops: return
            next_nodes = {}
            for edge_type in batch_data.edge_types:  # Iterate through edge types to find valid neighbors
                src_type, _, dst_type = edge_type
                if src_type == current_node_type and dst_type not in sampled_nodes:
                    src_nodes = batch_data[edge_type].edge_index[0].tolist()
                    dst_nodes = batch_data[edge_type].edge_index[1].tolist()
                    # print(edge_type, current_node_type, len(src_nodes), len(dst_nodes))
                    for src, dst in zip(src_nodes, dst_nodes):
                        if src in current_nodes:  # Ensure it's a valid node from the current set
                            if dst_type not in tmp_dict[src]:
                                tmp_dict[src][dst_type] = {}
                            tmp_dict[src][dst_type][dst] = {}

                            if dst_type not in next_nodes:
                                next_nodes[dst_type] = set()
                            next_nodes[dst_type].add(dst)
            for node in tmp_dict.keys():
                for next_node_type, nodes in next_nodes.items():   # Recursive call for the next hop
                    if next_node_type in tmp_dict[node].keys():
                        sample_neighbors(nodes, next_node_type, depth + 1, tmp_dict[node][next_node_type])

        sample_neighbors(set(target_nodes.tolist()), node_type, depth=0, tmp_dict=neighbor_dict[node_type])   # Start recursive sampling from target nodes
        return neighbor_dict

    def get_neighbor_embedding(self, neighbor_dict, embed_dict):

        def recursive_collect(node_type, node_id, sub_neighbors):
            """Recursively collect embeddings depth-first for a single node."""
            node_embedding = embed_dict[node_type][node_id].unsqueeze(0)  # Shape: (1, D)
            # Collect embeddings from deeper neighbors recursively
            neighbor_embeds = []
            for sub_type, sub_dict in sub_neighbors.items():
                for sub_id, sub_sub_neighbors in sub_dict.items():
                    neighbor_embeds.append(recursive_collect(sub_type, sub_id, sub_sub_neighbors))
            if neighbor_embeds:
                neighbor_embeds = torch.cat(neighbor_embeds)  # Concatenate along feature dimension
                node_embedding = torch.cat([node_embedding, neighbor_embeds])
            return node_embedding

        all_embeddings = []
        for target_type, targets in neighbor_dict.items():
            for target_id, neighbors in targets.items():
                all_embeddings.append(recursive_collect(target_type, target_id, neighbors))
        return torch.cat(all_embeddings) if all_embeddings else None

    def cl(self, batch: HeteroData, entity_table: NodeType, context=True, inference=False, add_noise=False,
        sigma_ratio: float = 0.1) -> Tensor:

        x_dict, batch_size = self.encode(batch, entity_table, add_noise=add_noise, sigma_ratio=sigma_ratio)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)

        # take only target table part
        node_embed = x_dict[entity_table][:batch_size]
        
        # ensure projector exists (stability)
        """if hasattr(self, "projector"):
            node_embed = self.projector(node_embed)"""

        

        # ---- 2) build neighbors ----
        try:
            if context:
                neighbors = self.recursive_sample(batch, entity_table,
                                                torch.arange(batch_size, device=node_embed.device),
                                                num_hops=1)
            else:
                neighbors = None
        except Exception:
            neighbors = None
        
        view_a = node_embed  # (B, D)
        view_b_list = []

        # ---- 3) build view B ----
        for i in range(batch_size):
            nb_emb = None

            if context and neighbors is not None and entity_table in neighbors:
                try:
                    neigh_info = neighbors[entity_table].get(i, {})
                    nb = self.get_neighbor_embedding(neigh_info, x_dict)
                    if nb is not None and nb.numel() > 0:
                        # mean aggregation
                        nb_emb = nb.mean(dim=0, keepdim=True)
                        
                except Exception:
                    nb_emb = None
                
            # fallback: noise augmentation
            if nb_emb is None:
                noise_scale = 0.1 * self.contrastive_temperature
                noise = torch.randn_like(view_a[i:i+1]) * noise_scale
                nb_emb = view_a[i:i+1] + noise

            view_b_list.append(nb_emb)

        view_b = torch.cat(view_b_list, dim=0)  # (B, D)

        # ---- 4) NT-Xent ----
        loss = self.nt_xent_loss(view_a, view_b, temperature=self.contrastive_temperature)

        # avoid None or NaN
        if loss is None or torch.isnan(loss):
            loss = torch.zeros((), device=node_embed.device)

        return loss

    def llama_cl(self, batch, entity_table):
        
        def row_to_text(row):
            parts = []
            for col in row.index:
                val = row[col]
                if pd.notna(val) and str(val) not in ['\\N', 'None']:
                    parts.append(f"{col} is {val}.")
            return " ".join(parts)

        batch_size = len(batch[entity_table].seed_time)

        # -------------------------
        # 1) Row-only embeddings
        # -------------------------
        row_embeds_list = []
        for i in range(batch_size):
            row = batch[entity_table].df.iloc[i]
            prompt = row_to_text(row)

            input_ids = torch.tensor(
                self.tokenizer(prompt, add_special_tokens=False).input_ids
            ).to(self.device)

            token_embeds = self.word_embedding(input_ids)     # llama word embeddings
            row_emb = token_embeds.mean(dim=0, keepdim=True)  # mean pooling
            row_emb = self.llama_projector(row_emb)           # projector -> grad
            row_embeds_list.append(row_emb)

        row_embeds = torch.cat(row_embeds_list, dim=0)  # (B, D)

        # -------------------------
        # 2) Row + neighbors embeddings
        # -------------------------
        neighbors = self.recursive_sample(batch, entity_table, torch.arange(batch_size), num_hops=1)
        row_neighbor_embeds_list = []

        for i in range(batch_size):
            row = batch[entity_table].df.iloc[i]
            main_text = row_to_text(row)

            # gather neighbor rows
            nbr_texts = []
            nbs = neighbors.get(entity_table, {}).get(i, {})

            for nbr_type, sub in nbs.items():
                cnt = 0
                for nbr_idx in sub.keys():
                    if cnt >= 3:
                        break
                    global_nid = batch[nbr_type].n_id[nbr_idx].item()
                    r = batch[nbr_type].df.iloc[global_nid]
                    t = row_to_text(r)
                    nbr_texts.append(t)
                    cnt += 1

            # concat neighbor info
            if nbr_texts:
                full_prompt = main_text + " Related: " + " ".join(nbr_texts)
            else:
                full_prompt = main_text

            input_ids = torch.tensor(
                self.tokenizer(full_prompt, add_special_tokens=False).input_ids
            ).to(self.device)

            token_embeds = self.word_embedding(input_ids)
            row_nb_emb = token_embeds.mean(dim=0, keepdim=True)
            row_nb_emb = self.llama_projector(row_nb_emb)
            row_neighbor_embeds_list.append(row_nb_emb)

        row_neighbor_embeds = torch.cat(row_neighbor_embeds_list, dim=0)

        # -------------------------
        # 3) Contrastive loss
        # -------------------------
        loss = self.nt_xent_loss(
            F.normalize(row_embeds, dim=-1),
            F.normalize(row_neighbor_embeds, dim=-1),
            temperature=self.contrastive_temperature
        )

        return loss




    def forward(
    self,
    batch,
    entity_table,
    demo_info=None,
    inference: bool = False,
    tta_mode: bool = False,
    add_noise: bool = False,
    sigma_ratio: float = 0.1,
    **kwargs
):
        """
        Masked-cell prediction with AUTOMATIC single-token mask selection.
        Compatible with ANY tokenizer.
        """

        device = self.device

        # ============================================================
        # Helper: auto find single-token mask token
        # ============================================================
        def find_single_token_word(tokenizer, max_scan=50000):

            for tok_id in range(min(tokenizer.vocab_size, max_scan)):

                text = tokenizer.decode([tok_id])

                if not text.strip():
                    continue
                if len(text) > 5:  # 太长的 token 不安全
                    continue

                ids = tokenizer(text, add_special_tokens=False).input_ids
                if len(ids) != 1:
                    continue
                if ids[0] != tok_id:
                    continue

                test_sentence = f"Label: {text}"
                test_ids = tokenizer(test_sentence, add_special_tokens=False).input_ids

                if tok_id in test_ids:
                    return text, tok_id

            raise RuntimeError("No atomic mask token found!")


        # ============================================================
        # 1. Graph Encoding + GNN
        # ============================================================
        x_dict, batch_size = self.encode(batch, entity_table,
                                        add_noise=add_noise,
                                        sigma_ratio=sigma_ratio)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        node_embed = self.projector(x_dict[entity_table][:batch_size])   # (B, D)
        def find_atomic_mask_token(tokenizer, max_scan=50000):
            vocab_size = getattr(tokenizer, "vocab_size", max_scan)
            limit = min(vocab_size, max_scan)
            for tok_id in range(limit):
                text = tokenizer.decode([tok_id])
                if not text.strip():
                    continue
                if len(text) > 5:
                    continue
                ids = tokenizer(text, add_special_tokens=False).input_ids
                if len(ids) == 1 and ids[0] == tok_id:
                    test = f"col is {text}"
                    if tok_id in tokenizer(test, add_special_tokens=False).input_ids:
                        return text, tok_id
            raise RuntimeError("no mask token found")



        if not hasattr(self, "mask_token_text"):
            self.mask_token_text, self.mask_token_id = find_atomic_mask_token(self.tokenizer)

        mask_token_text = self.mask_token_text
        mask_token_id = self.mask_token_id
        # yes/no
        def find_single_token_word(word_list):
            for w in word_list:
                ids = self.tokenizer(w, add_special_tokens=False).input_ids
                if len(ids) == 1:
                    return w, ids[0]
            raise RuntimeError(f"No single-token match found for {word_list}")

        yes_text, yes_id = find_single_token_word(["<yes>", "yes", "YES", "True"])
        no_text,  no_id  = find_single_token_word(["<no>", "no", "NO", "False"])

        # ============================================================
        # 3. prompt
        # ============================================================
        task_desc = description_dict[self.dataset][self.task.name]
        prompt = task_desc + f" Predict the masked label for this sample. Label: {mask_token_text}"

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)


        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        text_embeds = self.word_embedding(input_ids)[0]     # (T, D)
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
            raise ValueError(f"Mask token {mask_token_text} not found in prompt!")
        mask_pos = mask_positions.item()    # 0-based


        batch_embeds, batch_attn = [], []
        seq_lens = []

        prefix_len = 2   # bos + node

        for i in range(batch_size):
            seq = torch.cat([
                self.bos_embeds,                 # (1, D)
                node_embed[i].unsqueeze(0),      # (1, D)
                text_embeds                      # (T, D)
            ], dim=0)

            seq_lens.append(seq.size(0))
            batch_embeds.append(seq)
            batch_attn.append([1] * seq.size(0))

        max_len = max(seq_lens)

        # 前 padding
        for i in range(batch_size):
            pad_len = max_len - batch_embeds[i].size(0)
            if pad_len > 0:
                pad_blk = self.pad_embeds.repeat(pad_len, 1)
                batch_embeds[i] = torch.cat([pad_blk, batch_embeds[i]], dim=0)
                batch_attn[i]   = [0]*pad_len + batch_attn[i]

        inputs_embeds = torch.stack(batch_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attn, device=device)

        # ============================================================
        # 6. LLM forward
        # ============================================================
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits     # (B, L, V)

        mask_indices = []
        for i in range(batch_size):
            seq_len = seq_lens[i]
            pad_len = max_len - seq_len
            mask_idx = pad_len + prefix_len + mask_pos
            mask_indices.append(mask_idx)

        mask_indices = torch.tensor(mask_indices, device=device)

        mask_logits = logits[torch.arange(batch_size, device=device), mask_indices, :]

        yes_logits = mask_logits[:, yes_id]
        no_logits  = mask_logits[:, no_id]
        pred_logits = yes_logits - no_logits

        if tta_mode:
            with self.maybe_autocast():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )


            mask_logits = outputs.logits[
                torch.arange(batch_size, device=device),
                mask_indices,
                :
            ]

            yes_logits = mask_logits[:, yes_id]
            no_logits  = mask_logits[:, no_id]
            pred_logits = yes_logits - no_logits   # (B,)

            return pred_logits



        if inference:
            return pred_logits





    @staticmethod
    def focal_loss(logits, labels, gamma=2.0, alpha_weights=None):
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
        targets_one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()  # One-hot encoding
        ce_loss = -targets_one_hot * torch.log(probs)  # Cross-entropy loss
        loss = (1 - probs) ** gamma * ce_loss  # Apply focal scaling
        if alpha_weights is not None:
            loss *= alpha_weights  # Weight per class
        return loss.mean()

    def forward_dst_readout(self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType, ) -> Tensor:
        if self.id_awareness_emb is None:
            raise RuntimeError("id_awareness must be set True to use forward_dst_readout")
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight
        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time
        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[dst_table])

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param


    
class _ColStatsContext:

    def __init__(self, model: Model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 自动恢复源域统计信息
        self.model.encoder.node_to_col_stats = self.model._col_stats_backup
        if hasattr(self.model.encoder, "_build_encoders"):
            self.model.encoder._build_encoders()