from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import json
import os
import re
import random
from typing import Any, Dict, List
import contextlib
import random
import copy
import tempfile
import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict, Sigmoid, Sequential, Linear, Dropout
from transformers import AutoTokenizer, AutoModel
import torch_frame.data
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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


def load_causal_lm_with_rope_compat(model_type, **kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(model_type, **kwargs)
    except ValueError as exc:
        if "rope_scaling" not in str(exc):
            raise

        config_path = os.path.join(model_type, "config.json")
        if not os.path.exists(config_path):
            raise

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        rope_scaling = config_dict.get("rope_scaling")
        if not isinstance(rope_scaling, dict) or "factor" not in rope_scaling:
            raise

        config_dict["rope_scaling"] = {
            "type": "dynamic",
            "factor": float(rope_scaling["factor"]),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            compat_config_path = os.path.join(tmp_dir, "config.json")
            with open(compat_config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)
                f.write("\n")
            config = AutoConfig.from_pretrained(tmp_dir)

        print(
            "[WARN] Loaded Llama config with legacy rope_scaling compatibility. "
            "For exact Llama 3.x behavior, install transformers==4.45.2."
        )
        return AutoModelForCausalLM.from_pretrained(model_type, config=config, **kwargs)


def patch_accelerate_clear_device_cache() -> None:
    try:
        import accelerate.utils.memory as accelerate_memory
    except ImportError:
        return

    if hasattr(accelerate_memory, "clear_device_cache"):
        return

    def clear_device_cache(garbage_collection=False):
        if garbage_collection:
            import gc

            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accelerate_memory.clear_device_cache = clear_device_cache
    print(
        "[WARN] Patched accelerate.utils.memory.clear_device_cache for this "
        "accelerate/peft version combination."
    )
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


        name_to_cols = {}
        for t in all_tables:
            try:
                df = getattr(batch[t], "df", None)
                cols = [str(c) for c in _to_list(getattr(df, "columns", []))]
            except Exception:
                cols = []
            cols = cols[:max_cols]
            name_to_cols[t] = cols


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

        #self.gnn1 = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types, channels=channels, aggr=aggr, num_layers=2)
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



        #self.build_column_types(db)
         #self.build_value_vocab(db)



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

        # pretrain setup

        self.pretrain_mask_cell = pretrain_mask_cell


        self.pretrain_random_table = pretrain_random_table

        self.mask_ratio = mask_ratio

        self.mask_embed = Embedding(1, channels)
        self.column_keep = {}  
        # contrastive defaults
        self.contrastive_weight = 0.1
        self.contrastive_temperature = 0.1
        

        # =====================

        # =====================

        



        # https://huggingface.co/meta-llama/Llama-3.2-1B
        if model_type == 'gnn':
            self.model = None
            print('Using default GNNs without LLMs')
        elif model_type == './huggingface_cache/Llama-3.2-1B' or model_type == "./huggingface_cache/Llama-3.2-3B-Instruct":
            print('Loading LLAMA')
            self.num_demo = num_demo  
            self.dataset = dataset
            self.task = task
            self.max_new_tokens = max_new_tokens  




            
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False, padding_side="left")  # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side

            self.tokenizer.pad_token = self.tokenizer.eos_token  # for padding, https://huggingface.co/meta-llama/Meta-Llama-3-8B/discussions/36


            self.tokenizer.add_special_tokens({'mask_token': '<MASK>'})  # add masked token
            



            # Bind each distributed process to its own current GPU instead of always using GPU 0.
            device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
            model = load_causal_lm_with_rope_compat(model_type, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device_map)  # 16 instead of 32 with less memory!



            model.resize_token_embeddings(len(self.tokenizer))  # expand vocab due to '<MASK>', https://huggingface.co/docs/transformers/en/main_classes/tokenizer
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
            if llm_frozen:

                print("Freezing LLAMA!")
                for name, param in model.named_parameters():
                    param.requires_grad = False
            else: 
                print("Training LLAMA with LORA!")  # TODO: use_dora=True
                if hasattr(model, "gradient_checkpointing_enable"):
                    try:
                        model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                    except TypeError:
                        model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                patch_accelerate_clear_device_cache()
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)  

                lora_r: int = 8
                lora_alpha: int = 16
                lora_dropout: float = 0.05
                # lora_target_modules = ["q_proj", "k_proj", "v_proj"]
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
            """self.projector_row = Sequential(Linear(channels, 1024), Sigmoid(), Dropout(dropout), Linear(1024, out_dim),
                                        Dropout(dropout)).to(self.model.device)"""


            self.lm_head = MLP(out_dim, out_channels=out_channels, norm=norm, num_layers=1, dropout=dropout) if self.output_mlp else None
            # hidden_dim = LLM hidden size
            hidden_dim = self.model.config.hidden_size
            #self.classifier = torch.nn.Linear(hidden_dim, 1)



            self.table_start_embed = nn.Parameter(torch.randn(1, out_dim) * 0.02)
            self.table_end_embed   = nn.Parameter(torch.randn(1, out_dim) * 0.02)



            """self.classifier = nn.Linear(self.model.config.hidden_size, 1)
            self.loss_bce = nn.BCEWithLogitsLoss()"""





            #self.class_value_map = {}   # {table: {col: {value: id}}}
            #self.next_class_id = 0

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
            #self.bert_model = SentenceTransformer('./huggingface_cache/bert-base-uncased')
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
    ban_id=True
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
        # ----------------------------------------------------

        # ----------------------------------------------------
        if batch is None:
            return False
        if table_name not in batch:
            return False
        
        df = batch[table_name].df
        if column_name not in df.columns:
            return False

        col = df[column_name]

        # ----------------------------------------------------

        # ----------------------------------------------------


        values = col[col != '\\N'].astype(str)
        if len(values) == 0:
            return False

        # ----------------------------------------------------

        # ----------------------------------------------------
        missing_ratio = (col == '\\N').mean()
        if missing_ratio > 0.3:
            return False

        # ----------------------------------------------------

        # ----------------------------------------------------
        uniq = values.nunique(dropna=True)



        if uniq > 50:  
            return False

        uniq_simple = uniq <= max_unique

        # ----------------------------------------------------

        # ----------------------------------------------------
        avg_len = values.str.len().mean()
        len_simple = avg_len <= max_avg_len

        # ----------------------------------------------------

        # ----------------------------------------------------
        is_numeric = False
        try:
            pd.to_numeric(values)
            is_numeric = True
        except:
            is_numeric = False


        if is_numeric and uniq <= max_unique:
            return True

        # ----------------------------------------------------

        # ----------------------------------------------------
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


        #self.reset_parameters()

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
        #torch.nn.init.constant_(self.row_gate, 1.0) 
        """if self.row_gate_mlp is not None:
            self.row_gate_mlp.apply(initialize_weights)
        if self.table_fuse_mlp is not None:
            self.table_fuse_mlp.apply(initialize_weights)
        if self.classifier is not None:
            self.classifier.apply(initialize_weights)"""

        """if self.reg_head is not None:
            self.reg_head.apply(initialize_weights)
        if self.class_head is not None:
            self.class_head.apply(initialize_weights)"""

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
        #print(batch.tf_dict)
        """tf = batch.tf_dict["results"]
        print("attrs:", dir(tf))
        print("tensor?", hasattr(tf, "tensor"))
        print("data?", hasattr(tf, "data"))
        print("feats?", hasattr(tf, "feats"))
        print("values?", hasattr(tf, "values"))"""


        x_dict = self.encoder(batch.tf_dict, add_noise=add_noise, sigma_ratio=sigma_ratio,phase=phase, is_last_batch=is_last_batch)  # encode interactions within each table (tensor_frame)

        # batch_dict -> index of each node in seed time (different from batch.batch!)

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

    """def column_filter(self, df, df_name):
        if df_name not in self.column_keep:
            self.column_keep[df_name] = [col for col in df.columns if infer_series_stype(df[col]) in accept_stypes]
        return self.column_keep[df_name]"""

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

    



    def pretrain1(
    self, batch, entity_table, row_level_prob,
    add_noise: bool = False, epoch: int = 0, num_epochs: int = 100
):

        import copy
        import random
        import torch
        import torch_frame
        import pandas as pd
        from torch_frame import stype

        device = self.device

        # ================================================================

        # ================================================================
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

        # ================================================================

        # ================================================================
        progress = min(1.0, epoch / 50)
        DIFF_CELL = 0.3
        DIFF_ROW  = 0.7
        CELL_CONTEXT_KEEP_RATIO = 1.0 - DIFF_CELL * progress
        ROW_EASY_COL_PROB       = 1.0 - DIFF_ROW  * progress

        # ================================================================

        # ================================================================
        select_table = entity_table
        batch_size = len(batch[entity_table].seed_time)
        num_mask = int(batch_size * self.mask_ratio)
        if num_mask <= 0:
            return torch.tensor(0.0, device=device)

        ROW_RATIO  = 0.1
        CELL_RATIO = 0.1

        perm = torch.randperm(batch_size, device=device)
        row_count  = max(1, int(num_mask * ROW_RATIO))
        cell_count = max(1, int(num_mask * CELL_RATIO))
        mask_row  = perm[:row_count]
        mask_cell = perm[row_count: row_count + cell_count]

        # ================================================================

        # ================================================================
        batch_cell = copy.copy(batch)
        batch_row  = copy.copy(batch)

        batch_cell[entity_table].tf = copy.copy(batch[entity_table].tf)
        batch_cell[entity_table].tf.feat_dict = {
            k: v.clone() for k, v in batch[entity_table].tf.feat_dict.items()
        }


        valid_columns = [
            col for col, v in batch_cell[entity_table].tf._col_to_stype_idx.items()
            if v[0] != stype.timestamp
            and (batch_cell[entity_table].df[col] != "\\N").mean() > 0.6
        ]

        cell_target_cols = []
        if valid_columns:
            ncols = min(3, len(valid_columns))
            cell_target_cols = random.sample(valid_columns, ncols)

            for col_name in cell_target_cols:
                stype_idx = batch_cell[entity_table].tf._col_to_stype_idx[col_name]
                s_stype, c_idx = stype_idx
                feat = batch_cell[entity_table].tf.feat_dict[s_stype]
                if hasattr(feat, "values"):
                    mv = feat.values.clone()
                    off = feat.offset
                    mv[mask_cell, off[c_idx]:off[c_idx+1]] = 0
                    batch_cell[entity_table].tf.feat_dict[s_stype].values = mv
                else:
                    batch_cell[entity_table].tf.feat_dict[s_stype][mask_cell] = 0

        # Row mask
        x_cell, _ = self.encode(batch_cell, entity_table, add_noise=add_noise)
        x_row,  _ = self.encode(batch_row,  entity_table, add_noise=add_noise)
        x_row[select_table][mask_row] = self.mask_embed.weight

        # ================================================================

        # ================================================================
        x_row = self.gnn(x_row, batch_row.edge_index_dict)
        node_embed_row = self.projector(x_row[select_table][:batch_size])

        x_cell_gnn = self.gnn(x_cell, batch_cell.edge_index_dict)
        node_embed_cell = self.projector(x_cell_gnn[select_table][:batch_size])

        # ================================================================

        # ================================================================
        df = batch[select_table].df

        df_row  = df.iloc[batch[select_table].n_id[mask_row].cpu().numpy()]
        df_cell = df.iloc[batch[select_table].n_id[mask_cell].cpu().numpy()]

        filtered_row  = df_row[self.column_filter(df_row,  select_table, refresh=True)]
        filtered_cell = df_cell[self.column_filter(df_cell, select_table, refresh=True)]

        # ================================================================

        # ================================================================
        MAX_NEGATIVE = 2  

        def sample_negatives(column, true_value):
            col_series = df[column]
            cand = col_series[(col_series != "\\N") & (col_series.notna())]
            cand = cand[cand != true_value]
            uniq = cand.drop_duplicates().tolist()
            uniq = [str(x) for x in uniq]

            if len(uniq) == 0:
                return []

            if len(uniq) <= MAX_NEGATIVE:
                return uniq
            return random.sample(uniq, MAX_NEGATIVE)

        # ================================================================

        # ================================================================
        row_samples = []
        cell_samples = []

        # ============================================================

        # ============================================================
        for b_idx, (_, row) in zip(mask_row.tolist(), df_row.iterrows()):

            row_dict = row.to_dict()
            cols = [c for c, v in row_dict.items() if v not in ["\\N", None]]
            if not cols:
                continue

            easy_cols = [c for c in cols if self.is_easy_column(select_table, c, batch=batch)]
            hard_cols = [c for c in cols if c not in easy_cols]

            if random.random() < ROW_EASY_COL_PROB and easy_cols:
                tgt = random.choice(easy_cols)
            else:
                tgt = random.choice(hard_cols if hard_cols else easy_cols)

            true_val = row_dict[tgt]
            if true_val in ["\\N", None]:
                continue


            parts = []
            for col_name in filtered_row.columns:
                v = row_dict.get(col_name)
                if v in ["\\N", None]:
                    continue
                if col_name == tgt:
                    parts.append(f"{col_name} is {mask_token_text}.")
                else:
                    parts.append(f"{col_name} is {v}.")
            if not parts:
                continue

            ctx = " ".join(parts)


            pos_prompt = (
                ctx +
                f" The masked value is {true_val}. Is this correct?"
            )
            pos_pid = self.tokenizer(pos_prompt, add_special_tokens=False).input_ids
            try:
                mask_pos = pos_pid.index(mask_token_id)
            except:
                continue

            row_samples.append({
                "node": b_idx,
                "prompt_ids": pos_pid,
                "mask_pos": mask_pos,
                "label": 1.0,
                "mtype": "row",
            })

            # Negative samples
            neg_values = sample_negatives(tgt, true_val)
            for neg_val in neg_values:
                neg_prompt = (
                    ctx +
                    f" The masked value is {neg_val}. Is this correct?"
                )
                nid = self.tokenizer(neg_prompt, add_special_tokens=False).input_ids
                try:
                    nmask = nid.index(mask_token_id)
                except:
                    continue

                row_samples.append({
                    "node": b_idx,
                    "prompt_ids": nid,
                    "mask_pos": nmask,
                    "label": 0.0,
                    "mtype": "row",
                })

        # ============================================================

        # ============================================================
        if len(cell_target_cols) > 0:
            for b_idx, (_, rowv) in zip(mask_cell.tolist(), filtered_cell.iterrows()):
                full = df_cell.loc[rowv.name]
                row_dict_full = full.to_dict()

                for tgt in cell_target_cols:
                    if tgt not in filtered_cell.columns:
                        continue

                    true_val = row_dict_full.get(tgt)
                    if true_val in ["\\N", None]:
                        continue

                    cols_all = list(filtered_cell.columns)
                    random.shuffle(cols_all)
                    k = max(1, int(len(cols_all) * CELL_CONTEXT_KEEP_RATIO))
                    ctx_cols = cols_all[:k]
                    if tgt not in ctx_cols:
                        ctx_cols[0] = tgt

                    parts = []
                    for col_name in ctx_cols:
                        v = row_dict_full.get(col_name)
                        if v in ["\\N", None]:
                            continue
                        if col_name == tgt:
                            parts.append(f"{col_name} is {mask_token_text}.")
                        else:
                            parts.append(f"{col_name} is {v}.")
                    if not parts:
                        continue

                    ctx = " ".join(parts)

                    # Positive sample
                    pos_prompt = (
                        ctx +
                        f" The masked value is {true_val}. Is this correct?"
                    )
                    pid = self.tokenizer(pos_prompt, add_special_tokens=False).input_ids
                    try:
                        mask_pos = pid.index(mask_token_id)
                    except:
                        continue

                    cell_samples.append({
                        "node": b_idx,
                        "prompt_ids": pid,
                        "mask_pos": mask_pos,
                        "label": 1.0,
                        "mtype": "cell",
                    })

                    # Negative samples
                    neg_values = sample_negatives(tgt, true_val)
                    for nv in neg_values:
                        neg_prompt = (
                            ctx +
                            f" The masked value is {nv}. Is this correct?"
                        )
                        nid = self.tokenizer(neg_prompt, add_special_tokens=False).input_ids
                        try:
                            nmask = nid.index(mask_token_id)
                        except:
                            continue
                        cell_samples.append({
                            "node": b_idx,
                            "prompt_ids": nid,
                            "mask_pos": nmask,
                            "label": 0.0,
                            "mtype": "cell",
                        })

        if len(row_samples) == 0 and len(cell_samples) == 0:
            return torch.tensor(0.0, device=device)

        # ================================================================

        # ================================================================
        q_ids = self.tokenizer(
            " Please determine if the masked value is correct.",
            add_special_tokens=False, return_tensors="pt"
        ).input_ids[0].to(device)

        q_embed = self.word_embedding(q_ids)
        q_len = len(q_ids)

        # ================================================================

        # ================================================================
        def build_llm_batch(samples, node_embed, mtype: str):
            if len(samples) == 0:
                return None

            embeds_all = []
            attn_all = []
            mask_all = []
            label_all = []
            seq_lens = []

            for t in samples:
                pid = t["prompt_ids"]
                node = t["node"]
                mask_pos = t["mask_pos"]

                tok = self.word_embedding(torch.tensor(pid, device=device))
                ne = node_embed[node].unsqueeze(0)

                x = torch.cat([self.bos_embeds, ne, q_embed, tok], dim=0)


                global_mask = 1 + 1 + q_len + mask_pos

                embeds_all.append(x)
                mask_all.append(global_mask)
                label_all.append(t["label"])
                seq_lens.append(x.size(0))

            max_len = max(seq_lens)

            padded = []
            attns = []
            for i, x in enumerate(embeds_all):
                cur = x.size(0)
                pad_len = max_len - cur
                if pad_len > 0:
                    x = torch.cat([self.pad_embeds.repeat(pad_len, 1), x], dim=0)
                padded.append(x)
                attns.append([0]*pad_len + [1]*cur)

            return {
                "embeds": torch.stack(padded, 0).to(device),
                "attn": torch.tensor(attns, device=device),
                "mask_pos": torch.tensor(mask_all, device=device),
                "labels": torch.tensor(label_all, device=device),
            }

        row_batch = build_llm_batch(row_samples, node_embed_row, "row")
        cell_batch = build_llm_batch(cell_samples, node_embed_cell, "cell")

        # ================================================================

        # ================================================================
        def forward_binary(batch_dict):
            if batch_dict is None:
                return torch.tensor(0.0, device=device)

            embeds = batch_dict["embeds"]
            attn   = batch_dict["attn"]
            labels = batch_dict["labels"]
            mask_p = batch_dict["mask_pos"]

            B = embeds.size(0)

            with self.maybe_autocast():
                out = self.model(
                    inputs_embeds=embeds,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = out.hidden_states[-1]   
  # [B, L, D]

            idx = torch.arange(B, device=device)
            h_mask = hidden[idx, mask_p]      # [B, D]

            logits = self.classifier(h_mask).squeeze(-1)
            loss = self.loss_bce(logits, labels)
            return loss

        row_loss  = forward_binary(row_batch)
        cell_loss = forward_binary(cell_batch)

        # ================================================================
        # 10. Multi-task Loss
        # ================================================================
        w_row  = getattr(self, "loss_weight_row", 1.0)
        w_cell = getattr(self, "loss_weight_cell", 0.7)

        loss = w_row * row_loss + w_cell * cell_loss

        return loss, row_loss, cell_loss


    




    def pretrain(
    self, batch, entity_table, row_level_prob,
    add_noise: bool = False, epoch: int = 0, progress_lock_epoch: int = None, num_epochs: int = 100
):

        import copy
        import random
        import torch
        import torch_frame
        import pandas as pd
        from torch_frame import stype

        device = self.device
        IGNORE_INDEX = -100

        # ============================================================

        # ============================================================
        def find_atomic_mask_token(tokenizer, max_scan=50000):
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if vocab_size is None:
                vocab_size = max_scan

            limit = min(vocab_size, max_scan)

            for tok_id in range(limit):
                text = tokenizer.decode([tok_id])


                if not text.strip():
                    continue
                if len(text) > 5:
                    continue

                ids = tokenizer(text, add_special_tokens=False).input_ids
                if len(ids) != 1 or ids[0] != tok_id:
                    continue


                test_sentence = f"Column is {text}."
                test_ids = tokenizer(test_sentence, add_special_tokens=False).input_ids
                if tok_id in test_ids:
                    return text, tok_id

            raise RuntimeError("No atomic mask token found in vocab.")



        if not hasattr(self, "mask_token_text"):
            self.mask_token_text, self.mask_token_id = find_atomic_mask_token(self.tokenizer)

        mask_token_text = self.mask_token_text
        mask_token_id = self.mask_token_id
    
        # ============================================================

        # ============================================================
        progress = min(1.0, epoch / 50)




        DIFF_CELL = 0.5
        DIFF_ROW = 0.5


        CELL_CONTEXT_KEEP_RATIO = 1.0 - DIFF_CELL * progress  


        ROW_EASY_COL_PROB = 1.0 - DIFF_ROW * progress        

        # ============================================================

        # ============================================================
        select_table = entity_table
        batch_size = len(batch[entity_table].seed_time)
        num_mask = int(batch_size * self.mask_ratio)
        if num_mask <= 0:
            return torch.tensor(0.0, device=device)


        ROW_RATIO = 0.7
        CELL_RATIO = 0.3

        perm = torch.randperm(batch_size, device=device)
        row_count = max(1, int(num_mask * ROW_RATIO))
        cell_count = max(1, int(num_mask * CELL_RATIO))

        mask_row = perm[:row_count]
        mask_cell = perm[row_count:row_count + cell_count]

        # ============================================================

        # ============================================================
        batch_cell = copy.copy(batch)
        batch_row = copy.copy(batch)


        batch_cell[entity_table].tf = copy.copy(batch[entity_table].tf)
        batch_cell[entity_table].tf.feat_dict = {
            k: v.clone() for k, v in batch[entity_table].tf.feat_dict.items()
        }


        valid_columns = [
            col for col, v in batch_cell[entity_table].tf._col_to_stype_idx.items()
            if v[0] != stype.timestamp
            and (batch_cell[entity_table].df[col] != '\\N').mean() > 0.6
        ]


        cell_target_cols = []
        if valid_columns:
            num_cell_cols = min(3, len(valid_columns))  
            cell_target_cols = random.sample(valid_columns, num_cell_cols)

            for col_name in cell_target_cols:
                stype_idx = batch_cell[entity_table].tf._col_to_stype_idx[col_name]
                select_stype, col_idx = stype_idx
                feat = batch_cell[entity_table].tf.feat_dict[select_stype]

                if isinstance(feat, torch_frame.data.MultiEmbeddingTensor):
                    mask_values = feat.values.clone()
                    off = feat.offset
                    mask_values[mask_cell, off[col_idx]:off[col_idx+1]] = 0
                    batch_cell[entity_table].tf.feat_dict[select_stype].values = mask_values
                else:
                    batch_cell[entity_table].tf.feat_dict[select_stype][mask_cell] = 0


        x_cell, _ = self.encode(batch_cell, entity_table, add_noise=add_noise)

        x_row, _ = self.encode(batch_row, entity_table, add_noise=add_noise)
        x_row[select_table][mask_row] = self.mask_embed.weight  

        # ============================================================

        # ============================================================
        x_row = self.gnn(x_row, batch_row.edge_index_dict)
        node_embed_row = self.projector(x_row[select_table][:batch_size])

        x_cell_gnn = self.gnn(x_cell, batch_cell.edge_index_dict)
        node_embed_cell = self.projector(x_cell_gnn[select_table][:batch_size])

        # ============================================================

        # ============================================================
        df = batch[select_table].df

        df_row = df.iloc[batch[select_table].n_id[mask_row].cpu().numpy()]
        df_cell = df.iloc[batch[select_table].n_id[mask_cell].cpu().numpy()]

        filtered_row = df_row[self.column_filter(df_row, select_table, refresh=True)]
        filtered_cell = df_cell[self.column_filter(df_cell, select_table, refresh=True)]

        # ============================================================

        # ============================================================
        row_samples = []
        cell_samples = []


        def sample_negative_values(column_name, true_value, max_neg=2):
            col_series = df[column_name]

            cand = col_series[(col_series != '\\N') & (col_series.notna())]
            cand = cand[cand != true_value]
            if len(cand) == 0:
                return []
            uniq = cand.drop_duplicates().tolist()
            if len(uniq) <= max_neg:
                return [str(v) for v in uniq]
            return [str(v) for v in random.sample(uniq, max_neg)]


        for b_idx, (_, row) in zip(mask_row.tolist(), df_row.iterrows()):
            row_dict = row.to_dict()
            cols = [c for c, v in row_dict.items() if v not in ['\\N', None]]
            if not cols:
                continue


            easy_cols = [c for c in cols if self.is_easy_column(select_table, c, batch=batch)]
            hard_cols = [c for c in cols if c not in easy_cols]

            if random.random() < ROW_EASY_COL_PROB and easy_cols:
                tgt = random.choice(easy_cols)
            else:
                tgt = random.choice(hard_cols if hard_cols else easy_cols)

            true = row_dict[tgt]
            if true in ['\\N', None]:
                continue


            parts = []
            for col_name in filtered_row.columns:
                val = row_dict.get(col_name, None)
                if val in ['\\N', None]:
                    continue
                if col_name == tgt:

                    parts.append(f"{col_name} is {mask_token_text}.")
                else:
                    parts.append(f"{col_name} is {val}.")

            if not parts:
                continue

            context_text = " ".join(parts)


            prompt_text = (
                context_text
                + f" Please output the correct value of [{tgt}]."
            )
            #print(prompt_text)
            row_samples.append({
                "node": b_idx,
                "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                "answer_ids": self.tokenizer(str(true), add_special_tokens=False).input_ids,
                "type": "row",
            })


        if len(cell_target_cols) > 0:

            target_cols = [c for c in cell_target_cols if c in filtered_cell.columns]

            for b_idx, (_, row_view) in zip(mask_cell.tolist(), filtered_cell.iterrows()):
                full = df_cell.loc[row_view.name]
                row_dict_full = full.to_dict()

                for tgt in target_cols:
                    true = row_dict_full.get(tgt, None)
                    if true in ['\\N', None]:
                        continue


                    cols_all = list(filtered_cell.columns)
                    random.shuffle(cols_all)
                    k = max(1, int(len(cols_all) * CELL_CONTEXT_KEEP_RATIO))
                    context_cols = cols_all[:k]
                    if tgt not in context_cols:

                        context_cols[0] = tgt

                    parts = []
                    for col_name in context_cols:
                        val = row_dict_full.get(col_name, None)
                        if val in ['\\N', None]:
                            continue
                        if col_name == tgt:

                            parts.append(f"{col_name} is {mask_token_text}.")
                        else:
                            parts.append(f"{col_name} is {val}.")

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
                        "type": "cell",
                    })


        if len(row_samples) == 0 and len(cell_samples) == 0:
            return torch.tensor(0.0, device=device)

        # ============================================================

        # ============================================================
        q_row_ids = self.tokenizer(
            " Question: Predict the missing value in this row.",
            add_special_tokens=False, return_tensors='pt'
        ).input_ids[0].to(device)

        q_cell_ids = self.tokenizer(
            " Question: Predict the value of the masked cell.",
            add_special_tokens=False, return_tensors='pt'
        ).input_ids[0].to(device)

        q_embed_row = self.word_embedding(q_row_ids)
        q_embed_cell = self.word_embedding(q_cell_ids)

        q_len_row = q_row_ids.size(0)
        q_len_cell = q_cell_ids.size(0)

        # ============================================================

        # ============================================================
        def build_llm_batch(samples):
            batch_embeds = []
            batch_labels = []
            batch_mask = []
            seq_lens = []

            for t in samples:
                pid = t["prompt_ids"]
                aid = t["answer_ids"]
                mtype = t["type"]
                node = t["node"]

                if len(aid) == 0:
                    continue

                tok = self.word_embedding(torch.tensor(pid + aid, device=device))

                if mtype == "row":
                    ne = node_embed_row[node].unsqueeze(0)
                    q_emb = q_embed_row
                    q_len = q_len_row
                else:
                    ne = node_embed_cell[node].unsqueeze(0)
                    q_emb = q_embed_cell
                    q_len = q_len_cell

                x = torch.cat([self.bos_embeds, ne, q_emb, tok], dim=0)

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
                batch_mask.append([0] * pad + [1] * cur)

            embeds = torch.stack(batch_embeds, 0).to(device)
            attn = torch.tensor(batch_mask, device=device)
            labels = torch.tensor(batch_labels, device=device)
            return embeds, attn, labels


        row_embeds, row_attn, row_labels = build_llm_batch(row_samples)
        cell_embeds, cell_attn, cell_labels = build_llm_batch(cell_samples)

        # ============================================================

        # ============================================================
        row_loss = torch.tensor(0.0, device=device)
        cell_loss = torch.tensor(0.0, device=device)

        with self.maybe_autocast():
            if row_embeds is not None:
                out_row = self.model(
                    inputs_embeds=row_embeds,
                    attention_mask=row_attn,
                    labels=row_labels,
                    return_dict=True
                )
                row_loss = out_row.loss

            if cell_embeds is not None:
                out_cell = self.model(
                    inputs_embeds=cell_embeds,
                    attention_mask=cell_attn,
                    labels=cell_labels,
                    return_dict=True
                )
                cell_loss = out_cell.loss

        # ============================================================

        # ============================================================
        w_row = getattr(self, "loss_weight_row", 1.0)
        w_cell = getattr(self, "loss_weight_cell", 0.7)  

        loss = 0.7 * row_loss + 1 * cell_loss
        return loss, row_loss, cell_loss


     
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
        max_llm_samples = int(os.environ.get("REL_LLM_MAX_LLM_SAMPLES", "8"))
        if max_llm_samples > 0 and len(samples) > max_llm_samples:
            keep = torch.randperm(len(samples))[:max_llm_samples].tolist()
            samples = [samples[i] for i in keep]

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
            batch_mask.append([0] * (max_len - x.size(0)) + [1] * x.size(0))  

        embeds = torch.stack(batch_embeds, 0).to(device)
        attn = torch.tensor(batch_mask, device=device)
        labels = torch.tensor(batch_labels, device=device)
        return embeds, attn, labels

    def _build_llm_batchv6(self, samples, row_embed, q_embed, q_len, device):
        """
        samples: list of dict {node, prompt_ids, answer_ids}
        row_embed: [B, d]
        q_embed: [q_len, d]
        q_len: int
        """
        import torch

        IGNORE_INDEX = -100
        d = row_embed.size(-1)


        def embed_tokens(ids):
            ids = torch.tensor(ids, device=device, dtype=torch.long)
            return self.word_embedding(ids)  # [L, d]

        seq_embeds = []
        seq_labels = []
        seq_attn = []

        max_len = 0
        packed = []

        for s in samples:
            node = s["node"]
            prompt_ids = s["prompt_ids"]
            answer_ids = s["answer_ids"]


            row_tokens = self._make_row_tokens(row_embed[node])  # [3, d]


            p_embed = embed_tokens(prompt_ids)                   # [Lp, d]
            a_embed = embed_tokens(answer_ids)                   # [La, d]


            x = torch.cat([row_tokens, q_embed, p_embed, a_embed], dim=0)  # [L, d]



            L_row = row_tokens.size(0)  # 3
            Lq = q_embed.size(0)
            Lp = p_embed.size(0)
            La = a_embed.size(0)

            labels = torch.full((L_row + Lq + Lp + La,), IGNORE_INDEX, device=device, dtype=torch.long)

            labels[L_row + Lq + Lp : L_row + Lq + Lp + La] = torch.tensor(answer_ids, device=device, dtype=torch.long)

            packed.append((x, labels))
            max_len = max(max_len, x.size(0))

        if not packed:
            return None, None, None


        bs = len(packed)
        embeds = torch.zeros((bs, max_len, d), device=device, dtype=row_embed.dtype)
        labels = torch.full((bs, max_len), IGNORE_INDEX, device=device, dtype=torch.long)
        attn = torch.zeros((bs, max_len), device=device, dtype=torch.long)

        for i, (x, lab) in enumerate(packed):
            L = x.size(0)
            embeds[i, :L] = x
            labels[i, :L] = lab
            attn[i, :L] = 1

        return embeds, attn, labels

    def _build_llm_batch_lm_only(
    self,
    samples,
    row_embed=None,   
    q_embed=None,
    q_len=None,
    device=None,
):
        import torch

        IGNORE_INDEX = -100
        max_llm_samples = int(os.environ.get("REL_LLM_MAX_LLM_SAMPLES", "8"))
        if max_llm_samples > 0 and len(samples) > max_llm_samples:
            keep = torch.randperm(len(samples))[:max_llm_samples].tolist()
            samples = [samples[i] for i in keep]

        # embedding dim
        if row_embed is not None:
            d = row_embed.size(-1)
        else:
            d = q_embed.size(-1)

        def embed_tokens(ids):
            ids = torch.tensor(ids, device=device, dtype=torch.long)
            return self.word_embedding(ids)

        packed = []
        max_len = 0

        for s in samples:
            prompt_ids = s["prompt_ids"]
            answer_ids = s["answer_ids"]


            if row_embed is not None:
                node = s["node"]
                row_tokens = self._make_row_tokens(row_embed[node])  # [3, d]
            else:
                row_tokens = None


            p_embed = embed_tokens(prompt_ids)
            a_embed = embed_tokens(answer_ids)


            parts = []
            if row_tokens is not None:
                parts.append(row_tokens)
            if q_embed is not None:
                parts.append(q_embed)
            parts.append(p_embed)
            parts.append(a_embed)

            x = torch.cat(parts, dim=0)

            # --------- labels ---------
            L = x.size(0)
            labels = torch.full((L,), IGNORE_INDEX, device=device, dtype=torch.long)


            labels[L - len(answer_ids) : L] = torch.tensor(
                answer_ids, device=device, dtype=torch.long
            )

            packed.append((x, labels))
            max_len = max(max_len, L)

        if not packed:
            return None, None, None

        # --------- pad batch ---------
        bs = len(packed)
        embeds = torch.zeros((bs, max_len, d), device=device)
        labels = torch.full((bs, max_len), IGNORE_INDEX, device=device, dtype=torch.long)
        attn = torch.zeros((bs, max_len), device=device, dtype=torch.long)

        for i, (x, lab) in enumerate(packed):
            L = x.size(0)
            embeds[i, :L] = x
            labels[i, :L] = lab
            attn[i, :L] = 1

        return embeds, attn, labels


    def _make_row_tokens(self, row_vec: torch.Tensor) -> torch.Tensor:
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

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        CELL_RATIO = float(os.environ.get("REL_LLM_CELL_RATIO", "0.25"))
        CELL_RATIO = max(0.05, min(1.0, CELL_RATIO))

        if mask_indices is None:
            num_mask = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
            mask_row = torch.randperm(batch_size, device=device)[:num_mask]
        else:
            mask_row = mask_indices


        progress = min(1.0, epoch / 30.0)
        #epochs = math.ceil(epoch/1000) * 1000
        #progress = min(1.0, epochs / 50000.0) 


        raw_keep = 1.0 - float(b) * progress
        ROW_EASY_COL_PROB = float(max(0.25, min(1.0, raw_keep)))  

        #progress = min(1.0, epoch / 50.0)
        #print(b)
        #ROW_EASY_COL_PROB = 1.0 - float(b) * progress


        batch_row = copy.deepcopy(batch)
        x_row, _ = self.encode(batch_row, select_table, add_noise=add_noise)
        x_row[select_table][mask_row] = self.mask_embed.weight
        x_row = self.gnn(x_row, batch_row.edge_index_dict)
        node_embed_row = self.projector(x_row[select_table][:batch_size])

        # ---------------------------------------------------------

        # ---------------------------------------------------------
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
        max_target_cols = min(
            max_target_cols,
            int(os.environ.get("REL_LLM_MAX_TARGET_COLS", "1")),
        )

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        

        CELL_RATIO = 1.0

        if mask_indices is None:
            num_mask = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
            mask_cell = torch.randperm(batch_size, device=device)[:num_mask]
        else:
            mask_cell = mask_indices

        # ---------------------------------------------------------


        # ---------------------------------------------------------
        progress = min(1.0, epoch / 30.0)  


        MIN_KEEP = 0.15   
        MAX_KEEP = 1.0    


        raw_keep = MIN_KEEP + float(b) * progress

        CELL_CONTEXT_KEEP_RATIO = float(
            max(MIN_KEEP, min(MAX_KEEP, raw_keep))
        )


        # ---------------------------------------------------------

        # ---------------------------------------------------------
        batch_cell = copy.deepcopy(batch)
        batch_cell[select_table].tf = copy.copy(batch[select_table].tf)
        batch_cell[select_table].tf.feat_dict = {
            k: v.clone() for k, v in batch[select_table].tf.feat_dict.items()
        }
        tf = batch_cell[select_table].tf

        # ---------------------------------------------------------

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

        # ---------------------------------------------------------

        # ---------------------------------------------------------
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

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        x_cell, _ = self.encode(batch_cell, select_table, add_noise=add_noise)
        x_cell = self.gnn(x_cell, batch_cell.edge_index_dict)
        node_embed_cell = self.projector(x_cell[select_table][:batch_size])

        # ---------------------------------------------------------

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # No-GNN: MLP-based Row Context Aggregation
        # ---------------------------------------------------------


        # ---------------------------------------------------------

        # ---------------------------------------------------------

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


                    # print(context_text)


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

    def pretrain_cell_ada(
            self,
            batch: Any,
            entity_table: str,
            *,
            row_level_prob: float = 0.5,
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

        row_level_prob = float(max(0.0, min(1.0, row_level_prob)))

        MIN_KEEP = 0.15
        MAX_KEEP = 1.0
        CELL_CONTEXT_KEEP_RATIO = MIN_KEEP + (1.0 - row_level_prob) * (MAX_KEEP - MIN_KEEP)
        CELL_CONTEXT_KEEP_RATIO = float(max(MIN_KEEP, min(MAX_KEEP, CELL_CONTEXT_KEEP_RATIO)))

        batch_cell = copy.deepcopy(batch)
        batch_cell[select_table].tf = copy.copy(batch[select_table].tf)
        batch_cell[select_table].tf.feat_dict = {
            k: v.clone() for k, v in batch[select_table].tf.feat_dict.items()
        }
        tf = batch_cell[select_table].tf

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
                return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0,
                                                                                                        device=device)

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
                    prompt_text = context_text + f" Please output the correct value of [{tgt}]."

                    cell_samples.append({
                        "node": b_idx,
                        "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                        "answer_ids": self.tokenizer(str(true), add_special_tokens=False).input_ids,
                    })

        if len(cell_samples) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        q_ids = self.tokenizer(
            "Question: predict the value of the masked cell.",
            add_special_tokens=False, return_tensors="pt",
        ).input_ids[0].to(device)
        q_embed = self.word_embedding(q_ids)

        embeds, attn, labels = self._build_llm_batch(
            cell_samples, node_embed_cell, q_embed, q_ids.size(0), device
        )
        if embeds is None:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )
        return out.loss, out.loss, out.loss
    
    
    def pretrain_cell_gpu(
    self,
    batch: Any,
    entity_table: str,
    *,
    add_noise: bool = False,
    epoch: int = 0,
    b: float = 0.5,
    mask_indices: Optional[torch.Tensor] = None,
    max_target_cols: int = 3,
):

        import math
        import numpy as np
        import torch

        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[select_table].seed_time)

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        CELL_RATIO = 0.3
        if mask_indices is None:
            num_mask = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
            mask_cell = torch.randperm(batch_size, device=device)[:num_mask]
        else:
            mask_cell = mask_indices.to(device)

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        epochs = math.ceil(epoch / 1000) * 1000
        progress = min(1.0, epochs / 50000.0)
        raw_keep = 1.0 - float(b) * progress
        CELL_CONTEXT_KEEP_RATIO = float(max(0.25, min(1.0, raw_keep)))

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        tf = batch[select_table].tf
        df_full = batch[select_table].df

        # ---------------------------------------------------------
        # Helper: normalize cell value
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

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        valid_columns = []
        for col, stype_idx in tf._col_to_stype_idx.items():
            if not getattr(stype_idx, "__getitem__", None):
                continue
            stype_name, _ = stype_idx
            if str(stype_name) == "timestamp":
                continue
            col_series = df_full[col]
            if (col_series != "\\N").mean() <= 0.6:
                continue
            valid_columns.append(col)

        if len(valid_columns) == 0:
            return torch.tensor(0.0, device=device)

        num_cell_cols = min(max_target_cols, len(valid_columns))
        perm = torch.randperm(len(valid_columns))[:num_cell_cols].tolist()
        cell_target_cols = [valid_columns[i] for i in perm]

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        masked_backup = {}

        for col_name in cell_target_cols:
            stype_name, col_idx = tf._col_to_stype_idx[col_name]
            feat = tf.feat_dict[stype_name]

            if hasattr(feat, "values") and hasattr(feat, "offset"):
                off = feat.offset
                sl = slice(off[col_idx], off[col_idx + 1])

                masked_backup[col_name] = feat.values[mask_cell, sl].clone()
                feat.values[mask_cell, sl] = 0
            else:
                masked_backup[col_name] = feat[mask_cell].clone()
                feat[mask_cell] = 0

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        x_cell, _ = self.encode(batch, select_table, add_noise=add_noise)
        x_cell = self.gnn(x_cell, batch.edge_index_dict)
        node_embed_cell = self.projector(x_cell[select_table][:batch_size])

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        for col_name, backup in masked_backup.items():
            stype_name, col_idx = tf._col_to_stype_idx[col_name]
            feat = tf.feat_dict[stype_name]

            if hasattr(feat, "values") and hasattr(feat, "offset"):
                off = feat.offset
                sl = slice(off[col_idx], off[col_idx + 1])
                feat.values[mask_cell, sl] = backup
            else:
                feat[mask_cell] = backup

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        df = batch[select_table].df
        df_cell = df.iloc[batch[select_table].n_id[mask_cell].cpu().numpy()]
        filtered_cell = df_cell[self.column_filter(df_cell, select_table, refresh=True)]

        cell_samples = []

        target_cols = [c for c in cell_target_cols if c in filtered_cell.columns]
        if len(target_cols) == 0:
            return torch.tensor(0.0, device=device)

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
                true = normalize_val(row_dict_full.get(tgt, None))
                if true is None:
                    continue

                k = max(2, int(len(cols_all) * CELL_CONTEXT_KEEP_RATIO))
                k = min(k, len(cols_all))
                idx_perm = torch.randperm(len(cols_all))[:k].tolist()
                context_cols = [cols_all[i] for i in idx_perm]

                if tgt not in context_cols:
                    context_cols[0] = tgt

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

                prompt_text = (
                    " ".join(parts)
                    + f" Please output the correct value of [{tgt}]."
                )

                cell_samples.append({
                    "node": b_idx,
                    "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                    "answer_ids": self.tokenizer(str(true), add_special_tokens=False).input_ids,
                })

        if len(cell_samples) == 0:
            return torch.tensor(0.0, device=device)

        # ---------------------------------------------------------
        # LLM forward
        # ---------------------------------------------------------
        q_ids = self.tokenizer(
            "Question: predict the value of the masked cell.",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(device)

        q_embed = self.word_embedding(q_ids)

        embeds, attn, labels = self._build_llm_batch(
            cell_samples, node_embed_cell, q_embed, q_ids.size(0), device
        )

        if embeds is None:
            return torch.tensor(0.0, device=device)

        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )

        return out.loss, out.loss, out.loss

    
    def pretrain_cell_lm_contextual(
    self,
    batch: Any,
    entity_table: str,
    *,
    epoch: int = 0,
    b: float = 0.5,
    context_mode: str = "curriculum",  # ["curriculum", "full", "single"]
    mask_indices: Optional[torch.Tensor] = None,
    max_target_cols: int = 3,
) -> torch.Tensor:
        'Pretraining'

        import copy
        import math
        import numpy as np
        import torch

        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[select_table].seed_time)

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        if mask_indices is None:
            num_mask = max(1, int(batch_size * self.mask_ratio))
            mask_cell = torch.randperm(batch_size, device=device)[:num_mask]
        else:
            mask_cell = mask_indices

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        if context_mode == "curriculum":
            progress = min(1.0, epoch / 30.0)
            raw_keep = 1.0 - float(b) * progress
            KEEP_RATIO = float(max(0.25, min(1.0, raw_keep)))
        else:
            KEEP_RATIO = None  

        # ---------------------------------------------------------

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

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        batch_cell = copy.deepcopy(batch)
        tf = batch_cell[select_table].tf
        df_full = batch_cell[select_table].df

        valid_columns = []
        for col, stype_idx in tf._col_to_stype_idx.items():
            if not getattr(stype_idx, "__getitem__", None):
                continue
            stype_name, _ = stype_idx
            if str(stype_name) == "timestamp":
                continue
            if (df_full[col] != "\\N").mean() > 0.6:
                valid_columns.append(col)

        if len(valid_columns) == 0:
            return torch.tensor(0.0, device=device)

        perm = torch.randperm(len(valid_columns))[:min(max_target_cols, len(valid_columns))]
        target_cols = [valid_columns[i] for i in perm.tolist()]

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        df = batch[select_table].df
        df_cell = df.iloc[batch[select_table].n_id[mask_cell].cpu().numpy()]
        filtered_cell = df_cell[self.column_filter(df_cell, select_table, refresh=True)]

        cell_samples = []

        for b_idx, (row_index, _) in zip(mask_cell.tolist(), filtered_cell.iterrows()):
            row = df_cell.loc[row_index].to_dict()

            cols_all = [
                c for c in filtered_cell.columns
                if normalize_val(row.get(c, None)) is not None
            ]
            if len(cols_all) < 2:
                continue

            for tgt in target_cols:
                true = normalize_val(row.get(tgt, None))
                if true is None:
                    continue

                # -----------------------------

                # -----------------------------
                if context_mode == "full":
                    context_cols = cols_all[:]  

                elif context_mode == "single":
                    candidates = [c for c in cols_all if c != tgt]
                    if len(candidates) == 0:
                        continue
                    ctx = candidates[torch.randint(len(candidates), (1,)).item()]
                    context_cols = [ctx, tgt]

                elif context_mode == "curriculum":
                    k = max(2, int(len(cols_all) * KEEP_RATIO))
                    k = min(k, len(cols_all))
                    idx = torch.randperm(len(cols_all))[:k].tolist()
                    context_cols = [cols_all[i] for i in idx]
                    if tgt not in context_cols:
                        context_cols[0] = tgt
                else:
                    raise ValueError(f"Unknown context_mode: {context_mode}")

                # -----------------------------

                # -----------------------------
                parts = []
                for c in context_cols:
                    v = normalize_val(row.get(c, None))
                    if v is None:
                        continue
                    if c == tgt:
                        parts.append(f"{c} is {mask_token_text}.")
                    else:
                        parts.append(f"{c} is {v}.")
                if not parts:
                    continue

                prompt_text = (
                    " ".join(parts)
                    + f" Please output the correct value of [{tgt}]."
                )

                cell_samples.append({
                    "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                    "answer_ids": self.tokenizer(true, add_special_tokens=False).input_ids,
                })

        if len(cell_samples) == 0:
            return torch.tensor(0.0, device=device)

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        q_ids = self.tokenizer(
            "Question: predict the value of the masked cell.",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(device)

        embeds, attn, labels = self._build_llm_batch_lm_only(
            cell_samples,
            row_embed=None,  
            q_embed=self.word_embedding(q_ids),
            q_len=q_ids.size(0),
            device=device,
        )

        if embeds is None:
            return torch.tensor(0.0, device=device)

        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )

        return out.loss, out.loss, out.loss


    def pretrain_row_context1(
    self,
    batch,
    entity_table,
    *,
    add_noise=False,
    epoch=0,
):
        import torch, math

        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[select_table].seed_time)


        CELL_RATIO = 1.0
        NUM = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
        rows = torch.randperm(batch_size, device=device)[:NUM]
        rows = torch.randperm(batch_size, device=device)[:NUM]

        df = batch[select_table].df
        df_rows = df.iloc[batch[select_table].n_id[rows].cpu().numpy()]

        samples = []

        def norm(v):
            if v in ["\\N", None, "nan", "NaN"]:
                return None
            try:
                if isinstance(v, float) and math.isnan(v):
                    return None
            except: pass
            return str(v)

        for b_idx, (_, row) in zip(rows.tolist(), df_rows.iterrows()):
            row = row.to_dict()

            parts_full = []
            parts_mask = []

            for col, v in row.items():
                vv = norm(v)
                if vv is None:
                    continue
                parts_full.append(f"{col} is {vv}.")
                parts_mask.append(f"{col} is {mask_token_text}.")

            if not parts_full:
                continue

            full_text = " ".join(parts_full)
            masked_text = " ".join(parts_mask)

            prompt_text = (masked_text + f" Please output the correct value")

            samples.append({
                "node": b_idx,
                "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                "answer_ids": self.tokenizer(full_text, add_special_tokens=False).input_ids,
            })

        if not samples:
            return torch.tensor(0.0, device=device)

        # --- 2. GNN row embedding ---
        x_dict, _ = self.encode(batch, entity_table, add_noise=add_noise)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        row_embed = self.projector(x_dict[entity_table][:batch_size])


        q_ids = self.tokenizer(
            "Question: complete masked row.",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(device)

        q_embed = self.word_embedding(q_ids)


        embeds, attn, labels = self._build_llm_batch(
            samples, row_embed, q_embed, q_ids.size(0), device
        )

        if embeds is None:
            return torch.tensor(0.0, device=device)


        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )

        return out.loss, out.loss, out.loss


    def pretrain_row_context(
    self,
    batch,
    entity_table,
    *,
    add_noise=False,
    epoch=0,
):

        import torch, math

        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[select_table].seed_time)


        CELL_RATIO = 0.5
        NUM = max(1, int(int(batch_size * self.mask_ratio) * CELL_RATIO))
        rows = torch.randperm(batch_size, device=device)[:NUM]

        df = batch[select_table].df
        df_rows = df.iloc[batch[select_table].n_id[rows].cpu().numpy()]

        samples = []
        def norm(v):
            if v in ["\\N", None, "nan", "NaN"]:
                return None
            try:
                if isinstance(v, float) and math.isnan(v):
                    return None
            except: pass
            return str(v)


        for b_idx, (_, row) in zip(rows.tolist(), df_rows.iterrows()):
            row = row.to_dict()

            for col, v in row.items():
                vv = norm(v)
                if vv is None:
                    continue


                prompt_text = f"{col} is {mask_token_text}. Predict the value:"

                samples.append({
                    "node": b_idx,
                    "prompt_ids": self.tokenizer(prompt_text, add_special_tokens=False).input_ids,
                    "answer_ids": self.tokenizer(vv, add_special_tokens=False).input_ids,
                })

        if not samples:
            return torch.tensor(0.0, device=device)


        x_dict, _ = self.encode(batch, entity_table, add_noise=add_noise)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        row_embed = self.projector(x_dict[entity_table][:batch_size])  # [B, d]

        # 4) prefix prompt
        q_ids = self.tokenizer(
            "Question: fill the masked column value.",
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(device)

        q_embed = self.word_embedding(q_ids)


        embeds, attn, labels = self._build_llm_batch(
            samples, row_embed, q_embed, q_ids.size(0), device
        )

        if embeds is None:
            return torch.tensor(0.0, device=device)


        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )

        return out.loss, out.loss, out.loss



    def pretrain_row_contextv6(
    self,
    batch,
    entity_table,
    *,
    add_noise: bool = False,
    epoch: int = 0,
):

        import torch, math

        device = self.device
        mask_token_text, _ = self._ensure_mask_token()

        select_table = entity_table
        batch_size = len(batch[select_table].seed_time)

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        CELL_RATIO = 0.2
        num_rows = max(1, int(batch_size * self.mask_ratio * CELL_RATIO))
        rows = torch.randperm(batch_size, device=device)[:num_rows]

        df = batch[select_table].df
        seed_nid = batch[select_table].n_id[rows].cpu().numpy()
        df_rows = df.iloc[seed_nid]

        def norm(v):
            if v in ["\\N", None, "nan", "NaN"]:
                return None
            try:
                if isinstance(v, float) and math.isnan(v):
                    return None
            except Exception:
                pass
            s = str(v)
            return None if s.strip() == "" else s

        # ---------------------------------------------------------

        # ---------------------------------------------------------

        # batch-level question
        q_text = (
            "Question: Given latent TABLE ROW tokens <table_start> <tabular_node> <table_end> "
            "that constitute one row of a table (provided as embeddings), "
            "predict the masked column value."
        )
        q_ids = self.tokenizer(q_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        q_embed = self.word_embedding(q_ids)

        # column-level shared prefix
        col_prefix_text = (
            "Given TABLE ROW tokens <table_start> <tabular_node> <table_end> "
            "that constitute a row of table. "
            "Each TABLE ROW token contains the content within the CELL. "
        )
        col_prefix_ids = self.tokenizer(col_prefix_text, add_special_tokens=False).input_ids

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        samples = []

        for i, (_, row_series) in enumerate(df_rows.iterrows()):
            row_dict = row_series.to_dict()
            row_local = int(rows[i].item())

            for col, v in row_dict.items():
                vv = norm(v)
                if vv is None:
                    continue

                col_suffix_text = (
                    f"The value of column [{col}] is {mask_token_text}. "
                    f"Please predict the masked column value."
                )
                col_suffix_ids = self.tokenizer(col_suffix_text, add_special_tokens=False).input_ids

                samples.append({
                    "node": row_local,
                    "prompt_ids": col_prefix_ids + col_suffix_ids,  
                    "answer_ids": self.tokenizer(vv, add_special_tokens=False).input_ids,
                })

        if not samples:
            z = torch.tensor(0.0, device=device)
            return z, z, z

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        x_dict, _ = self.encode(batch, entity_table, add_noise=add_noise)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        row_embed = self.projector(x_dict[entity_table][:batch_size])

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        embeds, attn, labels = self._build_llm_batchv6(
            samples,
            row_embed,
            q_embed,
            q_ids.size(0),
            device,
        )

        if embeds is None:
            z = torch.tensor(0.0, device=device)
            return z, z, z

        # ---------------------------------------------------------

        # ---------------------------------------------------------
        out = self.model(
            inputs_embeds=embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )

        return out.loss, out.loss, out.loss





  


    
  

    def schema_pretrain(self, batch, entity_table):

        df = batch[entity_table].df
        columns = list(df.columns)


        schema_prompt = f"Columns: {', '.join(columns)}. Describe this table.\n"


        schema_target = (
            f"This table contains information about {entity_table}. "
            f"It includes columns such as {', '.join(columns)}."
        )
        #print("Schema prompt:", schema_prompt)
        #print("Schema target:", schema_target)

        # 4) tokenization
        prompt_ids = self.tokenizer(schema_prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(schema_target, add_special_tokens=False).input_ids

        input_ids = prompt_ids + target_ids + self.eos_user_id_list

        # labels: only supervise caption portion
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids + self.eos_id_list


        input_ids = torch.tensor(input_ids).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        inputs_embeds = self.word_embedding(input_ids)


        outputs = self.model(inputs_embeds=inputs_embeds.unsqueeze(0),
                     labels=labels.unsqueeze(0))

        loss = outputs.loss

        return loss

    def prepare_pretrain_batch(self, batch, entity_table, epoch=0, total_epochs=100, add_noise: bool = False):
        batch_size = len(batch[entity_table].seed_time)
        num_tokens_to_mask = int(batch_size * self.mask_ratio)
        perm = torch.randperm(batch_size).to(self.device)



        row_loss_est = torch.tensor([0.5], device=self.device)  
        cell_loss_est = torch.tensor([0.5], device=self.device)
        progress = torch.tensor([epoch / total_epochs], device=self.device)  

        gate_input = torch.cat([row_loss_est, cell_loss_est, progress], dim=0)  # shape [3]
        g = torch.sigmoid(self.row_gate_mlp(gate_input))
        g = 0.05 + 0.9 * g  

        row_count = int(num_tokens_to_mask * g.item())
        cell_count = num_tokens_to_mask - row_count
        mask_indices_row = perm[:row_count]
        mask_indices_cell = perm[row_count:row_count + cell_count]

        batch_cell = copy.copy(batch)
        batch_row = copy.copy(batch)

        # ---------------------- cell-level mask ----------------------
        batch_cell[entity_table].tf = copy.copy(batch[entity_table].tf)
        batch_cell[entity_table].tf.feat_dict = {k: v.clone() for k, v in batch[entity_table].tf.feat_dict.items()}

        valid_columns = [
            col for col, v in batch_cell[entity_table].tf._col_to_stype_idx.items()
            if v[0] != stype.timestamp and (batch_cell[entity_table].df[col] != '\\N').mean() > 0.5
        ]
        select_column = random.choice(valid_columns) if valid_columns else None

        if select_column:
            select_stype, select_idx = batch_cell[entity_table].tf._col_to_stype_idx[select_column]
            select_feat = batch_cell[entity_table].tf.feat_dict[select_stype]
            if isinstance(select_feat, torch_frame.data.MultiEmbeddingTensor):
                mask_values = select_feat.values.clone()
                offset = select_feat.offset
                mask_values[mask_indices_cell, offset[select_idx]: offset[select_idx + 1]] = 0
                batch_cell[entity_table].tf.feat_dict[select_stype].values = mask_values
            elif isinstance(select_feat, torch.Tensor):
                batch_cell[entity_table].tf.feat_dict[select_stype][mask_indices_cell] = 0

        x_dict_cell, _ = self.encode(batch_cell, entity_table, add_noise=add_noise)

        # ---------------------- row-level mask ----------------------
        x_dict_row, _ = self.encode(batch_row, entity_table, add_noise=add_noise)
        select_table = random.choice([i for i in x_dict_row.keys() if x_dict_row[i].numel() > 0]) \
                    if self.pretrain_random_table else entity_table
        x_dict_row[select_table][mask_indices_row] = self.mask_embed.weight


        x_dict_cell = self.gnn(x_dict_cell, batch_cell.edge_index_dict)
        x_dict_row = self.gnn(x_dict_row, batch_row.edge_index_dict)
        node_embed_cell = self.projector(x_dict_cell[select_table][:batch_size])
        node_embed_row = self.projector(x_dict_row[select_table][:batch_size])


        seed_df_cell = batch[select_table].df.iloc[batch[select_table].n_id[mask_indices_cell].cpu().numpy()]
        seed_df_row = batch[select_table].df.iloc[batch[select_table].n_id[mask_indices_row].cpu().numpy()]

        filtered_df_cell = seed_df_cell[self.column_filter(seed_df_cell, select_table, refresh=True)]
        filtered_df_row = seed_df_row[self.column_filter(seed_df_row, select_table, refresh=True)]
        if select_column in filtered_df_cell.columns:
            filtered_df_cell = filtered_df_cell[[select_column]]

        batch_input_ids, batch_label_input_ids, mask_types = [], [], []


        for _, row in filtered_df_row.iterrows():
            row_dict = list(row.to_dict().items())
            random.shuffle(row_dict)
            mask_types.append('row')
            prompts, label_tokens = [], []
            for col_name, col_value in row_dict:
                if col_value in ['\\N']:
                    continue
                other_values = [val for val in filtered_df_row[col_name].dropna().unique() if val != col_value and val != '\\N']
                if random.random() > 0.5 or len(other_values) < 1:
                    prompt = f'{col_name} is {col_value}.'
                    label = self.true_id
                else:
                    prompt = f'{col_name} is {random.choice(other_values)}.'
                    label = self.false_id
                prompts.append(prompt)
                label_tokens.append(label)
            full_prompt = " ".join(prompts)
            input_ids = self.tokenizer(full_prompt, add_special_tokens=False).input_ids + self.eos_user_id_list
            label_input_ids = [IGNORE_INDEX] * (len(input_ids) - len(label_tokens) - len(self.eos_id_list)) + label_tokens + self.eos_id_list
            batch_input_ids.append(input_ids)
            batch_label_input_ids.append(label_input_ids)


        for _, row in filtered_df_cell.iterrows():
            row_dict = list(row.to_dict().items())
            random.shuffle(row_dict)
            mask_types.append('cell')
            col_name, col_value = random.choice(row_dict)
            if col_value in ['\\N']:
                continue
            other_values = [val for val in filtered_df_cell[col_name].dropna().unique() if val != col_value and val != '\\N']
            if random.random() > 0.5 or len(other_values) < 1:
                prompt = f'{col_name} is {col_value}.'
                label = self.true_id
            else:
                prompt = f'{col_name} is {random.choice(other_values)}.'
                label = self.false_id
            input_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids + self.eos_user_id_list
            label_input_ids = [IGNORE_INDEX] * (len(input_ids) - 1 - len(self.eos_id_list)) + [label] + self.eos_id_list
            batch_input_ids.append(input_ids)
            batch_label_input_ids.append(label_input_ids)


        question_row = self.word_embedding(self.tokenizer(
            " Question: Are the statements in this row consistent? Answer Yes or No.",
            add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        question_cell = self.word_embedding(self.tokenizer(
            " Question: Is the cell value correct? Answer Yes or No.",
            add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))

        batch_inputs_embeds, batch_attention_mask = [], []
        for i, idx in enumerate(mask_indices_row):
            inputs_embeds = self.word_embedding(torch.tensor(batch_input_ids[i]).to(self.device))
            inputs_embeds = torch.cat([self.bos_embeds, node_embed_row[idx].unsqueeze(0), question_row, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        for i, idx in enumerate(mask_indices_cell):
            inputs_embeds = self.word_embedding(torch.tensor(batch_input_ids[row_count + i]).to(self.device))
            inputs_embeds = torch.cat([self.bos_embeds, node_embed_cell[idx].unsqueeze(0), question_cell, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # ===== padding =====
        max_length = max(max(x.shape[0] for x in batch_inputs_embeds), max(len(l) for l in batch_label_input_ids))
        for i in range(num_tokens_to_mask):
            pad_len = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_len, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_len + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * (max_length - len(batch_label_input_ids[i])) + batch_label_input_ids[i]

        return batch_inputs_embeds, batch_attention_mask, batch_label_input_ids, mask_types, node_embed_row, node_embed_cell, row_count, cell_count, g





    def pretrain_with_gating(self, batch, entity_table, row_level_prob=0.5, epoch=0, total_epochs=100,  add_noise=False):

        batch_inputs_embeds, batch_attention_mask, batch_label_input_ids, mask_types, node_embed_row, node_embed_cell, row_count, cell_count, g = \
            self.prepare_pretrain_batch(batch, entity_table, epoch, total_epochs, add_noise=add_noise)

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)


        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids
            )


        row_mask = torch.tensor([t=='row' for t in mask_types], device=self.device)
        cell_mask = ~row_mask
        row_loss = (outputs.loss * row_mask.float()).sum() / max(row_mask.sum(), 1)
        cell_loss = (outputs.loss * cell_mask.float()).sum() / max(cell_mask.sum(), 1)


        entropy = -(g * torch.log(g + 1e-9) + (1 - g) * torch.log(1 - g + 1e-9))


        target_g = torch.tensor(row_level_prob, device=self.device)
        target_loss = 0.1 * (g - target_g) ** 2


        final_loss = g * row_loss + (1 - g) * cell_loss + 0.1 * entropy + target_loss


        current_row_ratio = row_count / (row_count + cell_count + 1e-9)

        return final_loss, current_row_ratio


    # inside your Model class


# file: model_schema_pretrain.py


    # --------- Single-sample training ----------
    def schema_pretrain_multi_table(self, batch, db, main_table: str):
        facts = infer_schema_facts(db, batch, main_table)
        user_text = build_schema_prompt_generic(facts)
        tgt_text  = build_schema_target_generic(facts)
        #print("Schema facts:", facts)
        #print("User text:", user_text)
        #print("Target text:", tgt_text)
        
        max_len = _get_max_len_from_model(self.model)
        ids, labs, mask = _pack_causal_example(user_text, tgt_text, self.tokenizer, max_len)

        ids  = ids.to(self.device).unsqueeze(0)
        labs = labs.to(self.device).unsqueeze(0)
        mask = mask.to(self.device).unsqueeze(0)

        out = self.model(input_ids=ids, attention_mask=mask, labels=labs)
        return out.loss, out.loss, out.loss

    # --------- Mini-batch training ----------
    

    def schema_pretrain_multi_table_batch(self, items: List[Tuple[object, object, str]]):
        ids_list, labs_list, mask_list = [], [], []
        max_len = _get_max_len_from_model(self.model)

        for batch, db, main_table in items:
            facts = infer_schema_facts(db, batch, main_table)
            user_text = build_schema_prompt_generic(facts)
            tgt_text  = build_schema_target_generic(facts)
            ids, labs, mask = _pack_causal_example(user_text, tgt_text, self.tokenizer, max_len)
            ids_list.append(ids); labs_list.append(labs); mask_list.append(mask)

        ids_pad  = _pad_batch(ids_list,  pad_val=getattr(self.tokenizer, "pad_token_id", 0)).to(self.device)
        labs_pad = _pad_batch(labs_list, pad_val=IGNORE_INDEX).to(self.device)
        mask_pad = _pad_batch(mask_list, pad_val=0).to(self.device)

        out = self.model(input_ids=ids_pad, attention_mask=mask_pad, labels=labs_pad)
        return out.loss








    def lm_llama(self, batch, entity_table: str) -> Tuple[Tensor, Tensor, Tensor]:
        import json
    


        batch_size = len(batch[entity_table].seed_time)


        x_dict, _ = self.encode(batch, entity_table)

        x_dict[entity_table] = self.mask_embed.weight.expand_as(x_dict[entity_table])
        x_dict = self.gnn(x_dict, batch.edge_index_dict)

        node_embed = x_dict[entity_table][:batch_size]
        node_embed = self.projector(node_embed)                             # (B, D)


        batch_texts = []
        df_rows = batch[entity_table].df.iloc[:batch_size]
        for _, row in df_rows.iterrows():
            row_dict = {col: self._to_py(row[col]) for col in row.index}
            json_prompt = json.dumps(row_dict, ensure_ascii=False)
            prompt = (
                "You are given a JSON record of a database row.\n"
                "Understand its attributes and answer a prediction question.\n" + json_prompt
            )
            #print(prompt)
            batch_texts.append(prompt)


        tok = self.tokenizer(
            batch_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        ids: Tensor = tok.input_ids.to(self.device)                         # (B, T)
        attn_mask: Tensor = tok.attention_mask.to(self.device)              # (B, T)


        tok_embeds: Tensor = self.word_embedding(ids)                       # (B, T, D)
        text_embeds: Tensor = self._masked_token_avg(tok_embeds, attn_mask) # (B, D)


        row_embeds: Tensor = text_embeds + node_embed                       # (B, D)


        yes_embeds, no_embeds = self._get_yes_no_embeds()                   # (1, D), (1, D)

        return row_embeds, yes_embeds, no_embeds



    _yn_cache: Optional[Tuple[Tensor, Tensor]] = None

    # ---------- helpers ----------
    def _h_to_py(self, v: Any) -> Any:

        to_py = getattr(self, "_to_py", None)
        if callable(to_py):
            return to_py(v)


        try:
            if v in ["\\N", None]:
                return None
        except Exception:
            pass
        try:
        
            if pd.isna(v):
                return None
        except Exception:
            pass
        if hasattr(v, "isoformat"):
            try:
                return v.isoformat()
            except Exception:
                pass
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                pass
        return v

    def _h_masked_avg(self, token_embeds: Tensor, attn_mask: Tensor) -> Tensor:

        f = getattr(self, "_masked_token_avg", None)
        if callable(f):
            return f(token_embeds, attn_mask)
        mask = attn_mask.to(token_embeds.dtype).unsqueeze(-1)   # (B,T,1)
        summed = (token_embeds * mask).sum(dim=1)               # (B,D)
        denom = mask.sum(dim=1).clamp_min(1.0)                  # (B,1)
        return summed / denom

    def _h_get_yes_no(self) -> Tuple[Tensor, Tensor]:

        g = getattr(self, "_get_yes_no_embeds", None)
        if callable(g):
            return g()

        if self._yn_cache is not None:
            return self._yn_cache

        task_desc = description_dict[self.dataset][self.task.name]
        question = " Question: " + question_dict[self.dataset][self.task.name] + " Answer is: "

        tok_yes = self.tokenizer(task_desc + question + "Yes", add_special_tokens=False, return_tensors="pt")
        tok_no  = self.tokenizer(task_desc + question + "No",  add_special_tokens=False, return_tensors="pt")

        ids_yes, mask_yes = tok_yes.input_ids.to(self.device), tok_yes.attention_mask.to(self.device)
        ids_no,  mask_no  = tok_no.input_ids.to(self.device),  tok_no.attention_mask.to(self.device)

        we = self.word_embedding
        emb_yes = self._h_masked_avg(we(ids_yes), mask_yes)    # (1,D)
        emb_no  = self._h_masked_avg(we(ids_no),  mask_no)     # (1,D)

        self._yn_cache = (emb_yes, emb_no)
        return emb_yes, emb_no
        
    # ...existing code...
    def lm_llama_text(self, batch, entity_table: str) -> Tuple[Tensor, Tensor, Tensor]:

            device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            batch_size = len(batch[entity_table].seed_time)

            try:
                seed_idx = torch.arange(batch_size, device=device)
                neighbors = self.recursive_sample(batch, entity_table, seed_idx, num_hops=1)
            except Exception:
                neighbors = {}

            max_total_neighbors = 10
            max_neighbors_per_type = 6

            batch_texts: List[str] = []


            for i in range(batch_size):

                try:
                    global_idx = int(batch[entity_table].n_id[i].item())
                    row = batch[entity_table].df.iloc[global_idx]
                except Exception:
                    row = batch[entity_table].df.iloc[i]

                row_dict: Dict[str, Any] = {col: self._h_to_py(row[col]) for col in row.index}


                neighbor_json: Dict[str, List[Dict[str, Any]]] = {}
                try:
                    nbr_for_node = neighbors.get(entity_table, {}).get(i, {})
                except Exception:
                    nbr_for_node = {}

                candidates: List[Tuple[str, int]] = []
                for nbr_type, subdict in (nbr_for_node or {}).items():
                    cnt = 0

                    if isinstance(subdict, dict):
                        itr = list(subdict.keys())
                    else:
                        try:
                            itr = list(subdict)
                        except Exception:
                            itr = []
                    for nbr_local_id in itr:
                        if cnt >= max_neighbors_per_type:
                            break

                        try:
                            if nbr_type in getattr(batch, "node_types", []) and 0 <= int(nbr_local_id) < len(batch[nbr_type].n_id):
                                candidates.append((nbr_type, int(nbr_local_id)))
                                cnt += 1
                        except Exception:
                            continue


                seen = set()
                unique_candidates: List[Tuple[str, int]] = []
                for pair in candidates:
                    if pair not in seen:
                        unique_candidates.append(pair)
                        seen.add(pair)

                chosen = unique_candidates[:max_total_neighbors]

                for nbr_type, nbr_local_id in chosen:
                    try:
                        g_id = int(batch[nbr_type].n_id[nbr_local_id].item())
                        nrow = batch[nbr_type].df.iloc[g_id]
                        n_dict = {col: self._h_to_py(nrow[col]) for col in nrow.index}
                        neighbor_json.setdefault(f"add_{nbr_type}", []).append(n_dict)
                    except Exception:

                        continue
                import json
                full_dict = {**row_dict, **neighbor_json}
                
                batch_texts.append(json.dumps(full_dict, ensure_ascii=False))
                #print(batch_texts)

            tok = self.tokenizer(
                batch_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            ids: Tensor = tok.input_ids.to(device)                # (B,T)
            attn_mask: Tensor = tok.attention_mask.to(device)     # (B,T)

            tok_embeds: Tensor = self.word_embedding(ids)         # (B,T,D)
            row_embeds: Tensor = self._h_masked_avg(tok_embeds, attn_mask)  # (B,D)


            yes_embeds, no_embeds = self._h_get_yes_no()          # (1,D), (1,D)

            return row_embeds, yes_embeds, no_embeds

# ...existing code...



    
# 
# ...existing code...
# ...existing code...
    def lm_bert(self, batch, entity_table):

        batch_size = len(batch[entity_table].seed_time)


        neighbors = self.recursive_sample(batch, entity_table, torch.arange(batch_size), num_hops=1)
        max_total_neighbors = 2  
        max_neighbors_per_type = 3  


        sep_token = self.tokenizer.sep_token if getattr(self.tokenizer, "sep_token", None) else "[SEP]"

        batch_texts = []
        for i in range(batch_size):

            global_idx = batch[entity_table].n_id[i].item()
            row = batch[entity_table].df.iloc[global_idx]
            main_row_items = [f"The {col} value is {row[col]}" for col in row.index if row[col] not in ['\\N', None]]
            main_text = f"Main table ({entity_table}): " + f" {sep_token} ".join(main_row_items) if main_row_items else f"Main table ({entity_table}):"


            try:
                nbr_for_node = neighbors.get(entity_table, {}).get(i, {})  # dict of neighbor_type -> dict of ids
            except Exception:
                nbr_for_node = {}

            candidates = []
            for nbr_type, subdict in nbr_for_node.items():
                cnt = 0
                for nbr_id in subdict.keys():
                    if cnt >= max_neighbors_per_type:
                        break

                    if nbr_type in batch.node_types and nbr_id < len(batch[nbr_type].n_id):
                        candidates.append((nbr_type, nbr_id))
                        cnt += 1


            if candidates:
                random.shuffle(candidates)
                chosen = candidates[:max_total_neighbors]
            else:
                chosen = []


            neighbor_texts = []
            for nbr_type, nbr_id in chosen:
                try:
                    global_nid = batch[nbr_type].n_id[nbr_id].item()
                    nrow = batch[nbr_type].df.iloc[global_nid]
                    n_items = [f"The {col} value is {nrow[col]}" for col in nrow.index if nrow[col] not in ['\\N', None]]
                    if n_items:
                        neighbor_texts.append(f"Neighbor ({nbr_type}) row: " + f" {sep_token} ".join(n_items))
                except Exception:

                    continue


            if neighbor_texts:
                full_prompt = main_text + f" {sep_token} " + " ".join(neighbor_texts)
            else:
                full_prompt = main_text

            batch_texts.append(full_prompt)
            #print(f"[DEBUG] Row {i} prompt: {full_prompt}")

        row_embeds = self.encode_with_bert(batch_texts)  

        # -------- 4) yes/no prompt embedding --------
        task_desc = description_dict[self.dataset][self.task.name]
        question = " Question: " + question_dict[self.dataset][self.task.name] + " Answer is: "

        yes_text = question + "Yes"
        no_text = question + "No"

        yes_embeds = self.encode_with_bert([yes_text])  # (1, D)
        no_embeds = self.encode_with_bert([no_text])  # (1, D)

        return row_embeds, yes_embeds, no_embeds
    # ...existing code...


    def lm_robert(self, batch, entity_table):

        batch_size = len(batch[entity_table].seed_time)


        neighbors = self.recursive_sample(batch, entity_table, torch.arange(batch_size), num_hops=1)
        max_total_neighbors = 0  
        max_neighbors_per_type = 3  


        sep_token = self.tokenizer.sep_token if getattr(self.tokenizer, "sep_token", None) else "[SEP]"

        batch_texts = []
        for i in range(batch_size):

            global_idx = batch[entity_table].n_id[i].item()
            row = batch[entity_table].df.iloc[global_idx]
            main_row_items = [f"The {col} value is {row[col]}" for col in row.index if row[col] not in ['\\N', None]]
            main_text = f"Main table ({entity_table}): " + f" {sep_token} ".join(main_row_items) if main_row_items else f"Main table ({entity_table}):"


            try:
                nbr_for_node = neighbors.get(entity_table, {}).get(i, {})  # dict of neighbor_type -> dict of ids
            except Exception:
                nbr_for_node = {}

            candidates = []
            for nbr_type, subdict in nbr_for_node.items():
                cnt = 0
                for nbr_id in subdict.keys():
                    if cnt >= max_neighbors_per_type:
                        break

                    if nbr_type in batch.node_types and nbr_id < len(batch[nbr_type].n_id):
                        candidates.append((nbr_type, nbr_id))
                        cnt += 1


            if candidates:
                random.shuffle(candidates)
                chosen = candidates[:max_total_neighbors]
            else:
                chosen = []


            neighbor_texts = []
            for nbr_type, nbr_id in chosen:
                try:
                    global_nid = batch[nbr_type].n_id[nbr_id].item()
                    nrow = batch[nbr_type].df.iloc[global_nid]
                    n_items = [f"The {col} value is {nrow[col]}" for col in nrow.index if nrow[col] not in ['\\N', None]]
                    if n_items:
                        neighbor_texts.append(f"Neighbor ({nbr_type}) row: " + f" {sep_token} ".join(n_items))
                except Exception:

                    continue


            if neighbor_texts:
                full_prompt = main_text + f" {sep_token} " + " ".join(neighbor_texts)
            else:
                full_prompt = main_text

            batch_texts.append(full_prompt)
            #print(f"[DEBUG] Row {i} prompt: {full_prompt}")

        row_embeds = self.encode_with_roberta(batch_texts)  

        # -------- 4) yes/no prompt embedding --------
        task_desc = description_dict[self.dataset][self.task.name]
        question = " Question: " + question_dict[self.dataset][self.task.name] + " Answer is: "

        yes_text = task_desc + question + "Yes"
        no_text = task_desc + question + "No"

        yes_embeds = self.encode_with_roberta([yes_text])  # (1, D)
        no_embeds = self.encode_with_roberta([no_text])  # (1, D)

        return row_embeds, yes_embeds, no_embeds


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


        # ---- 1) normal forward ----
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


    def forward1(
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

        import torch
        device = self.device

        # ------------------------------------------------------------
        # 1. GNN + projector
        # ------------------------------------------------------------
        x_dict, batch_size = self.encode(batch, entity_table, add_noise, sigma_ratio)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        node_embed = self.projector(x_dict[entity_table][:batch_size])

        # ------------------------------------------------------------
        # 2. task_desc
        # ------------------------------------------------------------
        try:
            task_desc = self.description_dict[self.dataset][self.task.name]
        except:
            task_desc = "This is a binary classification task."

        task_ids = self.tokenizer(task_desc, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        task_embeds = self.word_embedding(task_ids)

        # ------------------------------------------------------------

        # ------------------------------------------------------------
        df = batch[entity_table].df
        nids = batch[entity_table].n_id[:batch_size].cpu().numpy()
        df_view = df.iloc[nids]

        valid_cols = self.column_filter(df_view, entity_table, refresh=True)

        seq_list, att_list, mask_pos_list = [], [], []

        for i in range(batch_size):
            row = df_view.iloc[i].to_dict()

            parts = []
            for col in valid_cols:
                val = row.get(col)
                if val not in ['\\N', None]:
                    parts.append(f"{col} is {val}.")

            if len(parts) == 0:
                parts = ["No valid feature."]

            prompt_text = " ".join(parts)


            final_text = prompt_text + f" The label is {self.mask_token_text}."

            prompt_ids = self.tokenizer(
                final_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0].to(device)


            try:
                mask_pos = prompt_ids.tolist().index(self.mask_token_id)
            except:
                continue

            prompt_embeds = self.word_embedding(prompt_ids)

            seq = torch.cat([
                self.bos_embeds,
                node_embed[i].unsqueeze(0),
                task_embeds,      
                prompt_embeds
            ], dim=0)

            mask_pos_global = 1 + 1 + len(task_ids) + mask_pos

            seq_list.append(seq)
            att_list.append([1]*seq.size(0))
            mask_pos_list.append(mask_pos_global)

        # ------------------------------------------------------------
        # 4. padding
        # ------------------------------------------------------------
        max_len = max(x.size(0) for x in seq_list)
        padded, attn = [], []

        for seq, att in zip(seq_list, att_list):
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                seq = torch.cat([self.pad_embeds.repeat(pad_len, 1), seq], 0)
                att  = [0]*pad_len + att
            padded.append(seq)
            attn.append(att)

        embeds = torch.stack(padded, 0)
        attn   = torch.tensor(attn, device=device)
        mask_pos_tensor = torch.tensor(mask_pos_list, device=device)

        # ------------------------------------------------------------

        # ------------------------------------------------------------
        with self.maybe_autocast():
            out = self.model(
                inputs_embeds=embeds,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True
            )
            hidden = out.hidden_states[-1]

        # ------------------------------------------------------------
        # 6. classifier(hidden_at_mask_pos)
        # ------------------------------------------------------------
        idx = torch.arange(hidden.size(0), device=device)
        h_mask = hidden[idx, mask_pos_tensor]
        logits = self.classifier(h_mask).squeeze(-1)

        if inference:
            return logits

        labels = batch[self.task.entity_table].y[:batch_size].float().to(device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss



    def forward1(
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

        import torch

        device = self.device

        # ============================================================

        # ============================================================
        x_dict, batch_size = self.encode(
            batch,
            entity_table,
            add_noise=add_noise,
            sigma_ratio=sigma_ratio,
        )
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        node_embed = self.projector(x_dict[entity_table][:batch_size])   # (B, D)

        # ============================================================

        # ============================================================
        if not hasattr(self, "label_token_0_id") or not hasattr(self, "label_token_1_id"):
            def find_single_token_from_list(tokenizer, candidates):
                for w in candidates:
                    ids = tokenizer(w, add_special_tokens=False).input_ids
                    if len(ids) == 1:
                        return w, ids[0]
                raise RuntimeError(f"No single-token match found for {candidates}")


            self.label_token_1_text, self.label_token_1_id = find_single_token_from_list(
                self.tokenizer, ["1", "yes", "<one>"]
            )
            self.label_token_0_text, self.label_token_0_id = find_single_token_from_list(
                self.tokenizer, ["0", "no", "<zero>"]
            )

        yes_id = self.label_token_1_id
        no_id  = self.label_token_0_id

        # ============================================================


        # ============================================================
        df = batch[entity_table].df
        n_ids = batch[entity_table].n_id[:batch_size].cpu().numpy()


        df_view = df.iloc[n_ids]
        try:
            valid_cols = self.column_filter(df_view, entity_table, refresh=False)
        except TypeError:

            valid_cols = self.column_filter(df_view, entity_table, refresh=True)

        seq_list = []
        attn_list = []
        seq_lens = []

        for i in range(batch_size):
            row = df.iloc[n_ids[i]]
            row_dict = row.to_dict()


            parts = []
            for col_name in valid_cols:
                val = row_dict.get(col_name, None)
                if val in ['\\N', None]:
                    continue
                parts.append(f"{col_name} is {val}.")

            if len(parts) == 0:

                context_text = "No valid feature."
            else:
                context_text = " ".join(parts)



            prompt_text = (
                " Question: Predict the binary label (0 or 1) for this row. "
                + context_text
                + " Answer with 0 or 1."
            )
            #print(prompt_text)
            input_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors='pt'
            ).input_ids[0].to(device)

            text_embeds = self.word_embedding(input_ids)   # (T, D)


            seq = torch.cat([
                self.bos_embeds,                    # (1, D)
                node_embed[i].unsqueeze(0),         # (1, D)
                text_embeds                         # (T, D)
            ], dim=0)                                # (L_i, D)

            seq_list.append(seq)
            attn_list.append([1] * seq.size(0))
            seq_lens.append(seq.size(0))

        # ============================================================

        # ============================================================
        max_len = max(seq_lens)
        for i, seq in enumerate(seq_list):
            pad_len = max_len - seq.size(0)
            if pad_len > 0:
                pad_blk = self.pad_embeds.repeat(pad_len, 1)
                seq_list[i] = torch.cat([pad_blk, seq], dim=0)
                attn_list[i] = [0] * pad_len + attn_list[i]

        inputs_embeds = torch.stack(seq_list, dim=0).to(device)      # (B, L, D)
        attention_mask = torch.tensor(attn_list, device=device)      # (B, L)

        # ============================================================

        # ============================================================
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits   # (B, L, V)


        next_logits = logits[:, -1, :]    # (B, V)

        yes_logits = next_logits[:, yes_id]
        no_logits  = next_logits[:, no_id]


        pred_logits = yes_logits - no_logits   # (B,)

        if inference:

            return pred_logits

        # ============================================================

        # ============================================================
        labels = batch[self.task.entity_table].y[:batch_size].to(device).float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, labels)

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
        pretrain_mode = kwargs.pop("pretrain_mode", None)
        if pretrain_mode == "row_context":
            return self.pretrain_row_context(
                batch,
                entity_table,
                add_noise=add_noise,
                epoch=kwargs.get("epoch", 0),
            )
        if pretrain_mode == "cell":
            return self.pretrain_cell(
                batch,
                entity_table,
                add_noise=add_noise,
                epoch=kwargs.get("epoch", 0),
                b=kwargs.get("b", 0.5),
            )
        if pretrain_mode == "llama_cl":
            return self.llama_cl(batch, entity_table)

        device = self.device

        # ============================================================
        # Helper: auto find single-token mask token
        # ============================================================
        def find_single_token_word(tokenizer, max_scan=50000):

            for tok_id in range(min(tokenizer.vocab_size, max_scan)):


                text = tokenizer.decode([tok_id])


                if not text.strip():
                    continue
                if len(text) > 5:  
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

        # ============================================================

        # ============================================================
        #mask_token_text, mask_token_id = find_single_token_word(self.tokenizer)
        # ================================================================

        # ================================================================
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

        # ============================================================

        # ============================================================
        mask_positions = (input_ids[0] == mask_token_id).nonzero(as_tuple=True)[0]
        if mask_positions.numel() == 0:
            raise ValueError(f"Mask token {mask_token_text} not found in prompt!")
        mask_pos = mask_positions.item()    # 0-based

        # ============================================================

        # ============================================================
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

        # ============================================================

        # ============================================================
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




    def forward1(self, batch: HeteroData, entity_table: NodeType, context=True, demo_info=None, inference: bool = False, tta_mode=False, add_noise: bool = False, sigma_ratio: float = 0.1) -> Tensor:

        x_dict, batch_size = self.encode(batch, entity_table, add_noise=add_noise, sigma_ratio=sigma_ratio)


        # num_sampled_nodes_dict ->  the number of sampled nodes for each node type at each layer (hop)
        """ {'user_friends': [0, 67636, 0], 'users': [512, 0, 2812], 'event_attendees': [0, 4751, 149], 'events': [0, 85, 4943], 'event_interest': [0, 224, 1]} """
        x_dict = self.gnn(x_dict, batch.edge_index_dict)  # interactions among different tables
        node_embed = x_dict[entity_table][:batch_size]
        if self.model is None: return self.head(node_embed)  # output prediction
        node_embed = self.projector(node_embed)

        # encode description, questions and labels   # TODO: pad at last/in the middle? pad id to like 0006086 of the same length?
        task_desc = description_dict[self.dataset][self.task.name]
        question = ' Question: ' + question_dict[self.dataset][self.task.name] + ' Answer: '  # https://huggingface.co/docs/transformers/tasks/prompting
        task_descs = self.tokenizer(task_desc, add_special_tokens=False)
        questions = self.tokenizer(question, add_special_tokens=False)
        #if not inference and not self.output_mlp: labels = self.label_tokenize(batch, entity_table)
        if (not inference) and (not self.output_mlp) and (not tta_mode):
            labels = self.label_tokenize(batch, entity_table)
        if context: neighbors = self.recursive_sample(batch, entity_table, torch.arange(batch_size), num_hops=1)
        if self.num_demo > 0 and demo_info is not None:
            # construct in-context demos
            demo_node_embeds, demo_labels = demo_info
            if self.task.task_type == TaskType.BINARY_CLASSIFICATION:  # balanced sampling
                mask = demo_labels == demo_labels[0].item()
                indices_A = torch.where(mask)[0]  # Indices for class 0
                indices_B = torch.where(~mask)[0]  # Indices for class 1
                count_A = indices_A.size(0)
                count_B = indices_B.size(0)
                num_demo_half = self.num_demo // 2
                extra = self.num_demo % 2
                assert count_A >= num_demo_half + extra and count_B >= num_demo_half + extra, "Not enough samples in one class"
                sampled_A = indices_A[torch.randint(0, count_A, (batch_size, num_demo_half), device=self.device)]  # (B, K_half)
                sampled_B = indices_B[torch.randint(0, count_B, (batch_size, num_demo_half), device=self.device)]  # (B, K_half)
                if extra:  # if M is odd, randomly choose which class to take the extra from for each B'
                    extra_class = torch.randint(0, 2, (batch_size,), device=self.device)
                    extra_A = indices_A[torch.randint(0, count_A, (batch_size,), device=self.device)]  # (B,)
                    extra_B = indices_B[torch.randint(0, count_B, (batch_size,), device=self.device)]  # (B,)
                    extra_samples = torch.where(extra_class, extra_B, extra_A).unsqueeze(1)  # (B, 1)
                    sampled_indices = torch.cat([sampled_A, sampled_B, extra_samples], dim=1)  # (B, K_half*2 + 1)
                else:
                    sampled_indices = torch.cat([sampled_A, sampled_B], dim=1)  # (B, K)
                shuffle_idx = torch.rand(batch_size, self.num_demo, device=self.device).argsort(dim=1)  # shuffle the indices
                sampled_indices = sampled_indices.gather(1, shuffle_idx)
            else:
                random_matrix = torch.rand(batch_size, len(demo_node_embeds), device=self.device)  # (B, B')
                sampled_indices = random_matrix.argsort(dim=1)[:, :self.num_demo]  # (B, K)
            demo_node_embeds = demo_node_embeds[sampled_indices]  # (B, K, D)
            demo_labels = demo_labels[sampled_indices]  # (B, K, 1)

        # print(neighbors)
        # tokenizer happens on CPU
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):  # TODO: do not need iteration (simplified)
            # Add bos & eos token
            input_ids = task_descs.input_ids + questions.input_ids + self.eos_user_id_list
            if (not inference) and (not self.output_mlp) and (not tta_mode):
                label_input_ids = labels.input_ids[i] + self.eos_id_list  # EOS ceases generation
                input_ids += label_input_ids

            # prioritize the entity details (which vary) over the static question.
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            if self.num_demo > 0 and demo_info is not None:
                demo_embeds = []
                for k in range(self.num_demo):
                    demo_embeds += [demo_node_embeds[i][k].unsqueeze(0), self.word_embedding(demo_labels[i][k])]
                demo_embeds.append(node_embed[i].unsqueeze(0))  # append the seed entity at last
                inputs_embeds = torch.cat([inputs_embeds[:-1], torch.cat(demo_embeds), inputs_embeds[-1:]])
            graph_prompt = node_embed[i].unsqueeze(0)
            if context:
                neighbor_embed = self.get_neighbor_embedding(neighbors[entity_table][i], x_dict)
                if neighbor_embed is not None:
                    neighbor_embed = self.projector(neighbor_embed)
                    # print(neighbor_embed.shape)
                    graph_prompt = torch.cat([graph_prompt, neighbor_embed])

            inputs_embeds = torch.cat([self.bos_embeds.to(self.device), graph_prompt.to(self.device), inputs_embeds.to(self.device)], dim=0)  # node embed after BOS

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            if (not inference) and (not self.output_mlp) and (not tta_mode):
                label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
                batch_label_input_ids.append(label_input_ids)  # auto-regressive + teacher forcing, https://github.com/XiaoxinHe/G-Retriever/issues/17

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.to(self.device).repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            if (not inference) and (not self.output_mlp) and (not tta_mode):
                batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]  # `inputs_embeds` contain `labels`

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.device)

                # ===============================

        # ===============================
        if self.output_mlp:

            with self.maybe_autocast():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )



            last_hidden = outputs.hidden_states[-1]      # (B, L, D)
            hidden = last_hidden[:, -1, :]               # (B, D)


            logits = self.mlp_head(hidden).view(-1)      # (B,)







            return logits


        if tta_mode:
            with self.maybe_autocast():
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=True
                )
            if self.output_mlp and self.lm_head is not None:
                hidden = outputs.hidden_states[-1][..., -1, :]
                logits = self.mlp_head(hidden).view(-1)
            else:
                logits = outputs.logits[..., :-1, :].contiguous()  
            return logits

        elif not inference:
            #########################
            # Training
            #########################
            label_input_ids = torch.tensor(batch_label_input_ids).to(self.device)

            if self.task.task_type != TaskType.BINARY_CLASSIFICATION:
                with self.maybe_autocast():
                    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=label_input_ids)
                return outputs.loss
            else:  # prevent over-fitting due to binary class imbalance
                with self.maybe_autocast():
                    outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
                # Shift so that tokens < n predict n, https://github.com/huggingface/transformers/issues/10480
                # https://discuss.huggingface.co/t/where-to-look-for-a-loss-definition-for-a-pretrained-model/26073/2
                logits = outputs.logits[..., :-1, :].contiguous()  
                labels = label_input_ids[..., 1:].contiguous()  # (B, L-1)
                valid_mask = (labels != IGNORE_INDEX)
                labels = labels[valid_mask]  # (2 * B), including the binary class + EOS
                logits = logits[valid_mask]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                focal_weight = (1 - probs).pow(self.gamma)
                loss = -focal_weight * probs.log()  # focal loss
                if self.alpha is not None:
                    class_weights = torch.ones(self.model.vocab_size).to(self.device)
                    class_weights[self.false_id] = self.alpha[0]
                    class_weights[self.true_id] = self.alpha[1]  # class weights
                    alpha_t = class_weights.gather(dim=0, index=labels)
                    loss = alpha_t * loss
                return loss.mean()

        #########################
        # Inference
        #########################
        with self.maybe_autocast():
            outputs = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=self.max_new_tokens, attention_mask=attention_mask, return_dict_in_generate=True,
                                            output_scores=True, use_cache=True,  # https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
                                            pad_token_id=self.tokenizer.pad_token_id)  # suppress hf warning

        if self.task.task_type == TaskType.BINARY_CLASSIFICATION:
            if self.output_probs:  # yes/no
                pred = outputs.scores[0][..., [self.false_id, self.true_id]]  # https://huggingface.co/docs/transformers/en/internal/generation_utils
                # print('before softmax:', pred)
                pred = torch.softmax(pred, dim=-1)[..., 1]  # output probs instead of 0/1, https://github.com/huggingface/transformers/issues/14498
                pred = torch.nan_to_num(pred, nan=0.5)
            else:
                seq = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                # print(seq)
                pred = torch.tensor([0.0 if i == 'No' else 1.0 for i in seq])
        elif self.task.task_type == TaskType.REGRESSION:
            seq = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            pred = []
            for i in seq:
                try:
                    pred.append(float(i))
                except ValueError:
                    pred.append(0.0)  # Skip invalid entries
            # print('Sequence: ', seq, 'Scores: ', pred)
            pred = torch.tensor(pred)
        return pred

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

        self.model.encoder.node_to_col_stats = self.model._col_stats_backup
        if hasattr(self.model.encoder, "_build_encoders"):
            self.model.encoder._build_encoders()
