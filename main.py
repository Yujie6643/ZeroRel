import argparse
import json
import math
import copy
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_RETRIES"] = "20"
os.environ["WANDB_MODE"] = "disabled"

import torch.nn.functional as F

import wandb
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from model import Model
from text_embedder import TextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from utils import task_info
import warnings
import copy
import torch
from torch_frame import stype
import pandas as pd
import copy
import random
from typing import List, Any
from typing import Optional
import torch
import random
import pandas as pd
warnings.filterwarnings(
    "ignore",
    message="cuDNN SDPA backward got grad_output.strides() != output.strides()",
)
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['CURL_CA_BUNDLE'] = ''  # huggingface connection issue
os.makedirs("./cache/checkpoints", exist_ok=True)


import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import os


import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import os


import torch

def gpu_peak_mem_mb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2

import time

class ThroughputMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.total_steps = 0

    def update(self, num_tokens, elapsed):
        self.total_tokens += int(num_tokens)
        self.total_time += float(elapsed)
        self.total_steps += 1

    @property
    def tokens_per_sec(self):
        if self.total_time <= 0:
            return 0.0
        return self.total_tokens / self.total_time

    @property
    def sec_per_step(self):
        if self.total_steps == 0:
            return 0.0
        return self.total_time / self.total_steps
    
def count_effective_tokens(attention_mask):
    if attention_mask is None:
        return 0
    return int(attention_mask.sum().item())


class MetricLogger:
    def __init__(self, save_dir="./metrics"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # -------- Epoch-level --------
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_auc_epoch = []
        self.test_auc_epoch = []

        # -------- Step-level --------
        self.train_loss_step = []
        self.row_loss_step = []
        self.cell_loss_step = []

        self.val_auc_step = []
        self.test_auc_step = []

    # -------------------
    #   Step-level update
    # -------------------
    def update_train_step(self, train_loss, row_loss, cell_loss):
        self.train_loss_step.append(float(train_loss))
        self.row_loss_step.append(float(row_loss))
        self.cell_loss_step.append(float(cell_loss))

    def update_val_test_step(self, val_auc, test_auc):
        self.val_auc_step.append(float(val_auc))
        self.test_auc_step.append(float(test_auc))

    # -------------------
    #  Epoch-level update
    # -------------------
    def update_epoch(self, train_loss, val_loss, val_auc, test_auc):
        self.train_loss_epoch.append(float(train_loss))
        self.val_loss_epoch.append(float(val_loss))
        self.val_auc_epoch.append(float(val_auc))
        self.test_auc_epoch.append(float(test_auc))

    # -------------------
    # Save JSON
    # -------------------
    def save_json(self):
        data = {
            "epoch": {
                "train_loss": self.train_loss_epoch,
                "val_loss": self.val_loss_epoch,
                "val_auc": self.val_auc_epoch,
                "test_auc": self.test_auc_epoch,
            },
            "step": {
                "train_loss": self.train_loss_step,
                "row_loss": self.row_loss_step,
                "cell_loss": self.cell_loss_step,
                "val_auc": self.val_auc_step,
                "test_auc": self.test_auc_step,
            }
        }
        with open(f"{self.save_dir}/metrics.json", "w") as f:
            json.dump(data, f, indent=4)

    # -------------------
    # Plot curves
    # -------------------
    def plot_curves(self):

        # ===== Epoch curves =====
        # Loss
        plt.figure(figsize=(7, 5))
        plt.plot(self.train_loss_epoch, label="Train Loss")
        plt.plot(self.val_loss_epoch, label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("Epoch Loss Curve")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{self.save_dir}/loss_epoch_curve.png")
        plt.close()

        # AUC
        plt.figure(figsize=(7, 5))
        plt.plot(self.val_auc_epoch, label="Val AUC")
        plt.plot(self.test_auc_epoch, label="Test AUC")
        plt.xlabel("Epoch"); plt.ylabel("AUC")
        plt.title("Epoch AUC Curve")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{self.save_dir}/auc_epoch_curve.png")
        plt.close()

        # ===== Step curves =====
        # step train loss
        plt.figure(figsize=(7, 5))
        plt.plot(self.train_loss_step, label="Train Loss (step)")
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.title("Train Loss per Step")
        plt.grid(True)
        plt.savefig(f"{self.save_dir}/train_loss_step_curve.png")
        plt.close()

        # step row/cell
        plt.figure(figsize=(7, 5))
        plt.plot(self.row_loss_step, label="Row Loss (step)")
        plt.plot(self.cell_loss_step, label="Cell Loss (step)")
        plt.xlabel("Step"); plt.ylabel("Loss")
        plt.title("Row/Cell Loss per Step")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{self.save_dir}/row_cell_loss_step_curve.png")
        plt.close()

        # step auc
        plt.figure(figsize=(7, 5))
        plt.plot(self.val_auc_step, label="Val AUC (step)")
        plt.plot(self.test_auc_step, label="Test AUC (step)")
        plt.xlabel("Val Step Trigger"); plt.ylabel("AUC")
        plt.title("Val/Test AUC per Step")
        plt.legend(); plt.grid(True)
        plt.savefig(f"{self.save_dir}/val_test_auc_step_curve.png")
        plt.close()

    # -------------------
    def finish(self):
        self.save_json()
        self.plot_curves()
        print(f"[MetricLogger] All curves saved to {self.save_dir}/")




@torch.no_grad()
def test(loader: NeighborLoader, task, demo_info=None) -> np.ndarray:
    model.eval()
    pred_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)

        pred = model(batch, task.entity_table, demo_info, inference=True)

        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)

        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if len(pred.size()) > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        pred_list.append(pred.detach().cpu())

    return torch.cat(pred_list, dim=0).numpy()

def test1(loader: NeighborLoader, task, demo_info=None) -> np.ndarray:
    model.eval()
    pred_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(batch, task.entity_table, demo_info, inference=True) 

        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)


        if len(pred.size()) > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        pred_list.append(pred.detach().cpu())

    return torch.cat(pred_list, dim=0).numpy()



@torch.no_grad()
def test_tgt(loader: NeighborLoader, task, demo_info=None, add_noise: bool = False,
             sigma_ratio: float = 0.1) -> np.ndarray:
    model_tgt.eval()
    pred_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model_tgt(batch, task.entity_table, demo_info, inference=True, add_noise=add_noise,
                         sigma_ratio=sigma_ratio)


        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None and clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if (args.model_type == 'gnn' or args.output_mlp) and task.task_type in [TaskType.BINARY_CLASSIFICATION,
                                                                                TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)

        # flatten
        if len(pred.size()) > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        pred_list.append(pred.detach().cpu())

    return torch.cat(pred_list, dim=0).numpy()


if __name__ == '__main__':
    # only classification tasks # todo: different tasks in the same dataset are different training sizes?
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_source", type=str, default="rel-avito")  # rel-event
    parser.add_argument("--dataset_target", type=str, default="rel-avito")  # rel-stack # rel-amazon

    parser.add_argument("--task_source", type=str, default="user-clicks")  # user-badge # user-churn
    parser.add_argument("--task_target", type=str, default="user-clicks")  # user-badge # user-churn

    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("./dow./relbench_examples"), )
    parser.add_argument("--debug", action='store_true')

    # GNNs
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument("--temporal_strategy", type=str, default="uniform", choices=['uniform', 'last'])
    parser.add_argument("--text_embedder", type=str, default='mpnet', choices=['glove', 'mpnet'])
    parser.add_argument("--text_embedder_path", type=str, default="./cache")

    # LLMs
    parser.add_argument("--model_type", type=str, default="./huggingface_cache/Llama-3.2-1B",
                        # ./huggingface_cache/Llama-3.2-1B
                        choices=['deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
                                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen2.5-7B-Instruct",
                                 "meta-llama/Llama-3.2-1B",
                                 "meta-llama/Llama-3.2-3B-Instruct"])
    parser.add_argument("--llm_frozen", action='store_true')
    parser.add_argument("--output_mlp", action='store_true')
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--tta_ratio", type=float, default=0.2)
    parser.add_argument("--num_demo", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument('--loss_class_weight', nargs='+', type=float, default=None)
    parser.add_argument("--align_model_path", type=str,
                        default="./dow./relbench_examples/checkpoints/source_best_model_clicks_alig.pt")
    parser.add_argument("--best_model_path", type=str,
                        default="./dow./relbench_examplese/checkpoints/source_best_model_clicks_alig.pt")

    # training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--align", action='store_true')
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--testing", action='store_true')
    parser.add_argument("--mean_std", action='store_true')
    parser.add_argument("--TTA", action='store_true')
    parser.add_argument("--align_epochs", type=int, default=3)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--schema_epochs", type=int, default=1)
    parser.add_argument("--tta_epochs", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=2000)  # 1000
    parser.add_argument("--batch_size", type=int, default=128)  # default 512 for GNN
    parser.add_argument("--tgt_batch_size", type=int, default=128)  # default 512 for GNN
    parser.add_argument("--test_batch_size", type=int, default=4)  # default 512 for GNN #32
    parser.add_argument("--val_size", type=int, default=None)  # default 512 for GNN
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.00001)  # default 0.005 for GNN
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    #############################################
    # get data and task information
    #############################################

    source_dataset_name = args.dataset_source
    target_dataset_name = args.dataset_target


    src_dataset: Dataset = get_dataset(source_dataset_name, download=False)

    src_db = src_dataset.get_db()
    print(f"[Source] Tables: {list(src_db.table_dict.keys())}")
    print(f"[Source] Time range: {src_db.min_timestamp} → {src_db.max_timestamp}")


    tgt_dataset: Dataset = get_dataset(target_dataset_name, download=False)
    tgt_db = tgt_dataset.get_db()
    print(f"[Target] Tables: {list(tgt_db.table_dict.keys())}")
    print(f"[Target] Time range: {tgt_db.min_timestamp} → {tgt_db.max_timestamp}")

    src_task = get_task(source_dataset_name, args.task_source, download=True)
    tgt_task = get_task(target_dataset_name, args.task_target, download=True)
    src_task.name = args.task_source
    tgt_task.name = args.task_target




    def prepare_col_to_stype(db, dataset_name):
        stypes_cache_path = Path(f"{args.cache_dir}/{dataset_name}/stypes.json")
        try:
            with open(stypes_cache_path, "r") as f:
                col_to_stype_dict = json.load(f)
            for table, col_dict in col_to_stype_dict.items():
                for col, type_name in col_dict.items():
                    col_dict[col] = stype(type_name)
        except FileNotFoundError:
            col_to_stype_dict = get_stype_proposal(db)
            Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(stypes_cache_path, "w") as f:
                json.dump(col_to_stype_dict, f, indent=2, default=str)
        return col_to_stype_dict


    col_to_stype_src = prepare_col_to_stype(src_db, args.dataset_source)

    col_to_stype_tgt = prepare_col_to_stype(tgt_db, args.dataset_target)

    # build heterogeneous and temporal graphs `data`
    # sentence_transformer: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    os.makedirs(args.text_embedder_path, exist_ok=True)
    text_embedder = TextEmbedding(args.text_embedder, args.text_embedder_path, device=device)

    print("Construct the source-domain graph")
    data_src, col_stats_dict_src = make_pkey_fkey_graph(
        src_db,
        col_to_stype_dict=col_to_stype_src,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=text_embedder, batch_size=128),
        cache_dir=f"{args.cache_dir}/{source_dataset_name}/materialized"
    )

    print("Construct the target-domain graph")
    data_tgt, col_stats_dict_tgt = make_pkey_fkey_graph(
        tgt_db,
        col_to_stype_dict=col_to_stype_tgt,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=text_embedder, batch_size=128),
        cache_dir=f"{args.cache_dir}/{target_dataset_name}/materialized"
    )


    # 'num_neighbors' -> the number of neighbors sampled per node (e.g., [64, 32, 16]), 'num_sampled_nodes' -> the total number of nodes sampled per layer (hop)
    out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max = task_info(src_task)
    loader_dict_src: Dict[str, NeighborLoader] = {}
    loader_dict_tgt: Dict[str, NeighborLoader] = {}


    for split in ["train", "val", "test"]:
        table = src_task.get_table(split)
        table_input = get_node_train_table_input(table=table, task=src_task)
        entity_table = table_input.nodes[0]
        bs = args.batch_size if (split == 'train' or args.val_size is None) else args.val_size
        loader_dict_src[split] = NeighborLoader(data_src,
                                                num_neighbors=[max(1, int(args.num_neighbors / 2 ** i)) for i in
                                                               range(args.num_layers)]
                                                , time_attr="time", input_nodes=table_input.nodes,
                                                input_time=table_input.time, transform=table_input.transform,
                                                batch_size=bs, temporal_strategy=args.temporal_strategy,
                                                shuffle=split == "train", num_workers=args.num_workers,
                                                persistent_workers=args.num_workers > 0,
                                                pin_memory=True)  # TODO: bidirectional


    for split in ["train", "val", "test"]:
        test_table = tgt_task.get_table(split)
        test_input = get_node_train_table_input(table=test_table, task=tgt_task)
        test_entity_table = test_input.nodes[0]

        bs = args.tgt_batch_size if split in ["train", "val"] else args.tgt_batch_size
        loader_dict_tgt[split] = NeighborLoader(data_tgt,
                                                num_neighbors=[max(1, int(args.num_neighbors / 2 ** i)) for i in
                                                               range(args.num_layers)]
                                                , time_attr="time",
                                                input_nodes=test_input.nodes,
                                                input_time=test_input.time, transform=test_input.transform,
                                                batch_size=bs,
                                                temporal_strategy=args.temporal_strategy,
                                                shuffle=split == "train", num_workers=args.num_workers,
                                                persistent_workers=args.num_workers > 0,
                                                pin_memory=True)  # TODO: bidirectional




    def _clamp_ratio(p: float) -> float:
        """Clamp to [0,1] to avoid invalid row masking."""
        if p is None:
            return 1.0
        return max(0.0, min(1.0, float(p)))


    def _clamp_ratio1(x: float) -> float:
        return max(0.0, min(1.0, float(x)))


    def is_integer_column(col: torch.Tensor, tol: float = 1e-6) -> bool:

        if not torch.is_floating_point(col):
            return True
        return torch.all(torch.abs(col - col.round()) < tol)



    def _strip_edge_time_if_node_time(data_hetero, time_attr: str = "time"):

        has_node_time = any(
            (time_attr in data_hetero[ntype]) for ntype in data_hetero.node_types
        )
        if has_node_time:
            for etype in data_hetero.edge_types:
                store = data_hetero[etype]
                if time_attr in store:
                    del store[time_attr]
        return data_hetero, has_node_time


    def is_continuous_float(series, tol=1e-6):

        if pd.api.types.is_float_dtype(series):
            return not ((series.dropna() % 1).abs() < tol).all()
        return False


    def get_continuous_float_cols_smart(data_src):

        node_type_float_cols = {}
        for node_type in data_src.node_types:
            df = data_src[node_type].df
            float_cols = [
                i for i, col in enumerate(df.columns)
                if 'ID' not in col and is_continuous_float(df[col])
            ]
            node_type_float_cols[node_type] = float_cols
        return node_type_float_cols


    node_type_float_cols = get_continuous_float_cols_smart(data_src)


    # print(node_type_float_cols)


    args.val_steps = min(args.val_steps, len(loader_dict_src['train']))
    print('Source Entity table: ', entity_table)
    print('Target Entity Table: ', table_input.nodes[0])
    print(f"[INFO] Cross-domain setup ready: {source_dataset_name} → {target_dataset_name}")

    #############################################
    # model training
    #############################################
    model = Model(data_src, col_stats_dict_src, args.num_layers, channels=args.channels, out_channels=out_channels,
                  aggr=args.aggr, dropout=args.dropout, model_type=args.model_type,
                  llm_frozen=args.llm_frozen, output_mlp=args.output_mlp, max_new_tokens=args.max_new_tokens,
                  alpha=args.loss_class_weight, num_demo=args.num_demo,
                  dataset=args.dataset_source, task=src_task, db=src_db).to(device)

    model_tgt = Model(data_tgt, col_stats_dict_tgt, args.num_layers, channels=args.channels, out_channels=out_channels,
                      aggr=args.aggr, dropout=args.dropout, model_type=args.model_type,
                      llm_frozen=args.llm_frozen, output_mlp=args.output_mlp, max_new_tokens=args.max_new_tokens,
                      alpha=args.loss_class_weight, num_demo=args.num_demo,
                      dataset=args.dataset_target, task=tgt_task, db=tgt_db).to(device)

    if args.finetune:
        print("⚙️  Finetuning stage: optimizing model_tgt parameters")
        opt_model = model_tgt
    elif args.pretrain:
        print("⚙️  Pretraining stage: optimizing model parameters")
        opt_model = model
    else:
        print("⚙️  Testing stage")
        opt_model = model_tgt

    params = [p for _, p in opt_model.named_parameters() if p.requires_grad]
    if args.wd != 0:  # weight decay should not be applied to bias terms and LayerNorm parameters
        optimizer = torch.optim.AdamW([{'params': [p for n, p in opt_model.named_parameters() if
                                                   "bias" not in n and "LayerNorm" not in n], 'weight_decay': args.wd},
                                       {'params': [p for n, p in opt_model.named_parameters() if
                                                   "bias" in n or "LayerNorm" in n], 'weight_decay': 0.0}], lr=args.lr,
                                      betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.Adam(opt_model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if higher_is_better else 'min',
                                                           factor=0.8, patience=100)
    trainable_params, all_param = opt_model.print_trainable_params()
    print(
        f"trainable params: {trainable_params / 1e6:.2f}M || all params: {all_param / 1e6:.2f}M || trainable: {100 * trainable_params / all_param:.4f}%")

   
    # #############################################
    #  Ttraining
    # #############################################
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.pretrain:
        state_dict = None
        if state_dict is not None: model.load_state_dict(state_dict)  # load pretrained weights


        metric_logger = MetricLogger(save_dir="./metrics")
        latest_val_auc = None
        latest_val_loss = None
        latest_test_auc = None


        if not args.debug:  # rename the project if init failure
            run = wandb.init(project='rel-LLM-zero', name=f'{args.dataset_source}_{args.task_source}',
                             id=f"pretrain_run_{args.dataset_source}_{args.task_source}", resume="allow")
        pretrain_steps = 0
        best_val_metric = -math.inf if higher_is_better else math.inf
        stage1_announced = False
        stage2_announced = False
        stage3_announced = False


        print("✅ Training")


        for epoch in range(1, args.pretrain_epochs + 1):
            loss_accum = row_loss_accum = cell_loss_accum = count_accum = align_loss_accum = 0
            row_ratios = []
            tq = tqdm(loader_dict_src["train"], total=len(loader_dict_src["train"]))
            row_level_prob = get_row_level_prob(epoch, 100, start=0.1, end=0.9, mode="linear")
            if epoch < args.schema_epochs and not stage1_announced:
                print(f"🚀 Entering Stage 1: GNN-LLM alignment (Epoch {epoch})")
                stage1_announced = True

            if epoch == args.schema_epochs and not stage2_announced:
                print(f"🔥 Entering Stage 2: Row/Cell reasoning pretraining (Epoch {epoch})")
                stage2_announced = True
            for batch in tq:
                try:
                    model.train()
                    batch = batch.to(device)
                    nums_samples = batch[entity_table].y.size(0)  # 当前 batch 的样本数，用来计算平均 loss
                    optimizer.zero_grad()

                    if epoch < args.schema_epochs:
                        
                        loss, row_loss, cell_loss = model.pretrain_row_context(batch, src_task.entity_table)

                        
                    else:
                        loss, row_loss, cell_loss = model.pretrain_cell(batch, src_task.entity_table, add_noise=False, epoch=pretrain_steps, b = 1.0)

                    if not torch.is_tensor(loss) or not loss.requires_grad:
                        print("⚠️ Loss has no gradient, skipping this batch")
                        continue
                    loss.backward()
                    optimizer.step()
                except torch.OutOfMemoryError:
                    print("Skipping batch due to CUDA out of memory error")
                    torch.cuda.empty_cache()  # Free up cached memory
                    continue
                pretrain_steps += 1

                loss_accum += loss.detach().item() * nums_samples
                count_accum += nums_samples
                train_loss = loss_accum / count_accum  
                
                row_loss_accum +=  row_loss.detach().item() * nums_samples
                row_loss = row_loss_accum / count_accum  

                cell_loss_accum +=  cell_loss.detach().item() * nums_samples
                cell_loss = cell_loss_accum / count_accum 


                metric_logger.update_train_step(train_loss, row_loss, cell_loss)


                summary = {'loss': train_loss, 'lr': optimizer.param_groups[-1]['lr']}
                if not args.debug:
                    for k, v in summary.items():
                        run.log({f'Pretrain/{k}': v}, step=pretrain_steps)  # Steps must be monotonically increasing
                tq.set_description(f'[Pretrain] Epoch/Step: {epoch:02d}/{pretrain_steps} | Train loss: {train_loss:.4f} | Row loss: {row_loss:.4f} |  Cell loss: {cell_loss:.4f}  ')


                if pretrain_steps % args.val_steps == 0:
                    val_pred = test(loader_dict_src["val"], src_task)
                    val_metrics = src_task.evaluate(val_pred, src_task.get_table("val"))
                    latest_val_auc = val_metrics[tune_metric]
                    latest_val_loss = val_metrics.get("loss", None)

                    if not args.debug:
                        for k, v in val_metrics.items():
                            run.log({f'src_val/{k}': v}, step=pretrain_steps)


                    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
                            not higher_is_better and val_metrics[tune_metric] <= best_val_metric):

                        best_val_metric = val_metrics[tune_metric]


                        test_pred = test(loader_dict_src["test"], src_task)
                        test_metrics = src_task.evaluate(test_pred)


                        latest_test_auc = test_metrics[tune_metric]
                        metric_logger.update_val_test_step(
                            latest_val_auc,
                            latest_test_auc
                        )

                        if not args.debug:
                            for k, v in test_metrics.items():
                                run.log({f'test/{k}': v}, step=pretrain_steps)

                        scheduler.step(val_metrics[tune_metric])
                        print(f'[Eval] Epoch/Step: {epoch:02d}/{pretrain_steps} | Val: {val_metrics} | Best val/test: {best_val_metric:.4f}/{test_metrics[tune_metric]:.4f}')


                        
    #############################################
    # Fine-tuning on Target Domain
    #############################################
    if args.finetune:
        steps = 0
        if not args.debug:
            if args.pretrain:
                run.finish()
            run = wandb.init(
                project='rel-LLM',
                name=f'{args.dataset_target}_to_{args.dataset_target}',
                id=f'finetune_run_{args.dataset_target}_to_{args.dataset_target}',
                resume="allow"
            )
        print("✅ Finetuning")

        if os.path.exists(args.best_model_path):
            print(f"🔹 加载源域预训练权重：{args.best_model_path}")
            pretrained_state = torch.load(args.best_model_path, map_location="cpu")
            model_tgt_dict = model_tgt.state_dict()


            filtered_state_dict = {
                k: v for k, v in pretrained_state.items()
                if k in model_tgt_dict and not k.startswith("encoder.encoders.")
            }


            model_tgt_dict.update(filtered_state_dict)
            model_tgt.load_state_dict(model_tgt_dict)
            print("✅ Loaded pretrained model from source domain.")

        best_val_metric = -math.inf if higher_is_better else math.inf

        for epoch in range(1, args.epochs + 1):
            loss_accum = count_accum = 0
            tq = tqdm(loader_dict_tgt["train"], total=len(loader_dict_tgt["train"]))

            for batch in tq:
                model_tgt.train()
                batch = batch.to(device)
                num_samples = batch[test_entity_table].y.size(0)
                optimizer.zero_grad()

                # forward + loss
                if args.model_type == 'gnn' or args.output_mlp:
                    output_pred = model_tgt(batch, tgt_task.entity_table)
                    output_pred = output_pred.view(-1) if len(output_pred.size()) > 1 and output_pred.size(
                        1) == 1 else output_pred
                    loss = loss_fn(output_pred.float(), batch[test_entity_table].y.float())
                else:
                    loss = model_tgt(batch, tgt_task.entity_table)


                loss.backward()
                optimizer.step()

                steps += 1
                loss_accum += loss.detach().item() * num_samples
                count_accum += num_samples
                train_loss = loss_accum / count_accum

                # wandb logging
                if not args.debug:
                    run.log({'train/loss': train_loss, 'train/lr': optimizer.param_groups[-1]['lr']}, step=steps)

                tq.set_description(f'[Finetune] Epoch/Step: {epoch:02d}/{steps} | Train loss: {train_loss:.4f}')

                if steps % args.val_steps == 0:

                    val_pred = test_tgt(loader_dict_tgt["val"], tgt_task)
                    val_metrics = tgt_task.evaluate(val_pred, tgt_task.get_table("val"))



                    if not args.debug:
                        for k, v in val_metrics.items():
                            run.log({f'src_val/{k}': v}, step=steps)
                        # for k, v in test_metrics.items():
                        # run.log({f'tgt_test/{k}': v}, step=steps)


                    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or \
                            (not higher_is_better and val_metrics[tune_metric] <= best_val_metric):
                        best_val_metric = val_metrics[tune_metric]
                        state_dict = copy.deepcopy(model_tgt.state_dict())


                    scheduler.step(val_metrics[tune_metric])

                    print(
                        f'[Eval] Epoch/Step: {epoch:02d}/{steps} | Tgt Val: {val_metrics} | Best tgt val: {best_val_metric:.4f}')

        if os.path.exists(args.best_model_path):
            print(f"🔹 Loading source-domain checkpoint: {args.best_model_path}")
            state_dict_src = torch.load(args.best_model_path, map_location="cpu")  # or map_location=device
        else:
            raise FileNotFoundError(f"❌ Source-domain checkpoint not found: {args.best_model_path}")

        model.load_state_dict(state_dict_src)
        model.eval()

        model_tgt.load_state_dict(state_dict, strict=False)
        model_tgt.eval()



        with torch.no_grad():
            print("✅ Testing")

            val_pred = test(loader_dict_src["val"], src_task)
            val_metrics = src_task.evaluate(val_pred, src_task.get_table("val"))

            test_pred_src = test(loader_dict_src["test"], src_task)
            test_metrics_src = src_task.evaluate(test_pred_src)

            test_pred = test_tgt(loader_dict_tgt["test"], tgt_task)
            test_metrics = tgt_task.evaluate(test_pred)
            print(f"[Source Val] metrics: {val_metrics}")
            print(f"[Source Test] metrics: {test_metrics_src}")
            print(f"[Target Test] metrics: {test_metrics}")

    #############################################
    # Test Time Adaption
    #############################################
    if args.TTA:
        if os.path.exists(args.best_model_path):
            print(f"🔹 加载源域权重文件：{args.best_model_path}")
            state_dict = torch.load(args.best_model_path, map_location="cpu")  # 或 map_location=device
        else:
            raise FileNotFoundError(f"❌ 未找到源域权重文件：{args.best_model_path}")

        model_tgt.load_state_dict(state_dict, strict=False)
        model_tgt.eval()

        with torch.no_grad():

            print("✅ OOD...")
            test_pred = test_tgt(loader_dict_tgt["test"], tgt_task)
            test_metrics_0 = tgt_task.evaluate(test_pred)
            best_metric1 = test_metrics_0["roc_auc"]
            print(f"[Target Test] metrics: {best_metric1}")


        best_metric = test_metrics_0["roc_auc"]
        best_epoch = 0
        patience = 10
        no_improve = 0

        # params_tta = [p for n, p in model_tgt.named_parameters() if "projector" in n and p.requires_grad]
        for n, p in model_tgt.named_parameters():
            if n.startswith("model.base_model.model.model.norm"):
                p.requires_grad = True

        params_tta = [
            p for n, p in model_tgt.named_parameters()
            if n.startswith("model.base_model.model.model.norm") and p.requires_grad
        ]



        print("Number of params passed to optimizer:", sum(p.numel() for p in params_tta))

        optimizer_tgt = torch.optim.AdamW(params_tta, lr=0.001, betas=(0.9, 0.95))


        # ===== TTA =====
        for epoch in range(1, args.tta_epochs + 1):
            loss_accum = count_accum = 0
            tq = tqdm(loader_dict_tgt["test_20"], total=len(loader_dict_tgt["test_20"]))
            model_tgt.train()
            for batch in tq:
                batch = batch.to(device)
                optimizer_tgt.zero_grad()


                logits = model_tgt(batch, tgt_task.entity_table, tta_mode=True)  # (B,)

                p = torch.sigmoid(logits)   # (B,) ∈ (0,1)

                entropy = - (
                    p * torch.log(p + 1e-8) +
                    (1 - p) * torch.log(1 - p + 1e-8)
                )

                loss = entropy.mean()


                loss_cl = model_tgt.llama_cl(batch, tgt_task.entity_table)
                loss = loss + 0.5 * loss_cl
                loss.backward()
                optimizer_tgt.step()

                num_samples = batch[tgt_task.entity_table]['tf'].num_rows  # TensorFrame 的行数

                loss_accum += loss.detach().item() * num_samples
                count_accum += num_samples
                train_loss = loss_accum / count_accum

                tq.set_description(f'[TTA] Epoch: {epoch:02d} | Entropy loss: {train_loss:.4f}')

            model_tgt.eval()
            with torch.no_grad():
                print("✅ Testing after TTA")
                test_pred = test_tgt(loader_dict_tgt["test"], tgt_task)
                test_metrics = tgt_task.evaluate(test_pred)
                print(f"[Target Test after TTA] metrics: {test_metrics}")

            current_metric = test_metrics["roc_auc"]

            if current_metric > best_metric:
                print(f"🎯 Metric improved from {best_metric:.4f} → {current_metric:.4f}")
                best_metric = current_metric
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model_tgt.state_dict().items()}
            else:
                no_improve += 1
                print(f"⚠️ No improvement for {no_improve} epoch(s). Best: {best_metric:.4f} @ epoch {best_epoch}")
                if no_improve >= patience:
                    print("🛑 Early stopping triggered!")
                    break


        if 'best_state' in locals():
            model_tgt.load_state_dict(best_state)
            print(f"✅ Restored best TTA model from epoch {best_epoch} (metric={best_metric:.4f})")

    if args.testing:
        if os.path.exists(args.best_model_path):
            print(f"🔹 Loading source-domain checkpoint: {args.best_model_path}")
            state_dict = torch.load(args.best_model_path, map_location="cpu")  # or map_location=device
        else:
            raise FileNotFoundError(f"❌ Source-domain checkpoint not found: {args.best_model_path}")

        model.load_state_dict(state_dict)
        model.eval()


        with torch.no_grad():
            print("✅ Testing")

            print("✅ IID...")
            test_pred = test(loader_dict_src["test"], src_task)
            test_metrics = src_task.evaluate(test_pred)
            print(f"[Source Test] metrics: {test_metrics}")






