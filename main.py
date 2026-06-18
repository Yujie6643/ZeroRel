import argparse
import atexit
import json
import math
import copy
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  
os.environ["CURL_CA_BUNDLE"] = ""  
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_DOWNLOAD_RETRIES"] = "20"
os.environ["WANDB_MODE"] = "disabled"
repo_root = Path(__file__).resolve().parent
for repo_relbench_cache in (
    repo_root / "dow." / "relbench",
    repo_root / "dow" / "relbench",
):
    if repo_relbench_cache.exists():
        os.environ.setdefault("RELBENCH_CACHE_DIR", str(repo_relbench_cache))
        break
print(f"[INFO] RELBENCH_CACHE_DIR={os.environ.get('RELBENCH_CACHE_DIR', '<default>')}")
# os.environ["HF_HOME"] = "/home/tyj/Rel-LLM-master/huggingface_cache"
import torch.nn.functional as F

import wandb
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import Model
from text_embedder import TextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from relbench.base import Dataset, Table, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from utils import task_info
import warnings

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


def distributed_is_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if distributed_is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if distributed_is_initialized() else 1


def is_main_process():
    return get_rank() == 0


def unwrap_model(module):
    return module.module if hasattr(module, "module") else module


def get_clean_state_dict(module):
    return unwrap_model(module).state_dict()


def normalize_state_dict_keys(state_dict):
    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_clean_state_dict(module, state_dict, strict=True):
    return unwrap_model(module).load_state_dict(
        normalize_state_dict_keys(state_dict),
        strict=strict,
    )


def limit_eval_table(table: Table, max_eval_samples: int | None) -> Table:
    if max_eval_samples is None or max_eval_samples <= 0 or len(table) <= max_eval_samples:
        return table
    return Table(
        df=table.df.iloc[:max_eval_samples].copy(),
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )


def make_capped_eval_table(task, split: str, max_eval_samples: int | None, seed: int = 42) -> Table:
    table = task.get_table(split, mask_input_cols=False)
    if max_eval_samples is None or max_eval_samples <= 0 or len(table) <= max_eval_samples:
        return table

    target_col = getattr(task, "target_col", None)
    if (
        target_col is None
        or target_col not in table.df
        or task.task_type not in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]
    ):
        return limit_eval_table(table, max_eval_samples)

    y = table.df[target_col].to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return limit_eval_table(table, max_eval_samples)

    max_eval_samples = min(max_eval_samples, len(table))
    rng = np.random.default_rng(seed)
    allocations = np.floor(counts / counts.sum() * max_eval_samples).astype(int)
    allocations = np.maximum(allocations, 1)

    while allocations.sum() > max_eval_samples:
        candidates = np.where(allocations > 1)[0]
        if len(candidates) == 0:
            break
        i = candidates[np.argmax(allocations[candidates])]
        allocations[i] -= 1

    while allocations.sum() < max_eval_samples:
        remaining = counts - allocations
        candidates = np.where(remaining > 0)[0]
        if len(candidates) == 0:
            break
        i = candidates[np.argmax(remaining[candidates])]
        allocations[i] += 1

    selected = []
    for cls, n in zip(classes, allocations):
        cls_indices = np.flatnonzero(y == cls)
        selected.append(rng.choice(cls_indices, size=min(int(n), len(cls_indices)), replace=False))
    selected_indices = np.sort(np.concatenate(selected))
    capped_table = Table(
        df=table.df.iloc[selected_indices].copy(),
        fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,
        pkey_col=table.pkey_col,
        time_col=table.time_col,
    )
    if is_main_process():
        capped_counts = capped_table.df[target_col].value_counts().sort_index().to_dict()
        print(f"[Eval] {split} capped to {len(capped_table)} stratified samples: {capped_counts}")
    return capped_table


def evaluate_limited(task, pred: np.ndarray, split: str, max_eval_samples: int | None, table: Table | None = None):
    table = limit_eval_table(table if table is not None else task.get_table(split, mask_input_cols=False), len(pred))
    if max_eval_samples is not None and max_eval_samples > 0:
        table = limit_eval_table(table, max_eval_samples)
        pred = pred[:len(table)]
    return task.evaluate(pred, table)


def save_checkpoint(state_dict, path: str) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def cleanup_distributed():
    if distributed_is_initialized():
        dist.destroy_process_group()


def partition_table_input_for_rank(table_input, split):
    if not distributed_is_initialized() or split != "train":
        return table_input

    rank = get_rank()
    world_size = get_world_size()
    node_type, nodes = table_input.nodes
    index = torch.arange(rank, nodes.numel(), world_size)
    count_device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    local_count = torch.tensor([index.numel()], device=count_device)
    dist.all_reduce(local_count, op=dist.ReduceOp.MIN)
    index = index[:int(local_count.item())]

    time = table_input.time[index] if table_input.time is not None else None
    target = table_input.target[index] if table_input.target is not None else None
    transform = None
    if table_input.transform is not None:
        transform = type(table_input.transform)(table_input.transform.entity, target)

    return table_input._replace(
        nodes=(node_type, nodes[index]),
        time=time,
        target=target,
        transform=transform,
    )


def per_rank_batch_size(batch_size):
    if batch_size is None:
        return None
    if not distributed_is_initialized():
        return batch_size
    return max(1, math.ceil(batch_size / get_world_size()))


def is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


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



"""@torch.no_grad()
def test(loader: NeighborLoader, demo_info=None) -> np.ndarray:
    model.eval()
    pred_list = []
    for test_batch in tqdm(loader):
        test_batch = test_batch.to(device)
        pred = model(test_batch, task.entity_table, demo_info, inference=True)
        if task.task_type == TaskType.REGRESSION:
            assert clamp_min is not None and clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if (args.model_type == 'gnn' or args.output_mlp) and task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)  # normalize to between 0 and 1

        pred = pred.view(-1) if len(pred.size()) > 1 and pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()"""


@torch.no_grad()
def test(loader: NeighborLoader, task, demo_info=None, max_eval_samples: int | None = None) -> np.ndarray:
    eval_model = unwrap_model(model)
    eval_model.eval()
    pred_list = []
    seen = 0

    for batch in tqdm(loader):
        batch = batch.to(device)


        pred = eval_model(batch, task.entity_table, demo_info, inference=True)
        #print(pred)


        if task.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTILABEL_CLASSIFICATION]:
            pred = torch.sigmoid(pred)
            #print(pred)

        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if len(pred.size()) > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        if max_eval_samples is not None and max_eval_samples > 0:
            remaining = max_eval_samples - seen
            if remaining <= 0:
                break
            pred = pred[:remaining]
        pred_list.append(pred.detach().cpu())
        seen += pred.numel() if pred.dim() == 1 else pred.size(0)
        if max_eval_samples is not None and max_eval_samples > 0 and seen >= max_eval_samples:
            break

    pred_np = torch.cat(pred_list, dim=0).numpy()
    if max_eval_samples is not None and max_eval_samples > 0 and is_main_process():
        print(f"[Eval] Using {len(pred_np)} samples for {task.entity_table} test/eval cap={max_eval_samples}")
    return pred_np

def test1(loader: NeighborLoader, task, demo_info=None) -> np.ndarray:
    eval_model = unwrap_model(model)
    eval_model.eval()
    pred_list = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = eval_model(batch, task.entity_table, demo_info, inference=True) 



        if task.task_type == TaskType.REGRESSION:
            pred = torch.clamp(pred, clamp_min, clamp_max)




        if len(pred.size()) > 1 and pred.size(1) == 1:
            pred = pred.view(-1)

        pred_list.append(pred.detach().cpu())

    return torch.cat(pred_list, dim=0).numpy()


@torch.no_grad()
def test_tgt(loader: NeighborLoader, task, demo_info=None, add_noise: bool = False,
             sigma_ratio: float = 0.1, max_eval_samples: int | None = None) -> np.ndarray:
    eval_model = unwrap_model(model_tgt)
    eval_model.eval()
    pred_list = []
    seen = 0

    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = eval_model(batch, task.entity_table, demo_info, inference=True, add_noise=add_noise,
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

        if max_eval_samples is not None and max_eval_samples > 0:
            remaining = max_eval_samples - seen
            if remaining <= 0:
                break
            pred = pred[:remaining]
        pred_list.append(pred.detach().cpu())
        seen += pred.numel() if pred.dim() == 1 else pred.size(0)
        if max_eval_samples is not None and max_eval_samples > 0 and seen >= max_eval_samples:
            break

    pred_np = torch.cat(pred_list, dim=0).numpy()
    if max_eval_samples is not None and max_eval_samples > 0 and is_main_process():
        print(f"[Eval] Using {len(pred_np)} samples for {task.entity_table} test/eval cap={max_eval_samples}")
    return pred_np


if __name__ == '__main__':
    # only classification tasks # todo: different tasks in the same dataset are different training sizes?
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="rel-hm")
    parser.add_argument("--dataset_source", type=str, default="rel-avito")  # rel-event
    parser.add_argument("--dataset_target", type=str, default="rel-avito")  # rel-stack # rel-amazon

    # parser.add_argument("--task", type=str, default="user-ignore")
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
    # huggingface-cli download --resume-download gpt2 --local-dir gpt2
    # huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir /ai/design/RelGraph/DeepSeek-R1-Distill-Qwen-32B
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
                        default="./cache/checkpoints/source_best_model_clicks_aligS066.pt")
    parser.add_argument("--best_model_path", type=str,
                        default="./cache/checkpoints/source_best_model_dnf.pt")

    # training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--align", action='store_true')
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--testing", action='store_true')
    parser.add_argument("--mean_std", action='store_true')
    parser.add_argument("--TTA", action='store_true')
    parser.add_argument("--align_epochs", type=int, default=3)
    parser.add_argument("--pretrain_epochs", type=int, default=100)
    parser.add_argument("--schema_epochs", type=int, default=1)
    parser.add_argument("--tta_epochs", type=int, default=100)
    parser.add_argument("--val_steps", type=int, default=500)  # 1000
    parser.add_argument("--max_eval_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)  # default 512 for GNN
    parser.add_argument("--tgt_batch_size", type=int, default=128)  # default 512 for GNN
    parser.add_argument("--test_batch_size", type=int, default=4)  # default 512 for GNN #32
    parser.add_argument("--val_size", type=int, default=None)  # default 512 for GNN
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.00001)  # default 0.005 for GNN
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--dist_backend", type=str, default="gloo" if os.name == "nt" else "nccl")
    args = parser.parse_args()
    args.needs_target_domain = (
        args.finetune
        or args.testing
        or args.TTA
        or args.align
        or args.mean_std
        or not args.pretrain
    )

    args.distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs.")
        if args.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={args.local_rank} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend)
        atexit.register(cleanup_distributed)
        original_batch_sizes = {
            "batch_size": args.batch_size,
            "tgt_batch_size": args.tgt_batch_size,
            "test_batch_size": args.test_batch_size,
            "val_size": args.val_size,
        }
        args.batch_size = per_rank_batch_size(args.batch_size)
        args.tgt_batch_size = per_rank_batch_size(args.tgt_batch_size)
        args.test_batch_size = per_rank_batch_size(args.test_batch_size)
        args.val_size = per_rank_batch_size(args.val_size)
    elif torch.cuda.is_available():
        if args.local_rank != 0:
            print(
                f"[WARN] Non-distributed run requested local_rank={args.local_rank}; "
                "using cuda:0."
            )
        args.local_rank = 0
        torch.cuda.set_device(0)

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    if args.distributed and is_main_process():
        print(
            f"[DDP] world_size={get_world_size()}, backend={args.dist_backend}, "
            f"visible_cuda={torch.cuda.device_count()}"
        )
        print(
            "[DDP] Per-rank batch sizes: "
            f"batch_size {original_batch_sizes['batch_size']} -> {args.batch_size}, "
            f"tgt_batch_size {original_batch_sizes['tgt_batch_size']} -> {args.tgt_batch_size}, "
            f"test_batch_size {original_batch_sizes['test_batch_size']} -> {args.test_batch_size}, "
            f"val_size {original_batch_sizes['val_size']} -> {args.val_size}"
        )

    #############################################
    # get data and task information
    #############################################


    source_dataset_name = args.dataset_source
    target_dataset_name = args.dataset_target


    src_dataset: Dataset = get_dataset(source_dataset_name, download=False)
    # dataset: Dataset = get_dataset(args.dataset, download=False)  # get dataset (database + temporal splitting times)

    # db = dataset.get_db()  # get database
    # print('Table names: ', list(db.table_dict.keys()))
    # print('Begin time: ', db.min_timestamp, 'End time: ', db.max_timestamp)
    # print('Val time: ', dataset.val_timestamp, 'Test time: ', dataset.test_timestamp)
    src_db = src_dataset.get_db()
    print(f"[Source] Tables: {list(src_db.table_dict.keys())}")
    print(f"Time range: {src_db.min_timestamp} -> {src_db.max_timestamp}")


    if args.needs_target_domain:
        tgt_dataset: Dataset = get_dataset(target_dataset_name, download=False)
        tgt_db = tgt_dataset.get_db()
    else:
        tgt_dataset = src_dataset
        tgt_db = src_db
    if args.needs_target_domain:
        print(f"[Target] Tables: {list(tgt_db.table_dict.keys())}")
    print(f"Time range: {tgt_db.min_timestamp} -> {tgt_db.max_timestamp}")

    # task = get_task(args.dataset, args.task, download=True)
    # task.name = args.task


    src_task = get_task(source_dataset_name, args.task_source, download=False)
    if args.needs_target_domain:
        tgt_task = get_task(target_dataset_name, args.task_target, download=False)
    else:
        tgt_task = src_task
    src_task.name = args.task_source
    if args.needs_target_domain:
        tgt_task.name = args.task_target

    # notebook: https://colab.research.google.com/github/snap-stanford/relbench/blob/main/tutorials/train_model.ipynb
    """stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
    try:  # configurate stype (modality) of each column, e.g., numerical/timestamp/categorical/text_embedded
        with open(stypes_cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
    except FileNotFoundError:
        col_to_stype_dict = get_stype_proposal(db)
        Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stypes_cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)"""


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
    # print(col_to_stype_src)
    col_to_stype_tgt = (
        prepare_col_to_stype(tgt_db, args.dataset_target)
        if args.needs_target_domain
        else None
    )
    # print(col_to_stype_tgt)

    # build heterogeneous and temporal graphs `data`
    # sentence_transformer: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
    os.makedirs(args.text_embedder_path, exist_ok=True)
    text_embedder = TextEmbedding(args.text_embedder, args.text_embedder_path, device=device)
    """data, col_stats_dict = make_pkey_fkey_graph(db, col_to_stype_dict=col_to_stype_dict, text_embedder_cfg=TextEmbedderConfig(text_embedder=text_embedder, batch_size=256),
                                                cache_dir=f"{args.cache_dir}/{args.dataset}/materialized")"""
    print('Building source graph')
    data_src, col_stats_dict_src = make_pkey_fkey_graph(
        src_db,
        col_to_stype_dict=col_to_stype_src,
        text_embedder_cfg=TextEmbedderConfig(text_embedder=text_embedder, batch_size=128),
        cache_dir=f"{args.cache_dir}/{source_dataset_name}/materialized"
    )
    # print(data_src)
    print('Building target graph')
    if args.needs_target_domain:
        data_tgt, col_stats_dict_tgt = make_pkey_fkey_graph(
            tgt_db,
            col_to_stype_dict=col_to_stype_tgt,
            text_embedder_cfg=TextEmbedderConfig(text_embedder=text_embedder, batch_size=128),
            cache_dir=f"{args.cache_dir}/{target_dataset_name}/materialized"
        )
    else:
        data_tgt = data_src
        col_stats_dict_tgt = col_stats_dict_src
        print("[INFO] Pure pretrain run: skipping target-domain graph materialization.")

    """print("Source columns stats:", col_stats_dict_src['users'])
    print("Target columns stats:", col_stats_dict_tgt['users'])
    print("Source stypes:", col_to_stype_src['users'])
    print("Target stypes:", col_to_stype_tgt['users'])


    print("[Source Graph] Keys:", data_src.keys())
    print("[Target Graph] Keys:", data_tgt.keys())
    print("Source columns:", list(col_stats_dict_src.keys())[:5])
    print("Target columns:", list(col_stats_dict_tgt.keys())[:5])"""

    # 'num_neighbors' -> the number of neighbors sampled per node (e.g., [64, 32, 16]), 'num_sampled_nodes' -> the total number of nodes sampled per layer (hop)
    out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max = task_info(src_task)
    loader_dict_src: Dict[str, NeighborLoader] = {}
    loader_dict_tgt: Dict[str, NeighborLoader] = {}
    eval_table_dict_src: Dict[str, Table] = {}
    eval_table_dict_tgt: Dict[str, Table] = {}


    for split in ["train", "val", "test"]:
        eval_table_dict_src[split] = make_capped_eval_table(
            src_task,
            split,
            args.max_eval_samples if split == "test" else None,
            seed=args.seed,
        )
        table = eval_table_dict_src[split]
        table_input = get_node_train_table_input(table=table, task=src_task)
        table_input = partition_table_input_for_rank(table_input, split)
        entity_table = table_input.nodes[0]
        if split == "train":
            bs = args.batch_size
        elif split == "val":
            bs = args.val_size if args.val_size is not None else args.batch_size
        else:
            bs = args.test_batch_size
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
        eval_table_dict_tgt[split] = make_capped_eval_table(
            tgt_task,
            split,
            args.max_eval_samples if split == "test" else None,
            seed=args.seed,
        )
        test_table = eval_table_dict_tgt[split]
        test_input = get_node_train_table_input(table=test_table, task=tgt_task)
        test_input = partition_table_input_for_rank(test_input, split)
        test_entity_table = test_input.nodes[0]

        bs = args.tgt_batch_size if split in ["train", "val"] else args.test_batch_size
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
    # batch = next(iter(loader_dict_tgt["test"]))
    # print(batch[tgt_task.entity_table])

    import copy

    import torch
    import random


    def _clamp_ratio(p: float) -> float:
        """Clamp to [0,1] to avoid invalid row masking."""
        if p is None:
            return 1.0
        return max(0.0, min(1.0, float(p)))


    def apply_covariate_shift_stable(
            data_shifted,
            tables=None,
            shift_ratio: float = 0.5,
            row_ratio: float = 1.0,
            id_col_names=None,
            scale_by_std: bool = True,
            eps: float = 1e-6,
            perturb_embedding: bool = False,
            emb_ratio: float = 0.02,
    ):
        row_ratio = _clamp_ratio(row_ratio)
        id_col_names = set(id_col_names or ['resultId', 'raceId', 'driverId', 'constructorId', 'statusId'])


        tables_to_shift = tables or [t for t in getattr(data_shifted, "node_types", [])]

        for table_name in tables_to_shift:
            if table_name not in data_shifted.node_types:
                continue
            node = data_shifted[table_name]
            if not hasattr(node, "tf"):

                continue

            tf = node.tf


            if stype.numerical in tf.feat_dict:
                feat = tf.feat_dict[stype.numerical]  # [N, D]
                col_names = list(tf.col_names_dict[stype.numerical])
                device = feat.device


                mask_id = torch.tensor([c in id_col_names for c in col_names],
                                       dtype=torch.bool, device=device)

                if scale_by_std:

                    col_std = feat.float().std(dim=0, unbiased=False)
                    col_std = torch.clamp(col_std, min=eps)
                    scale = shift_ratio * col_std
                else:
                    scale = torch.full((feat.size(1),), float(shift_ratio), device=device)

                noise = torch.randn_like(feat) * scale  
                noise[:, mask_id] = 0  

                if row_ratio < 1.0:
                    num_rows = feat.size(0)
                    row_mask = (torch.rand(num_rows, device=device) < row_ratio).unsqueeze(1)
                    noise = noise * row_mask  

                tf.feat_dict[stype.numerical] = feat + noise  


            """if perturb_embedding and (stype.embedding in tf.feat_dict):
                emb = tf.feat_dict[stype.embedding].values  # [N, D_emb]
                device = emb.device
                noise = torch.randn_like(emb) * emb_ratio
                if row_ratio < 1.0:
                    num_rows = emb.size(0)
                    row_mask = (torch.rand(num_rows, device=device) < row_ratio).unsqueeze(1)
                    noise = noise * row_mask
                tf.feat_dict[stype.embedding].values = emb + noise"""

        return data_shifted  # for chaining

    import torch
    from torch_frame import stype


    def _clamp_ratio1(x: float) -> float:
        return max(0.0, min(1.0, float(x)))


    def is_integer_column(col: torch.Tensor, tol: float = 1e-6) -> bool:
        if not torch.is_floating_point(col):
            return True
        return torch.all(torch.abs(col - col.round()) < tol)


    def apply_covariate_shift_semantic(
        data_shifted,
        tables=None,
        shift_ratio: float = 0.5,
        row_ratio: float = 1.0,
        id_col_names=None,
        scale_by_std: bool = True,
        eps: float = 1e-6,
        perturb_embedding: bool = False,
        emb_ratio: float = 0.02,
    ):

        row_ratio = _clamp_ratio1(row_ratio)
        id_col_names = set(
            id_col_names
            or ['resultId', 'raceId', 'driverId', 'constructorId', 'statusId']
        )

        tables_to_shift = tables or list(getattr(data_shifted, "node_types", []))

        for table_name in tables_to_shift:
            if table_name not in data_shifted.node_types:
                continue

            node = data_shifted[table_name]
            if not hasattr(node, "tf"):
                continue

            tf = node.tf

            # ============================================================
            # Numerical feature covariate shift
            # ============================================================
            if stype.numerical not in tf.feat_dict:
                continue

            feat = tf.feat_dict[stype.numerical]  # [N, D]
            col_names = list(tf.col_names_dict[stype.numerical])
            device = feat.device

            noise = torch.zeros_like(feat)


            if row_ratio < 1.0:
                row_mask = (
                    torch.rand(feat.size(0), device=device) < row_ratio
                ).unsqueeze(1)
            else:
                row_mask = None


            if scale_by_std:
                col_std = feat.float().std(dim=0, unbiased=False).clamp(min=eps)

            for j in range(feat.size(1)):
                col = feat[:, j]
                col_name = col_names[j]


                if col_name in id_col_names:
                    continue


                if is_integer_column(col):

                    max_shift = max(1, int(round(shift_ratio)))
                    int_noise = torch.randint(
                        low=-max_shift,
                        high=max_shift + 1,
                        size=col.shape,
                        device=device
                    )
                    noise[:, j] = int_noise.float()


                else:
                    if scale_by_std:
                        scale = shift_ratio * col_std[j]
                    else:
                        scale = shift_ratio
                    noise[:, j] = torch.randn_like(col) * scale


            if row_mask is not None:
                noise = noise * row_mask

            tf.feat_dict[stype.numerical] = feat + noise

            # ============================================================
            # Optional: embedding perturbation (rarely needed)
            # ============================================================
            if perturb_embedding and (stype.embedding in tf.feat_dict):
                emb = tf.feat_dict[stype.embedding].values
                emb_noise = torch.randn_like(emb) * emb_ratio

                if row_mask is not None:
                    emb_noise = emb_noise * row_mask

                tf.feat_dict[stype.embedding].values = emb + emb_noise

        return data_shifted


    import pandas as pd
    import copy
    import random
    from typing import List, Any



    # ======= BEGIN: resolve time conflict + robust test_20 =======
    from typing import Optional


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


   



    import pandas as pd


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

    def get_row_level_prob(epoch, num_epochs, start=0.3, end=0.5, mode="linear"):
        if mode == "linear":
            p = start + (end - start) * (epoch / num_epochs)
        elif mode == "sigmoid":
            progress = epoch / num_epochs
            factor = 1 / (1 + math.exp(-10 * (progress - 0.5)))
            p = start + (end - start) * factor
        else:
            p = start
        return min(max(p, start), end)



    import math

    def get_row_level_prob_decay(epoch, num_epochs, start=0.7, end=0.3, mode="cosine"):
        progress = epoch / num_epochs

        if mode == "linear":

            p = start - (start - end) * progress

        elif mode == "sigmoid":

            factor = 1 / (1 + math.exp(10 * (progress - 0.5)))
            p = end + (start - end) * factor

        elif mode == "cosine":

            factor = 0.5 * (1 + math.cos(math.pi * progress))
            p = end + (start - end) * factor

        else:
            p = start

        return min(max(p, min(start, end)), max(start, end))



    def get_block_level_prob(epoch, total_epochs, start=0.1, peak=0.3, end=0.0, mode="linear"):
        # 0 -> peak -> end
        mid_epoch = total_epochs // 2
        if epoch <= mid_epoch:

            return start + (peak - start) * (epoch / mid_epoch)
        else:

            return peak - (peak - end) * ((epoch - mid_epoch) / (total_epochs - mid_epoch))



    args.val_steps = min(args.val_steps, len(loader_dict_src['train']))
    print('Source Entity table: ', entity_table)
    print('Target Entity Table: ', table_input.nodes[0])
    print(f"[INFO] Cross-domain setup ready: {source_dataset_name} {target_dataset_name}")

    #############################################
    # model training
    #############################################
    model = Model(data_src, col_stats_dict_src, args.num_layers, channels=args.channels, out_channels=out_channels,
                  aggr=args.aggr, dropout=args.dropout, model_type=args.model_type,
                  llm_frozen=args.llm_frozen, output_mlp=args.output_mlp, max_new_tokens=args.max_new_tokens,
                  alpha=args.loss_class_weight, num_demo=args.num_demo,
                  dataset=args.dataset_source, task=src_task, db=src_db).to(device)

    needs_target_model = (
        args.finetune
        or args.testing
        or args.TTA
        or args.align
        or args.mean_std
        or not args.pretrain
    )
    model_tgt = None
    if needs_target_model:
        model_tgt = Model(data_tgt, col_stats_dict_tgt, args.num_layers, channels=args.channels, out_channels=out_channels,
                          aggr=args.aggr, dropout=args.dropout, model_type=args.model_type,
                          llm_frozen=args.llm_frozen, output_mlp=args.output_mlp, max_new_tokens=args.max_new_tokens,
                          alpha=args.loss_class_weight, num_demo=args.num_demo,
                          dataset=args.dataset_target, task=tgt_task, db=tgt_db).to(device)
    elif is_main_process():
        print("[INFO] Pure pretrain run: skipping target-domain model to save GPU memory.")

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=False)
        if hasattr(model, "_set_static_graph"):
            model._set_static_graph()
        if model_tgt is not None:
            model_tgt = DDP(model_tgt, device_ids=[args.local_rank], output_device=args.local_rank,
                            find_unused_parameters=False)
            if hasattr(model_tgt, "_set_static_graph"):
                model_tgt._set_static_graph()



    if args.finetune:
        print('Finetuning')
        opt_model = model_tgt
    elif args.pretrain:
        print('Pretrain model')
        opt_model = model
    else:
        print('Testing')
        opt_model = model_tgt

    if opt_model is None:
        raise RuntimeError("No model is available for the selected training stage.")

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
    trainable_params, all_param = unwrap_model(opt_model).print_trainable_params()
    print(
        f"trainable params: {trainable_params / 1e6:.2f}M || all params: {all_param / 1e6:.2f}M || trainable: {100 * trainable_params / all_param:.4f}%")

   
    # #############################################
    # # pretraining
    # #############################################
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.pretrain:
        state_dict = None  
        if state_dict is not None: load_clean_state_dict(model, state_dict)  # load pretrained weights

        metric_logger = MetricLogger(save_dir="./metrics")
        latest_val_auc = None
        latest_val_loss = None
        latest_test_auc = None


        if is_main_process() and not args.debug:  # rename the project if init failure
            run = wandb.init(project='rel-LLM-zero', name=f'{args.dataset_source}_{args.task_source}',
                             id=f"pretrain_run_{args.dataset_source}_{args.task_source}", resume="allow")
        pretrain_steps = 0  
        best_val_metric = -math.inf if higher_is_better else math.inf
        stage1_announced = False
        stage2_announced = False
        stage3_announced = False

        # epochs_no_improve = 0

        print('Pretraining')

        #throughput_meter = ThroughputMeter()
        for epoch in range(1, args.pretrain_epochs + 1):  
            loss_accum = row_loss_accum = cell_loss_accum = count_accum = align_loss_accum = 0
            row_ratios = []
            tq = tqdm(loader_dict_src["train"], total=len(loader_dict_src["train"]))
            row_level_prob = get_row_level_prob(epoch, 100, start=0.1, end=0.9, mode="linear")

            if epoch < args.schema_epochs and not stage1_announced:
                print(f"Stage 1: GNN-LLM {epoch} epoch")
                stage1_announced = True

            if epoch == args.schema_epochs and not stage2_announced:
                print(f"Stage 2: Row/Cell {epoch} epoch")

                stage2_announced = True
            #if epoch > args.schema_epochs and row_level_prob < 0.9:

            #if row_level_prob > 0.9 and not stage3_announced:

                #stage3_announced = True

            for batch in tq:
                try:  
                    model.train()
                    batch = batch.to(device)
                    nums_samples = batch[entity_table].y.size(0)  
                    optimizer.zero_grad()

                    """if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()"""

                    #peak_mem = gpu_peak_mem_mb()
                    #tq.set_postfix({"peakMB": f"{peak_mem:.1f}"})
                    

                    """if epoch < args.schema_epochs:
                        #loss = model.schema_pretrain(batch, src_task.entity_table)
                        loss = model.schema_pretrain_multi_table(batch, src_db, src_task.entity_table)
                        #loss = model.pretrain(batch, src_task.entity_table, row_level_prob, add_noise = True)
                    elif row_level_prob < 0.7:
                        loss = model.pretrain(batch, src_task.entity_table, row_level_prob, add_noise = False)
                    else:
                        loss, current_row_ratio = model.pretrain_with_gating( batch, src_task.entity_table, row_level_prob, epoch, total_epochs=args.pretrain_epochs,  add_noise = False)
                        row_ratios.append(current_row_ratio)"""

                    if epoch < args.schema_epochs:
                        
                        loss, row_loss, cell_loss = model(
                            batch,
                            src_task.entity_table,
                            pretrain_mode="row_context",
                            epoch=pretrain_steps,
                        )
                        #loss, row_loss, cell_loss = model.pretrain_row(batch, src_task.entity_table, add_noise=False, epoch=pretrain_steps, b = 0.4)
                        
                    else:
                        #new_epoch = epoch - args.schema_epochs+1
                        #loss, row_loss, cell_loss = model.pretrain(batch, src_task.entity_table, row_level_prob, add_noise=False, epoch=epoch)
                        #loss, row_loss, cell_loss = model.pretrain_row(batch, src_task.entity_table, add_noise=False, epoch=epoch, b = 0.8)
                        #loss_align, row_loss, cell_loss = model.pretrain_row_context(batch, src_task.entity_table, add_noise=False)
                        #loss_align, row_loss, cell_loss = model.pretrain_row_context1(batch, src_task.entity_table, add_noise=False)
                        #loss_align, row_loss, cell_loss = model.pretrain_row_contextv6(batch, src_task.entity_table)

                        loss, row_loss, cell_loss = model(
                            batch,
                            src_task.entity_table,
                            pretrain_mode="cell",
                            add_noise=False,
                            epoch=pretrain_steps,
                            b=1.0,
                        )
                        #loss, row_loss, cell_loss = model.pretrain_cell_lm_contextual(batch, src_task.entity_table, epoch=epoch, b = 1.0)

                        #loss, row_loss, cell_loss = model.pretrain_row_context1(batch, src_task.entity_table, add_noise=False)
                        #loss_align, row_loss, cell_loss = model.pretrain_row_context(batch, src_task.entity_table, add_noise=False)
                        #loss, row_loss, cell_loss = model.pretrain_cell_rowctx(batch, src_task.entity_table, add_noise=False, epoch=epoch, b = 1.0)
                
                        #loss =  loss1 + 0.2*loss_align

                    can_backward = torch.tensor(
                        [1 if torch.is_tensor(loss) and loss.requires_grad else 0],
                        device=device,
                    )
                    if args.distributed:
                        dist.all_reduce(can_backward, op=dist.ReduceOp.MIN)
                    if can_backward.item() == 0:
                        continue
                    loss.backward()
                    optimizer.step()
                    """if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    elapsed = time.time() - start_time"""

                except RuntimeError as exc:
                    if not is_cuda_oom(exc):
                        raise
                    if args.distributed:
                        raise
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

                #align_loss_accum +=  loss_align.detach().item() * nums_samples
                #loss_align = align_loss_accum / count_accum 

                metric_logger.update_train_step(train_loss, row_loss, cell_loss)


                summary = {'loss': train_loss, 'lr': optimizer.param_groups[-1]['lr']}
                if is_main_process() and not args.debug:  
                    for k, v in summary.items():
                        run.log({f'Pretrain/{k}': v}, step=pretrain_steps)  # Steps must be monotonically increasing
                #tq.set_description(f'[Pretrain] Epoch/Step: {epoch:02d}/{pretrain_steps} | Train loss: {train_loss:.4f} | Row loss: {row_loss:.4f} |  Cell loss: {cell_loss:.4f} |  Align loss: {loss_align:.4f}')
                """token_count = getattr(model, "last_llm_token_count", 0)

                throughput_meter.update(token_count, elapsed)"""
                
                tq.set_description(f'[Pretrain] Epoch/Step: {epoch:02d}/{pretrain_steps} | Train loss: {train_loss:.4f} | Row loss: {row_loss:.4f} |  Cell loss: {cell_loss:.4f}  ')


                if pretrain_steps % args.val_steps == 0:
                    val_pred = test(loader_dict_src["val"], src_task, max_eval_samples=None)
                    val_metrics = evaluate_limited(src_task, val_pred, "val", None, eval_table_dict_src["val"])
                    latest_val_auc = val_metrics[tune_metric]
                    latest_val_loss = val_metrics.get("loss", None)

                    if is_main_process() and not args.debug:
                        for k, v in val_metrics.items():
                            run.log({f'src_val/{k}': v}, step=pretrain_steps)


                    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or (
                            not higher_is_better and val_metrics[tune_metric] <= best_val_metric):

                        best_val_metric = val_metrics[tune_metric]


                        test_pred = test(loader_dict_src["test"], src_task, max_eval_samples=args.max_eval_samples)
                        test_metrics = evaluate_limited(
                            src_task,
                            test_pred,
                            "test",
                            args.max_eval_samples,
                            eval_table_dict_src["test"],
                        )


                        latest_test_auc = test_metrics[tune_metric]

                        metric_logger.update_val_test_step(
                            latest_val_auc,
                            latest_test_auc
                        )

                        if is_main_process():
                            save_checkpoint(get_clean_state_dict(model), args.best_model_path)

                        if is_main_process() and not args.debug:
                            for k, v in test_metrics.items():
                                run.log({f'test/{k}': v}, step=pretrain_steps)

                        scheduler.step(val_metrics[tune_metric])
                        print(f'[Eval] Epoch/Step: {epoch:02d}/{pretrain_steps} | Val: {val_metrics} | Best val/test: {best_val_metric:.4f}/{test_metrics[tune_metric]:.4f}')

            ###############################################################

            ###############################################################
            """metric_logger.update_epoch(
                train_loss,
                latest_val_loss if latest_val_loss is not None else 0,
                latest_val_auc if latest_val_auc is not None else 0.5,
                latest_test_auc if latest_test_auc is not None else 0.5
            )"""

        ###############################################################

        ###############################################################

                        
    #############################################
    # Fine-tuning on Target Domain
    #############################################
    if args.finetune:  
        steps = 0  
        if is_main_process() and not args.debug:
            if args.pretrain:
                run.finish()  
            run = wandb.init(
                project='rel-LLM',
                name=f'{args.dataset_target}_to_{args.dataset_target}',
                id=f'finetune_run_{args.dataset_target}_to_{args.dataset_target}',
                resume="allow"
            )
        print('Finetuning')

        if os.path.exists(args.best_model_path):
            print(f"Loading checkpoint: {args.best_model_path}")
            pretrained_state = normalize_state_dict_keys(torch.load(args.best_model_path, map_location="cpu"))
            model_tgt_base = unwrap_model(model_tgt)
            model_tgt_dict = model_tgt_base.state_dict()


            filtered_state_dict = {
                k: v for k, v in pretrained_state.items()
                if k in model_tgt_dict and not k.startswith("encoder.encoders.")
            }


            model_tgt_dict.update(filtered_state_dict)
            model_tgt_base.load_state_dict(model_tgt_dict)
            # model.load_state_dict(state_dict)
            print('Loaded pretrained model from source domain.')


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
                    # batch = next(iter(loader_dict_tgt["train"]))
                    # print(batch[tgt_task.entity_table].y[:10])
                    # print(batch[tgt_task.entity_table].tf.feat_dict.keys())

                loss.backward()
                optimizer.step()

                steps += 1
                loss_accum += loss.detach().item() * num_samples
                count_accum += num_samples
                train_loss = loss_accum / count_accum

                # wandb logging
                if is_main_process() and not args.debug:
                    run.log({'train/loss': train_loss, 'train/lr': optimizer.param_groups[-1]['lr']}, step=steps)

                tq.set_description(f'[Finetune] Epoch/Step: {epoch:02d}/{steps} | Train loss: {train_loss:.4f}')


                if steps % args.val_steps == 0:

                    val_pred = test_tgt(loader_dict_tgt["val"], tgt_task)  
                    val_metrics = tgt_task.evaluate(val_pred, tgt_task.get_table("val"))



                    # test_pred = test_tgt(loader_dict_tgt["test"], tgt_task)
                    # test_metrics = tgt_task.evaluate(test_pred)

                    if is_main_process() and not args.debug:
                        for k, v in val_metrics.items():
                            run.log({f'src_val/{k}': v}, step=steps)
                        # for k, v in test_metrics.items():
                        # run.log({f'tgt_test/{k}': v}, step=steps)


                    if (higher_is_better and val_metrics[tune_metric] >= best_val_metric) or \
                            (not higher_is_better and val_metrics[tune_metric] <= best_val_metric):
                        best_val_metric = val_metrics[tune_metric]
                        state_dict = copy.deepcopy(get_clean_state_dict(model_tgt))
                        # torch.save(model.state_dict(), args.best_model_path)

                    scheduler.step(val_metrics[tune_metric])

                    print(
                        f'[Eval] Epoch/Step: {epoch:02d}/{steps} | Tgt Val: {val_metrics} | Best tgt val: {best_val_metric:.4f}')

        if os.path.exists(args.best_model_path):
            print(f"Loading checkpoint: {args.best_model_path}")
            state_dict_src = torch.load(args.best_model_path, map_location="cpu")  
        else:
            raise FileNotFoundError(f"File not found: {args.best_model_path}")

        load_clean_state_dict(model, state_dict_src)
        unwrap_model(model).eval()

        load_clean_state_dict(model_tgt, state_dict, strict=False)
        unwrap_model(model_tgt).eval()



        with torch.no_grad():
            print('Testing')

            val_pred = test(loader_dict_src["val"], src_task, max_eval_samples=None)
            val_metrics = evaluate_limited(src_task, val_pred, "val", None, eval_table_dict_src["val"])



            test_pred_src = test(loader_dict_src["test"], src_task, max_eval_samples=args.max_eval_samples)
            test_metrics_src = evaluate_limited(
                src_task,
                test_pred_src,
                "test",
                args.max_eval_samples,
                eval_table_dict_src["test"],
            )



            test_pred = test_tgt(loader_dict_tgt["test"], tgt_task, max_eval_samples=args.max_eval_samples)
            test_metrics = evaluate_limited(
                tgt_task,
                test_pred,
                "test",
                args.max_eval_samples,
                eval_table_dict_tgt["test"],
            )
            print(f"[Source Val] metrics: {val_metrics}")
            print(f"[Source Test] metrics: {test_metrics_src}")
            print(f"[Target Test] metrics: {test_metrics}")

    #############################################
    # Test Time Adaption
    #############################################

    if args.TTA:
        if os.path.exists(args.best_model_path):
            print(f"Loading checkpoint: {args.best_model_path}")
            state_dict = torch.load(args.best_model_path, map_location="cpu")  
        else:
            raise FileNotFoundError(f"File not found: {args.best_model_path}")

        # model.load_state_dict(state_dict)
        # model.eval()

        load_clean_state_dict(model_tgt, state_dict, strict=False)
        unwrap_model(model_tgt).eval()


        # model_tgt.load_state_dict(state_dict)

        with torch.no_grad():


            print('Running OOD evaluation')
            test_pred = test_tgt(loader_dict_tgt["test"], tgt_task, max_eval_samples=args.max_eval_samples)
            test_metrics_0 = evaluate_limited(
                tgt_task,
                test_pred,
                "test",
                args.max_eval_samples,
                eval_table_dict_tgt["test"],
            )
            best_metric1 = test_metrics_0["roc_auc"]
            print(f"[Target Test] metrics: {best_metric1}")

        """for name, param in model_tgt.named_parameters():
            print(name)"""

        best_metric = test_metrics_0["roc_auc"]
        best_epoch = 0
        patience = 10  
        no_improve = 0

        # params_tta = [p for n, p in model_tgt.named_parameters() if "projector" in n and p.requires_grad]
        model_tgt_base = unwrap_model(model_tgt)
        for n, p in model_tgt_base.named_parameters():
            if n.startswith("model.base_model.model.model.norm"):
                p.requires_grad = True

        params_tta = [
            p for n, p in model_tgt_base.named_parameters()
            if n.startswith("model.base_model.model.model.norm") and p.requires_grad
        ]

        # params_tta = [p for n, p in model_tgt.named_parameters() if ("projector" in n or "norm" in n) and p.requires_grad]

        # print("Number of params passed to optimizer:", len(params_tta))

        print("Number of params passed to optimizer:", sum(p.numel() for p in params_tta))

        optimizer_tgt = torch.optim.AdamW(params_tta, lr=0.001, betas=(0.9, 0.95))


        """for k in model_dict.keys():
            print(k)"""




        for epoch in range(1, args.tta_epochs + 1):  
            loss_accum = count_accum = 0
            tq = tqdm(loader_dict_tgt["test_20"], total=len(loader_dict_tgt["test_20"]))
            model_tgt.train()
            for batch in tq:
                batch = batch.to(device)
                optimizer_tgt.zero_grad()




                """logits = model_tgt(batch, tgt_task.entity_table, tta_mode=True)
                probs = torch.softmax(logits, dim=-1)
                loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))"""
                logits = model_tgt(batch, tgt_task.entity_table, tta_mode=True)  # (B,)

                p = torch.sigmoid(logits)   

                entropy = - (
                    p * torch.log(p + 1e-8) +
                    (1 - p) * torch.log(1 - p + 1e-8)
                )

                loss = entropy.mean()

                # loss_cl = model_tgt.cl(batch, tgt_task.entity_table)
                loss_cl = model_tgt(batch, tgt_task.entity_table, pretrain_mode="llama_cl")
                loss = loss + 0.5 * loss_cl
                # print("Entropy loss:", loss.item())


                loss.backward()
                optimizer_tgt.step()
                # print(batch[tgt_task.entity_table].keys())

                # num_samples = batch[tgt_task.entity_table].x.size(0)

                num_samples = batch[tgt_task.entity_table]['tf'].num_rows  

                # num_samples = batch[tgt_task.entity_table]['df'].shape[0]

                loss_accum += loss.detach().item() * num_samples
                count_accum += num_samples
                train_loss = loss_accum / count_accum

                tq.set_description(f'[TTA] Epoch: {epoch:02d} | Entropy loss: {train_loss:.4f}')


            unwrap_model(model_tgt).eval()
            with torch.no_grad():
                print('Testing')
                test_pred = test_tgt(loader_dict_tgt["test"], tgt_task, max_eval_samples=args.max_eval_samples)
                test_metrics = evaluate_limited(
                    tgt_task,
                    test_pred,
                    "test",
                    args.max_eval_samples,
                    eval_table_dict_tgt["test"],
                )
                print(f"[Target Test after TTA] metrics: {test_metrics}")

            current_metric = test_metrics["roc_auc"]


            if current_metric > best_metric:
                print(f"Metric improved from {best_metric:.4f} {current_metric:.4f}")
                best_metric = current_metric
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in get_clean_state_dict(model_tgt).items()}
            else:
                no_improve += 1
                print(f"No improvement for {no_improve} epoch(s). Best: {best_metric:.4f} @ epoch {best_epoch}")
                if no_improve >= patience:
                    print('Early stopping triggered!')
                    break


        if 'best_state' in locals():
            load_clean_state_dict(model_tgt, best_state)
            print(f"Restored best TTA model from epoch {best_epoch} (metric={best_metric:.4f})")


        """model_tgt.eval()
        with torch.no_grad():
            test_pred = test_tgt(loader_dict_tgt["test"], tgt_task, add_noise=True, sigma_ratio=10.9, max_eval_samples=args.max_eval_samples)
            final_metrics = evaluate_limited(
                tgt_task,
                test_pred,
                "test",
                args.max_eval_samples,
                eval_table_dict_tgt["test"],
            )
            print(f"[Final Target Test after TTA] metrics: {final_metrics}")"""





    if args.testing:
        if os.path.exists(args.best_model_path):
            print(f"Loading checkpoint: {args.best_model_path}")
            state_dict = torch.load(args.best_model_path, map_location="cpu")  
        else:
            raise FileNotFoundError(f"File not found: {args.best_model_path}")

        load_clean_state_dict(model, state_dict)
        unwrap_model(model).eval()

        #model_tgt.load_state_dict(state_dict)
        #model_tgt.eval()

        

        with torch.no_grad():
            print('Testing')

            # val_pred = test(loader_dict_src["val"], src_task)
            # val_metrics = src_task.evaluate(val_pred, src_task.get_table("val"))


            print('IID...')
            
            test_pred = test(loader_dict_src["test"], src_task, max_eval_samples=args.max_eval_samples)
            test_metrics = evaluate_limited(
                src_task,
                test_pred,
                "test",
                args.max_eval_samples,
                eval_table_dict_src["test"],
            )
            #test_metrics_src1 = test_metrics["roc_auc"]
            print(f"[Source Test] metrics: {test_metrics}")



            

            """test_pred1 = test_tgt(loader_dict_tgt["test"], tgt_task)
            test_metrics1 = tgt_task.evaluate(test_pred1)
            best_metric1 = test_metrics1["roc_auc"]
            print(f"[Target Drift Test] metrics: {best_metric1}")"""

            # print(f"[Source Val] metrics: {val_metrics}")
            # print(f"[Source Test] metrics: {test_metrics_src}")


            'Running OOD evaluation'
