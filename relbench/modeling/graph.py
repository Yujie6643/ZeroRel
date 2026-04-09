import os
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_frame import stype
from torch_frame.config import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import sort_edge_index

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.modeling.utils import remove_pkey_fkey, to_unix_time


def make_pkey_fkey_graph(db: Database, col_to_stype_dict: Dict[str, Dict[str, stype]], text_embedder_cfg: Optional[TextEmbedderConfig] = None, cache_dir: Optional[str] = None, ) -> \
Tuple[HeteroData, Dict[str, Dict[str, Dict[StatType, Any]]]]:
    r"""Given a :class:`Database` object, construct a heterogeneous graph with primary-
    foreign key relationships, together with the column stats of each table.

    Args:
        db: A database object containing a set of tables.
        col_to_stype_dict: Column to stype for
            each table.
        text_embedder_cfg: Text embedder config.
        cache_dir: A directory for storing materialized tensor
            frames. If specified, we will either cache the file or use the
            cached file. If not specified, we will not use cached file and
            re-process everything from scratch without saving the cache.

    Returns:
        HeteroData: The heterogeneous :class:`PyG` object with
            :class:`TensorFrame` feature.
    """
    data = HeteroData()
    col_stats_dict = dict()
    if cache_dir is not None: os.makedirs(cache_dir, exist_ok=True)

    # 遍历数据库表
    # db.table_dict = {
    #     "users": Table(...),
    #     "events": Table(...),
    #     "event_attendees": Table(...),
    #     ...
    # }
    # key：表名（字符串）
    # value：Table 对象（封装了表的 DataFrame、主键、外键等信息）
    # table_name：字符串类型，表示表的名称，例如 "users"、"events"
    # table：封装了这张表的所有信息
    # table.df → Pandas DataFrame（实际数据）
    # table.pkey_col → 主键列名
    # table.fkey_col_to_pkey_table → 外键映射 {外键列名: 主键表名}
    # table.time_col → 可选时间列
    for table_name, table in db.table_dict.items():
        # Materialize the tables into tensor frames:
        df = table.df
        # Ensure that pkey is consecutive.
        if table.pkey_col is not None: # 检查主键是否连续（0..N-1），保证行索引和主键一致，便于图构建。
            assert (df[table.pkey_col].values == np.arange(len(df))).all()

        # 记录每列的模态类型（数值 / 类别 / 文本等）
        col_to_stype = col_to_stype_dict[table_name]  # table_name：字符串类型，表示表的名称，例如 "users"、"events"

        # 把主键列（pkey）和外键列（fkey）从输入特征里去掉。主外键ID不作为模型输入
        remove_pkey_fkey(col_to_stype, table)

        # 特殊情况：表格在去掉主键和外键列后没有剩余输入特征
        # 即便表本身没有输入特征，也能让每行成为图的节点，并通过外键构建图边
        if len(col_to_stype) == 0:  # Add constant feature in case df is empty:
            col_to_stype = {"__const__": stype.numerical} # 设置所有行的值都是 1 保证数据不空

            # 保留外键列用于构建边
            fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
            df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})  # 常数列 __const__ → 值全为 1

        path = (None if cache_dir is None else os.path.join(cache_dir, f"{table_name}.pt"))

        # 将 Pandas DataFrame 转成 TensorFrame（图神经网络可用的 tensor 表示）
        # df：当前表数据（去掉 pkey/fkey 或加上常数列）
        # col_to_stype：每列的模态类型（数值/类别/文本）
        # col_to_text_embedder_cfg：文本列嵌入配置，如果存在则生成 embedding
        # path：缓存路径，如果指定则保存 materialized tensor
        # .materialize() 会做：
        # 数值列 → float tensor
        # 类别列 → int tensor 或 embedding
        # 文本列 → embedding（batch 处理）
        # 保存 tensor 文件到 path（如果不为 None）
        # 把每张表的数据转成图神经网络可用的节点特征，同时保留原始信息和统计信息，为后续模型训练提供完整输入
        dataset = Dataset(df=df, col_to_stype=col_to_stype, col_to_text_embedder_cfg=text_embedder_cfg, ).materialize(path=path)

        # TensorFrame(
        #   num_cols=3,
        #   num_rows=8430002,
        #   numerical (1): ['Unnamed: 0'],
        #   categorical (1): ['status'],
        #   timestamp (1): ['start_time'],
        #   has_target=False,
        #   device='cpu',
        # )
        # 后处理数据  存储张量
        # ['__class__', '__copy__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__len__', '__lt__', '__module__', '__ne__', '__neq__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_apply', '_col_to_stype_idx', 'col_names_dict', 'cpu', 'cuda', 'device', 'feat_dict', 'get_col_feat', 'is_empty', 'num_cols', 'num_rows', 'stypes', 'to', 'validate', 'y']
        data[table_name].df = df  # Important! Easy to formulate contexts
        data[table_name].tf = dataset.tensor_frame
        """tf = data[table_name].tf
        # 假设列名是 'status'
        cols = ['Unnamed: 0', 'status', 'start_time']
        for col in cols:
            print(f"{col}: {tf.get_col_feat(col)[:5]}")"""

        # Unnamed: 0    event status  user_id              start_time
        # 7  2346692    yes     <NA> 2012-11-23 03:00:00.003
        # 8  2346692    yes     <NA> 2012-11-23 03:00:00.003
        # 9  2346692    yes     <NA> 2012-11-23 03:00:00.003
        # 10  2346692    yes     <NA> 2012-11-23 03:00:00.003
        # 11  2346692    yes     <NA> 2012-11-23 03:00:00.003
        # ...              ...      ...    ...      ...                     ...
        # 11245004  2329830     no     <NA> 2012-11-22 05:00:00.003
        # 11245005  2329830     no     <NA> 2012-11-22 05:00:00.003
        # 原始数据

        #print(data[table_name].tf)
        #print(data[table_name].df)
        # Unnamed: 0: tensor([[ 7.],
        #         [ 8.],
        #         [ 9.],
        #         [10.],
        #         [11.]])
        # status: tensor([[1],
        #         [1],
        #         [1],
        #         [1],
        #         [1]])
        # start_time: tensor([[[2012,   10,   22,    4,    3,    0,    0]],
        #
        #         [[2012,   10,   22,    4,    3,    0,    0]],
        #
        #         [[2012,   10,   22,    4,    3,    0,    0]],
        #
        #         [[2012,   10,   22,    4,    3,    0,    0]],
        col_stats_dict[table_name] = dataset.col_stats

        # 遍历 HeteroData 中每个节点类型（表）
        # 表: event_attendees
        #   节点数量: 8430002
        #   每列特征维度: [1, 1, 1]
        #   总特征维度: 3
        #
        # 表: event_attendees
        #   节点数量: 8430002
        #   每列特征维度: [1, 1, 1]
        #   总特征维度: 3
        #
        # 表: users
        #   节点数量: 37143
        #   每列特征维度: [2, 1, 1, 2]
        #   总特征维度: 6
        #
        # 表: event_attendees
        #   节点数量: 8430002
        #   每列特征维度: [1, 1, 1]
        #   总特征维度: 3
        #
        # 表: users
        #   节点数量: 37143
        #   每列特征维度: [2, 1, 1, 2]
        #   总特征维度: 6
        #
        # 表: user_friends
        #   节点数量: 30386403
        #   每列特征维度: [1]
        #   总特征维度: 1
        """for table_name in data.node_types:
            tf = data[table_name].tf  # TensorFrame 对象
            num_nodes = tf.num_rows

            # 计算特征维度
            feature_dims = [v.shape[1] if v.ndim > 1 else 1 for v in tf.feat_dict.values()]
            total_feature_dim = sum(feature_dims)

            print(f"表: {table_name}")
            print(f"  节点数量: {num_nodes}")
            print(f"  每列特征维度: {feature_dims}")
            print(f"  总特征维度: {total_feature_dim}\n")"""

        # Add time attribute:
        if table.time_col is not None:
            # tensor([1353639600, 1353639600, 1353639600,  ..., 1353560400, 1353560400,
            #         1351890000])
            data[table_name].time = torch.from_numpy(to_unix_time(table.df[table.time_col]))
            #print(data[table_name].time)

        # Add edges:
        for fkey_name, pkey_table_name in table.fkey_col_to_pkey_table.items():
            pkey_index = df[fkey_name]
            # Filter out dangling foreign keys
            mask = ~pkey_index.isna()
            fkey_index = torch.arange(len(pkey_index))
            # Filter dangling foreign keys:
            pkey_index = torch.from_numpy(pkey_index[mask].astype(int).values)
            fkey_index = fkey_index[torch.from_numpy(mask.values)]
            # Ensure no dangling fkeys
            assert (pkey_index < len(db.table_dict[pkey_table_name])).all()

            # fkey -> pkey edges
            edge_index = torch.stack([fkey_index, pkey_index], dim=0)
            edge_type = (table_name, f"f2p_{fkey_name}", pkey_table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

            # pkey -> fkey edges.
            # "rev_" is added so that PyG loader recognizes the reverse edges
            edge_index = torch.stack([pkey_index, fkey_index], dim=0)
            edge_type = (pkey_table_name, f"rev_f2p_{fkey_name}", table_name)
            data[edge_type].edge_index = sort_edge_index(edge_index)

    data.validate()
    return data, col_stats_dict


class AttachTargetTransform:
    r"""Attach the target label to the heterogeneous mini-batch.

    The batch consists of disjoins subgraphs loaded via temporal sampling. The same
    input node can occur multiple times with different timestamps, and thus different
    subgraphs and labels. Hence labels cannot be stored in the graph object directly,
    and must be attached to the batch after the batch is created.
    """

    def __init__(self, entity: str, target: Tensor):
        self.entity = entity
        self.target = target

    def __call__(self, batch: HeteroData) -> HeteroData:
        batch[self.entity].y = self.target[batch[self.entity].input_id]
        return batch


class NodeTrainTableInput(NamedTuple):
    r"""Training table input for node prediction.

    - nodes is a Tensor of node indices.
    - time is a Tensor of node timestamps.
    - target is a Tensor of node labels.
    - transform attaches the target to the batch.
    """

    nodes: Tuple[NodeType, Tensor]
    time: Optional[Tensor]
    target: Optional[Tensor]
    transform: Optional[AttachTargetTransform]


def get_node_train_table_input(table: Table, task: EntityTask, ) -> NodeTrainTableInput:
    r"""Get the training table input for node prediction."""

    nodes = torch.from_numpy(table.df[task.entity_col].astype(int).values)

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    target: Optional[Tensor] = None
    transform: Optional[AttachTargetTransform] = None
    if task.target_col in table.df:
        target_type = float
        if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            target_type = int
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            target = torch.from_numpy(np.stack(table.df[task.target_col].values))
        else:
            target = torch.from_numpy(table.df[task.target_col].values.astype(target_type))
        transform = AttachTargetTransform(task.entity_table, target)

    return NodeTrainTableInput(nodes=(task.entity_table, nodes), time=time, target=target, transform=transform, )


class LinkTrainTableInput(NamedTuple):
    r"""Training table input for link prediction.

    - src_nodes is a Tensor of source node indices.
    - dst_nodes is PyTorch sparse tensor in csr format.
        dst_nodes[src_node_idx] gives a tensor of destination node
        indices for src_node_idx.
    - num_dst_nodes is the total number of destination nodes.
        (used to perform negative sampling).
    - src_time is a Tensor of time for src_nodes
    """

    src_nodes: Tuple[NodeType, Tensor]
    dst_nodes: Tuple[NodeType, Tensor]
    num_dst_nodes: int
    src_time: Optional[Tensor]


def get_link_train_table_input(table: Table, task: RecommendationTask, ) -> LinkTrainTableInput:
    r"""Get the training table input for link prediction."""

    src_node_idx: Tensor = torch.from_numpy(table.df[task.src_entity_col].astype(int).values)
    exploded = table.df[task.dst_entity_col].explode()
    coo_indices = torch.from_numpy(np.stack([exploded.index.values, exploded.values.astype(int)]))
    sparse_coo = torch.sparse_coo_tensor(coo_indices, torch.ones(coo_indices.size(1), dtype=bool), (len(src_node_idx), task.num_dst_nodes), )
    dst_node_indices = sparse_coo.to_sparse_csr()

    time: Optional[Tensor] = None
    if table.time_col is not None:
        time = torch.from_numpy(to_unix_time(table.df[table.time_col]))

    return LinkTrainTableInput(src_nodes=(task.src_entity_table, src_node_idx), dst_nodes=(task.dst_entity_table, dst_node_indices), num_dst_nodes=task.num_dst_nodes,
        src_time=time, )
