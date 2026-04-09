from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import ModuleDict
import os
import torch_frame
from torch_frame import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import FeatureEncoder
from torch_frame.nn.encoder.stype_encoder import StypeEncoder


class StypeWiseFeatureEncoder(FeatureEncoder):
    r"""Feature encoder that transforms each stype tensor into embeddings and
    performs the final concatenation.

    Args:
        out_channels (int): Output dimensionality.
        col_stats
            (dict[str, dict[:class:`torch_frame.data.stats.StatType`, Any]]):
            A dictionary that maps column name into stats. Available as
            :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            A dictionary that maps :class:`torch_frame.stype` into
            :class:`torch_frame.nn.encoder.StypeEncoder` class. Only
            parent :class:`stypes <torch_frame.stype>` are supported
            as keys.
    """
    def __init__(
        self,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder],
    ) -> None:
        super().__init__()

        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.encoder_dict = ModuleDict()
        for stype, stype_encoder in stype_encoder_dict.items():
            if stype != stype.parent:
                if stype.parent in stype_encoder_dict:
                    msg = (
                        f"You can delete this {stype} directly since encoder "
                        f"for parent stype {stype.parent} is already declared."
                    )
                else:
                    msg = (f"To resolve the issue, you can change the key from"
                           f" {stype} to {stype.parent}.")
                raise ValueError(f"{stype} is an invalid stype to use in the "
                                 f"stype_encoder_dcit. {msg}")
            if stype not in stype_encoder.supported_stypes:
                raise ValueError(
                    f"{stype_encoder} does not support encoding {stype}.")

            if stype in col_names_dict:
                stats_list = [
                    self.col_stats[col_name]
                    for col_name in self.col_names_dict[stype]
                ]
                # Set lazy attributes
                stype_encoder.stype = stype
                stype_encoder.out_channels = out_channels
                stype_encoder.stats_list = stats_list
                self.encoder_dict[stype.value] = stype_encoder

    """def forward(self, tf: TensorFrame,  add_noise=False, sigma_ratio=0.1, phase="test") -> tuple[Tensor, list[str]]:
        all_col_names = []
        xs = []
        for stype in tf.stypes:
            feat = tf.feat_dict[stype]
            col_names = self.col_names_dict[stype]
            #x = self.encoder_dict[stype.value](feat, col_names)
            # ✅ 如果是 numerical 类型，则传入噪声参数
            # 对数值型特征加噪
            # ✅ 对数值型特征可选加噪（稳定版本）
            if stype == stype.numerical and add_noise:

                # ✅ 忽略 NaN，手动计算 mean/std
                mask = ~torch.isnan(feat)
                valid_feat = torch.where(mask, feat, torch.zeros_like(feat))
                count = mask.sum(dim=0, keepdim=True).clamp_min(1)
                mean_col = valid_feat.sum(dim=0, keepdim=True) / count
                var_col = ((torch.where(mask, feat, mean_col) - mean_col) ** 2).sum(dim=0, keepdim=True) / count
                std_col = torch.sqrt(var_col + 1e-6)

                # ✅ 对整列全 NaN 或异常值修正
                std_col[torch.isnan(std_col)] = 1.0
                mean_col[torch.isnan(mean_col)] = 0.0

                # ✅ 标准化 + 加噪 + 反标准化
                feat_norm = (feat - mean_col) / std_col
                noise = torch.randn_like(feat_norm) * sigma_ratio
                feat_noised = feat_norm + noise
                feat = feat_noised * std_col + mean_col
                #print(f"[Noise] sigma_ratio={sigma_ratio:.2f} | feat.std={torch.mean(std_col).item():.4f} | noise.std={noise.std().item():.4f}")

            x = self.encoder_dict[stype.value](feat, col_names)  # 其他类型照旧
            xs.append(x)
            all_col_names.extend(col_names)
        x = torch.cat(xs, dim=1)
        return x, all_col_names"""



    def forward(self, tf: TensorFrame, add_noise=False, sigma_ratio=0.3, phase="test", stats_file="/data/tyj/Rel-LLM-master/cache/noiseshift/train_stats_dnf.pt", is_last_batch=False,  enable_num_stats=False) -> \
    tuple[torch.Tensor, list[str]]:
        #print(f"[DEBUG] add_noise = {add_noise}, sigma_ratio = {sigma_ratio}")

        all_col_names = []
        xs = []
        mask_ratio = sigma_ratio
        for stype in tf.stypes:
            #print("stype",stype)
            #print("stype.value",stype.value)
            feat = tf.feat_dict[stype]
            col_names = self.col_names_dict[stype]
            #print("00")
            if enable_num_stats:
                #print("1")
                if stype == stype.numerical:

                    #print("2")
                    mask = ~torch.isnan(feat)
                    valid_feat = torch.where(mask, feat, torch.zeros_like(feat))
                    count = mask.sum(dim=0, keepdim=True).clamp_min(1)

                    
                    if add_noise:
                        #print("11")
                        feat_norm = (feat - mean_col) / std_col

                        noise = torch.randn_like(feat_norm) * sigma_ratio
                        #print(mean_col)
                        feat = feat_norm + noise * std_col + mean_col

                        #print("Before encoding:", feat.mean().item(), feat.std().item())
                        # 随机掩蔽 + 随机采样
                        """for j in range(feat.size(1)):
                            mask = torch.rand(feat.size(0), device=feat.device) < mask_ratio
                            if mask.sum() > 0:
                                candidate_idx = (~mask).nonzero(as_tuple=True)[0]
                                if len(candidate_idx) > 0:
                                    random_idx = candidate_idx[torch.randint(0, len(candidate_idx), (mask.sum(),))]
                                    feat[mask, j] = feat[random_idx, j]"""

                        """noise = (torch.rand_like(feat_norm) * 2 - 1) * sigma_ratio  # U(-0.1,0.1)
                        feat = feat_norm + noise * std_col + mean_col"""

                        # 对特征列 x_j 进行随机掩蔽与扰动，模拟分布偏移
                        '''mask = torch.rand_like(feat) < 0.1  # 随机采样率 0.1
                        random_indices = torch.randint(0, feat.size(0), (mask.sum(),), device=feat.device)
                        random_values = feat[random_indices]

                        # 使用随机采样值替换被掩蔽位置
                        feat = torch.where(mask, random_values, feat)'''

                
            # 编码
            x = self.encoder_dict[stype.value](feat, col_names)
            #print("After encoding:", x.mean().item(), x.std().item())

            xs.append(x)
            all_col_names.extend(col_names)

        x = torch.cat(xs, dim=1)
        return x, all_col_names


