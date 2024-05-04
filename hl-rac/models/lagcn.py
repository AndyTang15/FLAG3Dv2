import copy as cp
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleList , Sequential

import numpy as np
from typing import Tuple
from mmcv.cnn import build_activation_layer, build_norm_layer

class AAGCNBlock(BaseModule):
    """The basic block of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_nodes, num_nodes)`.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        residual (bool): Whether to use residual connection. Defaults to True.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A: torch.Tensor,
                 stride: int = 1,
                 residual: bool = True,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {
            k: v
            for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']
        }
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_aagcn')
        assert gcn_type in ['unit_aagcn']

        self.gcn = unit_aagcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(
                out_channels, out_channels, 9, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor,text) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.relu(self.tcn(self.gcn(x,text)) + self.residual(x))


class AAGCN(BaseModule):
    """AAGCN backbone, the attention-enhanced version of 2s-AGCN.

    Skeleton-Based Action Recognition with Multi-Stream
    Adaptive Graph Convolutional Networks.
    More details can be found in the `paper
    <https://arxiv.org/abs/1912.06971>`__ .

    Two-Stream Adaptive Graph Convolutional Networks for
    Skeleton-Based Action Recognition.
    More details can be found in the `paper
    <https://arxiv.org/abs/1805.07694>`__ .

    Args:
        graph_cfg (dict): Config for building the graph.
        in_channels (int): Number of input channels. Defaults to 3.
        base_channels (int): Number of base channels. Defaults to 64.
        data_bn_type (str): Type of the data bn layer. Defaults to ``'MVC'``.
        num_person (int): Maximum number of people. Only used when
            data_bn_type == 'MVC'. Defaults to 2.
        num_stages (int): Total number of stages. Defaults to 10.
        inflate_stages (list[int]): Stages to inflate the number of channels.
            Defaults to ``[5, 8]``.
        down_stages (list[int]): Stages to perform downsampling in
            the time dimension. Defaults to ``[5, 8]``.
        init_cfg (dict or list[dict], optional): Config to control
            the initialization. Defaults to None.

        Examples:
        >>> import torch
        >>> from mmaction.models import AAGCN
        >>> from mmaction.utils import register_all_modules
        >>>
        >>> register_all_modules()
        >>> mode = 'stgcn_spatial'
        >>> batch_size, num_person, num_frames = 2, 2, 150
        >>>
        >>> # openpose-18 layout
        >>> num_joints = 18
        >>> model = AAGCN(graph_cfg=dict(layout='openpose', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # nturgb+d layout
        >>> num_joints = 25
        >>> model = AAGCN(graph_cfg=dict(layout='nturgb+d', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # coco layout
        >>> num_joints = 17
        >>> model = AAGCN(graph_cfg=dict(layout='coco', mode=mode))
        >>> model.init_weights()
        >>> inputs = torch.randn(batch_size, num_person,
        ...                      num_frames, num_joints, 3)
        >>> output = model(inputs)
        >>> print(output.shape)
        >>>
        >>> # custom settings
        >>> # disable the attention module to degenerate AAGCN to AGCN
        >>> model = AAGCN(graph_cfg=dict(layout='coco', mode=mode),
        ...               gcn_attention=False)
        >>> model.init_weights()
        >>> output = model(inputs)
        >>> print(output.shape)
        torch.Size([2, 2, 256, 38, 18])
        torch.Size([2, 2, 256, 38, 25])
        torch.Size([2, 2, 256, 38, 17])
        torch.Size([2, 2, 256, 38, 17])
    """

    def __init__(self,
                 graph_cfg: Dict,
                 in_channels: int = 3,
                 base_channels: List[int] = [64,128,256,128,64,16],
                 base_strides: List[int] = [1,1,1,1,1,1],
                 data_bn_type: str = 'VC',
                 num_person: int = 2,
                 num_stages: int = 10,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        assert data_bn_type in ['MVC', 'VC', None]
        self.data_bn_type = data_bn_type
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_person = num_person
        self.num_stages = num_stages

        if self.data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif self.data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        modules = [
                AAGCNBlock(
                    in_channels,
                    base_channels[0],
                    A.clone(),
                    1,
                    residual=False,
                    **lw_kwargs[0])
            ]
        for i in range(num_stages-1):
            modules.append(
                AAGCNBlock(
                    base_channels[i],
                    base_channels[i+1],
                    A.clone(),
                    stride=base_strides[i],
                    **lw_kwargs[i+1]))

        self.gcn = ModuleList(modules)
        self.pooling_layer = nn.AvgPool1d(2,2)

    def forward(self, x: torch.Tensor, text) -> torch.Tensor:
        """Defines the computation performed at every call."""
        # N, M, T, V, C = x.size()
        N, C, T, V = x.size()
        x = x.permute(0,1,3,2).contiguous() # N,C,V,T
        x = x.view(N , C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T) # N,C,V,T
        x = x.permute(0,1,3,2).contiguous()# N,C,T,V

        for i in range(self.num_stages):
            x = self.gcn[i](x,text)
        N, C, T, V = x.size()
        x = x.permute(0,1,3,2).contiguous()
        x = x.view(N,C*V,T)

        return x
    

def k_adjacency(A: Union[torch.Tensor, np.ndarray],
                k: int,
                with_self: bool = False,
                self_factor: float = 1) -> np.ndarray:
    """Construct k-adjacency matrix.

    Args:
        A (torch.Tensor or np.ndarray): The adjacency matrix.
        k (int): The number of hops.
        with_self (bool): Whether to add self-loops to the
            k-adjacency matrix. The self-loops is critical
            for learning the relationships between the current
            joint and its k-hop neighbors. Defaults to False.
        self_factor (float): The scale factor to the added
            identity matrix. Defaults to 1.

    Returns:
        np.ndarray: The k-adjacency matrix.
    """
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(
        np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(edges: List[Tuple[int, int]], num_node: int) -> np.ndarray:
    """Get adjacency matrix from edges.

    Args:
        edges (list[tuple[int, int]]): The edges of the graph.
        num_node (int): The number of nodes of the graph.

    Returns:
        np.ndarray: The adjacency matrix.
    """
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[j, i] = 1
    return A


def normalize_digraph(A: np.ndarray, dim: int = 0) -> np.ndarray:
    """Normalize the digraph according to the given dimension.

    Args:
        A (np.ndarray): The adjacency matrix.
        dim (int): The dimension to perform normalization.
            Defaults to 0.

    Returns:
        np.ndarray: The normalized adjacency matrix.
    """
    # A is a 2D square array
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))

    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)

    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node: int,
                     edges: List[Tuple[int, int]],
                     max_hop: int = 1) -> np.ndarray:
    """Get n-hop distance matrix by edges.

    Args:
        num_node (int): The number of nodes of the graph.
        edges (list[tuple[int, int]]): The edges of the graph.
        max_hop (int): The maximal distance between two connected nodes.
            Defaults to 1.

    Returns:
        np.ndarray: The n-hop distance matrix.
    """
    A = np.eye(num_node)

    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    """The Graph to model the skeletons.

    Args:
        layout (str): must be one of the following candidates:
            'openpose', 'nturgb+d', 'coco'. Defaults to ``'coco'``.
        mode (str): must be one of the following candidates:
            'stgcn_spatial', 'spatial'. Defaults to ``'spatial'``.
        max_hop (int): the maximal distance between two connected
            nodes. Defaults to 1.
    """

    def __init__(self,
                 layout: str = 'coco',
                 mode: str = 'spatial',
                 max_hop: int = 1) -> None:

        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode

        assert layout in ['openpose', 'nturgb+d', 'coco','flag3d']

        self.set_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def set_layout(self, layout: str) -> None:
        """Initialize the layout of candidates."""

        if layout == 'openpose':
            self.num_node = 18
            self.inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                           (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                           (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            neighbor_base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
                             (7, 6), (8, 7), (9, 21), (10, 9), (11, 10),
                             (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                             (17, 1), (18, 17), (19, 18), (20, 19), (22, 8),
                             (23, 8), (24, 12), (25, 12)]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1
        elif layout == 'coco':
            self.num_node = 17
            self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                           (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                           (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
            self.center = 0
        elif layout == 'flag3d':
            self.num_node = 24
            neighbor_1base = [(7,6),(6,8),(6,5),(15,13),(13,11),(11,9),(9,5),(5,10),(10,12),(12,14),(14,16),(5,4),(4,3),(3,2),(2,1),(17,1),
                                (1,18),(17,19),(18,20),(19,21),(20,22),(21,23),(22,24)]
            self.inward = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.center = 0
        else:
            raise ValueError(f'Do Not Exist This Layout: {layout}')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self) -> np.ndarray:
        """ST-GCN spatial mode."""
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center

        A = []
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self) -> np.ndarray:
        """Standard spatial mode."""
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self) -> np.ndarray:
        """Construct an adjacency matrix for an undirected graph."""
        A = edge2mat(self.neighbor, self.num_node)
        return A[None]

class unit_aagcn(BaseModule):
    """The graph convolution unit of AAGCN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        A (torch.Tensor): The adjacency matrix defined in the graph
            with shape of `(num_subsets, num_joints, num_joints)`.
        coff_embedding (int): The coefficient for downscaling the embedding
            dimension. Defaults to 4.
        adaptive (bool): Whether to use adaptive graph convolutional layer.
            Defaults to True.
        attention (bool): Whether to use the STC-attention module.
            Defaults to True.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1,
                     override=dict(type='Constant', name='bn', val=1e-6)),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
                dict(type='ConvBranch', name='conv_d')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        coff_embedding: int = 4,
        adaptive: bool = True,
        attention: bool = True,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(
                type='Constant',
                layer='BatchNorm2d',
                val=1,
                override=dict(type='Constant', name='bn', val=1e-6)),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out'),
            dict(type='ConvBranch', name='conv_d')
        ]
    ) -> None:

        if attention:
            attention_init_cfg = [
                dict(
                    type='Constant',
                    layer='Conv1d',
                    val=0,
                    override=dict(type='Xavier', name='conv_sa')),
                dict(
                    type='Kaiming',
                    layer='Linear',
                    mode='fan_in',
                    override=dict(type='Constant', val=0, name='fc2c'))
            ]
            init_cfg = cp.copy(init_cfg)
            init_cfg.extend(attention_init_cfg)

        super(unit_aagcn, self).__init__(init_cfg=init_cfg)
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)
            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = ModuleList()
            self.conv_b = ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        # self.text_proj = nn.Sequential(nn.Linear(384,in_channels),nn.LayerNorm(in_channels))
        # self.cross_attention = nn.MultiheadAttention(24,1,batch_first=True,dropout=0.2)

    def forward(self, x: torch.Tensor,text) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.size()
        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
                    N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha + A1 * text
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class unit_tcn(BaseModule):
    """The basic unit of temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the temporal convolution kernel.
            Defaults to 9.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        dilation (int): Spacing between temporal kernel elements.
            Defaults to 1.
        norm (str): The name of norm layer. Defaults to ``'BN'``.
        dropout (float): Dropout probability. Defaults to 0.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
                dict(type='Constant', layer='BatchNorm2d', val=1),
                dict(type='Kaiming', layer='Conv2d', mode='fan_out')
            ]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: str = 'BN',
        dropout: float = 0,
        init_cfg: Union[Dict, List[Dict]] = [
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Kaiming', layer='Conv2d', mode='fan_out')
        ]
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] \
            if norm is not None else nn.Identity()

        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.drop(self.bn(self.conv(x)))


class mstcn(BaseModule):
    """The multi-scale temporal convolutional network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int): Number of middle channels. Defaults to None.
        dropout (float): Dropout probability. Defaults to 0.
        ms_cfg (list): The config of multi-scale branches. Defaults to
            ``[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']``.
        stride (int): Stride of the temporal convolution. Defaults to 1.
        init_cfg (dict or list[dict]): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mid_channels: int = None,
                 dropout: float = 0.,
                 ms_cfg: List = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3),
                                 '1x1'],
                 stride: int = 1,
                 init_cfg: Union[Dict, List[Dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(
                            kernel_size=(cfg[1], 1),
                            stride=(stride, 1),
                            padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(
                    branch_c,
                    branch_c,
                    kernel_size=cfg[0],
                    stride=stride,
                    dilation=cfg[1],
                    norm=None))
            branches.append(branch)

        self.branches = ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = Sequential(
            nn.BatchNorm2d(tin_channels), self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
