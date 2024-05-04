import torch.nn as nn
import torch
from models.stgcn import STGCN
from models.lagcn import AAGCN
import numpy as np
import torch.nn.functional as F
import math
import lightning as L
from torch.utils.data import DataLoader
from dataset import Flag3d
lossMSE = nn.MSELoss()

class LRAC(L.LightningModule):
    def __init__(self,num_frames,backbone,batch_size,
                 root_path,split,
                 trans_num_layer=3,mode="spatial",dropout=0.25,lr=1e-5,
                 ):
        super(LRAC, self).__init__()
        self.save_hyperparameters()
        if backbone == "gcn":
            self.backbone = STGCN(in_channels=3,graph_args={"layout":"flag3d","strategy": "spatial"},edge_importance_weighting=True)
        elif backbone == "agcn":
            self.backbone = AAGCN(in_channels=3,graph_cfg={"layout":"flag3d","mode": "{}".format(mode)},data_bn_type = 'VC', num_stages=6)
        else:
            self.backbone = None
        self.num_frames = num_frames
        self.lr = lr
        self.root_path = root_path
        self.split = split
        self.batch_size = batch_size
        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))
        self.sims = Similarity_matrix(att_dropout=dropout)
        self.conv3x3 = nn.Conv2d(in_channels=4,  
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(dropout)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=dropout, dim_ff=512, num_layers=trans_num_layer,
                                         num_frames=self.num_frames)
        self.FC = Prediction(512, 512, 256, 1,dropout=dropout)  
        self.text_encoder = nn.Sequential(nn.Linear(768,384),nn.LayerNorm(384),nn.Linear(384,24)) # For bert
        # self.text_encoder = nn.Sequential(nn.Linear(512,256),nn.LayerNorm(256),nn.Linear(256,24)) # For Clip

    def forward(self, x,text):
        B,T,V,C = x.shape # B,T,24,3
        x = x.view(B,T,C,V)
        x = x.permute(0,2,1,3).contiguous() # B,3,T,24
        text = self.text_encoder(text) # B,1,24
        x = self.backbone(x,text) # B,384,T
        x = x.permute(0,2,1).contiguous() # B,T,384
        x = F.relu(self.sims(x, x, x))
        ## x are the similarity matrixs
        x_matrix = x # B,4,256,256
        x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32(C),f,f]
        x = self.dropout1(x)
        x = x.permute(0, 2, 3, 1)  # [B,f,f,128]
        # --------- transformer encoder ------
        x = x.flatten(start_dim=2)  # ->[B,f,128*f]
        x = F.relu(self.input_projection(x))  # ->[B,f, 512]
        x = self.ln1(x)
        x = x.transpose(0, 1)  # [f,B,512]
        x = self.transEncoder(x)  #
        x = x.transpose(0, 1)  # ->[B,f, 512]
        x = self.FC(x)  # ->[b,f,1]
        x = x.squeeze(2)
        return x, x_matrix
    

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, text, y = batch
        count = torch.sum(y, dim=1).type(torch.FloatTensor).round()
        output, matrixs = self.forward(x,text)
        predict_count = torch.sum(output, dim=1).type(torch.FloatTensor)
        predict_density = output
        loss = lossMSE(predict_density, y)
        mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]
        
        gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
        acc = 0
        for item in gaps:
            if abs(item) <= 1:
                acc += 1
        OBO = acc / predict_count.flatten().shape[0]
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_obo", OBO, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, text, y = batch
        count = torch.sum(y, dim=1).type(torch.FloatTensor).round()
        output, matrixs = self.forward(x,text)
        predict_count = torch.sum(output, dim=1).type(torch.FloatTensor)
        predict_density = output
        loss = lossMSE(predict_density, y)
        mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]
        
        gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
        acc = 0
        for item in gaps:
            if abs(item) <= 1:
                acc += 1
        OBO = acc / predict_count.flatten().shape[0]

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_obo", OBO, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return None

    def test_step(self,batch,batch_idx):
        x, text, y = batch
        count = torch.sum(y, dim=1).type(torch.FloatTensor).round()
        output, matrixs = self.forward(x,text)
        predict_count = torch.sum(output, dim=1).type(torch.FloatTensor)
        predict_density = output
        loss = lossMSE(predict_density, y)
        mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]
        gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
        alpha = [0.4,0.6,0.8,1]
        OBO_List = []
        for i in range(len(alpha)):
            acc = 0
            for item in gaps:
                if abs(item) <= alpha[i]:
                    acc += 1
            OBO = acc / predict_count.flatten().shape[0]
            OBO_List.append(OBO)
            self.log("test_obo@{}".format(alpha[i]), OBO, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_obo@mean", np.mean(OBO_List), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        return DataLoader(Flag3d(path = self.root_path, split = self.split, num_frames = self.num_frames,),batch_size=self.batch_size,pin_memory=True,shuffle=True, num_workers=32)
    
    def val_dataloader(self):
        return DataLoader(Flag3d(path = self.root_path, split = self.split, num_frames = self.num_frames,),batch_size=self.batch_size,pin_memory=True,shuffle=False, num_workers=32)

class Similarity_matrix(nn.Module):
    ''' buliding similarity matrix by self-attention mechanism '''

    def __init__(self, num_heads=4, model_dim=512,att_dropout=0.):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = 384
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = attention(att_dropout=0.)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        # dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)  # [B,F,model_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        # [B,F,model_dim] ->  [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
        query = query.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        key = key.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        value = value.reshape(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix = self.attention(query, key, value, attn_mask)

        return matrix
    
class attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, scale=64, att_dropout=None):
        super().__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(att_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        # q: [B, head, F, model_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)  # [B,Head, F, F]
        if attn_mask:
            scores = scores.masked_fill_(attn_mask, -np.inf)
        scores = self.softmax(scores)
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores  # [B,head,F, F]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class Prediction(nn.Module):
    ''' predict the density map with densenet '''

    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim,dropout=0.25):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=dropout, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    
class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x