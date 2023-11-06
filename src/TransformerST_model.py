#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GraphConv, graclus, max_pool, global_mean_pool, JumpingKnowledge
import torch_geometric as tg
from src.gat_v2conv import GATv2Conv
from src.model_utils import InputPartlyTrainableLinear, PartlyTrainableParameter2D, get_fully_connected_layers, get_kl
import anndata
from typing import Any, Iterable, Mapping, Sequence, Tuple, Union
import numpy as np
import logging
from scipy.sparse import spmatrix
import networkx as nx
# from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Independent
from src.decoder import BasisODEDecoder,BasisDecoder
from torch.nn.functional import softplus
from src.submodules import *
from src.quantize_1d import VectorQuantizer
import torch_geometric as pyg
# from scETM.logging_utils import log_arguments
# from src.BaseCellModel import BaseCellModel
def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.1, eps=0.0001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t())) 
        return adj 

class ST_Transformer(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+params.feat_hidden2
        self.layer_num=3
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1,heads=1,dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList([tg.nn.TransformerConv(params.gcn_hidden1*1, params.gcn_hidden1,heads=1,dropout=params.p_drop) for i in range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1*1, params.gcn_hidden2,dropout=params.p_drop)
        self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1*1, params.gcn_hidden2,dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        x = F.relu(hidden1)
        # print(x.shape)
        # x = F.dropout(x, training=True)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, adj)
            x = F.relu(x)
            # x = F.dropout(x, training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,training):
        mu, logvar, feat_x = self.encode(x, adj,training)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, mu, logvar, de_feat, q, feat_x, gnn_z
class ST_Transformer_adaptive(nn.Module):
    def __init__(self, input_dim, params):
        super(ST_Transformer_adaptive, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2+64
        self.layer_num=3
        self.at=0.5
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))
        self.quantize = VectorQuantizer(self.latent_dim, 64, beta=0.25)
        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L1', full_block(self.latent_dim, input_dim, params.p_drop))
        # params.p_drop=0.5
        # GCN layers
        self.gc1 = tg.nn.TransformerConv(params.feat_hidden2, params.gcn_hidden1, heads=1, dropout=params.p_drop)
        self.conv_hidden = nn.ModuleList(
            [tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden1, heads=1, dropout=params.p_drop) for i in
             range(self.layer_num - 2)])
        self.gc2 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.gc3 = tg.nn.TransformerConv(params.gcn_hidden1 * 1, params.gcn_hidden2, dropout=params.p_drop)
        # self.jump = JumpingKnowledge(mode='cat')
        # self.lin1 = Linear(2 * hidden, hidden)
        # self.lin2 = Linear(hidden, num_classes)
        # self.pooling_type = pooling_type
        # self.no_cat = no_cat
        self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        # self.fc=nn.Linear(params.gcn_hidden2+params.feat_hidden2, 8)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2 + params.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
    def add_noise(self,x):
        noise=torch.randn(x.size())*0.4
        noisy_img=x+noise.cuda()
        return noisy_img
    def encode(self, x, adj,adj_prue,training):
        # x, edge_index = data.x, data.edge_index
        # edge_index=adj
        # x=self.add_noise(x)
        # print(x.shape,"wwwwwwwwwwwwwwwwwww")
        feat_x = self.encoder(x)
        # print(feat_x.shape,"wwwwwwwwwwww")
        # print(torch.isinf(feat_x).any(), "1111")
        hidden1,atten = self.gc1(feat_x, adj,return_attention_weights=True)
        # print(atten.shape,"wwwwwwwww")
        # atten1=atten
        # g=nx.Graph(atten[0],atten[1])
        atten=pyg.utils.remove_self_loops(atten[0],atten[1])
        atten=pyg.utils.to_dense_adj(atten[0],edge_attr=atten[1])
        atten=atten.squeeze(0).squeeze(-1)
        # print(torch.sum(atten,1),"wwwwwwww")
        # if torch.isnan(atten).any():
        #     print(atten1)
        # print(atten.shape,adj.shape,hidden1.shape,"wwwwwww")
        # adj_org = nx.adjacency_matrix(atten)
        # print(adj_org.shape)
        hidden1_prue,atten_prue = self.gc1(feat_x, adj_prue,return_attention_weights=True)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # print(atten.shape,hidden1.shape,"fdsfsdfsadf")
        hidden1=(1-self.at)*(torch.mm(atten,hidden1))+self.at*(torch.mm(atten_prue,hidden1_prue))
        # print(torch.isnan(hidden1).any(),"2")
        x = F.relu(hidden1)
        # x = F.relu(hidden1_prue)
        # x = F.dropout(x, p = 0.2,training=True)
        for i in range(self.layer_num - 2):
            x,atten = self.conv_hidden[i](x, adj,return_attention_weights=True)
            x_prue,atten_prue = self.conv_hidden[i](x, adj_prue,return_attention_weights=True)
            atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
            atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
            atten_prue = atten_prue.squeeze(0).squeeze(-1)
            atten = pyg.utils.remove_self_loops(atten[0], atten[1])
            atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
            atten = atten.squeeze(0).squeeze(-1)
            x=(1 - self.at) * (torch.mm(atten, x)) + self.at * (torch.mm(atten_prue, x_prue))
            x = F.relu(x)
            # x = F.dropout(x, p = 0.2,training=True)
        # x = F.normalize(x, p=2, dim=-1)
        hidden1=x
        # print(torch.isnan(hidden1).any(),"3")
        mu,atten=self.gc2(hidden1, adj,return_attention_weights=True)
        # logvar,atten_var=self.gc3(hidden1, adj,return_attention_weights=True)
        mu_prue,atten_prue = self.gc2(hidden1, adj_prue,return_attention_weights=True)
        # logvar_prue,atten_var_prue = self.gc3(hidden1, adj_prue,return_attention_weights=True)
        atten = pyg.utils.remove_self_loops(atten[0], atten[1])
        atten = pyg.utils.to_dense_adj(atten[0], edge_attr=atten[1])
        atten = atten.squeeze(0).squeeze(-1)
        atten_prue = pyg.utils.remove_self_loops(atten_prue[0], atten_prue[1])
        atten_prue = pyg.utils.to_dense_adj(atten_prue[0], edge_attr=atten_prue[1])
        atten_prue = atten_prue.squeeze(0).squeeze(-1)
        # atten_var_prue = pyg.utils.remove_self_loops(atten_var_prue[0], atten_var_prue[1])
        # atten_var_prue = pyg.utils.to_dense_adj(atten_var_prue[0], edge_attr=atten_var_prue[1])
        # atten_var_prue = atten_var_prue.squeeze(0).squeeze(-1)
        # atten_var = pyg.utils.remove_self_loops(atten_var[0], atten_var[1])
        # atten_var = pyg.utils.to_dense_adj(atten_var[0], edge_attr=atten_var[1])
        # atten_var = atten_var.squeeze(0).squeeze(-1)
        mu=(1-self.at)*torch.mm(atten,mu)+(self.at)*torch.mm(atten_prue,mu_prue)
        # logvar=(1-self.at)*torch.mm(atten_var,logvar)+(self.at)*torch.mm(atten_var_prue,logvar_prue)
        return mu,feat_x

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     for conv in self.conv_hidden:
    #         conv.reset_parameters()
    #     self.gc2.reset_parameters()
    #     self.encoder.reset_parameters()
    #     self.decoder.reset_parameters()
    #     self.gc3.reset_parameters()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,adj_prue,training):
        mu,feat_x = self.encode(x, adj,adj_prue,training)
        # gnn_z = self.reparameterize(mu, logvar)
        quant, _, info = self.quantize(feat_x)
        z = torch.cat((quant, mu), 1)
        de_feat = self.decoder(z)
        # print(gnn_z.shape)
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(q.shape,"wwwwwwwwwwwww")
        return z, de_feat, q, feat_x

