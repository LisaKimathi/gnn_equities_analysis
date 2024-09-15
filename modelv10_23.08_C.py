# -*- coding: utf-8 -*-
"""
from google.colab import drive
drive.mount("/content/drive")

!which python
!python --version

!pip install virtualenv

!virtualenv /content/drive/MyDrive/virtual_env
#!virtualenv\Scripts\activate
# %venv /content/drive/MyDrive/virtual_env

import sys
sys.path.append('/content/drive/MyDrive/virtual_env/lib/python3.10/site-packages/')
#!where python
!python --version

!source /content/drive/MyDrive/virtual_env/bin/activate
!pip install torch-scatter torch-sparse torch-cluster torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

import sys
sys.path.append('/content/drive/MyDrive/virtual_env/lib/python3.10/site-packages/')

import torch_geometric
print(torch_geometric.__file__)

# set up github manually
!pip install git+https://github.com/LisaKimathi/gnn_equities_analysis.git
"""

# Install required packages.
import os
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

import torch

os.environ['TORCH'] = torch.__version__
# print(torch.__version__)
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn import GCNConv, GATConv, GAE, VGAE, summary
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling

from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, r2_score

os.listdir()
# from typing import Optional, Tuple

# Data processing section
from torch_geometric.data import download_url

datalink = 'https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018/data'

year_one = 'G:/My Drive/Dissertation/Data/raw_data/2014_Financial_Data.csv'
year_two = 'G:/My Drive/Dissertation/Data/raw_data/2015_Financial_Data.csv'
year_three = 'G:/My Drive/Dissertation/Data/raw_data/2016_Financial_Data.csv'
year_four = 'G:/My Drive/Dissertation/Data/raw_data/2017_Financial_Data.csv'
year_five = 'G:/My Drive/Dissertation/Data/raw_data/2018_Financial_Data.csv'

df_one = pd.read_csv(year_one)
#print(df_one.head())

''' Done!
# Renaming existing unnamed column
file_paths = [
    '/content/drive/MyDrive/Dissertation_colab/raw_data/2014_Financial_Data.csv',
    '/content/drive/MyDrive/Dissertation_colab/raw_data/2015_Financial_Data.csv',
    '/content/drive/MyDrive/Dissertation_colab/raw_data/2016_Financial_Data.csv',
    '/content/drive/MyDrive/Dissertation_colab/raw_data/2017_Financial_Data.csv',
    '/content/drive/MyDrive/Dissertation_colab/raw_data/2018_Financial_Data.csv'
]
old_column_name = 'Unnamed: 0'
new_column_name = 'Company'
for file_path in file_paths:
    df = pd.read_csv(file_path)
    df = df.rename(columns={old_column_name: new_column_name})
    df.to_csv(file_path, index=False)
    print(f"Renamed '{old_column_name}' to '{new_column_name}' in {file_path}")
'''

df_two = pd.read_csv(year_two)
# merging_df = pd.merge(df_one, df_two, on=['Company', 'Sector'], how='inner')
# print(merging_df.head())
merged_df = pd.read_csv('G:/My Drive/Dissertation/Data/data/merge_df.csv')
# print(type(merge_df))
# print(merge_df.columns)
# print(merged_df.shape)  # 3777, 448
merged_df['Sector'] = merged_df['Sector'].astype('category').cat.codes
# print(merged_df['Sector'])
merged_df_indexed = merged_df.set_index('Company')
# print(sector_remain)  # Length: 3777, dtype: int8
# print(type(sector_remain))  # pandas series
# dtype=torch.float).unsqueeze(1)
numeric_columns = merged_df_indexed.select_dtypes(include=np.number).columns
merged_df2 = merged_df_indexed[numeric_columns]
# .apply(pd.to_numeric, errors='coerce')
# merge_df2 = merge_df.fillna(0)
# print('data types: ', merged_df2.dtypes)
# print('shape: ',merged_df2.shape)  # 3777, 447
# print('df: ', merged_df2.head())

merge_df = merged_df2.transpose()
# print(type(merge_df))
# print('data types t.: ', merge_df.dtypes)
# print('shape t.: ', merge_df.shape)   # 447, 3777
# print('df t. : ', merge_df.head())
# print('merge_df: ', merge_df)

# Add years to columns after merging
columns = merge_df.columns.tolist()
# print(columns)
'''
column_mapping = {}
for col in merge_df.columns:
    if col.endswith('_x'):
        column_mapping[col] = col[:-2] + '_2014'  # Replace '_x' with '_2014'
    elif col.endswith('_y'):
        column_mapping[col] = col[:-2] + '_2015'  # Replace '_y' with '_2015'
'''
# merge_df = merge_df.rename(columns=column_mapping)
# print(merge_df.columns)
'''
# For node features using year
year = []
for col in columns:
    if col.endswith('_x'):
        year.append(2014)
    elif col.endswith('_y'):
        year.append(2015)
        #col.replace("_y", "2015")
        #re.sub('_y', '2015', col)

year = []
for col in columns:
    if col.endswith('_2014'):
        year.append(2014)
    elif col.endswith('_2015'):
        year.append(2015)
# print(year)
# print(len(year))
'''
# merge_df.to_csv('G:/My Drive/Dissertation/Data/data/merge_df_t.csv', index=False)
# "------------------------------------------------------------------------------------------------------------"

# Table to Graph section
# Convert a df to tensor to be used in pytorch

def df_to_tensor(df):
    return torch.from_numpy(df.values)


# torch.from_numpy(df.values).to(torch.float)
# torch.tensor(df.values)
# torch.tensor(df.values).to(torch.float)
# torch.tensor(df.to_numpy())
# df_to_numpy = df.to_numpy()
# torch.from_numpy(df_to_numpy)


# merge_df[columns] = merge_df[columns].astype(float)
# numeric_columns = merge_df.select_dtypes(include=np.number).columns
# merge_df[numeric_columns] = merge_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
# print(type(merge_df[numeric_columns]))
df_tensor = df_to_tensor(merge_df)
# torch.save(df_tensor, 'G:/My Drive/Dissertation/Data/data/df_tensor.pt')
# print('df_tensor: ', df_tensor)
# print(df_tensor.shape)  # torch.Size([447, 3777])

# Convert relevant columns to numeric and convert to tensor
# print(merge_df.dtypes)
# numeric_columns = merge_df.select_dtypes(include=np.number).columns
# merge_df[numeric_columns] = merge_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
# merge_df[columns] = pd.to_numeric(merge_df[numeric_columns], errors='coerce')
'''
for col in merge_df.columns:
    try:
        merge_df[col] = pd.to_numeric(merge_df[col])
    except ValueError:
        pass  # Handle columns that can't be converted
print(merge_df.dtypes)
# merge_df = merge_df.fillna(0)
'''
# merged_df, merged_df_indexed, merged_df2, merge_df

# Nodes from financial metrics/row headings
for_nodes = merged_df.columns.drop('Company')
# print(for_nodes)
node_identifiers = for_nodes.tolist()
# print('ni: ', node_identifiers)
# print(type(node_identifiers))  # list
# print(len(node_identifiers))  # 447

# Node features
# used earlier...merged_df['Sector'] = merged_df['Sector'].astype('category').cat.codes
sector_feature = torch.tensor(merge_df.loc['Sector'], dtype=torch.float).unsqueeze(0)
# #FutureWarning: Series.__getitem__ treating keys as positions is deprecated.
# #In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior).
# #To access a value by position, use `ser.iloc[pos]`
# print('sf: ', sector_feature)
# print('sf_shape: ', sector_feature.shape)
# sf:  tensor([[3., 3., 3.,  ..., 9., 9., 9.]])
# sf_shape:  torch.Size([1, 3777])

# Convert company names (row headings) to numerical indices
# company_indices = pd.factorize(merge_df.index)[-1]
company_feature = torch.tensor(pd.factorize(merge_df.transpose().index)[0], dtype=torch.float).unsqueeze(0)
# print('cf: ', company_feature)
# print('cf_shape: ', company_feature.shape)  # torch.Size([1, 3777])

# For node features using year
year = []
for i in merge_df.index:
    if i.endswith('_2014'):
        year.append(2014)
    elif i.endswith('_2015'):
        year.append(2015)
# print(year)
# print(len(year))  # 444
'''
# year_feature = year
# print(type(year))
year_tensor = torch.tensor(year, dtype=torch.float)
yf_repeated = year_tensor.repeat_interleave(len(merge_df['Company']), dim=0)  # shape becomes (3777 * number of years,)
year_feature = yf_repeated.reshape(len(merge_df['Company']), -1)  # reshape to (3777, number of years)
# print('yf_shape: ', year_feature.shape)
# print('yf: ', year_feature)

# Feature shapes
sf_shape:  torch.Size([3777, 11])
cf_shape:  torch.Size([3777, 3777])
yf_shape:  torch.Size([3777, 444])

# print(type(year_feature))
# print('yf: ', year_feature)
# values_feature = df_tensor.values()
# values_feature = df_tensor.numpy()
# print('vf: ', values_feature)

node_features = np.hstack((sector_feature, company_numeric, year_feature)) # numpy array of node features
# print('nf: ', node_features)
nf_tensor = torch.tensor(node_features, dtype=torch.float)
# print('nf tensor: ', nf_tensor)
# print('nf_shape: ', nf_tensor.shape)   # torch.Size([3777, 4232])
# nf_df = pd.DataFrame(nf_tensor.numpy()
# nf_df.to_csv('G:/My Drive/Dissertation/Data/data/node_features.csv')

# print(f'Number of graphs: {len(merge_df)}')  # 3777 graphs
# ##################### print(f'Number of features: {nf_df.num_features}')

# Correlation matrix
corr_matrix = merge_df[numeric_columns].corr()     # pandas dataframe
corr_matrix_values = merge_df[numeric_columns].corr().values     # numpy array  # edge index shape: torch.Size([2, 4414])
# ######################## create a visualisation of the correlation matrix

# corr_matrix_values = corr_matrix.values     # numpy array
# corr_matrix = node_features().corr().values
# corr_matrix = torch.corrcoef(df_tensor)     # edge index shape: torch.Size([2, 7599124])
# corr_df = pd.DataFrame(corr_matrix.numpy())
# print('correlation matrix: ', corr_matrix)   # corr matrix shape:  (446, 446)
# print('corr_matrix_values: ', corr_matrix_values)
# print(type(corr_matrix))
# print(type(corr_matrix_values))
# print('min: ', corr_matrix.min(), ',max: ', corr_matrix.max())
# print(numpy.min(corr_matrix.values), corr_matrix.values.max())
# print('corr matrix shape: ', corr_matrix.shape)
# print(corr_matrix.shape[0])
# corr_matrix.to_csv('G:/My Drive/Dissertation/Data/data/corr_matrix_df.csv')

# Edge index
threshold = 0.5
edge_index = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])  # For undirected graph

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
# print('edge index shape: ', edge_index.shape)   # edge index shape: torch.Size([2, 4414])
# print('edge index: ', edge_index)

# Subgraphs
companies = merge_df['Company']
# print(companies)
graph_data_list = []
graph_list = []

for i, company in enumerate(companies):
    data = Data(x=nf_tensor, edge_index=edge_index)
    print(data)
    graph_data_list.append(data)
    print(graph_data_list)
    G = nx.Graph()
    G.add_nodes_from(node_identifiers)
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    print(G)
    graph_list.append(G)


dataset = Data(x=nf_tensor, edge_index=edge_index)    # creating a PyTorch Geometric Data object
# print(dataset)      # Data(x=[3777, 4232], edge_index=[2, 4414])

# Assign train and test masks
num_train = int(dataset.num_nodes*0.6)  
indices = torch.randperm(dataset.num_nodes)  # Create random permutation of indices
dataset.train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
dataset.train_mask[indices[:num_train]] = True
dataset.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool)
dataset.test_mask[indices[num_train:]] = True

# Check attributes of a data object in the graph
print(type(dataset))
print('----------------------------')
print(f'Number of nodes: {dataset.num_nodes}')    # 3777
print(f'Number of edges: {dataset.num_edges}')    # 4414
print(f'Average node degree: {dataset.num_edges / dataset.num_nodes:.2f}')    # 1.17
print(f'Number of training nodes: {dataset.train_mask.sum()}')   # 2266
print(f'Training node label rate: {int(dataset.train_mask.sum()) / dataset.num_nodes:.2f}')   # 0.60
print(f'Has isolated nodes: {dataset.has_isolated_nodes()}')    # True
print(f'Has self-loops: {dataset.has_self_loops()}')     # False
print(f'Is undirected: {dataset.is_undirected()}')    # True


# Create a graph
G = nx.Graph()
# G.add_nodes_from(node_identifiers)   # Graph with 447 nodes and 0 edges  # Graph with 9275 nodes and 4414 edges (using edge_index.t)
# G.add_nodes_from(dataset.num_nodes)
G.add_nodes_from(range(dataset.num_nodes))  # Graph with 12605 nodes and 4414 edges
# G.add_edges_from(edge_index.t())
edges = edge_index.t().tolist()   # Graph with 3777 nodes and 2207 edges
G.add_edges_from(edges)
print(G)

# Visualise graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42)  
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue')
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title("Graph Visualization")
plt.show()


# Visualise the graph
# from torch_geometric.utils import to_networkx
G = to_networkx(data, to_undirected = True)
visualize_graph(G, color = data.y)
'''

# Implement the Graph Neural Network section

'''
# Load graph (replace with your actual loading method)
G = nx.read_edgelist('your_graph_data.txt')

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)

# ... (Other feature calculations) ...
# Use these features in your link prediction model
# a

MAX_LOGSTD = 10
EPS = 1e-15

class VGAE(torch.nn.Module):
    """The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.reset_parameters()

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""  # noqa: D419
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def decode(self, *args, **kwargs) -> Tensor:
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Tensor = None) -> Tensor:
        """Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def kl_loss(self, mu: Tensor = None, logstd: Tensor = None) -> Tensor:
        """Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1))

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

# total_loss = recon_loss + kl_loss
'''
