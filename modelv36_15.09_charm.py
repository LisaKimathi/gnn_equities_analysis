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
import numpy as np
import pandas as pd
# import scipy
from scipy import stats
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import pickle

import torch

os.environ['TORCH'] = torch.__version__
# print(torch.__version__)
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GAE, VGAE, summary
from torch_geometric.nn import InnerProductDecoder
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.metrics.link_pred import LinkPredPrecision

from node2vec import Node2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score 

# os.listdir()
# from typing import Optional, Tuple


# Graph Neural Network model based on PyTorch Geometric's VGAE model
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x = self.conv1(x, edge_index).relu()
        x = F.dropout(self.conv1(x, edge_index).relu(), p=self.dropout, training=self.training)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd


MAX_LOGSTD = 10
EPS = 1e-15


class VGAEModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VGAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder()
        self.vgae = VGAE(encoder, decoder)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def reparameterize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, x, edge_index):
        z = self.vgae.encode(x, edge_index)
        adj = self.vgae.decoder.forward_all(z)
        return adj

    def encode(self, x, edge_index):
        self.__mu__, self.__logstd__ = self.encoder(x, edge_index)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparameterize(self.__mu__, self.__logstd__)
        return z

    def train_method(self, data, optimizer):
        self.train()
        optimizer.zero_grad()
        z = self.vgae.encode(data.x, data.edge_index)
        train_loss = self.vgae.recon_loss(z, train_data.pos_edge_label_index,
                                          train_data.neg_edge_label_index) + self.vgae.kl_loss(z)
        train_loss.backward()
        optimizer.step()
        return train_loss.item()

    def validate_method(self, data):
        self.eval()
        with torch.no_grad():
            z = self.vgae.encode(data.x, data.edge_index)
            # print('node embeddings :', z)
            val_loss = self.vgae.recon_loss(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)
        return val_loss.item()

    def test_method(self, data):
        self.eval()
        with torch.no_grad():
            z = self.vgae.encode(data.x, data.edge_index)
            '''pos_y = z.new_ones(pos_edge_index.size(1))
            neg_y = z.new_zeros(neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)

            pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

            return roc_auc_score(y, pred), average_precision_score(y, pred)'''


            auc = self.vgae.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
            print(auc)
            # pred = GAE(test)
            # roc_pred = GAE.test(z, est_data.pos_edge_label_index, test_data.neg_edge_label_index)
        return auc

    # pos_pred = model.vgae.decoder(z, data.pos_edge_label_index).view(-1)
    # neg_pred = model.vgae.decoder(z, data.neg_edge_label_index).view(-1)
    # preds = torch.cat([pos_pred, neg_pred], dim=0)
    # labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
    # precision = LinkPredPrecision(preds, labels)
    #    return precision.item()

    # pos_pred = model.vgae.decoder(z, data.pos_edge_label_index).view(-1)
    # neg_pred = model.vgae.decoder(z, data.neg_edge_label_index).view(-1)
    # preds = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
    # labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu().numpy()
    # avg_precision = average_precision_score(labels, preds)
    #  return avg_precision
'''
pos_pred = model.vgae.decoder(z, data.pos_edge_label_index).view(-1)
neg_pred = model.vgae.decoder(z, data.neg_edge_label_index).view(-1)
preds = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).cpu().numpy()
avg_precision = average_precision_score(labels, preds)
'''


# Function for data processing
def clean_data(files, raw_data_path, folder_path):
    # print(files)
    for file_name in files:
        file_path = os.path.join(raw_data_path, file_name)
        file = pd.read_csv(file_path)
        print(file.shape)
        print(file.isna())
        file.drop(columns=file.columns[-2:], inplace=True)
        file.dropna(axis=1, thresh=0.7 * len(file), inplace=True)
        file.dropna(axis=0, thresh=0.7 * len(file.columns), inplace=True)
        file.drop_duplicates()
        print((file == 0).all().any(axis=0))  # Checking for columns & rows with all zeros. False for columns & rows
        col_zero_proportion = (file == 0).mean()
        row_zero_proportion = (file == 0).mean(axis=1)
        cols_to_drop = col_zero_proportion[col_zero_proportion > 0.65].index  # 0.65 determined by ensuring that key columns aren't dropped
        file.drop(columns=cols_to_drop, inplace=True)
        rows_to_drop = row_zero_proportion[row_zero_proportion > 0.4].index  # 0.4 determined by need to have companies with more figures for better training
        file.drop(index=rows_to_drop, inplace=True)  # (3791, 215)
        file.reset_index(drop=True, inplace=True)
        print(file.index)
        print(file.shape)
        # file.describe()
        output_file_name = f"processed_{file_name}"
        output_file_path = os.path.join(folder_path, output_file_name)
        file.to_csv(output_file_path, index=False)
        print(f"Processed and saved: {output_file_name}")


# ################# may not be necessary
# Function to convert df to tensor
def df_to_tensor(df):
    return torch.from_numpy(df.values)


# Function to visualise graph
def visualise_graph(G):
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G,
                           seed=42)  # spring_layout positions nodes using Fruchterman-Reingold force-directed algorithm
    # nx.draw(G, pos, node_size=50, node_color="skyblue", edge_color="yellow")
    # node_colours = ["red" if node in handle_outliers(nodes_to_remove) else "purple" for node in G.nodes()]
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, node_color="purple", alpha=0.7, edge_cmap=plt.cm.Blues)
    plt.title("Graph Visualization")
    # plt.show()
    plt.savefig("graph_figure.png")  # Save the plot as an image file
    plt.close()


# Function to handle graph outliers
def handle_outliers(G, edge_index, n_neighbors=20, contamination=0.1, min_betweenness=0.01):
    # Outlier Detection and generating embeddings
    betweenness = nx.betweenness_centrality(G)
    central_nodes = {node for node, centrality in betweenness.items() if centrality >= min_betweenness}
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    # node2vec = Node2Vec(edge_index, embedding_dim=64, walk_length=30, context_size=10, walks_per_node=2)
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    e_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([e_model.wv[str(node)] for node in G.nodes()])
    outliers = lof.fit_predict(embeddings)
    outlier_nodes = {node for node, outlier_flag in zip(G.nodes(), outliers) if outlier_flag == -1}
    e_model.wv.save_word2vec_format('embeddings')  # save embeddings
    e_model.save('embedded_model')  # save model

    # Remove the outlier nodes from the graph
    nodes_to_remove = outlier_nodes - central_nodes
    G.remove_nodes_from(nodes_to_remove)
    # visualise_graph(G)

    # Visualize the resulting graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, node_color="orange", alpha=0.7, edge_cmap=plt.cm.Blues)
    plt.title('Graph after removing Outliers')
    # plt.show()
    plt.savefig("graph_after_outliers.png")
    plt.close()

    # ################ Visualise embedding??

    return G, nodes_to_remove


# Function to visualise losses
def visualise_losses(epochs, trainloss, valloss, tlabel, vlabel, plot_title, chart_name):
    plt.plot(epochs, trainloss, label=tlabel)
    plt.plot(epochs, valloss, label=vlabel)
    plt.title(plot_title)
    plt.xticks(np.arange(10, len(epochs)+1, 10))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(chart_name)
    plt.show()
    plt.close()


# Data processing
'''
datalink = 'https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018/data'
raw_data_path = 'G:/My Drive/Dissertation/Data/financial_data/raw_data/'
folder_path = 'G:/My Drive/Dissertation/Data/financial_data/'
files = os.listdir(raw_data_path)
clean_data(files, raw_data_path, folder_path)
'''
# Reading in the processed files
df_2014 = (pd.read_csv('G:/My Drive/Dissertation/Data/financial_data/processed_2014_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2014', axis=1)
df_2015 = (pd.read_csv('G:/My Drive/Dissertation/Data/financial_data/processed_2015_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2015', axis=1)
df_2016 = (pd.read_csv('G:/My Drive/Dissertation/Data/financial_data/processed_2016_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2016', axis=1)
df_2017 = (pd.read_csv('G:/My Drive/Dissertation/Data/financial_data/processed_2017_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2017', axis=1)
df_2018 = (pd.read_csv('G:/My Drive/Dissertation/Data/financial_data/processed_2018_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2018', axis=1)

# Creating the first merged data frame
first_merge = pd.merge(pd.merge(df_2014, df_2015, on=['Company', 'Sector'], how='inner'), df_2016,
                       on=['Company', 'Sector'], how='inner')
first_merge = first_merge.reset_index()
first_merge['Sector'] = first_merge['Sector'].astype('category').cat.codes
first_merge = first_merge.set_index('Company')
merge_df = first_merge.transpose()
# merge_df.to_csv('G:/My Drive/Dissertation/Data/financial_data/merge_df.csv', index=False)

# Creating the second merged data frame (for further testing)
second_merge = pd.merge(df_2017, df_2018, on=['Company', 'Sector'], how='inner')
second_merge = second_merge.reset_index()
second_merge['Sector'] = second_merge['Sector'].astype('category').cat.codes
second_merge = second_merge.set_index('Company')
merge_df2 = second_merge.transpose()
# merge_df2.to_csv('G:/My Drive/Dissertation/Data/financial_data/merge_df2.csv', index=False)
'''
# Creating the merged data frame 
dfs = [df_2014, df_2015, df_2016, df_2017, df_2018]
full_merge = ft.reduce(lambda left, right: pd.merge(left, right, on=['Company', 'Sector'], how='inner'), dfs)
print(full_merge.shape)  

full_merge = full_merge.reset_index()
full_merge['Sector'] = full_merge['Sector'].astype('category').cat.codes
full_merge = full_merge.set_index('Company')

merged_df = full_merge.transpose()
# merged_df.to_csv('G:/My Drive/Dissertation/Data/financial_data/merged_df.csv', index=False)
'''
"------------------------------------------------------------------------------------------------------------"

# Table to Graph section

# ########## doesn't seem necessary, not used anywhere 
# Convert df to tensor
df_tensor = df_to_tensor(merge_df)

# Nodes from financial metrics
node_identifiers = merge_df.index.difference(['Sector']).to_list()

# Correlation matrix
corr_df = merge_df.transpose()
corr_matrix = corr_df.corr()
# ######################## create a visualisation of the correlation matrix
# print(type(corr_matrix))                    # pandas DataFrame
# print('corr matrix shape: ', corr_matrix.shape)  # corr matrix shape:  (578, 578)
# print(len(corr_matrix))     # 578
# print('min: ', corr_matrix.min(), ',max: ', corr_matrix.max())  # ranges from -1 to 1
# corr_matrix.to_csv('G:/My Drive/Dissertation/Data/financial_data/corr_matrix_df_05.09.csv')

# Edge index
threshold = 0.5
edge_index = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
# print('edge index shape: ', edge_index.shape)   # edge index shape: torch.Size([2, 12914])
# print('edge index: ', edge_index)
# edge index:  tensor([[  1,   3,   1,  ..., 560, 561, 569],
#         [  3,   1,   4,  ..., 559, 569, 561]])
# ################################## visualise the adjacency matrix

# Node features
sector_feature = (torch.tensor(merge_df.loc['Sector'], dtype=torch.float).view(1, -1).
                  repeat(len(node_identifiers), 1))  # see warning
company_feature = (torch.tensor(pd.factorize(merge_df.columns)[0], dtype=torch.float).view(1, -1).
                   repeat(len(node_identifiers), 1))

# Financial statement feature (assigning financial statement tags to financial metrics)
# Extract list of financial metrics using df_2014
metrics_list = df_2014.columns.str.removesuffix('_2014').to_list()
# Matching metric to financial statement
profit_loss = ['Revenue', 'Cost of Revenue', 'Gross Profit', 'R&D Expenses', 'SG&A Expense', 'Operating Expenses', 'Operating Income',
               'Interest Expense', 'Earnings before Tax', 'Income Tax Expense', 'Net Income', 'Net Income Com', 'Weighted Average Shs Out',
               'Weighted Average Shs Out (Dil)',  'Consolidated Income', 'Earnings Before Tax Margin', 'Net Profit Margin']
financial_pos = ['Cash and cash equivalents', 'Short-term investments', 'Cash and short-term investments', 'Receivables', 'Inventories',
                 'Total current assets', 'Property, Plant & Equipment Net','Goodwill and Intangible Assets', 'Long-term investments', 'Tax assets',
                 'Total non-current assets', 'Total assets', 'Payables', 'Short-term debt', 'Total current liabilities', 'Long-term debt',
                 'Total debt', 'Deferred revenue', 'Tax Liabilities', 'Total non-current liabilities', 'Total liabilities', 'Other comprehensive income',
                 'Retained earnings (deficit)', 'Total shareholders equity', 'Investments', 'Other Liabilities', 'Depreciation & Amortization']
cashflow = ['Stock-based compensation', 'Operating Cash Flow', 'Capital Expenditure', 'Acquisitions and disposals', 'Investment purchases and sales',
            'Investing Cash flow', 'Issuance (repayment) of debt', 'Issuance (buybacks) of shares', 'Dividend payments',
            'Financing Cash Flow', 'Effect of forex changes on cash', 'Net cash flow / Change in cash', 'Free Cash Flow']
profit_loss_ratios = ['Revenue Growth', 'EPS', 'EPS Diluted', 'Dividend per Share', 'Gross Margin', 'EBITDA Margin', 'EBIT Margin', 'Profit Margin',
                      'Free Cash Flow margin', 'EBITDA', 'EBIT']
# Mapping financial metrics to the financial statement
fs_index = merge_df.index.difference(['Sector'])
metrics_dict = dict.fromkeys(metrics_list, 'metrics')
for key in metrics_dict:
    if key in profit_loss:
        metrics_dict[key] = 'profit_loss'
    elif key in financial_pos:
        metrics_dict[key] = 'financial_pos'
    elif key in cashflow:
        metrics_dict[key] = 'cash_flow'
    elif key in profit_loss_ratios:
        metrics_dict[key] = 'pl_ratio'
    else:
        metrics_dict[key] = 'ratio'

fs_mapping = {'profit_loss': 0, 'financial_pos': 1, 'cash_flow': 2, 'pl_ratio': 3, 'ratio': 4}
fs_feature = np.zeros(len(fs_index))
for i, metric in enumerate(fs_index):
    for base_metric, category in metrics_dict.items():
        if base_metric in metric:
            fs_feature[i] = fs_mapping[category]

financial_statement_feature = torch.tensor(fs_feature, dtype=torch.float).unsqueeze(1)

# Year feature


# Concatenate features
node_features = torch.cat((sector_feature, company_feature, financial_statement_feature), 1)
# print('nf_shape: ', node_features.shape)  # torch.Size([577, 6893])
# print('nf: ', node_features)
# nf_df = pd.DataFrame(node_features.numpy())
# nf_df.to_csv('G:/My Drive/Dissertation/Data/financial_data/node_features_11.09.csv')

scaler = StandardScaler()  # MinMaxScaler(),RobustScaler()
scaled_features = scaler.fit_transform(node_features)
# print(scaled_features.shape)  # (577, 6893)



dataset = Data(x=node_features, edge_index=edge_index)  # creating a PyTorch Geometric Data object
# print(dataset)      # Data(x=[577, 6893], edge_index=[2, 12914])
''' 
# Check attributes of data object in the graph
print(type(dataset))               # torch_geometric.data.data.Data
print(f'Number of features: {dataset.num_features}')   # 6892
print(f'Number of nodes: {dataset.num_nodes}')    # 577
print(f'Number of edges: {dataset.num_edges}')    # 12914
print(f'Average node degree: {dataset.num_edges / dataset.num_nodes:.2f}')    # 22.38
print(f'Has isolated nodes: {dataset.has_isolated_nodes()}')    # True
print(f'Has self-loops: {dataset.has_self_loops()}')     # False
print(f'Is undirected: {dataset.is_undirected()}')    # True
'''

# Create and visualise graph
Graph = nx.Graph()
Graph.add_nodes_from(range(dataset.num_nodes))  #
edges = edge_index.t().tolist()
Graph.add_edges_from(edges)
# print(Graph)         # Graph with 577 nodes and 6457 edges
visualise_graph(Graph)
# print(Graph.nodes)
# print(Graph.adj)

# Graph after removing outliers
# Graph_refined, removed_outliers = handle_outliers(Graph, edge_index)
# print(Graph_refined)  # Graph with 519 nodes and 6419 edges  Graph with 524 nodes and 6199 edges     Graph with 519 nodes and 6418 edges
# print(removed_outliers)

# Save and access the graph in latter code
# pickle.dump(Graph_refined, open('Graph_refined.pkl', 'wb'))
stored_graph = pickle.load(open('Graph_refined.pkl', 'rb'))

# Create dataset from refined graph
for node in stored_graph.nodes():
    stored_graph.nodes[node]['feature'] = node_features[node]
# Now convert the refined graph with features back to a PyTorch Geometric Data object
refined_dataset = from_networkx(stored_graph)
# print(refined_dataset)   # Data(edge_index=[2, 12786], feature=[519, 6893], num_nodes=519)
refined_dataset.x = torch.tensor(np.array([stored_graph.nodes[node]['feature'] for node in stored_graph.nodes()]))
# print(refined_dataset)  # Data(edge_index=[2, 12846], feature=[519, 6893], num_nodes=519, x=[519, 6893])
# Warning: Creating a tensor from a list of numpy.ndarrays is extremely slow.
# Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
refined_dataset = Data(x=refined_dataset.x, edge_index=refined_dataset.edge_index)
# print(refined_dataset.x.shape)  # torch.Size([519, 6893])
# print(refined_dataset)  # Data(x=[519, 6892], edge_index=[2, 12828])

data = refined_dataset



# Check that pos_edge_label_index and neg_edge_label_index exist in train_data, val_data, and test_data
# print(train_data)
# Data(x=[519, 6893], edge_index=[2, 7704], pos_edge_label=[3852], pos_edge_label_index=[2, 3852], neg_edge_label=[3852], neg_edge_label_index=[2, 3852])
# print(train_data.x)   # tensor
# print(train_data.x.size())  # torch.Size([519, 6893])

# edge_index, edge_weight, x.size(self.node_dim),
#                          ^^^^^^^^^^^^^^^^^^^^
# TypeError: 'int' object is not callable

'''
print("Train data positive edges:", hasattr(train_data, 'pos_edge_label_index'))  
print("Train data negative edges:", hasattr(train_data, 'neg_edge_label_index'))
print("Val data positive edges:", hasattr(val_data, 'pos_edge_label_index'))
print("Val data negative edges:", hasattr(val_data, 'neg_edge_label_index'))
print("Test data positive edges:", hasattr(test_data, 'pos_edge_label_index'))
print("Test data negative edges:", hasattr(test_data, 'neg_edge_label_index'))
'''
k_folds = 5
kf = KFold(n_splits=k_folds)
fold_aucs = []
fold_losses = []

in_channels = node_features.shape[1]
hidden_channels = 32
out_channels = 4
dropout=0.2

training_epochs = []
training_losses = []
validation_losses = []
test_output = []

for i, (train_index, val_index) in enumerate(kf.split(dataset.x)):
    print(f"Training on fold {i + 1}")

    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(refined_dataset)

    model = VGAEModel(Encoder(dataset.num_features, hidden_channels, out_channels), decoder=InnerProductDecoder())
    optimizer = Adam(model.parameters(), lr=0.01)

    with open('training_loss.csv', 'a') as f:
        f.write('Epoch,Loss,Val_loss,Test_AUC\n')

        for epoch in range(1, 201):
            loss = VGAEModel.train_method(model, train_data, optimizer)
            training_epochs.append(epoch)
            training_losses.append(loss)

            val_loss = VGAEModel.validate_method(model, val_data)
            validation_losses.append(val_loss)

            # test_aucs_pred = VGAEModel.test_method(model, test_data)
            # test_output.append(test_aucs_pred)

            f.write(f'{epoch},{loss:.4f},{val_loss:.4f}\n')
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}')
            # visualise_embedding()

        val_auc = VGAEModel.test_method(model, val_data)
        val_auc = val_auc[0]
        fold_aucs.append(val_auc)
        fold_losses.append(val_loss)
        print(f"Fold {i + 1}: Validation AUC = {val_auc:.4f}, Validation Loss = {val_loss:.4f}")

# After completing all folds, calculate the average AUC and loss
avg_auc = sum(fold_aucs) / k_folds
avg_loss = sum(fold_losses) / k_folds
print(f"Average AUC over {k_folds} folds: {avg_auc:.4f}")
print(f"Average Validation Loss over {k_folds} folds: {avg_loss:.4f}")

chart_epochs = training_epochs[9:]
chart_train_loss = training_losses[9:]
chart_valid_loss = validation_losses[9:]
# visualise_losses(chart_epochs, chart_train_loss, chart_valid_loss, 'Training loss', 'Validation loss', 'Training and Validation Loss', 'training validation loss chart.png')


'''
    for epoch in range(1, 201):
        loss = VGAEModel.train_method(model, dataset, optimizer)
        training_epochs.append(epoch)
        training_losses.append(loss)

        val_loss = validate_method(model, val_data)
        validation_losses.append(val_loss)

        test_auc = VGAEModel.test_method(model, test_data)
        test_aucs.append(test_auc)

        f.write(f'{epoch},{loss:.4f},{val_loss:.4f},{test_auc:.4f}\n')
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}, Test AUC = {test_auc:.4f}')
        # visualise_embedding()
    visualise_losses(training_epochs, training_losses, validation_losses, 'Training and Validation Loss',
                     'Training and Validation Loss', 'training validation loss chart.png')

    for epoch in range(1, 201):
        loss = VGAEModel.train_method(model, dataset, optimizer)
        training_epochs.append(epoch)
        training_losses.append(loss)
        f.write(f'{epoch},{loss:.4f}\n')
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss:.4f}')
        # visualise_embedding()
    visualise_losses(training_epochs, training_losses, 'Training Loss', 'Training Loss', 'training loss chart.png')
    
    
        plt.plot(epoch, train_loss, 't', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig('training loss chart.png')
'''
'''
 plt.plot(epoch, train_loss, 't', label='Training loss')
        plt.plot(epoch, val_loss, 'v', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig('training loss chart.png')
        plt.savefig('validation loss chart.png')
'''

'''
validation = VGAEModel.validate_method(
pred = output.argmax(dim=1)
print(pred)

test = VGAEModel.test_method(

   
# Loss = reconstruction loss (reconstruct the input as close as possible, eg: mean squared error,cross-entropy loss) + regularisation loss (ensures latent space is close to a normal distribution, eg: kl divergence)
# total_loss = recon_loss + kl_loss
'''
