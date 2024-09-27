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
import functools as ft
import pickle

import torch

os.environ['TORCH'] = torch.__version__
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score 


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
            auc = self.vgae.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
            print(auc)
        return auc


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
        file.drop(index=rows_to_drop, inplace=True)
        file.reset_index(drop=True, inplace=True)
        print(file.index)
        print(file.shape)
        # file.describe()
        output_file_name = f"processed_{file_name}"
        output_file_path = os.path.join(folder_path, output_file_name)
        file.to_csv(output_file_path, index=False)
        print(f"Processed and saved: {output_file_name}")


# Function to visualise graph
def visualise_graph(G):
    plt.figure(figsize=(8,8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, node_color="purple", alpha=0.7, edge_cmap=plt.cm.Blues)
    plt.title("Graph Visualization")
    plt.savefig("graph_figure.png")  # Save the plot as an image file
    plt.close()


# Function to handle graph outliers
def handle_outliers(G, n_neighbors=20, contamination=0.1, min_betweenness=0.01):
    betweenness = nx.betweenness_centrality(G)
    central_nodes = {node for node, centrality in betweenness.items() if centrality >= min_betweenness}
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    # node2vec = Node2Vec(edge_index, embedding_dim=64, walk_length=30, context_size=10, walks_per_node=2)   # if using PyG version
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    emb_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = np.array([emb_model.wv[str(node)] for node in G.nodes()])
    outliers = lof.fit_predict(embeddings)
    outlier_nodes = {node for node, outlier_flag in zip(G.nodes(), outliers) if outlier_flag == -1}
    emb_model.wv.save_word2vec_format('embeddings')  # save embeddings
    emb_model.save('embedded_model')  # save model

    # Remove the outlier nodes from the graph
    nodes_to_remove = outlier_nodes - central_nodes
    G.remove_nodes_from(nodes_to_remove)

    # Visualize the resulting graph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=50, node_color="orange", alpha=0.7, edge_cmap=plt.cm.Blues)
    plt.title('Graph after removing Outliers')
    # plt.show()
    plt.savefig("graph_after_outliers.png")
    plt.close()

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
datalink = 'https://www.kaggle.com/datasets/cnic92/200-financial-indicators-of-us-stocks-20142018/data'
raw_data_path = 'G:/My Drive/Dissertation/Data/financial_data/raw_data/'
folder_path = 'G:/My Drive/Dissertation/Data/financial_data/'
files = os.listdir(raw_data_path)
clean_data(files, raw_data_path, folder_path)

# Reading in the processed files
df_2014 = (pd.read_csv('C:/Users/richard/Desktop/LK/TECH7009 - Dissertation/Data/financial_data/processed_2014_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2014', axis=1)
df_2015 = (pd.read_csv('C:/Users/richard/Desktop/LK/TECH7009 - Dissertation/Data/financial_data/processed_2015_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2015', axis=1)
df_2016 = (pd.read_csv('C:/Users/richard/Desktop/LK/TECH7009 - Dissertation/Data/financial_data/processed_2016_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2016', axis=1)
df_2017 = (pd.read_csv('C:/Users/richard/Desktop/LK/TECH7009 - Dissertation/Data/financial_data/processed_2017_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2017', axis=1)
df_2018 = (pd.read_csv('C:/Users/richard/Desktop/LK/TECH7009 - Dissertation/Data/financial_data/processed_2018_Financial_Data.csv')).set_index(
    ['Company', 'Sector']).add_suffix('_2018', axis=1)

# Creating the merged data frame 
dfs = [df_2014, df_2015, df_2016, df_2017, df_2018]
full_merge = ft.reduce(lambda left, right: pd.merge(left, right, on=['Company', 'Sector'], how='inner'), dfs)
# print(full_merge.shape) # (3415, 959)
full_merge = full_merge.reset_index()
full_merge['Sector'] = full_merge['Sector'].astype('category').cat.codes
full_merge = full_merge.set_index('Company')

merged_df = full_merge.transpose()

# Table to Graph section
# Convert df to tensor
df_tensor = torch.from_numpy(merged_df.values)

# Nodes from financial metrics
node_identifiers = merged_df.index.difference(['Sector']).to_list()

# Correlation matrix
corr_df = merged_df.transpose()
corr_matrix = corr_df.corr()

# Edge index
threshold = 0.5
edge_index = []
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Node features
sector_feature = (torch.tensor(merged_df.loc['Sector'], dtype=torch.float).view(1, -1).
                  repeat(len(node_identifiers), 1))  # see warning
company_feature = (torch.tensor(pd.factorize(merged_df.columns)[0], dtype=torch.float).view(1, -1).
                   repeat(len(node_identifiers), 1))

# Financial statement feature (assigning financial statement tags to financial metrics)
metrics_list = df_2014.columns.str.removesuffix('_2014').to_list()   # Extract list of financial metrics using df_2014
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
fs_index = merged_df.index.difference(['Sector'])
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

# Concatenate features
node_features = torch.cat((sector_feature, company_feature, financial_statement_feature), 1)
scaler = MinMaxScaler()  # MinMaxScaler(),RobustScaler(), StandardScaler()
scaled_features = scaler.fit_transform(node_features)

dataset = Data(x=node_features, edge_index=edge_index)  # creating a PyTorch Geometric Data object
# Data(x=[959, 6831], edge_index=[2, 34446])

# Create and visualise graph
Graph = nx.Graph()
Graph.add_nodes_from(range(dataset.num_nodes))  #
edges = edge_index.t().tolist()
Graph.add_edges_from(edges)
visualise_graph(Graph)

# Graph after removing outliers
Graph_refined, removed_outliers = handle_outliers(Graph)

# Create dataset from refined graph
for node in Graph_refined.nodes():
    Graph_refined.nodes[node]['feature'] = node_features[node]
refined_dataset = from_networkx(Graph_refined)
refined_dataset.x = torch.tensor(np.array([Graph_refined.nodes[node]['feature'] for node in Graph_refined.nodes()]))
refined_dataset = Data(x=refined_dataset.x, edge_index=refined_dataset.edge_index)

data = refined_dataset

# Define training parameters
kfolds = 5
kf = KFold(n_splits=kfolds)
fold_aucs = []
fold_losses = []
fold_test_auc = []

in_channels = node_features.shape[1]
hidden_channels = 16
out_channels = 1
dropout=0.2

training_epochs = []
training_losses = []
validation_losses = []

for i, (train_index, val_index) in enumerate(kf.split(data.x)):
    print(f"Training on fold {i + 1}")

    transform = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, split_labels=True)
    train_data, val_data, test_data = transform(data)

    model = VGAEModel(Encoder(dataset.num_features, hidden_channels, out_channels, dropout), decoder=InnerProductDecoder())
    optimizer = Adam(model.parameters(), lr=0.01)

    with open('training_loss.csv', 'a') as f:
        f.write('Epoch,Loss,Val_loss,Test_AUC\n')

        for epoch in range(1, 201):
            loss = VGAEModel.train_method(model, train_data, optimizer)
            training_epochs.append(epoch)
            training_losses.append(loss)

            val_loss = VGAEModel.validate_method(model, val_data)
            validation_losses.append(val_loss)

            f.write(f'{epoch},{loss:.4f},{val_loss:.4f}\n')
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}')

        val_auc = VGAEModel.test_method(model, val_data)
        val_auc = val_auc[0]
        test_auc = VGAEModel.test_method(model, test_data)
        test_auc = test_auc[0]
        fold_aucs.append(val_auc)
        fold_losses.append(val_loss)
        fold_test_auc.append(test_auc)
        print(f"Fold {i + 1}: Validation AUC = {val_auc:.4f}, Validation Loss = {val_loss:.4f}, Test AUC = {test_auc:.4f}")
        VGAEModel.embeddings_visual(model, data)

        torch.save(VGAEModel.state_dict(), 'VGAEModel.pth')
        print('Model saved')

# Calculate the average AUC and loss for all folds
avg_auc = sum(fold_aucs) / kfolds
avg_loss = sum(fold_losses) / kfolds
avg_test_auc = sum(fold_test_auc) / kfolds
print(f"Average Validation AUC over {kfolds} folds: {avg_auc:.4f}")
print(f"Average Validation Loss over {kfolds} folds: {avg_loss:.4f}")
print(f"Average Test AUC over {kfolds} folds: {avg_test_auc:.4f}")

chart_epochs = training_epochs[9:]
chart_train_loss = training_losses[9:]
chart_valid_loss = validation_losses[9:]
visualise_losses(chart_epochs, chart_train_loss, chart_valid_loss, 'Training loss', 'Validation loss', 'Training and Validation Loss', 'training validation loss chart.png')
# torch.save(model, 'VGAEModel_complete.pth')
