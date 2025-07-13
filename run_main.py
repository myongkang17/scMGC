import pandas as pd
import scanpy as sc
import numpy as np
import torch
import torch.nn as nn
import argparse
from graph_function import get_adj, knn_graph
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from post_clustering import acc, nmi, DI_calcu, JC_calcu
from MAE_Contrastive import MAE_Contrastive
from process_data import read_dataset, normalize_data, geneSelection
import h5py
from loss_function import total_loss, mae_loss, contrastive_loss
import random
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

class scMGC(nn.Module):
    def __init__(self, args, in_dim, z_emb_size, dropout_rate, hidden_dims):
        super(scMGC, self).__init__()
        self.args = args
        self.mae = MAE_Contrastive(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            z_dim=z_emb_size,
            out_dim=in_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, x1, x2, adj):
        # Generate two views with different masks
        z1, x_hat1, mask1 = self.mae(x1, adj)
        z2, x_hat2, mask2 = self.mae(x2, adj)

        z_final = z1

        return z_final, z1, z2, x_hat1, x_hat2, mask1, mask2


##Set random seeds
def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args, device, x_scRNA, adj, n_clusters):
    # Define model architecture
    hidden_dims = [2048, 256, 64]
    model = scMGC(
        args=args,
        in_dim=x_scRNA.shape[1],
        z_emb_size=16,
        dropout_rate=0.1,
        hidden_dims=hidden_dims
    )
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-5)
    train_loss_list = []

    best_loss = float('inf')
    best_model_state = None

    for epoch in range(args.epoch):
        # Generate two views with different masks
        x1 = x_scRNA
        x2 = x_scRNA.clone()  # Create second view

        # Forward pass
        _, z1, z2, x_hat1, x_hat2, mask1, mask2 = model(x1, x2, adj)

        # Compute loss
        loss = total_loss(
            z1=z1,
            z2=z2,
            x_hat1=x_hat1,
            x_hat2=x_hat2,
            x1=x1,
            x2=x2,
            mask1=mask1,
            mask2=mask2,
            alpha= 0.9 # Weight for reconstruction vs contrastive loss
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())
        print(f"epoch {epoch} loss={loss.item():.4f}")

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

        # Check convergence
        if len(train_loss_list) >= 2:
            if abs(train_loss_list[-1] - train_loss_list[-2]) / train_loss_list[-2] < 1e-5:
                print("converged!!!")
                print(epoch)
                break

    # Save best model
    torch.save(best_model_state, 'best_model.pth')
    return best_model_state


def inference(model, x_scRNA, adj, n_clusters, label_groundTruth):
    # Set model to eval mode
    model.eval()

    with torch.no_grad():
        # Generate two views with different masks
        x1 = x_scRNA
        x2 = x_scRNA.clone()

        # Forward pass
        z_final, _, _, _, _, _, _ = model(x1, x2, adj)

        # Get final embeddings
        emb = z_final.data.cpu().numpy()






        # Clustering
        print('====== Do clustering on embedding output')
        print(emb.shape)
        kmeans = KMeans(n_clusters=n_clusters, random_state=100)
        label_pred = kmeans.fit_predict(emb)



        # Evaluation
        print('dataset=%s,ACC=%.4f, NMI=%.4f, ARI=%.4f' %
              (args.dataname,
               acc(label_groundTruth, label_pred),
               nmi(label_groundTruth, label_pred),
               adjusted_rand_score(label_groundTruth, label_pred)))




def load_data(args):
    data_h5 = h5py.File(args.dataname)
    x = np.array(data_h5['X']).astype(np.float32)
    y = np.array(data_h5['Y'])

    print("标签数据类型:", y.dtype)

    importantGenes = geneSelection(x, n=3000, plot=False)
    x = x[:, importantGenes]

    adata = sc.AnnData(x)

    adata.obs['Group'] = y

    # ================= load sc data
    adata = read_dataset(adata, transpose=False, test_split=False, copy=True)

    if (args.type == 'TPM'):
        x1 = normalize_data(adata, size_factors=False, normalize_input=True, logtrans_input=True)
    else:
        x1 = normalize_data(adata, size_factors=True, normalize_input=True, logtrans_input=True)

    x_scRNA = x1.X.astype(np.float32)
    x_scRNAraw = x1.raw.X.astype(np.float32)
    x_scRNA_size_factor = x1.obs['size_factors'].values.astype(np.float32)

    # ================= compute and save adj
    graph_dict = get_adj(x_scRNA, k=30, pca=260)
    adj = graph_dict['adj']

    label_groundTruth = y
    Y = label_groundTruth.astype(np.float32)
    Y = Y.squeeze()
    n_clusters = len(set(Y))

    return x_scRNA, adj, n_clusters, label_groundTruth





if __name__ == "__main__":
    same_seeds(2023)

    parser = argparse.ArgumentParser(description='scMGC')
    parser.add_argument("--dataname", default=r"D:\PythonProject\datasets\human_kidney.h5", type=str)
    parser.add_argument('--epoch', type=int, default=60, help='Number of epochs to train scMGC of scRNA_seq.')
    parser.add_argument('--beta', type=float, default=0.001,
                        help='weight to determine the importance of topological structure')
    parser.add_argument('--training_scMGC', type=bool, default=True, help='Training scMGC')
    parser.add_argument('--k', type=bool, default=30, help='k')
    parser.add_argument('--type', type=str, default="count", help='Type of the data')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load data
    x_scRNA, adj, n_clusters, label_groundTruth = load_data(args)

    # Move data to device
    adj = adj.to(device)
    x_scRNA = torch.from_numpy(x_scRNA).to(device)
    if args.training_scMGC:
        # Train model
        best_model_state = train(args, device, x_scRNA, adj, n_clusters)
        # Load best model for inference
        model = scMGC(
            args=args,
            in_dim=x_scRNA.shape[1],
            z_emb_size=16,
            dropout_rate=0.1,
            hidden_dims=[2048, 256, 64]
        )
        model.load_state_dict(best_model_state)
        model.to(device)

        # Perform inference
        inference(model, x_scRNA, adj, n_clusters, label_groundTruth)
