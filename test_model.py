import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from torch import nn
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score

# model architecture
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiheadAttentionLayer, self).__init__()

        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)

        self.multihead_attn = nn.MultiheadAttention(1, num_heads,batch_first=True)

    def forward(self, input):
        query = self.query_proj(input)
        key = self.key_proj(input)
        value = self.value_proj(input)
        query = query.unsqueeze(2)
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = attn_output.squeeze(2)
        return attn_output

class FeatureNetWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureNetWithAttention, self).__init__()
        self.self_attention = MultiheadAttentionLayer(input_dim, num_heads=1)
        self.fc2 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super(CrossAttention,self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,batch_first=True)

    def forward(self, x, y):
        x = x.unsqueeze(2)
        y = y.unsqueeze(2)
        cross_attention_output, attention_weights = self.cross_attention(query=x,key=y,value=y)

        return cross_attention_output, attention_weights

class CombinedNetWithCrossAttention(nn.Module):
    def __init__(self, kmer_net, physico_net, motif_net, combined_hidden_dim):
        super(CombinedNetWithCrossAttention, self).__init__()
        self.kmer_net = kmer_net
        self.physico_net = physico_net
        self.motif_net = motif_net

        self.cross_attention_kp = CrossAttention(1, num_heads=1)
        self.fc_cross_attention_kp = nn.Linear(128,128)

        self.cross_attention_kpm = CrossAttention(1,num_heads=1)
        self.fc_kpm = nn.Linear(128, combined_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(combined_hidden_dim, combined_hidden_dim//4)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(combined_hidden_dim//4 , 2)

    def forward(self, kmer_x, physico_x, motif_x):
        kmer_out = self.kmer_net(kmer_x)
        physico_out = self.physico_net(physico_x)
        motif_out = self.motif_net(motif_x)

        kp_out,attn_weights_kp = self.cross_attention_kp(kmer_out, physico_out)
        kpm_out,attn_weights_kpm = self.cross_attention_kpm(kp_out.squeeze(2),motif_out)
        out = self.fc_kpm(kpm_out.squeeze(2))
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out,(attn_weights_kp, attn_weights_kpm)


def calculate_entropy(prob_distribution):
    return -np.sum(prob_distribution * np.log2(prob_distribution + 1e-12))

kmer_df = pd.read_csv('kmer_feature.csv')
physico_df = pd.read_csv('protein_feature.csv')
motif_df = pd.read_csv('motif_feature.csv')
X_k_mer_data = pd.concat([kmer_df.iloc[:,2:258],kmer_df.iloc[:,259:515]],axis=1)
X_physical_chemical_properties_data = physico_df.iloc[:,:-1]
X_motif = motif_df.iloc[:,3:]
labels = motif_df.iloc[:, 2]

X_motif = np.where(X_motif == 0, np.nan, X_motif)
X_motif = np.nan_to_num(X_motif, nan=np.nanmean(X_motif, axis=0))

if np.isnan(X_motif).any():
    print("Warning: X_motif still contains NaN values after processing.")
    print(f"Number of NaN values in X_motif: {np.isnan(X_motif).sum()}")
    X_motif = np.nan_to_num(X_motif, nan=0.0)

labels = labels.values

scaler_kmer = joblib.load('scaler_kmer.pkl')
scaler_physico = joblib.load('scaler_physico.pkl')
scaler_motif = joblib.load('scaler_motif.pkl')

X_kmer = scaler_kmer.transform(X_k_mer_data)
X_physico = scaler_physico.transform(X_physical_chemical_properties_data)
X_motif = scaler_motif.transform(X_motif)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kmer_tensor = torch.from_numpy(X_kmer).float().to(device)
physico_tensor = torch.from_numpy(X_physico).float().to(device)
motif_tensor = torch.from_numpy(X_motif).float().to(device)

model_path = "model.pt"
net = torch.load(model_path)
net = net.to(device)
net.eval()

with torch.no_grad():
    outputs,attn_metric = net(kmer_tensor, physico_tensor, motif_tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()

entropies = [calculate_entropy(prob) for prob in probs]

results = []
for idx in range(len(probs)):
    prob_0 = probs[idx, 0]
    prob_1 = probs[idx, 1]
    results.append({
        'Sample_Index': idx,
        'True_Label': labels[idx],
        'Prob_0': prob_0,
        'Prob_1': prob_1,
        'Entropy': entropies[idx]
    })
df = pd.DataFrame(results)
df = df[['Sample_Index', 'True_Label', 'Prob_0', 'Prob_1', 'Entropy']]

df.to_csv("test_result.csv", index=False)