import os
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sympy.polys.polyconfig import query
from torch.nn import Linear, Sequential, ReLU, Sigmoid, Softmax
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, Dataset,random_split
from torch.utils.data.dataset import T_co, Subset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import joblib
from sklearn.preprocessing import RobustScaler

class DataSetVersion3(Dataset):
    def __init__(self, kmer_physico_data_path, motif_data_path):
        kmer_physico_df = pd.read_csv(kmer_physico_data_path)
        motif_df = pd.read_csv(motif_data_path)
        X_k_mer_data = kmer_physico_df.iloc[:, 165:]
        X_physical_chemical_properties_data = kmer_physico_df.iloc[:, 3:165]
        X_motif = motif_df.iloc[:, 3:]

        X_motif = np.where(X_motif == 0, np.nan, X_motif)
        X_motif = np.nan_to_num(X_motif, nan=np.nanmean(X_motif, axis=0))

        if np.isnan(X_motif).any():
            print("Warning: X_motif still contains NaN values after processing.")
            print(f"Number of NaN values in X_motif: {np.isnan(X_motif).sum()}")
            X_motif = np.nan_to_num(X_motif, nan=0.0)

        scaler_kmer = StandardScaler()
        scaler_physico = StandardScaler()
        scaler_motif = RobustScaler()
        X_k_mer_data = scaler_kmer.fit_transform(X_k_mer_data)
        X_physical_chemical_properties_data = scaler_physico.fit_transform(X_physical_chemical_properties_data)
        X_motif = scaler_motif.fit_transform(X_motif)

        joblib.dump(scaler_kmer, 'scaler_kmer.pkl')
        joblib.dump(scaler_physico,
                    'scaler_physico.pkl')
        joblib.dump(scaler_motif, 'scaler_motif.pkl')

        labels = motif_df.iloc[:, 2]  

        self.X_k_mer_data = torch.from_numpy(X_k_mer_data).float()  
        self.X_physical_chemical_properties_data = torch.from_numpy(X_physical_chemical_properties_data).float()
        self.X_motif = torch.from_numpy(X_motif).float()
        self.labels = torch.from_numpy(labels.values).long()  # 确保标签为 long 类型
        self.len = X_k_mer_data.shape[0]

    def __getitem__(self, index):
        return self.X_k_mer_data[index], self.X_physical_chemical_properties_data[index], self.X_motif[index], self.labels[index]

    def __len__(self):
        return self.len

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
        return attn_output, attn_weights

class FeatureNetWithAttention(nn.Module):
    def __init__(self, input_dim):
        super(FeatureNetWithAttention, self).__init__()
        self.self_attention = MultiheadAttentionLayer(input_dim, num_heads=1)
        self.fc2 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        x,attn_self_metrics = self.self_attention(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x,attn_self_metrics

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
        kmer_out,kmer_self_attn = self.kmer_net(kmer_x)
        physico_out,physico_self_attn = self.physico_net(physico_x)
        motif_out,motif_self_attn = self.motif_net(motif_x)

        kp_out,attn_weights_kp = self.cross_attention_kp(kmer_out, physico_out)
        kpm_out,attn_weights_kpm = self.cross_attention_kpm(kp_out.squeeze(2),motif_out)

        out = self.fc_kpm(kpm_out.squeeze(2))
        out = self.relu(out)
        out = self.dropout(out)  
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out,(kmer_self_attn,physico_self_attn,motif_self_attn,attn_weights_kp, attn_weights_kpm)

def train_model(model, criterion, optimizer, train_loader, test_loader, n_epochs,device,sample_num,folder_path):
    best_val_f1 = 0.0
    best_metrics = {
        'accuracy': 0.0,
        'specificity': 0.0,
        'sensitivity': 0.0,
        'f1_score': 0.0,
        'mcc': 0.0,
        'auc': 0.0
    }

    best_fpr = None
    best_tpr = None
    best_auc = 0.0

    train_losses = []
    val_accuracies = []
    val_specificitys = []
    val_sensitivitys = []
    val_f1_scores = []
    val_mccs = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for kmer_data, physico_data, motif_data, labels in train_loader:
            kmer_data, physico_data, motif_data, labels = kmer_data.to(device), physico_data.to(device), motif_data.to(
                device), labels.to(device)
            outputs,attn_metrics = model(kmer_data, physico_data, motif_data)
            batch_losses = criterion(outputs, labels)
            optimizer.zero_grad()
            batch_losses.backward()
            optimizer.step() 
            running_loss += batch_losses.item()  
        train_losses.append(running_loss / sample_num)  

        val_accuracy, val_specificity, val_sensitivity, val_f1, val_mcc,val_auc, fpr, tpr = evaluate_model(model, test_loader, device)
        val_accuracies.append(val_accuracy)
        val_specificitys.append(val_specificity)
        val_sensitivitys.append(val_sensitivity)
        val_f1_scores.append(val_f1)
        val_mccs.append(val_mcc)
        if val_f1 > best_val_f1 or (val_f1 == 0.0 and best_val_f1 == 0.0): 

            best_val_f1 = val_f1
           
            best_metrics = {
                'accuracy': val_accuracy,
                'specificity': val_specificity,
                'sensitivity': val_sensitivity,
                'f1_score': val_f1,
                'mcc': val_mcc,
                'auc': val_auc,
            }
            best_fpr = fpr
            best_tpr = tpr
            best_auc = val_auc

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path = f"{folder_path}/attn_dropout_kfold_model_{current_time}.pt"
    torch.save(model, model_path)
    metrics_path = f"{folder_path}/attn_dropout_kfold_metrics_{current_time}.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {best_metrics['accuracy']}\n")
        f.write(f"Specificity: {best_metrics['specificity']}\n")
        f.write(f"Sensitivity: {best_metrics['sensitivity']}\n")
        f.write(f"F1 Score: {best_metrics['f1_score']}\n")
        f.write(f"MCC: {best_metrics['mcc']}\n")
        f.write(f"AUC: {best_metrics['auc']}\n")
    return train_losses, val_accuracies, val_specificitys, val_sensitivitys, val_f1_scores,val_mccs,(best_fpr, best_tpr, best_auc)

def evaluate_model(model, data_loader,device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for kmer_data, physico_data, motif_data, labels in data_loader:
            kmer_data, physico_data, motif_data, labels = kmer_data.to(device), physico_data.to(device), motif_data.to(
                device), labels.to(device)
            outputs,attn_metrics = model(kmer_data, physico_data, motif_data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            all_preds.extend(preds) 
            all_labels.extend(labels.cpu().numpy()) 
            all_probs.extend(probs[:, 1].cpu().numpy()) 
    tn,fp,fn,tp = metrics.confusion_matrix(all_labels, all_preds).ravel()

    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    fpr,tpr,thresholds = metrics.roc_curve(all_labels, all_probs)
    roc_auc = metrics.auc(fpr, tpr)
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity  = recall_score(all_labels, all_preds,zero_division=0)
    f1 = f1_score(all_labels, all_preds,zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return accuracy, specificity, sensitivity, f1, mcc,roc_auc, fpr,tpr


def plot_results_kfold_with_roc(all_train_losses, all_val_accuracies, all_val_specificitys,
            all_val_sensitivitys, all_val_f1_scores,all_val_mccs,all_val_rocs):

    k_folds = len(all_train_losses)
    epochs = range(1, len(all_train_losses[0]) + 1)
    plt.figure(figsize=(24, 18))

    # Training Loss
    plt.subplot(3, 2, 1)
    for fold in range(k_folds):
        plt.plot(epochs, all_train_losses[fold], label=f"Fold {fold + 1}")
    plt.title("Training Loss", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.ylim(0, 0.1)
    plt.legend(fontsize=10)


    # Validation Accuracy
    plt.subplot(3, 2, 2)
    for fold in range(k_folds):
        plt.plot(epochs, all_val_accuracies[fold], label=f"Fold {fold + 1}")
    plt.title("Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10)

    # Validation Specificity
    plt.subplot(3, 2, 3)
    for fold in range(k_folds):
        plt.plot(epochs, all_val_specificitys[fold], label=f"Fold {fold + 1}")
        plt.title("Validation Specificity", fontsize=14)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Specificity", fontsize=12)
        plt.legend(fontsize=10)

    # Validation Recall
    plt.subplot(3, 2, 4)
    for fold in range(k_folds):
        plt.plot(epochs, all_val_sensitivitys[fold], label=f"Fold {fold + 1}")
    plt.title("Validation Sensitivitys", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Sensitivity", fontsize=12)
    plt.legend(fontsize=10)

    # Validation F1 Score
    plt.subplot(3, 2, 5)
    for fold in range(k_folds):
        plt.plot(epochs, all_val_f1_scores[fold], label=f"Fold {fold + 1}")
    plt.title("Validation F1 Score", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.legend(fontsize=10)

    plt.subplot(3, 2, 6)
    for fold in range(k_folds):

        fpr, tpr, auc_value = all_val_rocs[fold]  # 直接解包每个 fold 的最佳 ROC 数据
        plt.plot(fpr, tpr, label=f"Fold {fold + 1}(AUC={auc_value:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label="Random Guess")
    plt.title("ROC Curve", fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(folder_path, f"attn_dropout_kfold_{current_time}.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")

    plt.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 5000  
    batch_size = 32 
    all_train_losses = []
    all_val_accuracies = []
    all_val_specificitys = []
    all_val_sensitivitys = []
    all_val_f1_scores = []
    all_val_mccs = []
    all_val_rocs = []

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = f'train_model_{current_time}'
    if os.path.exists(folder_path):
        print("文件夹已存在...")
    else:
        os.mkdir(folder_path)


    data_version3 = DataSetVersion3(kmer_physico_data_path="updated_PHP_kmer_protein_features.csv",
                                    motif_data_path="updated_PHP_motif_feature.csv")

    kmer_input_dim = 512
    physico_input_dim = 162
    motif_input_dim = 1280


    kf = KFold(n_splits=5,shuffle=True,random_state=42)
    for train_idx,test_idx in kf.split(data_version3):
        train_subset = Subset(data_version3, train_idx)
        test_subset = Subset(data_version3, test_idx)

        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kmer_net = FeatureNetWithAttention(input_dim=kmer_input_dim).to(device)
        physico_net = FeatureNetWithAttention(input_dim=physico_input_dim).to(device)
        motif_net = FeatureNetWithAttention(input_dim=motif_input_dim).to(device)
        combined_net = CombinedNetWithCrossAttention(kmer_net, physico_net, motif_net, combined_hidden_dim=128).to(device)


        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(combined_net.parameters(), lr=0.001)

        train_losses, val_accuracies, val_specificitys, val_sensitivitys, val_f1_scores, val_mccs, best_roc = train_model(
            combined_net, criterion, optimizer, train_loader, test_loader, n_epochs, device, 672, folder_path
        )
        all_train_losses.append(train_losses)
        all_val_accuracies.append(val_accuracies)
        all_val_specificitys.append(val_specificitys)
        all_val_sensitivitys.append(val_sensitivitys)
        all_val_f1_scores.append(val_f1_scores)
        all_val_mccs.append(val_mccs)
        all_val_rocs.append(best_roc)

        plot_results_kfold_with_roc(
            all_train_losses, all_val_accuracies, all_val_specificitys,
            all_val_sensitivitys, all_val_f1_scores, all_val_mccs, all_val_rocs
        )