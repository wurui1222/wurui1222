from Bio import SeqIO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler  # 导入RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
import shap
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
import seaborn as sns
# 读取Excel文件
data = pd.read_excel(r"D:\数据集\combined_output.xlsx")  # 替换为你的文件路径

# 提取序列和标签
sequences = data['Sequence'].values  # 假设列名是 'Sequence'
labels = data['Label'].values  # 假设列名是 'Label'
# 筛选正样本和负样本
positive_samples = data[data['Label'] == 1]  # 获取正样本 (Label = 1)
negative_samples = data[data['Label'] == 0]  # 获取负样本 (Label = 0)
positive_samples_repeated = pd.concat([positive_samples] * 2, ignore_index=True)
data_balanced = pd.concat([negative_samples, positive_samples_repeated], ignore_index=True)
data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
# 定义特殊token
CLS_TOKEN = 'CLS'
SEP_TOKEN = 'SEP'
PAD_TOKEN = 'PAD'
UNK_TOKEN = 'UNK'
MASK_TOKEN = 'MASK'
# 确保生成的编码字典包含所有可能的k-mer，并避免超出嵌入层大小
def create_encoding(k):
    encoding = {}
    for i in range(4**k):
        kmer = ''.join(['ATCG'[i // (4**j) % 4] for j in range(k)])
        encoding[kmer] = i
    encoding.update({
        CLS_TOKEN: 4**k,
        SEP_TOKEN: 4**k + 1,
        PAD_TOKEN: 4**k + 2,
        UNK_TOKEN: 4**k + 3,
        MASK_TOKEN: 4**k + 4
    })
    return encoding
# 创建 k-mer 编码
def kmer_encode(seq, k=4):
    seq = seq.upper()
    kmer_tokens = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    kmer_tokens = [CLS_TOKEN] + kmer_tokens + [SEP_TOKEN]
    return kmer_tokens
def tokenize(kmer_tokens, encoding):
    return [encoding.get(token, encoding[UNK_TOKEN]) for token in kmer_tokens]
max_length = 48
k = 4 # 定义k的值
encoding = create_encoding(k)  # 创建编码字典
# 使用平衡后的数据集
sequences_balanced = data_balanced['Sequence'].values  # 使用平衡后的序列
labels_balanced = data_balanced['Label'].values        # 使用平衡后的标签
# 创建 k-mer 编码字典
tokenized_sequences = [tokenize(kmer_encode(seq, k), encoding) for seq in sequences_balanced]
tokenized_sequences = np.array(tokenized_sequences, dtype=float)
# 填充序列到固定长度，避免负值
padded_sequences = np.array([
    np.pad(seq[:max_length], (0, max(0, max_length - len(seq))), 'constant', constant_values=encoding[PAD_TOKEN]) 
    for seq in tokenized_sequences
])
y = np.array(labels_balanced, dtype=int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# 手动设置类别权重，假设负类为 1，正类为 10
class_weights = torch.tensor([1.0, 10.0], dtype=torch.float).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print("Class weights:", class_weights)

# 打印训练集大小
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# 检查 X 和 y 的形状
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# 创建PyTorch数据集和数据加载器
class DNADataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long).to(device)  # 转移到GPU
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建训练和测试数据集
train_dataset = DNADataset(X_train, y_train)
test_dataset = DNADataset(X_test, y_test)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 输出训练集和测试集的信息
print("Number of batches in train loader:", len(train_loader))
print("Number of batches in test loader:", len(test_loader))


 
# 定义PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
         
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 8, in_planes, 1, bias=False)

        self.conv1x1 = nn.Conv1d(in_planes, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        attention_weights = self.sigmoid(out)
        return attention_weights * x, attention_weights

d_model=256
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1x1 = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)
        self.conv2d = nn.Conv1d(d_model, d_model, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1x1(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)

        x = nn.Conv1d(2, d_model, kernel_size=1, bias=False).to(x.device)(x)

        x = self.conv2d(x)
        
        attention_weights = self.sigmoid(x)
        return attention_weights * x, attention_weights


class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention(in_channels=in_channel)

    def forward(self, x):
        x, ca_weights = self.ca(x)
        x, sa_weights = self.sa(x)
        return x, ca_weights, sa_weights  # 返回两种注意力权重



class TransformerModelWithLPA(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModelWithLPA, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.lpa = LPA(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(1, 0)  
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 2, 0)  
        src, ca_weights, sa_weights = self.lpa(src) 
        src = src.permute(2, 0, 1)  
        output = self.transformer_encoder(src)

        output = self.dropout(output)
        output = torch.mean(output, dim=0)  # 序列池化
      
        output = self.fc(output)
        return output, ca_weights, sa_weights  # 返回两种权重

# 初始化模型
input_dim = len(encoding)  
model = TransformerModelWithLPA(input_dim=input_dim, 
                         output_dim=2, 
                         d_model=256, 
                         nhead=8, 
                         num_layers=3, 
                         dim_feedforward=512, 
                         dropout=0.5).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score

def train(model, iterator, optimizer, criterion, epoch):
    epoch_accuracy = 0
    epoch_loss = 0
    model.train()
    all_predictions = []
    all_labels = []

    for batch_idx, batch in enumerate(iterator):
        optimizer.zero_grad()
        X, y = batch
        X, y = X.to(device), y.to(device)
        y = y.long()

                # 获取模型输出和注意力权重
        predictions, ca_weights, sa_weights = model(X)  

        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        predicted = (predictions.argmax(dim=1) == y).float()
        epoch_accuracy += predicted.sum().item()

        epoch_loss += loss.item()

        all_predictions.extend(predictions.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_loss = epoch_loss / len(iterator)

    return epoch_accuracy / len(iterator.dataset) * 100, avg_loss
def decode_kmer(indices, encoding):
    """将索引解码为k-mer字符串"""
    reverse_encoding = {v: k for k, v in encoding.items()}
    return [reverse_encoding.get(int(idx), UNK_TOKEN) for idx in indices]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib

plt.rcParams['font.family'] = 'Times New Roman'

def process_attention_weights(batch_X, transformer_attention_weights, encoding):
    """
    处理注意力权重并与k-mer对齐
    返回结构：
    [
        {
            'kmers': [kmer1, kmer2,...],
            'transformer_attention': [w1, w2,...]
        },
        ...
    ]
    """
    batch_results = []
    for i in range(batch_X.size(0)):
        # 解码原始序列
        raw_indices = batch_X[i].cpu().numpy()
        kmers = decode_kmer(raw_indices, encoding)
        
        # 获取 Transformer 注意力权重
        transformer_attention = transformer_attention_weights[i].mean(dim=0)  
        valid_length = min(len([k for k in kmers if k not in [CLS_TOKEN, SEP_TOKEN, PAD_TOKEN]]), len(kmers))
        
        batch_results.append({
            'kmers': kmers[:valid_length],
            'transformer_attention': transformer_attention[:valid_length].tolist()
        })
        
    return batch_results

def plot_attention_pattern(sample_data, epoch, sample_idx=0):
    """绘制 Transformer 注意力权重热力图"""
    data = sample_data[sample_idx]
    
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    # Transformer 注意力
    plt.subplot(1, 1, 1)
    sns.heatmap([data['transformer_attention']], annot=False, cmap="YlGnBu",
                xticklabels=data['kmers'], yticklabels=False)
    
    plt.xticks(fontsize=20, fontname='Times New Roman')
    plt.title(f"Attention - Sample {sample_idx + 1}", fontsize=20, fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig(f'attention weights_sample{sample_idx + 1}.png', dpi=300)
    plt.close()

def print_top_kmers(sample_data, top_n=5):
    """打印高注意力k-mer"""
    for idx, data in enumerate(sample_data[:6]): 
        print(f"\nSample {idx + 1} Top Attention k-mers:")  
        
        sorted_kmers = sorted(zip(data['kmers'], data['transformer_attention']), key=lambda x: x[1], reverse=True)[:top_n]
        
        for kmer, weight in sorted_kmers:
            print(f"{kmer}: {weight:.4f}")

import os
import matplotlib.pyplot as plt
import numpy as np

def combine_attention_weights(ca_weights, sa_weights):
    return ca_weights + sa_weights  

def visualize_lpa_attention(model, dataloader, device, num_samples=3, save_dir='./attention_images'):
    """
    可视化LPA模块的通道和空间注意力，并与k-mer序列对齐
    参数:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备(cpu/cuda)
        num_samples: 可视化的样本数
        save_dir: 保存图像的目录
    """
    model.eval()
    
    # 创建保存图像的目录
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= 1:
                break
                
            inputs = inputs.to(device)
            outputs, ca_weights, sa_weights = model(inputs)
            
            # 获取原始k-mer序列
            input_seqs = []
            for seq_idx in range(min(num_samples, inputs.size(0))):
                seq_tokens = []
                for token_idx in inputs[seq_idx].cpu().numpy():
                    if token_idx == encoding[PAD_TOKEN]:
                        break
                    for kmer, idx in encoding.items():
                        if idx == token_idx:
                            seq_tokens.append(kmer)
                            break
                # 合并k-mer为连续序列
                full_seq = seq_tokens[0]
                for kmer in seq_tokens[1:]:
                    full_seq += kmer[-1]
                input_seqs.append(full_seq.replace(CLS_TOKEN, "").replace(SEP_TOKEN, ""))
            
            # 转换权重到CPU numpy数组
            ca_weights = ca_weights.cpu().numpy()
            sa_weights = sa_weights.cpu().numpy()
            
            # ========== 通道注意力可视化 ==========
            plt.figure(figsize=(20, 8*num_samples))
            
            # 设置全局字体
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 14
            
            for i in range(min(num_samples, inputs.size(0))):
                current_seq = input_seqs[i]
                
                # 通道注意力可视化
                plt.subplot(num_samples, 1, i+1)
                plt.bar(range(len(ca_weights[i].squeeze())), ca_weights[i].squeeze())
                plt.title(f'Sample {i+1} Channel Attention: {current_seq[:8]}...{current_seq[-8:]}', 
                          fontsize=20, fontname='Times New Roman')
                plt.xlabel('Channel Index', fontsize=18, fontname='Times New Roman')
                plt.ylabel('Attention Weight', fontsize=18, fontname='Times New Roman')
                plt.xticks(fontsize=14, fontname='Times New Roman')
                plt.yticks(fontsize=14, fontname='Times New Roman')
                
                plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'channel_attention_batch_{batch_idx}.png'), dpi=600)
            plt.show()
                        # ========== 空间注意力可视化 ==========
            subplot_height = 8
            fig_width = 20

            total_height = subplot_height * num_samples

            plt.figure(figsize=(fig_width, total_height))

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 14

            for i in range(min(num_samples, inputs.size(0))):
                current_seq = input_seqs[i]
                seq_len = len(current_seq)
                kmer_positions = [current_seq[j:j+4] for j in range(len(current_seq)-3)]  
                
                spatial_weights = sa_weights[i].mean(axis=0)[:seq_len] 
                
                step = max(1, seq_len//15) 
                positions = range(0, seq_len, step)
                
                positions = [pos for pos in positions if pos < len(kmer_positions)]
                
                labels = [f"{pos}\n{kmer_positions[pos]}" for pos in positions]
                
                if len(positions) != len(labels):
                    diff = len(positions) - len(labels)
                    positions = positions[:len(labels)]  
                
                ax = plt.subplot(num_samples, 1, i+1)
                ax.set_position([0.1, (total_height - (i+1)*subplot_height)/total_height, 0.8, subplot_height/total_height])
                plt.plot(spatial_weights)
                plt.title(f'Sample {i+1} Spatial Attention', 
          fontsize=24, 
          fontname='Times New Roman',
          loc='left')  
                plt.ylabel('Attention Weight', fontsize=22, fontname='Times New Roman')
                
                plt.xticks(positions, labels, rotation=45, fontsize=24, fontname='Times New Roman')
                plt.yticks(fontsize=24, fontname='Times New Roman')
                
                plt.grid(True)
                peaks = (spatial_weights[1:-1] > spatial_weights[:-2]) & (spatial_weights[1:-1] > spatial_weights[2:])
                peaks = np.concatenate(([False], peaks, [False]))  
                for pos in range(1, len(spatial_weights)-1): 
                    if peaks[pos]: 
                        if pos < len(kmer_positions):  
                            plt.annotate(kmer_positions[pos], 
                                        (pos, spatial_weights[pos]),
                                        textcoords="offset points",
                                        xytext=(0, 10),
                                        ha='center',
                                        fontsize=22,  
                                        fontname='Times New Roman')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'spatial_attention_batch_{batch_idx}.png'), dpi=600)
            plt.show()
visualize_lpa_attention(model, test_loader, device, num_samples=2)




from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model, iterator, criterion, epoch):
    epoch_accuracy = 0
    epoch_loss = 0  # 用来累计每个 epoch 的测试损失
    model.eval()
    all_predictions = []
    all_labels = []
    attention_data = []
    TP = FP = TN = FN = 0
    precision_list = []
    recall_list = []

    with torch.no_grad():
        for batch in iterator:
            X, y = batch
            X, y = X.to(device), y.to(device)
            y = y.long()

            predictions, ca_weights, sa_weights = model(X)
            transformer_attention_weights = combine_attention_weights(ca_weights, sa_weights)
                        # 处理注意力权重
            batch_results = process_attention_weights(X, transformer_attention_weights, encoding)
            attention_data.extend(batch_results)
            loss = criterion(predictions, y)
            epoch_loss += loss.item()
            
            probas = torch.softmax(predictions, dim=1)[:, 1]  

            predicted = (probas > 0.3).float() 
            epoch_accuracy += (predicted == y).sum().item()
            all_predictions.extend(probas.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            TP += ((predicted == 1) & (y == 1)).sum().item()  # True Positives
            TN += ((predicted == 0) & (y == 0)).sum().item()  # True Negatives
            FP += ((predicted == 1) & (y == 0)).sum().item()  # False Positives
            FN += ((predicted == 0) & (y == 1)).sum().item()  # False Negatives
     
    if len(attention_data) > 0:
        
         for sample_idx in range(min(6, len(attention_data))): 
             plot_attention_pattern(attention_data, epoch, sample_idx=sample_idx)
        
        
         print_top_kmers(attention_data)
    # 计算 Specificity
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0  
    precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
    min_len = min(len(precision), len(recall), len(thresholds) + 1)
    precision = precision[:min_len]
    recall = recall[:min_len]

    # 记录每个阈值下的 precision, recall 
    precision_list.extend(precision)
    recall_list.extend(recall)

    # 计算 F1-score作为阈值优化标准
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0)
    fixed_threshold = 0.3  
    fixed_predicted = (np.array(all_predictions) > fixed_threshold).astype(int)
    TP_fixed = ((fixed_predicted == 1) & (np.array(all_labels) == 1)).sum()
    FP_fixed = ((fixed_predicted == 1) & (np.array(all_labels) == 0)).sum()
    FN_fixed = ((fixed_predicted == 0) & (np.array(all_labels) == 1)).sum()

    fixed_precision = TP_fixed / (TP_fixed + FP_fixed) if (TP_fixed + FP_fixed) != 0 else 0
    fixed_recall = TP_fixed / (TP_fixed + FN_fixed) if (TP_fixed + FN_fixed) != 0 else 0

    
    prc_auc = auc(recall, precision)
    auc_score = roc_auc_score(all_labels, all_predictions)

    
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, fixed_predicted)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.savefig(f'confusion_matrix_epoch_{epoch+1}.png')
    plt.close()

    # 绘制 PRC 曲线
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PRC curve (AUC = {prc_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Epoch {epoch+1}')
    plt.legend(loc="lower left")
    plt.savefig(f'prc_curve_epoch_{epoch+1}.png')
    plt.close()

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - Epoch {epoch+1}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_epoch_{epoch+1}.png')
    plt.close()

    # 在最后一轮保存 PRC 曲线的 precision 和 recall 到 CSV
    if epoch == N_EPOCHS - 1:  # 判断是否是最后一轮
        prc_data = {
            'Precision': precision_list,
            'Recall': recall_list
        }
        prc_df = pd.DataFrame(prc_data)
        prc_df.to_csv(f'prc_data_epoch_{epoch+1}.csv', index=False)

    # 返回 Test Accuracy, Precision, AUC 分数以及测试损失
    test_accuracy = (fixed_predicted == np.array(all_labels)).sum() / len(all_labels) * 100
    avg_test_loss = epoch_loss / len(iterator)  # 计算平均测试损失
    return test_accuracy, auc_score, prc_auc, fixed_precision, fixed_recall, specificity, avg_test_loss, fpr, tpr

# 训练循环
N_EPOCHS = 1
all_auc_scores = []  # 用于记录每个 epoch 的 AUC 得分
all_prc_scores = []  # 用于记录每个 epoch 的 PRC AUC 得分
all_train_losses = []  # 用于记录每个 epoch 的训练损失
all_recalls = []  # 用于记录每个 epoch 的召回率
all_precisions = []  # 用于记录每个 epoch 的精确度
all_test_losses = []  # 用于记录每个 epoch 的测试损失
all_fprs = []  # 用于记录每个 epoch 的 FPR
all_tprs = []  # 用于记录每个 epoch 的 TPR

for epoch in range(N_EPOCHS):
    # 训练
    train_accuracy, train_loss = train(model, train_loader, optimizer, criterion, epoch)
    
    # 评估
    test_accuracy, auc_score, prc_auc, best_precision, best_recall, specificity, test_loss, fpr, tpr = evaluate(model, test_loader, criterion, epoch)
    
    # 保存每个 epoch 的 AUC、PRC AUC 得分和召回率
    all_auc_scores.append(auc_score)
    all_prc_scores.append(prc_auc)
    all_train_losses.append(train_loss)
    all_recalls.append(best_recall)  # 保存召回率
    all_precisions.append(best_precision)  # 保存精确度
    all_test_losses.append(test_loss)  # 保存测试损失
    all_fprs.append(fpr)  # 保存 FPR
    all_tprs.append(tpr)  # 保存 TPR
    
    # 输出每个 epoch 的训练准确率，测试准确率，AUC，PRC AUC，精确度，损失等
    print(f'Epoch: {epoch+1:02}, Train Accuracy: {train_accuracy:.3f}, Train Loss: {train_loss:.4f}, '
      f'Test Accuracy: {test_accuracy:.3f}, Test Loss: {test_loss:.4f}, AUC Score: {auc_score:.3f}, '
      f'PRC AUC: {prc_auc:.3f}, Recall: {best_recall:.3f}, '
      f'FPR: {fpr}, TPR: {tpr}')

import matplotlib.pyplot as plt
def plot_train_metrics():
    epochs = range(1, N_EPOCHS + 1)
    
    # Plot Training Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, all_train_losses, label='Training Loss', color='b', linestyle='-', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig('train_loss.png')  # Save the figure
    plt.show()

    # Plot Training Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, all_recalls, label='Training Accuracy', color='g', linestyle='-', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig('train_accuracy.png')  # Save the figure
    plt.show()

def plot_roc_curve():
    # Assuming all_fprs and all_tprs are lists of FPRs and TPRs for each epoch
    plt.figure(figsize=(10, 6))
    
    # Plot the ROC curve for each epoch
    for epoch in range(N_EPOCHS):
        plt.plot(all_fprs[epoch], all_tprs[epoch], label=f'Epoch {epoch+1} (AUC = {all_auc_scores[epoch]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')  # Save the figure
    plt.show()

def plot_prc_curve():
    # Plot the PRC curve for the last epoch
    plt.figure(figsize=(10, 6))
    plt.plot(all_recalls, all_precisions, color='blue', lw=2, label=f'PRC curve (AUC = {all_prc_scores[-1]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('prc_curve.png')  # Save the figure
    plt.show()
plot_train_metrics()
plot_roc_curve()
plot_prc_curve()

# import pandas as pd

# # 创建一个空的DataFrame来存储每个epoch的结果
# results_df = pd.DataFrame(columns=['Epoch', 'AUC Score', 'PRC AUC', 'Recall', 'Precision', 'Specificity', 'Test Loss', 'FPR', 'TPR'])

# # 用于保存所有epoch的每个FPR和TPR的列表
# all_fpr_list = []
# all_tpr_list = []
# all_auc_list = []

# for epoch in range(N_EPOCHS):
#     # 训练
#     train_accuracy, train_loss = train(model, train_loader, optimizer, criterion, epoch)

#     # 评估
#     test_accuracy, auc_score, prc_auc, best_precision, best_recall, specificity, test_loss, fpr, tpr = evaluate(model, test_loader, criterion, epoch)

#     # 将FPR和TPR存储为字符串，方便存储到DataFrame
#     fpr_str = ', '.join(map(str, fpr))  # 将FPR列表转为字符串
#     tpr_str = ', '.join(map(str, tpr))  # 将TPR列表转为字符串

#     # 将每个epoch的结果转换为DataFrame
#     epoch_results = pd.DataFrame([{
#         'Epoch': epoch + 1,
#         'AUC Score': auc_score,
#         'PRC AUC': prc_auc,
#         'Recall': best_recall,
#         'Precision': best_precision,
#         'Specificity': specificity,
#         'Test Loss': test_loss,
#         'FPR': fpr_str,  # 存储FPR为字符串
#         'TPR': tpr_str   # 存储TPR为字符串
#     }])

#     # 使用pd.concat()连接DataFrame
#     results_df = pd.concat([results_df, epoch_results], ignore_index=True)

#     # 保存每个epoch的FPR、TPR和AUC到列表
#     all_fpr_list.append(fpr)
#     all_tpr_list.append(tpr)
#     all_auc_list.append(auc_score)

#     # 输出每个epoch的信息...
#     # 你的输出代码...

# # 找到AUROC最高的epoch
# max_auc_idx = results_df['AUC Score'].idxmax()
# max_auc_row = results_df.loc[max_auc_idx]

# # 将最高AUROC的参数保存到CSV文件中
# max_auc_row.to_csv('best_auc_params.csv', index=False)

# # 保存所有epoch的结果到CSV
# results_df.to_csv('all_epoch_results.csv', index=False)

# # 保存每个epoch的FPR、TPR以及AUC到文件
# # 将FPR, TPR, AUC的列表转换为DataFrame
# roc_data = pd.DataFrame({
#     'Epoch': range(1, N_EPOCHS + 1),
#     'FPR': all_fpr_list,
#     'TPR': all_tpr_list,
#     'AUC': all_auc_list
# })

# # 保存到CSV文件
# roc_data.to_csv('roc_curve_data.csv', index=False)









    # # 保存测试集预测和真实标签
    # df_test = pd.DataFrame({
    #     'Prediction': all_predictions, 
    #     'True Label': all_labels,
    #     'Sequence': all_sequences
    # })
    # df_test.to_csv(f'test_results_epoch_{epoch+1}.csv', index=False)
    
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
# import torch.optim as optim

# # 超参数候选值
# param_grid = {
#     'lr': [0.0001],
#     'batch_size': [16, 32, 64],
#     'num_layers': [2, 3, 4],
#     'd_model': [128, 256],
#     'nhead': [4, 8],
#     'dim_feedforward': [256, 512]
# }

# # 定义训练和评估函数
# def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device):
#     # 训练阶段
#     model.train()
#     epoch_accuracy = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         X, y = batch
#         X, y = X.to(device), y.to(device)
#         predictions = model(X)
#         loss = criterion(predictions.squeeze(), y)
#         loss.backward()
#         optimizer.step()
        
#         # 计算准确率
#         predicted = (predictions.squeeze().round() == y).float()
#         epoch_accuracy += predicted.sum().item()

#     # 评估阶段
#     model.eval()
#     all_predictions = []
#     all_labels = []
#     with torch.no_grad():
#         for batch in test_loader:
#             X, y = batch
#             X, y = X.to(device), y.to(device)
#             predictions = model(X)
#             all_predictions.extend(predictions.squeeze().round().cpu().numpy())
#             all_labels.extend(y.cpu().numpy())

#     # 计算评估指标
#     precision = precision_score(all_labels, all_predictions, average='binary')
#     recall = recall_score(all_labels, all_predictions, average='binary')
#     f1 = f1_score(all_labels, all_predictions, average='binary')
#     auc = roc_auc_score(all_labels, all_predictions)

#     return epoch_accuracy / len(train_loader.dataset) * 100, precision, recall, f1, auc
# from sklearn.model_selection import KFold

# # 创建KFold交叉验证
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # 存储每种超参数配置的结果
# results = []

# # 进行超参数搜索
# for lr in param_grid['lr']:
#     for batch_size in param_grid['batch_size']:
#         for num_layers in param_grid['num_layers']:
#             for d_model in param_grid['d_model']:
#                 for nhead in param_grid['nhead']:
#                     for dim_feedforward in param_grid['dim_feedforward']:
#                         print(f"Training with lr={lr}, batch_size={batch_size}, num_layers={num_layers}, "
#                               f"d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}")

#                         # 对于每种超参数组合，执行KFold交叉验证
#                         fold_results = []
#                         for train_index, test_index in kf.split(padded_sequences):
#                             X_train_fold, X_test_fold = padded_sequences[train_index], padded_sequences[test_index]
#                             y_train_fold, y_test_fold = y[train_index], y[test_index]

#                             # 使用RandomUnderSampler来处理不平衡数据
#                             rus = RandomUnderSampler(random_state=42)
#                             X_train_resampled, y_train_resampled = rus.fit_resample(X_train_fold, y_train_fold)

#                             # 创建数据加载器
#                             train_dataset = DNADataset(X_train_resampled, y_train_resampled)
#                             test_dataset = DNADataset(X_test_fold, y_test_fold)
#                             train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#                             test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#                             # 初始化模型
#                             model = TransformerModel(input_dim=input_dim,
#                                                      output_dim=1,
#                                                      d_model=d_model,
#                                                      nhead=nhead,
#                                                      num_layers=num_layers,
#                                                      dim_feedforward=dim_feedforward,
#                                                      dropout=0.5).to(device)

#                             # 定义损失函数和优化器
#                             optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
#                             criterion = nn.BCELoss()

#                             # 训练和评估
#                             train_acc, precision, recall, f1, auc = train_and_evaluate(
#                                 model, train_loader, test_loader, optimizer, criterion, device)

#                             fold_results.append((train_acc, precision, recall, f1, auc))

#                         # 计算每种配置的平均结果
#                         avg_train_acc = np.mean([r[0] for r in fold_results])
#                         avg_precision = np.mean([r[1] for r in fold_results])
#                         avg_recall = np.mean([r[2] for r in fold_results])
#                         avg_f1 = np.mean([r[3] for r in fold_results])
#                         avg_auc = np.mean([r[4] for r in fold_results])

#                         results.append({
#                             'lr': lr,
#                             'batch_size': batch_size,
#                             'num_layers': num_layers,
#                             'd_model': d_model,
#                             'nhead': nhead,
#                             'dim_feedforward': dim_feedforward,
#                             'avg_train_acc': avg_train_acc,
#                             'avg_precision': avg_precision,
#                             'avg_recall': avg_recall,
#                             'avg_f1': avg_f1,
#                             'avg_auc': avg_auc
#                         })

# # 找到最佳配置
# best_config = max(results, key=lambda x: x['avg_auc'])  # 根据AUC选择最佳配置
# print(f"Best config based on AUC: {best_config}")
  
     # 早停法
    #early_stopping(test_loss, model)
#if early_stopping.early_stop:
#print("Early stopping")
#break
