import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pandas as pd
import logging
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from joblib import dump, load
import math
import pickle
import uuid
import json
from datetime import datetime

# ==============================================================================
# 参数配置
# ==============================================================================
# 'SOFTMATCH','FLEXMATCH','FREEMATCH','ESMatch'
METHODS_TO_RUN = ['SOFTMATCH','FLEXMATCH','FREEMATCH','ESMATCH']
DATASETS_TO_RUN = ['qic']
TARGET_TOTAL_SIZE = 5000          # 总数据量（训练+验证+测试）
TEST_SET_SIZE = 1000              # 测试集大小
VALIDATION_SIZE = 250             # 验证集大小
TRAIN_SET_SIZE = 3750             # 训练集大小 
LABELED_CONFIGS =[500]
NUM_RUNS = 3

# 模型与特征
MLP_HIDDEN_LAYERS = (256, 64)
PCA_N_COMPONENTS = 512
MLP_DROPOUT_RATE = 0.5

# 训练超参数
UNSUPERVISED_LOSS_WEIGHT = 1 
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 50
LOGITS_RAMPUP_EPOCHS = 0   
MAX_TRAINING_EPOCHS = 2000

SAVE_WEIGHT_SNAPSHOTS = True
WEIGHT_SNAPSHOT_INTERVAL = 50

# FlexMatch参数
FLEXMATCH_EMA_ALPHA = 0.999 
FLEXMATCH_CONFIDENCE_THRESHOLD = 0.95

# SoftMatch 参数 
SOFTMATCH_EMA_ALPHA = 0.999

# FreeMatch参数
FREEMATCH_EMA_ALPHA = 0.999

# ESMatch 参数 
ESMATCH_EMA_ALPHA = 0.999
ESMATCH_DIST_REG_LOSS_WEIGHT = 1

# ==============================================================================
# Ablation 控制开关 (消融实验专用)
# 默认全为 False 时，运行的是完整版 ESM-DA (Entropy-Guided Soft Margin + DA)
# ==============================================================================
# 1. 验证柔性掩码的保护作用
# 设为 True 时，去掉 Sigmoid 软化，退化为传统一刀切的硬截断掩码 (Hard Truncation)。
ABLATION_WO_SOFT = False

# 2. 验证高维特征分布对齐的抗长尾能力
# 设为 True 时，关闭 KL 散度正则项，退化为纯伪标签学习。
ABLATION_WO_DA = False

# 早停配置
USE_EARLY_STOPPING = True
ES_PATIENCE = 200
ES_ACC_MIN_DELTA = 0.01
ES_LOSS_MIN_DELTA = 0.01

# 文件路径
WEIGHTS_DIR = "weights"
BEST_MODEL_PATH = os.path.join(WEIGHTS_DIR, f"best_model_weights_{uuid.uuid4().hex[:8]}.pth")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

_FEATURE_LABEL_MEM_CACHE = {}
_PCA_MODEL_MEM_CACHE = {}
_PCA_REDUCED_MEM_CACHE = {}

def _cache_key(*parts):
    return "|".join(str(p) for p in parts)

def _atomic_save_numpy(path, array):
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "wb") as f:
        np.save(f, array)
    os.replace(tmp_path, path)

def _atomic_dump_pickle(path, obj):
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)

def _atomic_dump_joblib(path, obj):
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
    dump(obj, tmp_path)
    os.replace(tmp_path, path)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================== 数据加载与特征提取 ==============================
def load_data(dataset):
    """
    统一数据划分策略：
      - 总数据量: TARGET_TOTAL_SIZE = 5000
      - 测试集:   TEST_SET_SIZE    = 1000
      - 验证集:   VALIDATION_SIZE  = 250
      - 训练集:   TRAIN_SET_SIZE   = 3750  (= 5000 - 1000 - 250)
    QIC 数据集没有官方测试集，从训练集中统一划分测试集和验证集。
    """
    base_dir = os.path.join("data", dataset.lower())
    
    if dataset.lower() == 'sogou':
        train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'), names=['label', 'title', 'content'], encoding='utf-8')
        test_df  = pd.read_csv(os.path.join(base_dir, 'test.csv'),  names=['label', 'title', 'content'], encoding='utf-8')
        
        train_df['text']  = train_df['title'].fillna('') + " " + train_df['content'].fillna('')
        train_df['label'] = train_df['label'].astype(int) - 1
        test_df['text']   = test_df['title'].fillna('') + " " + test_df['content'].fillna('')
        test_df['label']  = test_df['label'].astype(int) - 1
        num_classes = 5
        
        # 从训练文件中取出 TRAIN_SET_SIZE + VALIDATION_SIZE 条
        need = TRAIN_SET_SIZE + VALIDATION_SIZE
        if len(train_df) > need:
            _, train_df = train_test_split(train_df, test_size=need, random_state=42, stratify=train_df['label'])
        train_df, val_df = train_test_split(train_df, test_size=VALIDATION_SIZE, random_state=42, stratify=train_df['label'])
        
        # 从测试文件中取 TEST_SET_SIZE 条
        if len(test_df) > TEST_SET_SIZE:
            _, test_df = train_test_split(test_df, test_size=TEST_SET_SIZE, random_state=42, stratify=test_df['label'])
        
    elif dataset.lower() == 'qic':
        # 官方提供的 QIC 测试集没有标注 (label)，因此测试集从训练集划分；验证集仍从官方 dev 文件读取
        qic_label_map = {
            "治疗方案": 0, "病情诊断": 1, "指标解读": 2, "就医建议": 3,
            "疾病表述": 4, "病因分析": 5, "注意事项": 6, "后果表述": 7,  
            "医疗费用": 8, "其他": 9, "功效作用": 10
        }
        train_source_df = pd.read_json(os.path.join(base_dir, 'KUAKE-QIC_train.json'), encoding='utf-8')
        val_df          = pd.read_json(os.path.join(base_dir, 'KUAKE-QIC_dev.json'),   encoding='utf-8')

        for df in [train_source_df, val_df]:
            df.rename(columns={'query': 'text'}, inplace=True)
            df['label'] = df['label'].map(qic_label_map)
            df.dropna(subset=['label', 'text'], inplace=True)
            df['label'] = df['label'].astype(int)
        num_classes = 11

        # 从 train.json 取 TRAIN_SET_SIZE + TEST_SET_SIZE = 4750 条，再分出测试集 1000
        need = TRAIN_SET_SIZE + TEST_SET_SIZE  # 3750 + 1000 = 4750
        if len(train_source_df) > need:
            _, train_source_df = train_test_split(
                train_source_df, test_size=need, random_state=42, stratify=train_source_df['label']
            )
        train_df, test_df = train_test_split(
            train_source_df, test_size=TEST_SET_SIZE, random_state=42, stratify=train_source_df['label']
        )

        # 验证集从官方 dev 文件取 VALIDATION_SIZE 条
        if len(val_df) > VALIDATION_SIZE:
            _, val_df = train_test_split(val_df, test_size=VALIDATION_SIZE, random_state=42, stratify=val_df['label'])

    elif dataset.lower() == 'csl':
        csl_label_map = {
            '工学': 0, '理学': 1, '农学': 2, '医学': 3, '管理学': 4,
            '法学': 5, '教育学': 6, '经济学': 7, '文学': 8, '艺术学': 9,
            '历史学': 10, '军事学': 11, '哲学': 12
        }
        train_df = pd.read_csv(os.path.join(base_dir, 'train.tsv'), sep='\t', names=['raw_text', 'label_name'], encoding='utf-8')
        val_df   = pd.read_csv(os.path.join(base_dir, 'dev.tsv'),   sep='\t', names=['raw_text', 'label_name'], encoding='utf-8')
        test_df  = pd.read_csv(os.path.join(base_dir, 'test.tsv'),  sep='\t', names=['raw_text', 'label_name'], encoding='utf-8')
        
        for df in [train_df, val_df, test_df]:
            df['text']  = df['raw_text'].str.replace(r'^to category\s*', '', regex=True).str.strip()
            df['label'] = df['label_name'].map(csl_label_map)
            df.dropna(subset=['label', 'text'], inplace=True)
            df['label'] = df['label'].astype(int)
        num_classes = 13
        
        if len(train_df) > TRAIN_SET_SIZE:
            _, train_df = train_test_split(train_df, test_size=TRAIN_SET_SIZE, random_state=42, stratify=train_df['label'])
        if len(val_df) > VALIDATION_SIZE:
            _, val_df = train_test_split(val_df, test_size=VALIDATION_SIZE, random_state=42, stratify=val_df['label'])
        if len(test_df) > TEST_SET_SIZE:
            _, test_df = train_test_split(test_df, test_size=TEST_SET_SIZE, random_state=42, stratify=test_df['label'])

    logger.info(
        f"[{dataset}] 数据划分完成: 训练={len(train_df)}, 验证={len(val_df)}, 测试={len(test_df)}"
    )
    return (
        train_df['text'].tolist(), train_df['label'].tolist(),
        val_df['text'].tolist(),   val_df['label'].tolist(),
        test_df['text'].tolist(),  test_df['label'].tolist(),
        num_classes
    )

def get_attention_weighted_embedding(text, batch_size=32, model=None, tokenizer=None, device=None):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask']
        scores = hidden.mean(dim=2)
        scores = torch.softmax(scores, dim=1) * mask
        scores = scores / scores.sum(dim=1, keepdim=True)
        embedding = (hidden * scores.unsqueeze(-1)).sum(dim=1)
    return embedding.cpu().numpy()

def extract_features_with_attention(data, desc, model, tokenizer, device):
    features = []
    for i in tqdm(range(0, len(data), 32), desc=f"提取 {desc} 特征"):
        batch = data[i:i+32]
        vecs = get_attention_weighted_embedding(batch, 32, model, tokenizer, device)
        features.append(vecs)
    return np.concatenate(features, axis=0)

def _atomic_save_npz(path, **arrays):
    """原子写 .npz：先写临时文件，完成后原子替换目标路径。"""
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)

def load_or_build_pca_features(dataset, model, tokenizer, device, n_labeled=None, n_components=PCA_N_COMPONENTS):
    """
    只缓存 PCA 降维后的特征，每个分集一个 .npz 文件（特征+标签合并存储）。
    num_classes 存入 train.npz，无需额外 .pkl 文件。
    若缓存已存在则直接加载，否则提取 BERT 特征、拟合 PCA、降维后缓存。

    缓存目录: cache/<dataset>/
    文件命名规则（标注样本数为唯一标识）:
      train_L{n_labeled}.npz  包含字段: feat (N, dim), label (N,), num_classes ()
      val_L{n_labeled}.npz    包含字段: feat, label
      test_L{n_labeled}.npz   包含字段: feat, label
    """
    pca_cache_dir = os.path.join("cache", dataset.lower())
    os.makedirs(pca_cache_dir, exist_ok=True)

    tag = f"L{n_labeled}_D{n_components}"
    train_cache = os.path.join(pca_cache_dir, f'train_{tag}.npz')
    val_cache   = os.path.join(pca_cache_dir, f'val_{tag}.npz')
    test_cache  = os.path.join(pca_cache_dir, f'test_{tag}.npz')

    if all(os.path.exists(p) for p in [train_cache, val_cache, test_cache]):
        tr = np.load(train_cache)
        va = np.load(val_cache)
        te = np.load(test_cache)
        return (
            (tr['feat'], tr['label']),
            (va['feat'], va['label']),
            (te['feat'], te['label']),
            int(tr['num_classes'])
        )

    # ---- 缓存不存在：提取 BERT 特征 ----
    logger.info(f"[{dataset}] 缓存未找到（标注样本数={n_labeled}，维数={n_components}），开始提取 BERT 特征...")
    train_text, train_label, val_text, val_label, test_text, test_label, num_classes = load_data(dataset)

    train_raw = extract_features_with_attention(train_text, f"{dataset}_train", model, tokenizer, device)
    val_raw   = extract_features_with_attention(val_text,   f"{dataset}_val",   model, tokenizer, device)
    test_raw  = extract_features_with_attention(test_text,  f"{dataset}_test",  model, tokenizer, device)

    if n_components >= 768:
        logger.info(f"[{dataset}] 目标维度 {n_components} >= 768，跳过 PCA，直接使用原生特征...")
        train_feat = train_raw
        val_feat   = val_raw
        test_feat  = test_raw
    else:
        # ---- 在训练集上拟合 PCA，避免数据泄露 ----
        logger.info(f"[{dataset}] 拟合 PCA（{n_components} 维）并降维...")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(train_raw)
        train_feat = pca.transform(train_raw)
        val_feat   = pca.transform(val_raw)
        test_feat  = pca.transform(test_raw)

    # ---- 原子写缓存：每个分集一个 .npz ----
    logger.info(f"[{dataset}] 写入缓存（标注样本数={n_labeled}）...")
    _atomic_save_npz(train_cache, feat=train_feat, label=np.array(train_label), num_classes=np.array(num_classes))
    _atomic_save_npz(val_cache,   feat=val_feat,   label=np.array(val_label))
    _atomic_save_npz(test_cache,  feat=test_feat,  label=np.array(test_label))

    return (
        (train_feat, np.array(train_label)),
        (val_feat,   np.array(val_label)),
        (test_feat,  np.array(test_label)),
        num_classes
    )

# ============================== 模型与增强 ==============================
class FixMatchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, dropout_rate):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev = h
        self.feature_extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)
        self.feature_dim = prev  # 记录最后一层特征的维度 (64)

    def forward(self, x, use_dropout=True):
        if use_dropout: self.train()
        else: self.eval()
        f = self.feature_extractor(x)
        logits = self.head(f)
        return logits, None, f, None

class FeatureAugmentation:
    def __init__(self, noise_std=0.05, dropout_rate=0.2): 
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate  

    def weak_augment(self, x):
        return x + torch.randn_like(x) * self.noise_std

    def strong_augment(self, x):
        if self.dropout_rate > 0:
            mask = torch.rand_like(x) > self.dropout_rate
            x = x * mask
        return x

# ============================== 基类：半监督分类器 ==============================
class BaseSSLClassifier:
    def __init__(self, input_dim=PCA_N_COMPONENTS, hidden_layers=MLP_HIDDEN_LAYERS, 
                 num_classes=2, lambda_u=UNSUPERVISED_LOSS_WEIGHT, epochs=MAX_TRAINING_EPOCHS,
                 batch_size=BATCH_SIZE, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 warmup_epochs=WARMUP_EPOCHS, logits_rampup_epochs=LOGITS_RAMPUP_EPOCHS,
                 mlp_dropout_rate=MLP_DROPOUT_RATE, device=None,
                 use_early_stopping=USE_EARLY_STOPPING, es_patience=ES_PATIENCE,
                 es_acc_min_delta=ES_ACC_MIN_DELTA, es_loss_min_delta=ES_LOSS_MIN_DELTA):
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_u = lambda_u
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr, self.weight_decay = lr, weight_decay
        self.warmup_epochs, self.logits_rampup_epochs = warmup_epochs, logits_rampup_epochs
        self.use_early_stopping = use_early_stopping
        self.es_patience, self.es_acc_min_delta, self.es_loss_min_delta = \
            es_patience, es_acc_min_delta, es_loss_min_delta

        self.model = FixMatchMLP(input_dim, hidden_layers, num_classes, mlp_dropout_rate).to(self.device)
        self.augmentation = FeatureAugmentation()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        
        self.history = {
            'sup_loss': [], 'unsup_loss': [], 'val_accuracy': [],
            'effective_utilization': [],   # 统一语义：无标签数据被利用的程度（硬掩码均值 / 软权重均值）
            'pseudo_entropy': [],          # 伪标签平均熵（硬标签=0，软标签>0，可跨方法对比）
            'weight_snapshots': [],        # 每隔 50 epoch 记录一次全量无标注样本权重分布 [(epoch, weights), ...]
        }
        self.num_classes = num_classes
    def get_method_name(self):
        return "BASE"

    def initialize_strategy(self, X_l_t, y_l_t):
        pass

    def update_epoch_params(self, epoch):
        pass

    def post_epoch_update(self, avg_sup):
        pass

    def _unsup_extra_step(self, weak_x, strong_x, strong_logits, pseudo, mask, label_dist):
        """
        扩展接口：供子类实现额外的无监督 Loss (如 Distribution Alignment, 特征聚类等)
        """
        return torch.tensor(0.0).to(self.device)

    def compute_pseudo_labels(self, weak_x, idx_ulb=None):
        raise NotImplementedError

    def get_pseudo_distribution(self, pseudo_labels):
        """将硬标签转为分布，或直接返回软标签分布"""
        if pseudo_labels.dim() == 2:
            return pseudo_labels
        return F.one_hot(pseudo_labels, num_classes=self.num_classes).float()
    
    def register_buffer(self, name, tensor):
        setattr(self, name, tensor.to(self.device))

    def _ramp_up(self, epoch, final_value, rampup_epochs, warmup_epochs):
        if epoch < warmup_epochs:
            return 0.0
        elif epoch < warmup_epochs + rampup_epochs:
            return final_value * (epoch - warmup_epochs) / rampup_epochs
        else:
            return final_value

    def _eval_val(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)
            logits, _, _, _ = self.model(X_val_t, use_dropout=False)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == y_val_t).float().mean().item()
        return acc

    def fit(self, X_labeled, y_labeled, X_unlabeled, y_u_true, X_val, y_val):
        X_l_t = torch.FloatTensor(X_labeled).to(self.device)
        y_l_t = torch.LongTensor(y_labeled).to(self.device)
        X_u_t = torch.FloatTensor(X_unlabeled).to(self.device)
        y_u_true_t = torch.LongTensor(y_u_true).to(self.device) if y_u_true is not None else None

        self.initialize_strategy(X_l_t, y_l_t)
       
        best_val_acc, best_sup_loss = 0.0, float('inf')
        es_cnt_val, es_cnt_loss = 0, 0

        n_l, n_u = len(X_labeled), len(X_unlabeled)
        max_b = max(n_l // self.batch_size, n_u // self.batch_size, 1)


        for epoch in range(self.epochs):
            l_idx = torch.randperm(n_l).to(self.device) if n_l > 0 else torch.tensor([])
            u_idx = torch.randperm(n_u).to(self.device)
            lambda_u_ramped = self._ramp_up(epoch, self.lambda_u, self.logits_rampup_epochs, self.warmup_epochs)
            self.update_epoch_params(epoch)

            sup_sum = unsup_sum = util_sum = entropy_sum = 0.0
            batches = 0

            # 当前 epoch 训练过程中实际使用到的无标签权重
            epoch_weight_chunks = []

            for b in range(max_b):
                self.model.train()
                self.optimizer.zero_grad()
                
                # ========== 监督学习 ==========
                s_l = (b * self.batch_size) % n_l
                e_l = min(s_l + self.batch_size, n_l)
                if n_l > 0:
                    logits_l, _, _, _ = self.model(X_l_t[l_idx[s_l:e_l]])
                    sup_loss = F.cross_entropy(logits_l, y_l_t[l_idx[s_l:e_l]])
                else:
                    sup_loss = torch.tensor(0.0).to(self.device)

                # ========== 无监督学习 ==========
                s_u = (b * self.batch_size) % n_u
                e_u = min(s_u + self.batch_size, n_u)
                idx_u = u_idx[s_u:e_u]
               
                if epoch >= self.warmup_epochs and n_u > 0:
                    weak_x = self.augmentation.weak_augment(X_u_t[idx_u])
                    self.model.eval()
                    with torch.no_grad():
                        pseudo, mask, conf, batch_util, batch_ent = self.compute_pseudo_labels(weak_x, idx_u)
                        util_sum += batch_util
                        entropy_sum += batch_ent

                        # 保存训练过程中实际用于无监督损失的权重
                        if SAVE_WEIGHT_SNAPSHOTS and epoch >= self.warmup_epochs:
                            epoch_weight_chunks.append(mask.detach().cpu())

                        # 判断是否为软标签（ESMatch 返回分布）
                        is_soft_label = (pseudo.dim() == 2)

                    self.model.train()
                    strong_x = self.augmentation.strong_augment(X_u_t[idx_u])
                    strong_logits, _, strong_feat, _ = self.model(strong_x)  # 同时拿logits和特征，避免二次前向
                    
                    # 根据标签类型选择损失函数
                    if is_soft_label:
                        # 软标签：使用 KL 散度或软交叉熵
                        log_probs = F.log_softmax(strong_logits, dim=1)
                        unsup_loss = -(pseudo * log_probs).sum(dim=1)  # 软交叉熵
                        unsup_loss = (unsup_loss * mask).mean()
                    else:
                        # 硬标签：标准交叉熵
                        unsup_loss = (F.cross_entropy(strong_logits, pseudo, reduction='none') * mask).mean()

                    weighted_unsup_loss = lambda_u_ramped * unsup_loss

                    # ========== 子类专属步骤==========
                    # 传入当前 batch 有标签数据的模型预测分布（AdaMatch 动态对齐目标）
                    with torch.no_grad():
                        p_labeled_batch = torch.softmax(logits_l.detach(), dim=1).mean(dim=0)  # [C]
                    extra_loss = self._unsup_extra_step(
                        weak_x, strong_x, strong_logits, pseudo, mask, p_labeled_batch
                    )

                    total_loss = (
                        sup_loss
                        + weighted_unsup_loss
                        + extra_loss
                    )
                else:
                    total_loss = sup_loss
                    weighted_unsup_loss = torch.tensor(0.0).to(self.device)
                    
                total_loss.backward()
                self.optimizer.step()
                sup_sum += sup_loss.item()
                unsup_sum += weighted_unsup_loss.item()
                batches += 1

            val_acc = self._eval_val(X_val, y_val)
            avg_sup, avg_unsup = sup_sum/batches, unsup_sum/batches
            self.history['sup_loss'].append(avg_sup)
            self.history['unsup_loss'].append(avg_unsup)
            self.history['val_accuracy'].append(val_acc)
            self.history['effective_utilization'].append(util_sum / batches)
            self.history['pseudo_entropy'].append(entropy_sum / batches)

            # 保存当前 epoch 的训练权重分布快照
            if (
                SAVE_WEIGHT_SNAPSHOTS
                and epoch >= self.warmup_epochs
                and epoch % WEIGHT_SNAPSHOT_INTERVAL == 0
                and len(epoch_weight_chunks) > 0
            ):
                epoch_weights = torch.cat(epoch_weight_chunks, dim=0).numpy()
                self.history['weight_snapshots'].append((epoch, epoch_weights))

            self.post_epoch_update(avg_sup)

            if self.use_early_stopping :
                if val_acc > best_val_acc + self.es_acc_min_delta:
                    best_val_acc, best_sup_loss, es_cnt_val, es_cnt_loss = val_acc, avg_sup, 0, 0
                    torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                else:
                    es_cnt_val += 1
                    if avg_sup < best_sup_loss - self.es_loss_min_delta:
                        best_sup_loss, es_cnt_loss = avg_sup, 0
                        torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                    else:
                        es_cnt_loss += 1
                    if es_cnt_val >= self.es_patience and es_cnt_loss >= self.es_patience:
                        break

            self.scheduler.step()

        if os.path.exists(BEST_MODEL_PATH):
            self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=self.device))

        avg_effective_util = sum(self.history['effective_utilization']) / len(self.history['effective_utilization']) if self.history['effective_utilization'] else 0.0

        return self, epoch + 1, avg_effective_util

# ============================== 策略实现 ==============================
class FreeMatchClassifier(BaseSSLClassifier):
    def __init__(self, ema_alpha=FREEMATCH_EMA_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.ema_alpha = ema_alpha
        
        # 缓冲区初始化
        self.register_buffer('tau_global', torch.tensor(1.0 / self.num_classes))
        # p_model (p̃_t)：类别预测概率的 EMA
        self.register_buffer('p_model', torch.ones(self.num_classes) / self.num_classes)
        # h_model (h̃_t)：被选中样本的伪标签直方图 EMA，用于 SAF 归一化
        self.register_buffer('h_model', torch.ones(self.num_classes) / self.num_classes)

    def get_method_name(self):
        return "FreeMatch"

    def _unsup_extra_step(self, weak_x, strong_x, strong_logits, pseudo, mask, label_dist=None):
        """SAF 损失：-H(SumNorm(p̃_t/h̃_t), SumNorm(p̄/h̄))，直方图归一化消除类别不平衡影响。"""
        mean_probs = torch.softmax(strong_logits, dim=1).mean(dim=0)  # p̄ [C]

        with torch.no_grad():
            sel = mask.bool()
            pseudo_hard = pseudo if pseudo.dim() == 1 else pseudo.argmax(dim=1)
            if sel.sum() > 0:
                batch_h = torch.zeros(self.num_classes, device=self.device)
                for c in range(self.num_classes):
                    batch_h[c] = (pseudo_hard[sel] == c).float().sum()
                batch_h = batch_h / (batch_h.sum() + 1e-8)
            else:
                batch_h = torch.ones(self.num_classes, device=self.device) / self.num_classes
            self.h_model = self.ema_alpha * self.h_model + (1 - self.ema_alpha) * batch_h

        def sum_norm(p, h):
            r = p / (h + 1e-8)
            return r / (r.sum() + 1e-8)

        q_ema = sum_norm(self.p_model, self.h_model)
        q_cur = sum_norm(mean_probs, batch_h)
        return (q_ema * torch.log(q_cur + 1e-8)).sum()

    def compute_pseudo_labels(self, weak_x, idx_ulb=None):
        # 1. 获取弱增强的预测
        weak_logits, _, _, _ = self.model(weak_x)
        probs = torch.softmax(weak_logits.detach(), dim=1)
        conf, pseudo = torch.max(probs, dim=1)

        with torch.no_grad():
            # --- 2. 更新全局状态 (SAT 核心) ---
            # 全局阈值更新：EMA(当前 Batch 的平均最大概率)
            current_batch_max_mean = conf.mean()
            self.tau_global = self.ema_alpha * self.tau_global + (1 - self.ema_alpha) * current_batch_max_mean

            # 类别权重更新：EMA(当前 Batch 的全类别概率均值)
            current_batch_p_model = probs.mean(dim=0)
            self.p_model = self.ema_alpha * self.p_model + (1 - self.ema_alpha) * current_batch_p_model

            # --- 3. 计算类特定阈值 ---
            max_p = self.p_model.max()
            mod = self.p_model / (max_p + 1e-8)
            tau_local = self.tau_global * mod
            
            # --- 4. 硬阈值掩码生成 ---
            sample_tau = tau_local[pseudo]
            mask = conf.ge(sample_tau).float()

        return pseudo, mask, conf, mask.mean().item(), 0.0

class FlexMatchClassifier(BaseSSLClassifier):
    def __init__(self, ulb_dest_len, ema_alpha=FLEXMATCH_EMA_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.ulb_dest_len = ulb_dest_len  # 必须传入未标记数据的总长度
        self.p_cutoff = FLEXMATCH_CONFIDENCE_THRESHOLD  # 固定基础阈值
        
        # 1. 完全复现：全局状态维护
        # selected_label 存储每个全量未标记样本的伪标签，初始化为 -1
        self.register_buffer("selected_label", torch.ones(ulb_dest_len, dtype=torch.long) * -1)
        # classwise_acc 存储定义的“相对学习进度”
        self.register_buffer("classwise_acc", torch.zeros(self.num_classes))

    def get_method_name(self):
        return "FLEXMATCH"

    @torch.no_grad()
    def update_classwise_acc(self):
        """基于全量样本的相对计数"""
        # 统计已选中的伪标签分布
        pseudo_counter = torch.bincount(self.selected_label[self.selected_label != -1], minlength=self.num_classes)
        max_cnt = torch.max(pseudo_counter).item()
        if max_cnt > 0:
            # 核心公式：当前类数量 / 最大类数量
            self.classwise_acc = pseudo_counter.float() / max_cnt
        else:
            self.classwise_acc = torch.zeros(self.num_classes).to(self.selected_label.device)

    def compute_pseudo_labels(self, weak_x, idx_ulb=None):
        """注意：实现必须传入样本在全量集中的索引 idx_ulb"""
        if idx_ulb is None:
            # 未提供索引时，创建临时索引以保持兼容性
            idx_ulb = torch.arange(len(weak_x)).to(weak_x.device)
        self.model.eval()
        with torch.no_grad():
            weak_logits, _, _, _ = self.model(weak_x)
            probs = torch.softmax(weak_logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)

            # --- Step 1: 计算动态阈值 ---
            # 凸函数公式：scaling = acc / (2 - acc)
            scaling_factor = self.classwise_acc / (2.0 - self.classwise_acc)
            dynamic_thresholds = self.p_cutoff * scaling_factor
            
            # --- Step 2: 生成最终掩码 ---
            sample_thresholds = dynamic_thresholds[pseudo_labels]
            final_mask = max_probs.ge(sample_thresholds).float()

            # --- Step 3: 更新全局状态 ---
            select_mask = max_probs.ge(self.p_cutoff)
            if select_mask.any():
                self.selected_label[idx_ulb[select_mask]] = pseudo_labels[select_mask]
            self.update_classwise_acc()
        return pseudo_labels, final_mask, max_probs, final_mask.mean().item(), 0.0
        
class ESMatchClassifier(BaseSSLClassifier):
    def __init__(self, ema_alpha=ESMATCH_EMA_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.ema_alpha = ema_alpha
        
        # 1. tau_global: 追踪全局平均置信度，作为底层进度基准
        self.register_buffer('tau_global', torch.tensor(0.0))
        # 2. p_model: 追踪各类别的专属平均置信度，用于计算非对称阈值
        self.register_buffer('p_model', torch.ones(self.num_classes) / self.num_classes)
        # 3. class_entropy: 动态跟踪每个类别的归一化香农熵，用于提供最优的自适应掩码温度 (Temperature)
        self.register_buffer('class_entropy', torch.ones(self.num_classes))
    
    def initialize_strategy(self, X_l_t, y_l_t):
        pass

    def get_method_name(self):
        # 此处 ESMatch 可改名为更贴切的缩写，如 ESM-DA (Entropy-Guided Soft Margin with Dist Alignment)
        # 暂时保留名称保证兼容性
        return "ESMATCH"

    def compute_pseudo_labels(self, weak_x, idx_ulb=None):
        self.model.eval()
        with torch.no_grad():
            logits, _, _, _ = self.model(weak_x, use_dropout=False)
            log_p_model = F.log_softmax(logits, dim=1)
            soft_labels = torch.exp(log_p_model)

            conf, pseudo = soft_labels.max(dim=1)
            
            # --- 1. 更新全局基础阈值 (Global Threshold) ---
            batch_mean_conf = conf.mean()
            if self.tau_global.item() == 0:
                self.tau_global.fill_(batch_mean_conf.item())
            else:
                new_tau = self.ema_alpha * self.tau_global + (1.0 - self.ema_alpha) * batch_mean_conf
                self.tau_global.fill_(new_tau.item())

            # --- 2. 状态追踪 (Progress & Entropy Tracking) ---
            # 采用 tau_global 统一过滤各类的“精英群体”。
            # 该群体的熵天然缩放至 [0.01, 0.15] 区间，适配 Sigmoid 掩码的温度区间。
            high_conf_mask = (conf >= self.tau_global).float()
            for c in range(self.num_classes):
                c_idx = (pseudo == c).bool() & high_conf_mask.bool()
                if c_idx.any():
                    # Z_t: 该类高置信度群体的平均置信度 (学习进度)
                    c_conf = conf[c_idx]
                    Z_t = c_conf.mean()
                    
                    # H_norm: 归一化香农熵 (反映类别的内生难度)
                    c_soft_labels = soft_labels[c_idx]
                    c_entropy = -(c_soft_labels * torch.log(c_soft_labels + 1e-8)).sum(dim=1).mean()
                    H_norm = c_entropy / math.log(self.num_classes)
                    
                    # 使用 EMA 平滑更新长期状态
                    self.class_entropy[c] = self.ema_alpha * self.class_entropy[c] + (1 - self.ema_alpha) * H_norm.item()
                    self.p_model[c] = self.ema_alpha * self.p_model[c] + (1 - self.ema_alpha) * Z_t

            # --- 3. 非对称动态阈值 (Asymmetric Scaling) ---
            # 缓解长尾效应：学习进度越差的类别，其专属阈值越低
            norm_p = self.p_model / (self.p_model.max() + 1e-8)
            tau_local = self.tau_global * norm_p
            sample_threshold = tau_local[pseudo]

            # --- 4. 信息熵动态柔性掩码 (Entropy-Guided Soft Mask, ESM) ---
            if ABLATION_WO_SOFT:
                mask = (conf >= sample_threshold).float()  # 退化为硬截断
            else:
                # 以该类的动态信息熵作为温度参数，自适应调节 Sigmoid 掩码的宽度。
                # 高熵对应平滑掩码，缓冲梯度震荡；低熵对应陡峭掩码，提升确信样本的梯度权重。
                dynamic_width = self.class_entropy[pseudo].clamp(min=0.05)
                mask = torch.sigmoid((conf - sample_threshold) / dynamic_width)

            entropy = -(soft_labels * torch.log(soft_labels + 1e-8)).sum(dim=1)

        utilization = (conf >= sample_threshold).float().mean().item()
        return soft_labels, mask, conf, utilization, entropy.mean().item()

    def _unsup_extra_step(self, weak_x, strong_x, strong_logits, pseudo, mask, label_dist):
        if ESMATCH_DIST_REG_LOSS_WEIGHT > 0 and not ABLATION_WO_DA:
            probs = torch.softmax(strong_logits, dim=1)
            w = mask.detach().clamp(min=0)
            w_sum = w.sum().clamp(min=1e-8)
            p_unlabeled = (probs * w.unsqueeze(1)).sum(dim=0) / w_sum
            p_target = label_dist
            
            # 单向 KL 散度 (The "Defibrillator" Effect)
            # 当 p_unlabeled 趋近于 0 时产生的“巨大梯度”是打破长尾死锁的必要机制！
            # 只有极大的惩罚，才能强迫模型在 768 维空间中将丢弃的尾部类别重新拉回流形。
            da_loss = F.kl_div(
                torch.log(p_unlabeled.clamp(min=1e-8)),
                p_target.clamp(min=1e-8),
                reduction='sum'
            )
            
            return ESMATCH_DIST_REG_LOSS_WEIGHT * da_loss

        return torch.tensor(0.0).to(self.device)
class SoftMatchClassifier(BaseSSLClassifier):
    def __init__(self, ema_alpha=SOFTMATCH_EMA_ALPHA, n_sigma=2, **kwargs):
        super().__init__(**kwargs)
        # USB 默认使用非常平滑的动量 0.999
        self.ema_alpha = ema_alpha
        self.n_sigma = n_sigma # USB 默认值
        
        # 1. p_model: 分布对齐 (USB 的 SoftMatch 类中包含此逻辑)
        self.register_buffer("p_model", torch.ones(self.num_classes) / self.num_classes)
        
        # 2. 统计量初始化 (参考 USB)
        self.register_buffer("prob_max_mu", torch.tensor(1.0 / self.num_classes))
        self.register_buffer("prob_max_var", torch.tensor(0.1))

    def get_method_name(self):
        return "SOFTMATCH"

    def get_pseudo_distribution(self, pseudo_labels):
        return self._last_probs_aligned

    def compute_pseudo_labels(self, weak_x, idx_ulb=None):
        # 获取预测
        weak_logits, _, _, _ = self.model(weak_x)
        probs = torch.softmax(weak_logits.detach(), dim=-1)

        # --- 步骤 A: 分布对齐 (DA) ---
        # 保持对齐逻辑，这在 USB 完整代码中也是存在的
        probs_aligned = probs * (1.0 / self.p_model)
        probs_aligned = probs_aligned / probs_aligned.sum(dim=-1, keepdim=True)

        self._last_probs_aligned = probs_aligned.detach()

        with torch.no_grad():
            # 更新预测分布
            self.p_model = self.ema_alpha * self.p_model + (1 - self.ema_alpha) * probs.mean(dim=0)

        # 2. 计算最大概率
        max_probs, pseudo_labels = probs_aligned.max(dim=-1)

        # --- 步骤 B: 更新统计量 (严格遵循 USB 逻辑) ---
        with torch.no_grad():
            cur_mu = torch.mean(max_probs)
            cur_var = torch.var(max_probs, unbiased=True)
            self.prob_max_mu = self.ema_alpha * self.prob_max_mu + (1 - self.ema_alpha) * cur_mu.item()
            self.prob_max_var = self.ema_alpha * self.prob_max_var + (1 - self.ema_alpha) * cur_var.item()

        # --- 步骤 C: USB 标准权重计算 (Truncated Gaussian) ---
        mu = self.prob_max_mu
        var = self.prob_max_var.clamp(min=1e-8)
        
        # USB 核心公式：只对低于 mu 的部分计算高斯衰减
        # n_sigma 控制曲线的陡峭程度
        diff = torch.clamp(max_probs - mu, max=0.0) ** 2
        soft_weight = torch.exp(-(diff / (2 * var / (self.n_sigma ** 2))))

        # 伪标签熵：使用对齐后的平滑分布计算
        entropy = -(probs_aligned * torch.log(probs_aligned + 1e-8)).sum(dim=-1).mean().item()

        # 统计指标
        return pseudo_labels, soft_weight.detach(), max_probs, soft_weight.mean().item(), entropy

# ============================== 实验流程 ==============================
def train_and_evaluate(dataset, n_labeled, run, method, model, tokenizer, device):
    set_seed(42 + run)
    
    # 1. 加载或构建缓存特征（按标注样本数分文件）
    (X_train_full, y_train_full), (X_val, y_val), (X_test, y_test), num_classes = \
        load_or_build_pca_features(dataset, model, tokenizer, device, n_labeled=n_labeled)

    # 2. 从独立划分的训练集中切分出 Labeled 和 Unlabeled
    X_labeled, X_unlabeled, y_labeled, y_u_true = train_test_split(
        X_train_full, y_train_full,
        train_size=n_labeled, random_state=42+run, stratify=y_train_full
    )

    # 记录当前运行的标签分布情况
    unique_labels, counts = np.unique(y_labeled, return_counts=True)
    dist_info = {int(k): int(v) for k, v in zip(unique_labels, counts)}
    
    # 4. 初始化分类器
    params = {'input_dim': PCA_N_COMPONENTS, 'num_classes': num_classes, 'device': device}
    
    if method == 'FREEMATCH': clf = FreeMatchClassifier(**params)
    elif method == 'FLEXMATCH': clf = FlexMatchClassifier(ulb_dest_len=len(X_unlabeled), **params)
    elif method == 'SOFTMATCH': clf = SoftMatchClassifier(**params)
    elif method == 'ESMATCH': clf = ESMatchClassifier(**params)
    else:
        logger.error(f"未知方法: {method}")
        return None

    # 5. 模型训练
    clf, final_epoch, avg_effective_util = clf.fit(
        X_labeled, y_labeled, X_unlabeled, y_u_true, X_val, y_val
    )

    # 6. 训练结束后对全量无标签数据做一次整体评估
    clf.model.eval()
    detailed_info = {}
    with torch.no_grad():
        X_u_t_full = torch.FloatTensor(X_unlabeled).to(device)
        weak_x_full = clf.augmentation.weak_augment(X_u_t_full)
        pseudo_full, mask_full, conf_full, final_eff_util, avg_entropy = clf.compute_pseudo_labels(weak_x_full)
        is_soft = (pseudo_full.dim() == 2)

        # 获取分布用于 KL 散度计算
        soft_dist = clf.get_pseudo_distribution(pseudo_full)

        # 2. KL 散度
        label_counts = torch.bincount(torch.LongTensor(y_labeled), minlength=num_classes).float()
        label_dist_ref = (label_counts / label_counts.sum()).to(device)
        mean_dist = soft_dist.mean(dim=0)
        kl_to_label_dist = F.kl_div(torch.log(label_dist_ref + 1e-8), mean_dist, reduction='sum').item()

        # 3. 有效利用率（训练结束后全量评估，由 compute_pseudo_labels 统一提供）
        effective_utilization = final_eff_util

        # 4. ECE（Expected Calibration Error）
        #    用模型在测试集上的置信度 vs 实际准确率来计算
        X_test_t = torch.FloatTensor(X_test).to(device)
        test_logits, _, _, _ = clf.model(X_test_t, use_dropout=False)
        test_probs = torch.softmax(test_logits, dim=1)
        test_conf, test_pred = test_probs.max(dim=1)
        test_correct = (test_pred == torch.LongTensor(y_test).to(device)).float()

        n_bins = 10
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
        ece = torch.tensor(0.0, device=device)
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (test_conf > lo) & (test_conf <= hi)
            if in_bin.sum() > 0:
                bin_acc = test_correct[in_bin].mean()
                bin_conf = test_conf[in_bin].mean()
                ece += (in_bin.float().mean()) * (bin_acc - bin_conf).abs()
        ece = ece.item()

        # 保存机制分析所需信息
        if is_soft:
            # ESMatch 返回的是软标签分布 [N, C]，取 argmax 作为 hard pseudo-label
            pseudo_hard_full = torch.argmax(pseudo_full, dim=1)
            pseudo_soft_full = pseudo_full
        else:
            # SoftMatch / FlexMatch / FreeMatch 返回的是 hard pseudo-label
            pseudo_hard_full = pseudo_full
            pseudo_soft_full = soft_dist
        pseudo_correct_full = (pseudo_hard_full == torch.LongTensor(y_u_true).to(device)).float()
        detailed_info = {
            'weights': mask_full.cpu().numpy(),
            'scores': conf_full.cpu().numpy(),
            'dist_at_run': dist_info,
            'pseudo_labels': pseudo_hard_full.cpu().numpy(),
            'pseudo_probs': pseudo_soft_full.cpu().numpy(),
            'true_labels': np.asarray(y_u_true),
            'pseudo_correct': pseudo_correct_full.cpu().numpy()
        }

    # 7. 测试集评估
    with torch.no_grad():
        pred = test_pred.cpu().numpy()

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    logger.info(
        f"[{dataset}] [{method}] Run {run+1}: Acc={acc:.4f}, F1(W)={f1:.4f}, "
        f"ECE={ece:.4f}, PseudoEntropy={avg_entropy:.4f}, "
        f"FinalEffUtil={effective_utilization:.4f}, KL_label={kl_to_label_dist:.4f}"
    )

    return {
        'dataset': dataset,
        'accuracy': acc,
        'f1': f1,
        'method': method,
        'labeled': n_labeled,
        'epochs': final_epoch,
        'history': clf.history,
        'avg_effective_utilization': avg_effective_util,
        'effective_utilization': effective_utilization,
        'ece': ece,
        'avg_pseudo_entropy': avg_entropy,
        'kl_to_label_dist': kl_to_label_dist,
        'label_dist': dist_info,
        'detailed_info': detailed_info
    }

def _pca_cache_exists(dataset, n_labeled):
    """检查指定数据集和标注样本数的缓存（3 个 .npz）是否已完整存在。"""
    pca_cache_dir = os.path.join("cache", dataset.lower())
    tag = f"L{n_labeled}_D{PCA_N_COMPONENTS}"
    required = [
        os.path.join(pca_cache_dir, f'train_{tag}.npz'),
        os.path.join(pca_cache_dir, f'val_{tag}.npz'),
        os.path.join(pca_cache_dir, f'test_{tag}.npz'),
    ]
    return all(os.path.exists(p) for p in required)

def main():
    # 检查并创建 weights 文件夹
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        logger.info(f"已创建权重目录: {WEIGHTS_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 缓存检查：遍历所有 (数据集, 标注量) 组合，缺失时才加载 BERT ----
    missing_configs = [
        (ds, nl) for ds in DATASETS_TO_RUN for nl in LABELED_CONFIGS
        if not _pca_cache_exists(ds, nl)
    ]
    if missing_configs:
        tokenizer  = BertTokenizer.from_pretrained('./bert-base-chinese')
        bert_model = BertModel.from_pretrained('./bert-base-chinese').to(device)
        bert_model.eval()  # 关闭 Dropout，确保特征提取确定性

        # 提前为所有缺失缓存的配置构建特征，完成后立即释放显存
        for ds, nl in missing_configs:
            logger.info(f"[{ds}] 开始提取并缓存特征（标注量={nl}）...")
            load_or_build_pca_features(ds, bert_model, tokenizer, device, n_labeled=nl)

        # 释放 BERT 模型显存，后续训练不再需要
        del bert_model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bert_model = tokenizer = None
    else:
        bert_model = tokenizer = None

    all_results = []
    
    # 开始多轮实验循环
    for method in METHODS_TO_RUN:
        for dataset in DATASETS_TO_RUN:
            for n_labeled in LABELED_CONFIGS:
                print(f"\n>>> 运行方法: {method} | 数据集: {dataset} | 标注量: {n_labeled}")
                
                for run in range(NUM_RUNS):
                    # 调用训练与评估逻辑
                    res = train_and_evaluate(dataset, n_labeled, run, method, bert_model, tokenizer, device)  # bert_model/tokenizer 可为 None（缓存命中时）
                    if res:
                        all_results.append(res)
    # 统计与汇总结果
    if all_results:
        df = pd.DataFrame(all_results)
        
        summary_rows = []
        for dataset in DATASETS_TO_RUN:
            print(f"\n" + "="*70)
            print(f"【 数据集: {dataset} 】")
            if 'ESMATCH' in METHODS_TO_RUN:
                print(f"【 Ablation 控制开关 (ESMatch) 】")
                print(f"  - w/o Soft Mask (退化为硬截断)  : {ABLATION_WO_SOFT}")
            print("="*70)
            
            dataset_df = df[df['dataset'] == dataset]
            
            summary = dataset_df.groupby(['method', 'labeled']).agg({
                'accuracy': ['mean', 'std'],
                'f1': ['mean', 'std'],
                'ece': ['mean', 'std'],
                'avg_pseudo_entropy': ['mean', 'std'],
                'kl_to_label_dist': ['mean', 'std'],
                'avg_effective_utilization': ['mean', 'std'],
                'effective_utilization': ['mean', 'std'],
                'epochs': ['mean']
            })
            summary = summary.sort_index()

            header = (f"{'Method':<12} | {'Labeled':<7} | {'Acc (Mean±Std)':<20} | "
                      f"{'F1-W (Mean±Std)':<20} | {'ECE':<12} | "
                      f"{'PseudoEntropy':<16} | {'KL_label':<14} | "
                      f"{'AvgEffUtil':<14} | {'FinalEffUtil'}")
            print(header)
            print("-" * len(header))
            
            for (method_name, n_lab), row in summary.iterrows():
                acc_mean  = row[('accuracy', 'mean')]
                acc_std   = row[('accuracy', 'std')]
                f1_mean   = row[('f1', 'mean')]
                f1_std    = row[('f1', 'std')]
                ece_mean  = row[('ece', 'mean')]
                ece_std   = row[('ece', 'std')]
                ent_mean  = row[('avg_pseudo_entropy', 'mean')]
                ent_std   = row[('avg_pseudo_entropy', 'std')]
                kll_mean  = row[('kl_to_label_dist', 'mean')]
                kll_std   = row[('kl_to_label_dist', 'std')]
                util_mean = row[('avg_effective_utilization', 'mean')]
                util_std  = row[('avg_effective_utilization', 'std')]
                final_util_mean = row[('effective_utilization', 'mean')]
                final_util_std  = row[('effective_utilization', 'std')]
                epochs    = row[('epochs', 'mean')]

                print(f"{method_name:<12} | Labeled={n_lab:<2} | "
                      f"{acc_mean:.4f}±{acc_std:.4f}     | "
                      f"{f1_mean:.4f}±{f1_std:.4f}     | "
                      f"{ece_mean:.4f}±{ece_std:.4f} | "
                      f"{ent_mean:.4f}±{ent_std:.4f}   | "
                      f"{kll_mean:.4f}±{kll_std:.4f} | "
                      f"{util_mean:.3f}±{util_std:.3f} | "
                      f"{final_util_mean:.3f}±{final_util_std:.3f}")

                summary_rows.append({
                    'method': method_name, 'labeled': int(n_lab), 'dataset': dataset,
                    'acc_mean': acc_mean, 'acc_std': acc_std,
                    'f1_mean': f1_mean, 'f1_std': f1_std,
                    'ece_mean': ece_mean, 'ece_std': ece_std,
                    'pseudo_entropy_mean': ent_mean, 'pseudo_entropy_std': ent_std,
                    'kl_to_label_dist_mean': kll_mean, 'kl_to_label_dist_std': kll_std,
                    'effective_utilization_mean': util_mean, 'effective_utilization_std': util_std,
                    'effective_utilization_mean': final_util_mean, 'effective_utilization_std': final_util_std,
                    'avg_epochs': epochs
                })
            print("-" * len(header))

        # 保存原始实验数据（pickle）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        methods_tag = '_'.join(METHODS_TO_RUN)
        datasets_tag = '_'.join(DATASETS_TO_RUN)

        result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
        os.makedirs(result_dir, exist_ok=True)

        raw_output = os.path.join(result_dir, f"results_{datasets_tag}_{methods_tag}_{timestamp}.pkl")
        for r in all_results:
            new_history = {}
            for k, vals in r['history'].items():
                if k == 'weight_snapshots':
                    new_history[k] = [
                        (int(ep), w.tolist() if hasattr(w, 'tolist') else w)
                        for ep, w in vals
                    ]
                else:
                    new_history[k] = [float(v) for v in vals]
            r['history'] = new_history
            r['detailed_info'] = {
                k: v.tolist() if hasattr(v, 'tolist') else v
                for k, v in r['detailed_info'].items()
            }
        with open(raw_output, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"原始实验数据已保存至: {raw_output}")

        # 保存汇总指标（JSON）
        summary_output = os.path.join(result_dir, f"summary_{datasets_tag}_{methods_tag}_{timestamp}.json")
        with open(summary_output, 'w', encoding='utf-8') as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        logger.info(f"汇总指标已保存至: {summary_output}")

        # 保存汇总指标（CSV）
        csv_output = os.path.join(result_dir, f"summary_{datasets_tag}_{methods_tag}_{timestamp}.csv")
        pd.DataFrame(summary_rows).to_csv(csv_output, index=False, encoding='utf-8-sig')
        logger.info(f"汇总指标 CSV 已保存至: {csv_output}")

    # 运行结束，删除权重文件
    if os.path.exists(BEST_MODEL_PATH):
        os.remove(BEST_MODEL_PATH)

if __name__ == "__main__":
    main()