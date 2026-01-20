import random
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

# ==============================================================================
# 参数配置
# ==============================================================================
# 'SOFTMATCH','DASH','SLD','FIXMATCH'
METHODS_TO_RUN = ['SOFTMATCH','DASH','SLD','FIXMATCH']
DATASETS_TO_RUN = ['computer']
TARGET_TOTAL_SIZE = 2000
TRAIN_SET_SIZE_CONFIG = 1200
LABELED_CONFIGS = [40]
NUM_RUNS = 100
VALIDATION_SIZE = 40
TEST_SET_SIZE = 800

# 模型与特征
MLP_HIDDEN_LAYERS = (128, 64, 32)
PCA_N_COMPONENTS = 32
MLP_DROPOUT_RATE = 0.8

# 训练超参数
UNSUPERVISED_LOSS_WEIGHT = 1.0 
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3
WARMUP_EPOCHS = 10
LOGITS_RAMPUP_EPOCHS = 90   
MAX_TRAINING_EPOCHS = 2000

# FixMatch参数
FIXMATCH_CONFIDENCE_THRESHOLD = 0.95

# DASH参数
DASH_GAMMA = 1.005
DASH_C = 1.0

# SoftMatch 参数 
SOFTMATCH_TEMPERATURE = 0.5 
SOFTMATCH_EMA_ALPHA = 0.999 

# 早停配置
USE_EARLY_STOPPING = True
ES_PATIENCE = 300
ES_ACC_MIN_DELTA = 0.01
ES_LOSS_MIN_DELTA = 0.01
ES_SKIP_FIRST_EPOCHS = 100

# 文件路径
BEST_MODEL_PATH = 'best_model_weights1.pth'
SHOPPING_FILE_PATH = 'online_shopping_10_cats.csv'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

MAPPING = {
    'book': '书籍', 'tablet': '平板', 'mobile': '手机', 'fruit': '水果',
    'shampoo': '洗发水', 'water_heater': '热水器', 'mengniu': '蒙牛',
    'clothes': '衣服', 'computer': '计算机', 'hotel': '酒店',
}

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
    if dataset not in MAPPING:
        logger.error(f"不支持的数据集：{dataset}")
        return None, None, None
    try:
        df = pd.read_csv(SHOPPING_FILE_PATH, encoding='utf-8')
        cat = MAPPING[dataset]
        df = df[df['cat'] == cat]
        if df.empty:
            logger.error(f"数据集 {dataset} ({cat}) 无数据")
            return None, None, None
        N_original = len(df)
        if N_original >= TARGET_TOTAL_SIZE:
            df = df.sample(n=TARGET_TOTAL_SIZE, random_state=42).reset_index(drop=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        data = df['review'].tolist()
        labels = df['label'].astype(int).tolist() 
        return data, labels, 2
    except Exception as e:
        logger.error(f"加载 {dataset} 失败: {e}")
        return None, None, None

def get_attention_weighted_embedding(text, batch_size=32, model=None, tokenizer=None, device=None):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask']
        scores = hidden.mean(dim=2)
        scores = torch.softmax(scores, dim=1) * mask
        scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-10)
        embedding = (hidden * scores.unsqueeze(-1)).sum(dim=1)
    return embedding.cpu().numpy()

def extract_features_with_attention(data, dataset, model, tokenizer, device):
    path = f'bert_features_shopping_{dataset}_T{TARGET_TOTAL_SIZE}.npy'
    if os.path.exists(path):
        return np.load(path)
    features = []
    for i in tqdm(range(0, len(data), 32), desc=f"提取 {dataset} 特征"):
        batch = data[i:i+32]
        vecs = get_attention_weighted_embedding(batch, 32, model, tokenizer, device)
        features.append(vecs)
    features = np.concatenate(features, axis=0)
    np.save(path, features)
    return features

def apply_pca(features, n_components=PCA_N_COMPONENTS, dataset='default'):
    path = f'pca_model_{dataset}_{n_components}_T{TARGET_TOTAL_SIZE}.joblib'
    try:
        pca = load(path)
        return pca.transform(features)
    except:
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(features)
        dump(pca, path)
        return reduced

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
                 es_acc_min_delta=ES_ACC_MIN_DELTA, es_loss_min_delta=ES_LOSS_MIN_DELTA,
                 es_skip_first_epochs=ES_SKIP_FIRST_EPOCHS): 
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_u = lambda_u 
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.logits_rampup_epochs = logits_rampup_epochs
        self.num_classes = num_classes
        self.use_early_stopping = use_early_stopping
        self.es_patience = es_patience
        self.es_acc_min_delta = es_acc_min_delta
        self.es_loss_min_delta = es_loss_min_delta
        self.es_skip_first_epochs = es_skip_first_epochs
        
        self.model = FixMatchMLP(input_dim, hidden_layers, num_classes, mlp_dropout_rate).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.augmentation = FeatureAugmentation()
        self.history = {'sup_loss': [], 'unsup_loss': [], 'acceptance_ratio': [], 
                        'loss_factor_mean': [], 'val_accuracy': [], 'pseudo_acc': []}

    def _ramp_up(self, epoch, max_w, ramp_dur, warmup):
        if epoch < warmup: return 0.0
        passed = epoch - warmup
        return max_w * min(1.0, passed / ramp_dur) if ramp_dur > 0 else max_w

    def _eval_val(self, X_val, y_val):
        if X_val is None or len(X_val) == 0: return 0.0
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _, _, _ = self.model(X_val_t, use_dropout=False)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
        return accuracy_score(y_val, pred)

    def compute_pseudo_labels(self, weak_x):
        raise NotImplementedError
    
    def get_method_name(self):
        raise NotImplementedError

    def fit(self, X_labeled, y_labeled, X_unlabeled, y_unlabeled_true=None, X_val=None, y_val=None):
        X_l_t = torch.FloatTensor(X_labeled).to(self.device)
        y_l_t = torch.LongTensor(y_labeled).to(self.device)
        X_u_t = torch.FloatTensor(X_unlabeled).to(self.device)
        y_u_true_t = torch.LongTensor(y_unlabeled_true).to(self.device) if y_unlabeled_true is not None else None

        self.initialize_strategy(X_l_t, y_l_t)
        best_val_acc, best_sup_loss, es_cnt_val, es_cnt_loss = -1.0, float('inf'), 0, 0
        final_epoch = self.epochs
        if os.path.exists(BEST_MODEL_PATH): os.remove(BEST_MODEL_PATH)

        n_l, n_u = len(X_labeled), len(X_unlabeled)
        max_b = max(n_l // self.batch_size, n_u // self.batch_size, 1)

        for epoch in range(self.epochs):
            l_idx = torch.randperm(n_l).to(self.device) if n_l > 0 else torch.tensor([])
            u_idx = torch.randperm(n_u).to(self.device)
            lambda_u_ramped = self._ramp_up(epoch, self.lambda_u, self.logits_rampup_epochs, self.warmup_epochs)
            self.update_epoch_params(epoch)

            sup_sum = unsup_sum = pseudo_acc_sum = base_mask_sum = loss_factor_sum = 0.0
            batches = epoch_pseudo_acc_batches = 0

            for b in range(max_b):
                self.model.train()
                self.optimizer.zero_grad()
                
                # 监督学习
                s_l = (b * self.batch_size) % n_l
                e_l = min(s_l + self.batch_size, n_l)
                if n_l > 0:
                    logits_l, _, _, _ = self.model(X_l_t[l_idx[s_l:e_l]])
                    sup_loss = F.cross_entropy(logits_l, y_l_t[l_idx[s_l:e_l]])
                else:
                    sup_loss = torch.tensor(0.0).to(self.device)

                # 无监督学习
                s_u = (b * self.batch_size) % n_u
                e_u = min(s_u + self.batch_size, n_u)
                idx_u = u_idx[s_u:e_u]
                
                if epoch >= self.warmup_epochs and n_u > 0:
                    weak_x = self.augmentation.weak_augment(X_u_t[idx_u])
                    self.model.eval()
                    with torch.no_grad():
                        pseudo, mask, conf, bm_mean, lf_mean = self.compute_pseudo_labels(weak_x)
                        base_mask_sum += bm_mean
                        loss_factor_sum += lf_mean
                        if y_u_true_t is not None:
                            acc_mask = (mask > 1e-5).float()
                            if acc_mask.sum() > 0:
                                correct = (pseudo[acc_mask > 0] == y_u_true_t[idx_u][acc_mask > 0]).sum().item()
                                pseudo_acc_sum += correct / acc_mask.sum().item()
                                epoch_pseudo_acc_batches += 1

                    self.model.train()
                    strong_logits, _, _, _ = self.model(self.augmentation.strong_augment(X_u_t[idx_u]))
                    unsup_loss = (F.cross_entropy(strong_logits, pseudo, reduction='none') * mask).mean()
                    weighted_unsup_loss = lambda_u_ramped * unsup_loss
                    total_loss = sup_loss + weighted_unsup_loss
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
            self.history['acceptance_ratio'].append(base_mask_sum/batches)
            self.history['loss_factor_mean'].append(loss_factor_sum/batches)
            self.history['pseudo_acc'].append(pseudo_acc_sum/epoch_pseudo_acc_batches if epoch_pseudo_acc_batches > 0 else 0.0)

            self.post_epoch_update(avg_sup)

            if self.use_early_stopping and epoch >= self.es_skip_first_epochs:
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
                    final_epoch = epoch + 1
                    break

        if os.path.exists(BEST_MODEL_PATH):
            self.model.load_state_dict(torch.load(BEST_MODEL_PATH))
            os.remove(BEST_MODEL_PATH)
        return self, final_epoch, np.mean(self.history['acceptance_ratio']), np.mean(self.history['pseudo_acc'])

    def initialize_strategy(self, X_l_t, y_l_t): pass
    def update_epoch_params(self, epoch): pass
    def post_epoch_update(self, avg_sup): pass

# ============================== 策略实现 ==============================
class FixMatchClassifier(BaseSSLClassifier):
    def __init__(self, confidence_threshold=FIXMATCH_CONFIDENCE_THRESHOLD, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
    def get_method_name(self): return "FIXMATCH"
    def compute_pseudo_labels(self, weak_x):
        weak_logits, _, _, _ = self.model(weak_x)
        probs = torch.softmax(weak_logits, dim=1)
        conf, pseudo = torch.max(probs, dim=1)
        mask = (conf >= self.confidence_threshold).float()
        return pseudo, mask, conf, mask.mean().item(), 0.0

class DASHClassifier(BaseSSLClassifier):
    def __init__(self, gamma=DASH_GAMMA, C=DASH_C, **kwargs):
        super().__init__(**kwargs)
        self.gamma, self.C, self.rho_hat, self.current_epoch = gamma, C, 0.0, 0
    def get_method_name(self): return "DASH"
    def initialize_strategy(self, X_l_t, y_l_t):
        if len(X_l_t) > 0:
            self.model.eval()
            with torch.no_grad():
                logits, _, _, _ = self.model(X_l_t, use_dropout=False)
                self.rho_hat = F.cross_entropy(logits, y_l_t, reduction='none').mean().item()
    def update_epoch_params(self, epoch): self.current_epoch = epoch
    def compute_pseudo_labels(self, weak_x):
        weak_logits, _, _, _ = self.model(weak_x)
        conf, pseudo = torch.max(torch.softmax(weak_logits, dim=1), dim=1)
        loss_per = F.cross_entropy(weak_logits, pseudo, reduction='none')
        rho_t = self.C * (self.gamma ** (-self.current_epoch)) * self.rho_hat
        mask = (loss_per <= rho_t).float()
        return pseudo, mask, conf, mask.mean().item(), 0.0

class SLDClassifier(BaseSSLClassifier):
    def __init__(self, temperature=0.5, **kwargs): 
        super().__init__(**kwargs)
        self.temperature = temperature
    def get_method_name(self): return "SLD"
    def compute_pseudo_labels(self, weak_x):
        weak_logits, _, _, _ = self.model(weak_x)
        probs = torch.softmax(weak_logits, dim=1)
        conf, pseudo = torch.max(probs, dim=1)
        with torch.no_grad():
            num_classes = probs.shape[1]
            counts = torch.bincount(pseudo, minlength=num_classes).float()
            dist = counts / (counts.sum() + 1e-6)
            current_mu, current_std = conf.mean(), conf.std() + 1e-6
            bias_factor = (dist[pseudo] - 1.0 / num_classes).clamp(min=0)
            adaptive_tau = torch.clamp(current_mu - current_std + bias_factor, min=0.75)
            dynamic_range = (conf.max() - conf.min()) + 1e-6
        conf_mod = torch.sigmoid((conf - adaptive_tau) / (self.temperature * dynamic_range))
        mask = conf_mod * conf.detach()
        return pseudo, mask.detach(), conf, conf_mod.mean().item(), conf_mod.mean().item()
        
class SoftMatchClassifier(BaseSSLClassifier):
    def __init__(self, temperature=SOFTMATCH_TEMPERATURE, ema_alpha=SOFTMATCH_EMA_ALPHA, **kwargs):
        super().__init__(**kwargs)
        self.temperature, self.ema_alpha, self.tau_ema = temperature, ema_alpha, 0.999
    def get_method_name(self): return "SOFTMATCH"
    def compute_pseudo_labels(self, weak_x):
        weak_logits, _, _, _ = self.model(weak_x)
        conf, pseudo = torch.max(torch.softmax(weak_logits, dim=1), dim=1)
        self.tau_ema = self.ema_alpha * self.tau_ema + (1.0 - self.ema_alpha) * conf.mean().item()
        tau_clamp = max(1e-6, min(1.0, self.tau_ema))
        soft_weight = torch.exp(-(1.0 - conf) / (tau_clamp * self.temperature))
        return pseudo, soft_weight.detach(), conf, (soft_weight >= 0.5).float().mean().item(), soft_weight.mean().item()

# ============================== 实验流程 ==============================
def train_and_evaluate(dataset, n_labeled, run, method, model, tokenizer, device):
    set_seed(42 + run)
    data, labels, num_classes = load_data(dataset)
    if not data: return None
    features = extract_features_with_attention(data, dataset, model, tokenizer, device)
    reduced = apply_pca(features, n_components=PCA_N_COMPONENTS, dataset=dataset)

    X_train_full, X_test, y_train_full, y_test = train_test_split(reduced, labels, test_size=TEST_SET_SIZE, random_state=42+run)
    X_pool, X_val, y_pool, y_val = train_test_split(X_train_full, y_train_full, test_size=VALIDATION_SIZE, random_state=42+run)
    X_labeled, X_unlabeled, y_labeled, y_u_true = train_test_split(X_pool, y_pool, train_size=n_labeled, random_state=42+run)

    dist_info = {int(k): int(v) for k, v in zip(*np.unique(y_labeled, return_counts=True))}
    
    params = {'input_dim': PCA_N_COMPONENTS, 'num_classes': num_classes, 'device': device}
    if method == 'FIXMATCH': clf = FixMatchClassifier(**params)
    elif method == 'DASH': clf = DASHClassifier(**params)
    elif method == 'SLD': clf = SLDClassifier(**params)
    elif method == 'SOFTMATCH': clf = SoftMatchClassifier(**params)

    # 执行模型训练
    clf, final_epoch, avg_acc_ratio, avg_ps_acc = clf.fit(X_labeled, y_labeled, X_unlabeled, y_u_true, X_val, y_val)
    
    clf.model.eval()
    detailed_info = {}
    with torch.no_grad():
        X_u_t = torch.FloatTensor(X_unlabeled).to(device)
        # 对无标签数据进行弱增强前向传播
        weak_x = clf.augmentation.weak_augment(X_u_t)
        
        # 获取伪标签、软权重（mask）以及原始置信度（conf）
        pseudo, mask, conf, _, _ = clf.compute_pseudo_labels(weak_x)
        
        # 记录每个样本的具体指标
        detailed_info = {
            'weights': mask.cpu().numpy(),          # 软门控产生的权重
            'confidences': conf.cpu().numpy(),      # 模型原始置信度
            'is_correct': (pseudo == torch.LongTensor(y_u_true).to(device)).cpu().numpy(), # 预测是否正确
            'dist_at_run': dist_info                # 该运行轮次的类别分布
        }
    
    # 测试集评估
    with torch.no_grad():
        test_logits, _, _, _ = clf.model(torch.FloatTensor(X_test).to(device), False)
        pred = torch.argmax(test_logits, dim=1).cpu().numpy()
    
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred, average='binary')
    logger.info(f"[{dataset}] [{method}] Run {run+1}: Acc={acc:.4f}, Dist={dist_info}")

    # 返回包含样本级数据的完整结果字典
    return {
        'dataset': dataset, 
        'accuracy': acc, 
        'f1': f1, 
        'method': method, 
        'labeled': n_labeled, 
        'epochs': final_epoch, 
        'history': clf.history, 
        'avg_acceptance_ratio': avg_acc_ratio, 
        'avg_pseudo_acc': avg_ps_acc, 
        'label_dist': dist_info,
        'detailed_info': detailed_info 
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
    model = BertModel.from_pretrained('./bert-base-chinese').to(device)
    
    all_results = []
    for method in METHODS_TO_RUN:
        for dataset in DATASETS_TO_RUN:
            for n_labeled in LABELED_CONFIGS:
                for run in range(NUM_RUNS):
                    res = train_and_evaluate(dataset, n_labeled, run, method, model, tokenizer, device)
                    if res: all_results.append(res)

    if all_results:
        df = pd.DataFrame(all_results)
        
        # 遍历数据集进行格式化打印
        for dataset in DATASETS_TO_RUN:
            print(f"\n【数据集: {dataset}】")
            print("-" * 120)
            
            dataset_df = df[df['dataset'] == dataset]
            
            # 按方法和标注数分组计算
            summary = dataset_df.groupby(['method', 'labeled']).agg({
                'accuracy': ['mean', 'std'],
                'f1': ['mean', 'std'],
                'avg_acceptance_ratio': ['mean', 'std'],
                'avg_pseudo_acc': ['mean'],
                'epochs': ['mean']
            })
            summary = summary.sort_index()
            
            for (method, n_labeled), row in summary.iterrows():
                acc_mean = row[('accuracy', 'mean')]
                acc_std = row[('accuracy', 'std')]
                f1_mean = row[('f1', 'mean')]
                f1_std = row[('f1', 'std')]
                recept_mean = row[('avg_acceptance_ratio', 'mean')]
                recept_std = row[('avg_acceptance_ratio', 'std')]
                ps_acc = row[('avg_pseudo_acc', 'mean')]
                epochs = row[('epochs', 'mean')]    
                print(f"{method:<12} | Labeled= {n_labeled:<2} | "
                      f"Acc: {acc_mean:.4f}±{acc_std:.4f} | "
                      f"F1: {f1_mean:.4f}±{f1_std:.4f} | "
                      f"Acceptance: {recept_mean:.3f}±{recept_std:.3f} | "
                      f"PseudoAcc: {ps_acc:.3f} | "
                      f"Epochs: {epochs:.1f}")
            
            print("-" * 120)

        output_file = f"results_{'_'.join(METHODS_TO_RUN)}_{'_'.join(DATASETS_TO_RUN)}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"结果已保存至 {output_file}")

if __name__ == '__main__':
    main()
