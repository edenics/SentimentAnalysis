import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import logging
import os
from collections import Counter
from transformers import BertTokenizer, BertModel
import json

# 设置全局变量
TRAIN_SIZE = 0.4  # 训练集比例
NOISE_RATE = 0.1  # 标签噪声率
FIXED_CONFIDENCE_THRESHOLD_1 = 0.8  # 固定置信度阈值1
FIXED_CONFIDENCE_THRESHOLD_2 = 0.9  # 固定置信度阈值2
NUM_RUNS = 10  # 实验运行次数
MLP_HIDDEN_LAYERS = (20, 10)  # MLP 隐藏层结构
MIN_DISTANCE_CHANGE = 0.01  # 早停条件：最小距离变化阈值
MAX_DISTANCE_THRESHOLD = 2.4  # 最大距离阈值
GAMMA = 1.27  # 距离阈值衰减系数
DISTANCE_SCALING_FACTOR = 1  # 距离阈值缩放系数
INIT_DISTANCE_SCALING_FACTOR = 0.6

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置全局随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# 文件路径
file_path = 'ChnSentiCorp_htl_all.csv'
feature_file = 'bert_features_hybrid_v5.npy'

# 读取数据
labels = []
data = []
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        header = lines[0]
        data_lines = lines[1:]
        random.shuffle(data_lines)
        for line in data_lines:
            parts = line.strip().split(',', 1)
            if len(parts) > 1:
                labels.append(int(parts[0]))
                data.append(parts[1].strip())
except FileNotFoundError:
    logging.error(f"文件 {file_path} 未找到，请检查路径！")
    exit()

# 加载或提取特征
if not os.path.exists(feature_file):
    logging.info(f"特征文件 {feature_file} 未找到，重新提取特征...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    model.eval()
    features = []
    with torch.no_grad():
        for text in data:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model(**inputs)
            features.append(outputs.last_hidden_state[:, 0, :].numpy())  # 使用 [CLS] 标记的向量
    s_array = np.vstack(features)
    np.save(feature_file, s_array)
    logging.info(f"特征提取完成，保存到 {feature_file}，形状：{s_array.shape}")
else:
    s_array = np.load(feature_file)
    logging.info(f"加载特征矩阵形状：{s_array.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    s_array, labels, train_size=TRAIN_SIZE, random_state=42, stratify=labels
)
y_train = np.array(y_train)
y_test = np.array(y_test)
logging.info(f"训练数据量：{len(X_train)}，测试数据量：{len(X_test)}")

# PCA 降维（仅在训练集上拟合）
pca = PCA(n_components=20, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# 引入标签噪声
def add_label_noise(labels, seed=None):
    if seed is not None:
        random.seed(seed)
    noisy_labels = labels.copy()
    if NOISE_RATE == 0:
        return noisy_labels
    
    # 计算正负标签的数量
    positive_labels = [i for i, label in enumerate(labels) if label == 1]
    negative_labels = [i for i, label in enumerate(labels) if label == 0]
    num_positive = len(positive_labels)
    num_negative = len(negative_labels)
    
    # 按比例计算正负标签的噪声数量
    num_noise_positive = max(1, int(num_positive * NOISE_RATE)) if num_positive > 0 else 0
    num_noise_negative = max(1, int(num_negative * NOISE_RATE)) if num_negative > 0 else 0
    num_noise = num_noise_positive + num_noise_negative
    
    # 随机选择正负标签的索引进行翻转
    noise_indices_positive = random.sample(positive_labels, min(num_noise_positive, num_positive)) if num_positive > 0 else []
    noise_indices_negative = random.sample(negative_labels, min(num_noise_negative, num_negative)) if num_negative > 0 else []
    noise_indices = noise_indices_positive + noise_indices_negative
    
    # 翻转选中的正负标签
    for idx in noise_indices:
        noisy_labels[idx] = 1 - noisy_labels[idx]
    
    return noisy_labels

# 算法 A（DASH）
class PaperDASHClassifier:
    def __init__(self, base_estimators, tau_0=1.0, gamma=GAMMA, tau_min=0.05, max_iter=100, verbose=False):
        self.base_estimators = base_estimators
        self.tau_0 = tau_0
        self.gamma = gamma
        self.tau_min = tau_min
        self.max_iter = max_iter
        self.verbose = verbose
        self.threshold = None
        self.transduction_ = None
        self.initial_loss_threshold = None
        self.convergence_iteration = 0
        self.weights = None

    def _estimate_pseudo_label_quality(self, predictions, y_unlabeled_true, final_mask, unlabeled_indices, labeled_size):
        pseudo_labels = predictions[final_mask]
        if len(pseudo_labels) > 0:
            selected_indices = unlabeled_indices[final_mask] - labeled_size  # Adjust for labeled offset
            true_labels = y_unlabeled_true[selected_indices]
            correct = np.sum(pseudo_labels == true_labels)
            total = len(pseudo_labels)
            return correct / total if total > 0 else 0.0
        return 0.0

    def _fit_iteration(self, X, y, unlabeled_indices, y_unlabeled_true, iteration, labeled_size):
        labeled_mask = y != -1
        if len(unlabeled_indices) == 0:
            return y

        self.base_estimators[0].fit(X[labeled_mask], y[labeled_mask])
        probas = self.base_estimators[0].predict_proba(X[unlabeled_indices])

        n_samples = len(unlabeled_indices)
        confidences = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)
        consistent_mask = np.ones(n_samples, dtype=bool)

        log_loss = -np.log(confidences + 1e-9)
        if iteration == 0 and self.initial_loss_threshold is None:
            labeled_probas = self.base_estimators[0].predict_proba(X[labeled_mask])
            labeled_confidences = np.max(labeled_probas, axis=1)
            self.initial_loss_threshold = np.mean(-np.log(labeled_confidences + 1e-9))
            if self.verbose:
                logging.info(f"[A] 迭代 {iteration}, 初始损失阈值: {self.initial_loss_threshold:.3f}")

        C = 1.0001
        dynamic_loss_threshold = C * (self.gamma ** -(iteration)) * self.initial_loss_threshold
        loss_threshold = max(self.tau_min, dynamic_loss_threshold)

        high_conf_mask = log_loss <= loss_threshold
        final_mask = high_conf_mask & consistent_mask

        Q_t = self._estimate_pseudo_label_quality(predictions, y_unlabeled_true, final_mask, unlabeled_indices, labeled_size)
        pseudo_labels = np.full(n_samples, -1, dtype=int)
        pseudo_labels[final_mask] = predictions[final_mask]
        y[unlabeled_indices[final_mask]] = pseudo_labels[final_mask]

        if self.verbose:
            logging.info(f"[A] 迭代 {iteration}, 伪标签数: {np.sum(final_mask)}, 伪标签质量: {Q_t:.4f}, 损失阈值: {loss_threshold:.3f}")
        return y

    def fit(self, X, y, y_unlabeled_true, labeled_size):
        self.transduction_ = y.copy()
        unlabeled_indices = np.where(y == -1)[0]
        for iteration in range(self.max_iter):
            y_old = self.transduction_.copy()
            self.transduction_ = self._fit_iteration(X, self.transduction_, unlabeled_indices, y_unlabeled_true, iteration, labeled_size)
            if np.array_equal(y_old, self.transduction_):
                self.convergence_iteration = iteration + 1
                if self.verbose:
                    logging.info(f"[A] 迭代 {iteration+1} 收敛，终止训练")
                break
        labeled_mask = self.transduction_ != -1
        self.weights = [1.0]
        self.base_estimators[0].fit(X[labeled_mask], self.transduction_[labeled_mask])
        return self

    def predict(self, X):
        probas = self.base_estimators[0].predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return predictions

# 算法 B（基于距离阈值，优化初始阈值，保留早停）
class ImprovedDASHClassifier:
    def __init__(self, base_estimators, tau_0=1.0, gamma=GAMMA, tau_min=0.05, max_iter=100, verbose=True, ablation='full'):
        self.base_estimators = base_estimators
        self.tau_0 = tau_0
        self.gamma = gamma
        self.tau_min = tau_min
        self.max_iter = max_iter
        self.verbose = verbose
        self.ablation = ablation
        self.transduction_ = None
        self.previous_transduction_ = None
        self.initial_loss_threshold = None
        self.initial_distance = None
        self.convergence_iteration = 0
        self.weights = None
        self.mean_distances = []
        self.prev_mean_dist = None
        self.distance_threshold = None
        self.prev_threshold = None

    def _calculate_distance_threshold(self, iteration, X, labeled_mask, unlabeled_indices):
        distances = euclidean_distances(X[unlabeled_indices], X[labeled_mask])
        min_distances = np.min(distances, axis=1)
        mean_dist = np.mean(min_distances)
        std_dist = np.std(min_distances)
        self.mean_distances.append(mean_dist)
        
        if iteration == 0:
            self.initial_distance = mean_dist
            threshold = mean_dist - INIT_DISTANCE_SCALING_FACTOR * std_dist
            threshold = min(threshold, MAX_DISTANCE_THRESHOLD)  # 限制不超过最大阈值
        else:
            threshold = DISTANCE_SCALING_FACTOR * self.gamma ** (-iteration) * self.initial_distance
            threshold = min(threshold, MAX_DISTANCE_THRESHOLD)

        if self.verbose:
            logging.info(f"[B-{self.ablation}] 迭代 {iteration}, min_distances 均值: {mean_dist:.3f}, 标准差: {std_dist:.3f}, 距离阈值: {threshold:.3f}")
        return threshold, mean_dist

    def _estimate_pseudo_label_quality(self, predictions, y_unlabeled_true, final_mask, unlabeled_indices, labeled_size):
        pseudo_labels = predictions[final_mask]
        if len(pseudo_labels) > 0:
            selected_indices = unlabeled_indices[final_mask] - labeled_size  # Adjust for labeled offset
            true_labels = y_unlabeled_true[selected_indices]
            correct = np.sum(pseudo_labels == true_labels)
            total = len(pseudo_labels)
            return correct / total if total > 0 else 0.0
        return 0.0

    def _fit_iteration(self, X, y, unlabeled_indices, y_unlabeled_true, iteration, labeled_size):
        labeled_mask = y != -1
        if len(unlabeled_indices) == 0:
            return y

        self.base_estimators[0].fit(X[labeled_mask], y[labeled_mask])
        probas = self.base_estimators[0].predict_proba(X[unlabeled_indices])

        n_samples = len(unlabeled_indices)
        confidences = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)

        log_loss = -np.log(confidences + 1e-9)
        if iteration == 0 and self.initial_loss_threshold is None:
            labeled_probas = self.base_estimators[0].predict_proba(X[labeled_mask])
            labeled_confidences = np.max(labeled_probas, axis=1)
            self.initial_loss_threshold = np.mean(-np.log(labeled_confidences + 1e-9))
            if self.verbose:
                logging.info(f"[B-{self.ablation}] 迭代 {iteration}, 初始损失阈值: {self.initial_loss_threshold:.3f}")

        C = 1.0001
        dynamic_loss_threshold = C * (self.gamma ** -(iteration)) * self.initial_loss_threshold
        loss_threshold = max(self.tau_min, dynamic_loss_threshold)

        self.prev_threshold = self.distance_threshold
        self.distance_threshold, mean_dist = self._calculate_distance_threshold(iteration, X, labeled_mask, unlabeled_indices)
        distances = euclidean_distances(X[unlabeled_indices], X[labeled_mask])
        min_distances = np.min(distances, axis=1)
        dist_mask = min_distances <= self.distance_threshold

        high_conf_mask = log_loss <= loss_threshold
        final_mask = dist_mask & high_conf_mask

        Q_t = self._estimate_pseudo_label_quality(predictions, y_unlabeled_true, final_mask, unlabeled_indices, labeled_size)
        pseudo_labels = np.full(n_samples, -1, dtype=int)
        pseudo_labels[final_mask] = predictions[final_mask]

        self.previous_transduction_ = y.copy()
        y[unlabeled_indices[final_mask]] = pseudo_labels[final_mask]

        if self.verbose:
            logging.info(f"[B-{self.ablation}] 迭代 {iteration}, 伪标签数: {np.sum(final_mask)}, 伪标签质量: {Q_t:.4f}, 距离阈值: {self.distance_threshold:.3f}, 损失阈值: {loss_threshold:.3f}")
        return y

    def fit(self, X, y, y_unlabeled_true, labeled_size):
        self.transduction_ = y.copy()
        if self.transduction_ is None:
            raise ValueError("transduction_ 初始化失败，检查输入 y 是否有效")
        unlabeled_indices = np.where(y == -1)[0]
        for iteration in range(self.max_iter):
            if self.transduction_ is None:
                raise ValueError(f"迭代 {iteration} 前 transduction_ 为 None")
            y_old = self.transduction_.copy()
            self.transduction_ = self._fit_iteration(X, self.transduction_, unlabeled_indices, y_unlabeled_true, iteration, labeled_size)
            self.convergence_iteration = iteration + 1

            # 早停条件：仅对 full 模式启用基于 min_distances 均值变化的早停
            if self.ablation == 'full':
                mean_dist = self.mean_distances[-1] if self.mean_distances else None
                if iteration > 0 and self.prev_mean_dist is not None and abs(mean_dist - self.prev_mean_dist) < MIN_DISTANCE_CHANGE:
                    if self.verbose:
                        logging.info(f"[B-{self.ablation}] 迭代 {iteration}, min_distances 均值变化 {abs(mean_dist - self.prev_mean_dist):.4f} 小于 {MIN_DISTANCE_CHANGE}，终止训练")
                    if iteration > 0 and self.previous_transduction_ is not None:
                        self.transduction_ = self.previous_transduction_
                    break
                self.prev_mean_dist = mean_dist

            # 其他早停条件：检查伪标签是否收敛
            if np.array_equal(y_old, self.transduction_):
                if self.verbose:
                    logging.info(f"[B-{self.ablation}] 迭代 {iteration+1} 收敛，终止训练")
                break

            unlabeled_indices = np.where(y == -1)[0]

        labeled_mask = self.transduction_ != -1
        self.weights = [1.0]
        self.base_estimators[0].fit(X[labeled_mask], self.transduction_[labeled_mask])
        return self

    def predict(self, X):
        probas = self.base_estimators[0].predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return predictions

# 算法 C（基于固定置信度）
class FixedConfidenceDASHClassifier:
    def __init__(self, base_estimators, fixed_confidence, max_iter=150, verbose=True, ablation=''):
        self.base_estimators = base_estimators
        self.fixed_confidence = fixed_confidence
        self.max_iter = max_iter
        self.verbose = verbose
        self.transduction_ = None
        self.convergence_iteration = 0
        self.weights = None

    def _estimate_pseudo_label_quality(self, predictions, y_unlabeled_true, final_mask, unlabeled_indices, labeled_size):
        pseudo_labels = predictions[final_mask]
        if len(pseudo_labels) > 0:
            selected_indices = unlabeled_indices[final_mask] - labeled_size  # Adjust for labeled offset
            true_labels = y_unlabeled_true[selected_indices]
            correct = np.sum(pseudo_labels == true_labels)
            total = len(pseudo_labels)
            return correct / total if total > 0 else 0.0
        return 0.0

    def _fit_iteration(self, X, y, unlabeled_indices, y_unlabeled_true, iteration, labeled_size):
        labeled_mask = y != -1
        if len(unlabeled_indices) == 0:
            return y

        self.base_estimators[0].fit(X[labeled_mask], y[labeled_mask])
        probas = self.base_estimators[0].predict_proba(X[unlabeled_indices])

        n_samples = len(unlabeled_indices)
        confidences = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)

        high_conf_mask = confidences >= self.fixed_confidence
        final_mask = high_conf_mask

        Q_t = self._estimate_pseudo_label_quality(predictions, y_unlabeled_true, final_mask, unlabeled_indices, labeled_size)
        pseudo_labels = np.full(n_samples, -1, dtype=int)
        pseudo_labels[final_mask] = predictions[final_mask]
        y[unlabeled_indices[final_mask]] = pseudo_labels[final_mask]

        if self.verbose:
            logging.info(f"[C-{self.ablation}] 迭代 {iteration}, 伪标签数: {np.sum(final_mask)}, 伪标签质量: {Q_t:.4f}, 置信度阈值: {self.fixed_confidence:.3f}, 伪标签分布: {Counter(pseudo_labels[final_mask])}")
        return y

    def fit(self, X, y, y_unlabeled_true, labeled_size):
        self.transduction_ = y.copy()
        unlabeled_indices = np.where(y == -1)[0]
        for iteration in range(self.max_iter):
            y_old = self.transduction_.copy()
            self.transduction_ = self._fit_iteration(X, self.transduction_, unlabeled_indices, y_unlabeled_true, iteration, labeled_size)
            self.convergence_iteration = iteration + 1

            # 仅检查伪标签是否收敛
            if np.array_equal(y_old, self.transduction_):
                if self.verbose:
                    logging.info(f"[C-{self.ablation}] 迭代 {iteration+1} 收敛，终止训练")
                break

            unlabeled_indices = np.where(y == -1)[0]

        labeled_mask = self.transduction_ != -1
        self.weights = [1.0]
        self.base_estimators[0].fit(X[labeled_mask], self.transduction_[labeled_mask])
        return self

    def predict(self, X):
        probas = self.base_estimators[0].predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return predictions

# 单模型预测器
class SingleModelPredictor:
    def __init__(self, base_estimators):
        self.base_estimators = base_estimators

    def fit(self, X, y):
        self.base_estimators[0].fit(X, y)
        return self

    def predict(self, X):
        probas = self.base_estimators[0].predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return predictions

# 训练和评估函数
def train_and_evaluate_with_mlp():
    labeled_ratios = [0.01, 0.02, 0.03]
    a_test_acc_list = []
    a_test_f1_list = []
    a_pseudo_acc_list = []
    a_pseudo_count_list = []
    a_convergence_list = []
    b_full_test_acc_list = []
    b_full_test_f1_list = []
    b_full_pseudo_acc_list = []
    b_full_pseudo_count_list = []
    b_full_convergence_list = []
    b_no_stop_test_acc_list = []
    b_no_stop_test_f1_list = []
    b_no_stop_pseudo_acc_list = []
    b_no_stop_pseudo_count_list = []
    b_no_stop_convergence_list = []
    c_fixed_08_test_acc_list = []
    c_fixed_08_test_f1_list = []
    c_fixed_08_pseudo_acc_list = []
    c_fixed_08_pseudo_count_list = []
    c_fixed_08_convergence_list = []
    c_fixed_09_test_acc_list = []
    c_fixed_09_test_f1_list = []
    c_fixed_09_pseudo_acc_list = []
    c_fixed_09_pseudo_count_list = []
    c_fixed_09_convergence_list = []
    ratio_list = []
    mean_distances_no_stop = []

    for labeled_ratio in labeled_ratios:
        n_labeled = max(4, int(len(X_train) * labeled_ratio))
        ratio = n_labeled / len(X_train)
        ratio_list.append(ratio)
        logging.info(f"标记比例: {ratio*100:.2f}% ({n_labeled} 样本)")
        run_acc_a = []
        run_f1_a = []
        run_pseudo_acc_a = []
        run_pseudo_count_a = []
        run_convergence_a = []
        run_acc_b_full = []
        run_f1_b_full = []
        run_pseudo_acc_b_full = []
        run_pseudo_count_b_full = []
        run_convergence_b_full = []
        run_acc_b_no_stop = []
        run_f1_b_no_stop = []
        run_pseudo_acc_b_no_stop = []
        run_pseudo_count_b_no_stop = []
        run_convergence_b_no_stop = []
        run_acc_c_fixed_08 = []
        run_f1_c_fixed_08 = []
        run_pseudo_acc_c_fixed_08 = []
        run_pseudo_count_c_fixed_08 = []
        run_convergence_c_fixed_08 = []
        run_acc_c_fixed_09 = []
        run_f1_c_fixed_09 = []
        run_pseudo_acc_c_fixed_09 = []
        run_pseudo_count_c_fixed_09 = []
        run_convergence_c_fixed_09 = []
        mean_distances_runs = []

        for run in range(NUM_RUNS):
            set_seed(42 + run)
            X_labeled, X_unlabeled, y_labeled, y_unlabeled_true = train_test_split(
                X_train, y_train, train_size=n_labeled, random_state=42 + run, stratify=y_train
            )

            y_labeled_noisy = add_label_noise(y_labeled, seed=42 + run)

            X_all = np.vstack((X_labeled, X_unlabeled))
            y_unlabeled = np.array([-1] * len(X_unlabeled))
            y_all = np.concatenate([y_labeled_noisy, y_unlabeled])

            base_estimators = [MLPClassifier(hidden_layer_sizes=MLP_HIDDEN_LAYERS, max_iter=2000, random_state=42 + run)]

            # 算法 A
            dash_a = PaperDASHClassifier(base_estimators, tau_0=1.0, gamma=GAMMA, tau_min=0.05, max_iter=100, verbose=False)
            dash_a.fit(X_all, y_all, y_unlabeled_true, labeled_size=len(X_labeled))
            pred_a = dash_a.predict(X_test)
            acc_a = accuracy_score(y_test, pred_a)
            f1_a = f1_score(y_test, pred_a, average='weighted')
            run_acc_a.append(acc_a)
            run_f1_a.append(f1_a)
            run_convergence_a.append(dash_a.convergence_iteration)

            pseudo_labels_a = dash_a.transduction_[-len(X_unlabeled):]
            labeled_mask_a = pseudo_labels_a != -1
            pseudo_acc_a = accuracy_score(y_unlabeled_true[labeled_mask_a], pseudo_labels_a[labeled_mask_a]) if np.sum(labeled_mask_a) > 0 else 0.0
            pseudo_count_a = np.sum(labeled_mask_a)
            run_pseudo_acc_a.append(pseudo_acc_a)
            run_pseudo_count_a.append(pseudo_count_a)
            logging.info(f"[A] 运行 {run}, 伪标签准确率: {pseudo_acc_a:.4f}, 伪标签样本数: {pseudo_count_a}, 测试集准确率: {acc_a:.4f}, 测试集F1: {f1_a:.4f}, 迭代次数: {dash_a.convergence_iteration}")

            # 算法 B（有早停）
            dash_b_full = ImprovedDASHClassifier(
                base_estimators, tau_0=1.0, gamma=GAMMA, tau_min=0.05, max_iter=100, verbose=True, ablation='full'
            )
            dash_b_full.fit(X_all, y_all, y_unlabeled_true, labeled_size=len(X_labeled))
            pred_b_full = dash_b_full.predict(X_test)
            acc_b_full = accuracy_score(y_test, pred_b_full)
            f1_b_full = f1_score(y_test, pred_b_full, average='weighted')
            run_acc_b_full.append(acc_b_full)
            run_f1_b_full.append(f1_b_full)
            run_convergence_b_full.append(dash_b_full.convergence_iteration)

            pseudo_labels_b_full = dash_b_full.transduction_[-len(X_unlabeled):]
            labeled_mask_b_full = pseudo_labels_b_full != -1
            pseudo_acc_b_full = accuracy_score(y_unlabeled_true[labeled_mask_b_full], pseudo_labels_b_full[labeled_mask_b_full]) if np.sum(labeled_mask_b_full) > 0 else 0.0
            pseudo_count_b_full = np.sum(labeled_mask_b_full)
            run_pseudo_acc_b_full.append(pseudo_acc_b_full)
            run_pseudo_count_b_full.append(pseudo_count_b_full)
            logging.info(f"[B-full] 运行 {run}, 伪标签准确率: {pseudo_acc_b_full:.4f}, 伪标签样本数: {pseudo_count_b_full}, 测试集准确率: {acc_b_full:.4f}, 测试集F1: {f1_b_full:.4f}, 迭代次数: {dash_b_full.convergence_iteration}")

            # 算法 B（无早停）
            dash_b_no_stop = ImprovedDASHClassifier(
                base_estimators, tau_0=1.0, gamma=GAMMA, tau_min=0.05, max_iter=100, verbose=False, ablation='no_early_stop'
            )
            dash_b_no_stop.fit(X_all, y_all, y_unlabeled_true, labeled_size=len(X_labeled))
            pred_b_no_stop = dash_b_no_stop.predict(X_test)
            acc_b_no_stop = accuracy_score(y_test, pred_b_no_stop)
            f1_b_no_stop = f1_score(y_test, pred_b_no_stop, average='weighted')
            run_acc_b_no_stop.append(acc_b_no_stop)
            run_f1_b_no_stop.append(f1_b_no_stop)
            run_convergence_b_no_stop.append(dash_b_no_stop.convergence_iteration)

            pseudo_labels_b_no_stop = dash_b_no_stop.transduction_[-len(X_unlabeled):]
            labeled_mask_b_no_stop = pseudo_labels_b_no_stop != -1
            pseudo_acc_b_no_stop = accuracy_score(y_unlabeled_true[labeled_mask_b_no_stop], pseudo_labels_b_no_stop[labeled_mask_b_no_stop]) if np.sum(labeled_mask_b_no_stop) > 0 else 0.0
            pseudo_count_b_no_stop = np.sum(labeled_mask_b_no_stop)
            run_pseudo_acc_b_no_stop.append(pseudo_acc_b_no_stop)
            run_pseudo_count_b_no_stop.append(pseudo_count_b_no_stop)
            logging.info(f"[B-no_early_stop] 运行 {run}, 伪标签准确率: {pseudo_acc_b_no_stop:.4f}, 伪标签样本数: {pseudo_count_b_no_stop}, 测试集准确率: {acc_b_no_stop:.4f}, 测试集F1: {f1_b_no_stop:.4f}, 迭代次数: {dash_b_no_stop.convergence_iteration}")

            mean_distances_runs.append(dash_b_no_stop.mean_distances)

            # 算法 C（固定置信度 0.8）
            dash_c_fixed_08 = FixedConfidenceDASHClassifier(
                base_estimators, fixed_confidence=FIXED_CONFIDENCE_THRESHOLD_1, max_iter=150, verbose=False, ablation='fixed_0.8'
            )
            dash_c_fixed_08.fit(X_all, y_all, y_unlabeled_true, labeled_size=len(X_labeled))
            pred_c_fixed_08 = dash_c_fixed_08.predict(X_test)
            acc_c_fixed_08 = accuracy_score(y_test, pred_c_fixed_08)
            f1_c_fixed_08 = f1_score(y_test, pred_c_fixed_08, average='weighted')
            run_acc_c_fixed_08.append(acc_c_fixed_08)
            run_f1_c_fixed_08.append(f1_c_fixed_08)
            run_convergence_c_fixed_08.append(dash_c_fixed_08.convergence_iteration)

            pseudo_labels_c_fixed_08 = dash_c_fixed_08.transduction_[-len(X_unlabeled):]
            labeled_mask_c_fixed_08 = pseudo_labels_c_fixed_08 != -1
            pseudo_acc_c_fixed_08 = accuracy_score(y_unlabeled_true[labeled_mask_c_fixed_08], pseudo_labels_c_fixed_08[labeled_mask_c_fixed_08]) if np.sum(labeled_mask_c_fixed_08) > 0 else 0.0
            pseudo_count_c_fixed_08 = np.sum(labeled_mask_c_fixed_08)
            run_pseudo_acc_c_fixed_08.append(pseudo_acc_c_fixed_08)
            run_pseudo_count_c_fixed_08.append(pseudo_count_c_fixed_08)
            logging.info(f"[C-fixed_0.8] 运行 {run}, 伪标签准确率: {pseudo_acc_c_fixed_08:.4f}, 伪标签样本数: {pseudo_count_c_fixed_08}, 测试集准确率: {acc_c_fixed_08:.4f}, 测试集F1: {f1_c_fixed_08:.4f}, 迭代次数: {dash_c_fixed_08.convergence_iteration}")

            # 算法 C（固定置信度 0.9）
            dash_c_fixed_09 = FixedConfidenceDASHClassifier(
                base_estimators, fixed_confidence=FIXED_CONFIDENCE_THRESHOLD_2, max_iter=150, verbose=False, ablation='fixed_0.9'
            )
            dash_c_fixed_09.fit(X_all, y_all, y_unlabeled_true, labeled_size=len(X_labeled))
            pred_c_fixed_09 = dash_c_fixed_09.predict(X_test)
            acc_c_fixed_09 = accuracy_score(y_test, pred_c_fixed_09)
            f1_c_fixed_09 = f1_score(y_test, pred_c_fixed_09, average='weighted')
            run_acc_c_fixed_09.append(acc_c_fixed_09)
            run_f1_c_fixed_09.append(f1_c_fixed_09)
            run_convergence_c_fixed_09.append(dash_c_fixed_09.convergence_iteration)

            pseudo_labels_c_fixed_09 = dash_c_fixed_09.transduction_[-len(X_unlabeled):]
            labeled_mask_c_fixed_09 = pseudo_labels_c_fixed_09 != -1
            pseudo_acc_c_fixed_09 = accuracy_score(y_unlabeled_true[labeled_mask_c_fixed_09], pseudo_labels_c_fixed_09[labeled_mask_c_fixed_09]) if np.sum(labeled_mask_c_fixed_09) > 0 else 0.0
            pseudo_count_c_fixed_09 = np.sum(labeled_mask_c_fixed_09)
            run_pseudo_acc_c_fixed_09.append(pseudo_acc_c_fixed_09)
            run_pseudo_count_c_fixed_09.append(pseudo_count_c_fixed_09)
            logging.info(f"[C-fixed_0.9] 运行 {run}, 伪标签准确率: {pseudo_acc_c_fixed_09:.4f}, 伪标签样本数: {pseudo_count_c_fixed_09}, 测试集准确率: {acc_c_fixed_09:.4f}, 测试集F1: {f1_c_fixed_09:.4f}, 迭代次数: {dash_c_fixed_09.convergence_iteration}")

        # 计算均值和标准差
        for metric, run_data, avg_list in [
            ('acc_a', run_acc_a, a_test_acc_list),
            ('f1_a', run_f1_a, a_test_f1_list),
            ('pseudo_acc_a', run_pseudo_acc_a, a_pseudo_acc_list),
            ('pseudo_count_a', run_pseudo_count_a, a_pseudo_count_list),
            ('convergence_a', run_convergence_a, a_convergence_list),
            ('acc_b_full', run_acc_b_full, b_full_test_acc_list),
            ('f1_b_full', run_f1_b_full, b_full_test_f1_list),
            ('pseudo_acc_b_full', run_pseudo_acc_b_full, b_full_pseudo_acc_list),
            ('pseudo_count_b_full', run_pseudo_count_b_full, b_full_pseudo_count_list),
            ('convergence_b_full', run_convergence_b_full, b_full_convergence_list),
            ('acc_b_no_stop', run_acc_b_no_stop, b_no_stop_test_acc_list),
            ('f1_b_no_stop', run_f1_b_no_stop, b_no_stop_test_f1_list),
            ('pseudo_acc_b_no_stop', run_pseudo_acc_b_no_stop, b_no_stop_pseudo_acc_list),
            ('pseudo_count_b_no_stop', run_pseudo_count_b_no_stop, b_no_stop_pseudo_count_list),
            ('convergence_b_no_stop', run_convergence_b_no_stop, b_no_stop_convergence_list),
            ('acc_c_fixed_08', run_acc_c_fixed_08, c_fixed_08_test_acc_list),
            ('f1_c_fixed_08', run_f1_c_fixed_08, c_fixed_08_test_f1_list),
            ('pseudo_acc_c_fixed_08', run_pseudo_acc_c_fixed_08, c_fixed_08_pseudo_acc_list),
            ('pseudo_count_c_fixed_08', run_pseudo_count_c_fixed_08, c_fixed_08_pseudo_count_list),
            ('convergence_c_fixed_08', run_convergence_c_fixed_08, c_fixed_08_convergence_list),
            ('acc_c_fixed_09', run_acc_c_fixed_09, c_fixed_09_test_acc_list),
            ('f1_c_fixed_09', run_f1_c_fixed_09, c_fixed_09_test_f1_list),
            ('pseudo_acc_c_fixed_09', run_pseudo_acc_c_fixed_09, c_fixed_09_pseudo_acc_list),
            ('pseudo_count_c_fixed_09', run_pseudo_count_c_fixed_09, c_fixed_09_pseudo_count_list),
            ('convergence_c_fixed_09', run_convergence_c_fixed_09, c_fixed_09_convergence_list),
        ]:
            if 'pseudo_acc' in metric:
                total_samples = sum(run_pseudo_count_a if 'a' in metric else run_pseudo_count_b_full if 'b_full' in metric else run_pseudo_count_b_no_stop if 'b_no_stop' in metric else run_pseudo_count_c_fixed_08 if 'c_fixed_08' in metric else run_pseudo_count_c_fixed_09)
                avg_value = sum(acc * count for acc, count in zip(run_data, run_pseudo_count_a if 'a' in metric else run_pseudo_count_b_full if 'b_full' in metric else run_pseudo_count_b_no_stop if 'b_no_stop' in metric else run_pseudo_count_c_fixed_08 if 'c_fixed_08' in metric else run_pseudo_count_c_fixed_09)) / total_samples if total_samples > 0 else 0.0
            elif 'pseudo_count' in metric:
                avg_value = int(np.mean(run_data))
            elif 'convergence' in metric:
                avg_value = np.mean(run_data)
            else:
                avg_value = np.mean(run_data)
            std_value = np.std(run_data) if len(run_data) > 1 else 0.0
            avg_list.append((avg_value, std_value))

        # 计算 mean_distances 平均值
        max_iterations = max(len(distances) for distances in mean_distances_runs)
        mean_distances_avg = []
        for i in range(max_iterations):
            values = [distances[i] if i < len(distances) else None for distances in mean_distances_runs]
            valid_values = [v for v in values if v is not None]
            mean_value = round(np.mean(valid_values), 2) if valid_values else None
            mean_distances_avg.append(mean_value)
        mean_distances_no_stop.append(mean_distances_avg[:21])

        logging.info(f"[B-no_early_stop] 平均 mean_distances: {mean_distances_avg[:21]}")

        logging.info(f"[A] 平均准确率: {a_test_acc_list[-1][0]:.4f} ± {a_test_acc_list[-1][1]:.4f}, 平均F1: {a_test_f1_list[-1][0]:.4f} ± {a_test_f1_list[-1][1]:.4f}, 伪标签准确率: {a_pseudo_acc_list[-1][0]:.4f}, 伪标签计数: {int(a_pseudo_count_list[-1][0])}, 平均迭代次数: {a_convergence_list[-1][0]:.1f}")
        logging.info(f"[B-full] 平均准确率: {b_full_test_acc_list[-1][0]:.4f} ± {b_full_test_acc_list[-1][1]:.4f}, 平均F1: {b_full_test_f1_list[-1][0]:.4f} ± {b_full_test_f1_list[-1][1]:.4f}, 伪标签准确率: {b_full_pseudo_acc_list[-1][0]:.4f} ± {b_full_pseudo_acc_list[-1][1]:.4f}, 伪标签计数: {int(b_full_pseudo_count_list[-1][0])}, 平均迭代次数: {b_full_convergence_list[-1][0]:.1f}")
        logging.info(f"[B-no_early_stop] 平均准确率: {b_no_stop_test_acc_list[-1][0]:.4f} ± {b_no_stop_test_acc_list[-1][1]:.4f}, 平均F1: {b_no_stop_test_f1_list[-1][0]:.4f} ± {b_no_stop_test_f1_list[-1][1]:.4f}, 伪标签准确率: {b_no_stop_pseudo_acc_list[-1][0]:.4f} ± {b_no_stop_pseudo_acc_list[-1][1]:.4f}, 伪标签计数: {int(b_no_stop_pseudo_count_list[-1][0])}, 平均迭代次数: {b_no_stop_convergence_list[-1][0]:.1f}")
        logging.info(f"[C-fixed_0.8] 平均准确率: {c_fixed_08_test_acc_list[-1][0]:.4f} ± {c_fixed_08_test_acc_list[-1][1]:.4f}, 平均F1: {c_fixed_08_test_f1_list[-1][0]:.4f} ± {c_fixed_08_test_f1_list[-1][1]:.4f}, 伪标签准确率: {c_fixed_08_pseudo_acc_list[-1][0]:.4f} ± {c_fixed_08_pseudo_acc_list[-1][1]:.4f}, 伪标签计数: {int(c_fixed_08_pseudo_count_list[-1][0])}, 平均迭代次数: {c_fixed_08_convergence_list[-1][0]:.1f}")
        logging.info(f"[C-fixed_0.9] 平均准确率: {c_fixed_09_test_acc_list[-1][0]:.4f} ± {c_fixed_09_test_acc_list[-1][1]:.4f}, 平均F1: {c_fixed_09_test_f1_list[-1][0]:.4f} ± {c_fixed_09_test_f1_list[-1][1]:.4f}, 伪标签准确率: {c_fixed_09_pseudo_acc_list[-1][0]:.4f} ± {c_fixed_09_pseudo_acc_list[-1][1]:.4f}, 伪标签计数: {int(c_fixed_09_pseudo_count_list[-1][0])}, 平均迭代次数: {c_fixed_09_convergence_list[-1][0]:.1f}")

    return {
        'a_test_acc_list': a_test_acc_list,
        'a_test_f1_list': a_test_f1_list,
        'a_pseudo_acc_list': a_pseudo_acc_list,
        'a_pseudo_count_list': a_pseudo_count_list,
        'a_convergence_list': a_convergence_list,
        'b_full_test_acc_list': b_full_test_acc_list,
        'b_full_test_f1_list': b_full_test_f1_list,
        'b_full_pseudo_acc_list': b_full_pseudo_acc_list,
        'b_full_pseudo_count_list': b_full_pseudo_count_list,
        'b_full_convergence_list': b_full_convergence_list,
        'b_no_stop_test_acc_list': b_no_stop_test_acc_list,
        'b_no_stop_test_f1_list': b_no_stop_test_f1_list,
        'b_no_stop_pseudo_acc_list': b_no_stop_pseudo_acc_list,
        'b_no_stop_pseudo_count_list': b_no_stop_pseudo_count_list,
        'b_no_stop_convergence_list': b_no_stop_convergence_list,
        'c_fixed_08_test_acc_list': c_fixed_08_test_acc_list,
        'c_fixed_08_test_f1_list': c_fixed_08_test_f1_list,
        'c_fixed_08_pseudo_acc_list': c_fixed_08_pseudo_acc_list,
        'c_fixed_08_pseudo_count_list': c_fixed_08_pseudo_count_list,
        'c_fixed_08_convergence_list': c_fixed_08_convergence_list,
        'c_fixed_09_test_acc_list': c_fixed_09_test_acc_list,
        'c_fixed_09_test_f1_list': c_fixed_09_test_f1_list,
        'c_fixed_09_pseudo_acc_list': c_fixed_09_pseudo_acc_list,
        'c_fixed_09_pseudo_count_list': c_fixed_09_pseudo_count_list,
        'c_fixed_09_convergence_list': c_fixed_09_convergence_list,
        'ratio_list': ratio_list,
        'mean_distances_no_stop': mean_distances_no_stop
    }

# 训练全监督基线
def train_full_supervised():
    run_acc_full = []
    run_f1_full = []
    for run in range(NUM_RUNS):
        set_seed(42 + run)
        base_classifier = [MLPClassifier(hidden_layer_sizes=MLP_HIDDEN_LAYERS, max_iter=2000, random_state=42 + run)]
        clf = SingleModelPredictor(base_classifier)
        clf.fit(X_train, y_train)
        pred_full = clf.predict(X_test)
        acc_full = accuracy_score(y_test, pred_full)
        f1_full = f1_score(y_test, pred_full, average='weighted')
        run_acc_full.append(acc_full)
        run_f1_full.append(f1_full)
    acc_full = np.mean(run_acc_full)
    f1_full = np.mean(run_f1_full)
    std_acc_full = np.std(run_acc_full) if len(run_acc_full) > 1 else 0.0
    std_f1_full = np.std(run_f1_full) if len(run_f1_full) > 1 else 0.0
    logging.info(f"全监督（全部训练集）准确率: {acc_full:.4f} ± {std_acc_full:.4f}, F1分数: {f1_full:.4f} ± {std_f1_full:.4f}")
    return (acc_full, std_acc_full), (f1_full, std_f1_full)

# 训练部分数据基线
def train_partial_supervised_with_noise():
    labeled_ratios = [0.01, 0.02, 0.03]
    partial_acc_noise_list = []
    partial_f1_noise_list = []

    for labeled_ratio in labeled_ratios:
        n_labeled = max(4, int(len(X_train) * labeled_ratio))
        run_acc_noise = []
        run_f1_noise = []
        for run in range(NUM_RUNS):
            set_seed(42 + run)
            X_labeled, _, y_labeled, _ = train_test_split(
                X_train, y_train, train_size=n_labeled, random_state=42 + run, stratify=y_train
            )

            y_labeled_noisy = add_label_noise(y_labeled, seed=42 + run)

            base_classifier = [MLPClassifier(hidden_layer_sizes=MLP_HIDDEN_LAYERS, max_iter=2000, random_state=42 + run)]
            clf_noise = SingleModelPredictor(base_classifier)
            clf_noise.fit(X_labeled, y_labeled_noisy)
            pred_noise = clf_noise.predict(X_test)
            acc_noise = accuracy_score(y_test, pred_noise)
            f1_noise = f1_score(y_test, pred_noise, average='weighted')
            run_acc_noise.append(acc_noise)
            run_f1_noise.append(f1_noise)

        avg_acc_noise = np.mean(run_acc_noise)
        avg_f1_noise = np.mean(run_f1_noise)
        std_acc_noise = np.std(run_acc_noise) if len(run_acc_noise) > 1 else 0.0
        std_f1_noise = np.std(run_f1_noise) if len(run_f1_noise) > 1 else 0.0
        partial_acc_noise_list.append((avg_acc_noise, std_acc_noise))
        partial_f1_noise_list.append((avg_f1_noise, std_f1_noise))

        logging.info(f"全监督（部分数据）平均准确率: {avg_acc_noise:.4f} ± {std_acc_noise:.4f}, 平均F1分数: {avg_f1_noise:.4f} ± {std_f1_noise:.4f}")

    return partial_acc_noise_list, partial_f1_noise_list

# 主实验函数
def train_and_evaluate():
    (acc_full, std_acc_full), (f1_full, std_f1_full) = train_full_supervised()
    partial_acc_noise_list, partial_f1_noise_list = train_partial_supervised_with_noise()
    result = train_and_evaluate_with_mlp()

    final_result = {
        'acc_full': (acc_full, std_acc_full),
        'f1_full': (f1_full, std_f1_full),
        'partial_acc_noise_list': partial_acc_noise_list,
        'partial_f1_noise_list': partial_f1_noise_list,
        'ratio_list': result['ratio_list'],
        'a_test_acc_list': result['a_test_acc_list'],
        'a_test_f1_list': result['a_test_f1_list'],
        'a_pseudo_acc_list': result['a_pseudo_acc_list'],
        'a_pseudo_count_list': result['a_pseudo_count_list'],
        'a_convergence_list': result['a_convergence_list'],
        'b_full_test_acc_list': result['b_full_test_acc_list'],
        'b_full_test_f1_list': result['b_full_test_f1_list'],
        'b_full_pseudo_acc_list': result['b_full_pseudo_acc_list'],
        'b_full_pseudo_count_list': result['b_full_pseudo_count_list'],
        'b_full_convergence_list': result['b_full_convergence_list'],
        'b_no_stop_test_acc_list': result['b_no_stop_test_acc_list'],
        'b_no_stop_test_f1_list': result['b_no_stop_test_f1_list'],
        'b_no_stop_pseudo_acc_list': result['b_no_stop_pseudo_acc_list'],
        'b_no_stop_pseudo_count_list': result['b_no_stop_pseudo_count_list'],
        'b_no_stop_convergence_list': result['b_no_stop_convergence_list'],
        'c_fixed_08_test_acc_list': result['c_fixed_08_test_acc_list'],
        'c_fixed_08_test_f1_list': result['c_fixed_08_test_f1_list'],
        'c_fixed_08_pseudo_acc_list': result['c_fixed_08_pseudo_acc_list'],
        'c_fixed_08_pseudo_count_list': result['c_fixed_08_pseudo_count_list'],
        'c_fixed_08_convergence_list': result['c_fixed_08_convergence_list'],
        'c_fixed_09_test_acc_list': result['c_fixed_09_test_acc_list'],
        'c_fixed_09_test_f1_list': result['c_fixed_09_test_f1_list'],
        'c_fixed_09_pseudo_acc_list': result['c_fixed_09_pseudo_acc_list'],
        'c_fixed_09_pseudo_count_list': result['c_fixed_09_pseudo_count_list'],
        'c_fixed_09_convergence_list': result['c_fixed_09_convergence_list'],
        'mean_distances_no_stop': result['mean_distances_no_stop']
    }

    model_name = 'Single Model (MLP)'
    print(f"\n表 1：情感分类精度对比 ({model_name})")
    print("| 方法                     | 标记样本量 (比例) | 测试集Acc       | 测试集F1       |")
    print("|--------------------------|------------------|----------------|----------------|")
    print(f"| 全监督                   | 100%             | {acc_full:.4f} ± {std_acc_full:.4f} | {f1_full:.4f} ± {std_f1_full:.4f} |")
    for n_labeled, ratio, (acc_noise, std_acc_noise), (f1_noise, std_f1_noise), (acc_a, std_acc_a), (f1_a, std_f1_a), (acc_b_full, std_acc_b_full), (f1_b_full, std_f1_b_full), (acc_b_no_stop, std_acc_b_no_stop), (f1_b_no_stop, std_f1_b_no_stop), (acc_c_fixed_08, std_acc_c_fixed_08), (f1_c_fixed_08, std_f1_c_fixed_08), (acc_c_fixed_09, std_acc_c_fixed_09), (f1_c_fixed_09, std_f1_c_fixed_09) in zip(
        [max(4, int(len(X_train) * r)) for r in [0.01, 0.02, 0.03]],
        final_result['ratio_list'],
        final_result['partial_acc_noise_list'],
        final_result['partial_f1_noise_list'],
        final_result['a_test_acc_list'],
        final_result['a_test_f1_list'],
        final_result['b_full_test_acc_list'],
        final_result['b_full_test_f1_list'],
        final_result['b_no_stop_test_acc_list'],
        final_result['b_no_stop_test_f1_list'],
        final_result['c_fixed_08_test_acc_list'],
        final_result['c_fixed_08_test_f1_list'],
        final_result['c_fixed_09_test_acc_list'],
        final_result['c_fixed_09_test_f1_list']
    ):
        print(f"| 全监督（部分数据）        | {n_labeled} ({ratio*100:.2f}%) | {acc_noise:.4f} ± {std_acc_noise:.4f} | {f1_noise:.4f} ± {std_f1_noise:.4f} |")
        print(f"| 算法 A (DASH)            | {n_labeled} ({ratio*100:.2f}%) | {acc_a:.4f} ± {std_acc_a:.4f} | {f1_a:.4f} ± {std_f1_a:.4f} |")
        print(f"| 算法 B (带早停)          | {n_labeled} ({ratio*100:.2f}%) | {acc_b_full:.4f} ± {std_acc_b_full:.4f} | {f1_b_full:.4f} ± {std_f1_b_full:.4f} |")
        print(f"| 算法 B (无早停)          | {n_labeled} ({ratio*100:.2f}%) | {acc_b_no_stop:.4f} ± {std_acc_b_no_stop:.4f} | {f1_b_no_stop:.4f} ± {std_f1_b_no_stop:.4f} |")
        print(f"| 算法 C (固定置信度0.8)   | {n_labeled} ({ratio*100:.2f}%) | {acc_c_fixed_08:.4f} ± {std_acc_c_fixed_08:.4f} | {f1_c_fixed_08:.4f} ± {std_f1_c_fixed_08:.4f} |")
        print(f"| 算法 C (固定置信度0.9)   | {n_labeled} ({ratio*100:.2f}%) | {acc_c_fixed_09:.4f} ± {std_acc_c_fixed_09:.4f} | {f1_c_fixed_09:.4f} ± {std_f1_c_fixed_09:.4f} |")

    print(f"\n表 2：伪标签数量、准确率、测试集性能和迭代次数对比 ({model_name})")
    print("| 标记样本量 (比例) | 方法                     | 伪标签数 | 伪标签Acc | 测试集Acc | 测试集F1 | 迭代次数 |")
    print("|-------------------|-------------------------|----------|-----------|----------|----------|----------|")
    for n_labeled, ratio, (pseudo_count_a, _), (pseudo_acc_a, std_pseudo_a), (acc_a, std_acc_a), (f1_a, std_f1_a), (pseudo_count_b_full, _), (pseudo_acc_b_full, std_pseudo_b_full), (acc_b_full, std_acc_b_full), (f1_b_full, std_f1_b_full), (pseudo_count_b_no_stop, _), (pseudo_acc_b_no_stop, std_pseudo_b_no_stop), (acc_b_no_stop, std_acc_b_no_stop), (f1_b_no_stop, std_f1_b_no_stop), (pseudo_count_c_fixed_08, _), (pseudo_acc_c_fixed_08, std_pseudo_c_fixed_08), (acc_c_fixed_08, std_acc_c_fixed_08), (f1_c_fixed_08, std_f1_c_fixed_08), (pseudo_count_c_fixed_09, _), (pseudo_acc_c_fixed_09, std_pseudo_c_fixed_09), (acc_c_fixed_09, std_acc_c_fixed_09), (f1_c_fixed_09, std_f1_c_fixed_09), (convergence_a_mean, convergence_a_std), (convergence_b_full_mean, convergence_b_full_std), (convergence_b_no_stop_mean, convergence_b_no_stop_std), (convergence_c_fixed_08_mean, convergence_c_fixed_08_std), (convergence_c_fixed_09_mean, convergence_c_fixed_09_std) in zip(
        [max(4, int(len(X_train) * r)) for r in [0.01, 0.02, 0.03]],
        final_result['ratio_list'],
        final_result['a_pseudo_count_list'],
        final_result['a_pseudo_acc_list'],
        final_result['a_test_acc_list'],
        final_result['a_test_f1_list'],
        final_result['b_full_pseudo_count_list'],
        final_result['b_full_pseudo_acc_list'],
        final_result['b_full_test_acc_list'],
        final_result['b_full_test_f1_list'],
        final_result['b_no_stop_pseudo_count_list'],
        final_result['b_no_stop_pseudo_acc_list'],
        final_result['b_no_stop_test_acc_list'],
        final_result['b_no_stop_test_f1_list'],
        final_result['c_fixed_08_pseudo_count_list'],
        final_result['c_fixed_08_pseudo_acc_list'],
        final_result['c_fixed_08_test_acc_list'],
        final_result['c_fixed_08_test_f1_list'],
        final_result['c_fixed_09_pseudo_count_list'],
        final_result['c_fixed_09_pseudo_acc_list'],
        final_result['c_fixed_09_test_acc_list'],
        final_result['c_fixed_09_test_f1_list'],
        final_result['a_convergence_list'],
        final_result['b_full_convergence_list'],
        final_result['b_no_stop_convergence_list'],
        final_result['c_fixed_08_convergence_list'],
        final_result['c_fixed_09_convergence_list']
    ):
        print(f"| {n_labeled} ({ratio*100:.2f}%) | 算法 A (DASH)            | {int(pseudo_count_a)} | {pseudo_acc_a:.4f} ± {std_pseudo_a:.4f} | {acc_a:.4f} ± {std_acc_a:.4f} | {f1_a:.4f} ± {std_f1_a:.4f} | {convergence_a_mean:.1f} |")
        print(f"| {n_labeled} ({ratio*100:.2f}%) | 算法 B (带早停)          | {int(pseudo_count_b_full)} | {pseudo_acc_b_full:.4f} ± {std_pseudo_b_full:.4f} | {acc_b_full:.4f} ± {std_acc_b_full:.4f} | {f1_b_full:.4f} ± {std_f1_b_full:.4f} | {convergence_b_full_mean:.1f} |")
        print(f"| {n_labeled} ({ratio*100:.2f}%) | 算法 B (无早停)          | {int(pseudo_count_b_no_stop)} | {pseudo_acc_b_no_stop:.4f} ± {std_pseudo_b_no_stop:.4f} | {acc_b_no_stop:.4f} ± {std_acc_b_no_stop:.4f} | {f1_b_no_stop:.4f} ± {std_f1_b_no_stop:.4f} | {convergence_b_no_stop_mean:.1f} |")
        print(f"| {n_labeled} ({ratio*100:.2f}%) | 算法 C (固定置信度0.8)   | {int(pseudo_count_c_fixed_08)} | {pseudo_acc_c_fixed_08:.4f} ± {std_pseudo_c_fixed_08:.4f} | {acc_c_fixed_08:.4f} ± {std_acc_c_fixed_08:.4f} | {f1_c_fixed_08:.4f} ± {std_f1_c_fixed_08:.4f} | {convergence_c_fixed_08_mean:.1f} |")
        print(f"| {n_labeled} ({ratio*100:.2f}%) | 算法 C (固定置信度0.9)   | {int(pseudo_count_c_fixed_09)} | {pseudo_acc_c_fixed_09:.4f} ± {std_pseudo_c_fixed_09:.4f} | {acc_c_fixed_09:.4f} ± {std_acc_c_fixed_09:.4f} | {f1_c_fixed_09:.4f} ± {std_f1_c_fixed_09:.4f} | {convergence_c_fixed_09_mean:.1f} |")

    chart_config_distance = {
        "type": "line",
        "data": {
            "labels": list(range(21)),
            "datasets": [
                {
                    "label": "1%",
                    "data": [round(float(result['mean_distances_no_stop'][0][i]), 2) if i < len(result['mean_distances_no_stop'][0]) else None for i in range(21)],
                    "borderColor": "#000000",
                    "backgroundColor": "#000000",
                    "pointStyle": "circle",
                    "pointRadius": 4,
                    "borderWidth": 0.8,
                    "fill": False
                },
                {
                    "label": "2%",
                    "data": [round(float(result['mean_distances_no_stop'][1][i]), 2) if i < len(result['mean_distances_no_stop'][1]) else None for i in range(21)],
                    "borderColor": "red",
                    "backgroundColor": "red",
                    "pointStyle": "rect",
                    "pointRadius": 4,
                    "borderWidth": 0.8,
                    "fill": False
                },
                {
                    "label": "3%",
                    "data": [round(float(result['mean_distances_no_stop'][2][i]), 2) if i < len(result['mean_distances_no_stop'][2]) else None for i in range(21)],
                    "borderColor": "blue",
                    "backgroundColor": "blue",
                    "pointStyle": "triangle",
                    "pointRadius": 4,
                    "borderWidth": 0.8,
                    "fill": False
                }
            ]
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": False,
            "aspectRatio": 0.5,
            "title": {
                "display": True,
                "text": "距离随迭代变化 (不同标记比例)"
            },
            "legend": {
                "display": True,
                "position": "top",
                "labels": {
                    "boxWidth": 10,
                    "fontColor": "#000000"
                }
            },
            "scales": {
                "x": {
                    "title": {
                        "display": True,
                        "text": "迭代次数"
                    },
                    "ticks": {
                        "min": 0,
                        "max": 20,
                        "stepSize": 1
                    }
                },
                "y": {
                    "title": {
                        "display": True,
                        "text": "平均距离"
                    },
                    "ticks": {
                        "min": 0,
                        "max": 3,
                        "stepSize": 0.3
                    }
                }
            }
        }
    }
    print(f"\n### 图表: 距离随迭代变化 (不同标记比例)")
    print(f"```chartjs")
    print(json.dumps(chart_config_distance, indent=2))
    print(f"```")

    return final_result

# 运行实验
if __name__ == "__main__":
    result_mlp = train_and_evaluate()
