import os
import cv2
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from xgboost import XGBClassifier
import concurrent.futures
from sklearn.neural_network import MLPClassifier  # 新增：导入MLPClassifier

train_dir = '/root/autodl-fs/spectrogram/train'
test_dir = '/root/autodl-fs/spectrogram/test'

def data_augmentation(img):
    augmented_images = [img]

    # 添加高斯噪声
    noise = np.random.normal(0.0001, 0.005, img.shape)
    augmented_images.append(np.clip(img + noise, 0, 1))

    # 水平翻转
    augmented_images.append(np.fliplr(img))

    # 垂直翻转
    augmented_images.append(np.flipud(img))
    return augmented_images

def extract_features_from_spectrogram(spectrogram):
    # 使用 librosa 提取频谱图的 MFCC 特征
    mfccs = librosa.feature.mfcc(S=spectrogram, sr=22050, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # 组合所有特征
    features = np.hstack([mfccs_mean])
    return features

def process_image(img_path, label):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = img / 255.0  # 标准化频谱图
        augmented_images = data_augmentation(img)
        features = []
        for aug_img in augmented_images:
            img_features = extract_features_from_spectrogram(aug_img)
            features.append(img_features)
        return features, int(label)
    return None

def load_and_extract_features(folder):
    features = []
    labels = []
    print(f"加载并提取 {folder} 中的特征...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for label in os.listdir(folder):
            label_path = os.path.join(folder, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    img_path = os.path.join(label_path, filename)
                    futures.append(executor.submit(process_image, img_path, label))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                img_features, img_label = result
                features.extend(img_features)
                labels.extend([img_label] * len(img_features))

    return np.array(features), np.array(labels)

if __name__ == "__main__":
    # 加载训练和测试集并提取特征
    print("开始加载并提取训练和测试数据的特征...")
    X_train, y_train = load_and_extract_features(train_dir)
    X_test, y_test = load_and_extract_features(test_dir)

    # 特征选择
    selector = SelectFromModel(RandomForestClassifier(n_estimators=1000, random_state=42), threshold='median')
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # 定义模型
    rf = RandomForestClassifier(
        n_estimators=1000,  # 树的数量
        max_depth=10,      # 树的最大深度
        min_samples_split=10,  # 最小样本分裂数
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=1000,  # 树数量
        learning_rate=0.01,  # 学习率
        max_depth=6,         # 树的最大深度
        reg_lambda=1,           # L2 正则化
        alpha=0.001,             # L1 正则化
        random_state=42
    )

    svm = SVC(
        kernel='rbf',     # 核函数类型
        C=0.001,               # 正则化参数
        probability=True,    # 预测概率
        random_state=42
    )

    mlp = MLPClassifier(  # 新增：定义MLP模型
        hidden_layer_sizes=(64,32),  # 隐藏层大小
        activation='relu',           # 激活函数
        solver='adam',              # 优化算法
        max_iter=1000,               # 最大迭代次数
        random_state=42,
        alpha=0.001
    )

    # 创建投票分类器
    print('创建投票器......')
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm), ('mlp', mlp)],
        voting='soft',
        weights = [1, 1, 1, 1]
    )

    # 使用StratifiedKFold进行交叉验证
    print('开始进行交叉检验......')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(voting_clf, X_train_selected, y_train, cv=skf, n_jobs=-1)
    print(f"交叉验证分数: {cv_scores}")
    print(f"平均交叉验证分数: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

    # 在整个训练集上训练模型
    print('开始训练')
    voting_clf.fit(X_train_selected, y_train)

    # 预测与评估
    print('开始预测与评估')
    y_pred = voting_clf.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'准确率: {accuracy * 100:.2f}%')
    print("分类报告:")
    print(classification_report(y_test, y_pred))

    # 训练模型以进行特征重要性分析
    print('开始特征重要性分析')
    rf.fit(X_train_selected, y_train)

    # 获取特征重要性
    feature_importances = rf.feature_importances_

    # 打印特征重要性
    print("特征重要性分析:")
    for name, model in voting_clf.named_estimators_.items():
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name} 特征重要性:")
            feature_importance = model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            print("Top 10 most important features:")
            for idx in sorted_idx[-10:]:
                print(f"Feature {idx}: {feature_importance[idx]:.4f}")