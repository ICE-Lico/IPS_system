from flask import Flask, render_template
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import io
import base64

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化 Flask 应用
app = Flask(__name__)


# 1. 数据加载与预处理
def load_and_preprocess_data(file_path):
    """加载Excel数据并进行预处理，将所有特征转换为数值类型"""
    df = pd.read_excel(file_path)
    print(f"数据维度: {df.shape}")
    print(f"原始列名: {df.columns.tolist()}")

    # 检查原始数据缺失情况
    missing_values = df.isnull().sum()
    print("\n缺失值统计：")
    print(missing_values[missing_values > 0])

    # 处理缺失值
    df = df.fillna({
        '应用层协议': 'unknown',
        '威胁类别': 'unknown',
        '威胁名称': 'unknown',
        '详细信息': '',
        '源端口': 'unknown',
        '目的端口': 'unknown'
    })

    # 检查关键特征的缺失值
    critical_features = ['源IP', '目的IP', '发现时间', '威胁等级']
    for feature in critical_features:
        if feature in df.columns and df[feature].isnull().any():
            print(f"警告：关键特征 '{feature}' 仍有缺失值，将删除对应行")
            df = df.dropna(subset=[feature])

    # 时间特征提取
    df['发现时间'] = pd.to_datetime(df['发现时间'])
    df['小时'] = df['发现时间'].dt.hour
    df['星期'] = df['发现时间'].dt.dayofweek
    df['月份'] = df['发现时间'].dt.month
    df = df.drop(['发现时间'], axis=1)

    # IP地址转整数
    def ip_to_int(ip):
        return sum(int(x) << (8 * i) for i, x in enumerate(reversed(ip.split('.'))))

    df['源IP_int'] = df['源IP'].apply(ip_to_int)
    df['目的IP_int'] = df['目的IP'].apply(ip_to_int)
    df = df.drop(['源IP', '目的IP'], axis=1)

    # 端口类型标签编码（知名端口=0，动态端口=1）
    port_type_mapping = {'知名端口': 0, '动态端口': 1}
    df['源端口'] = df['源端口'].map(port_type_mapping)
    df['目的端口'] = df['目的端口'].map(port_type_mapping)

    # 处理可能的映射失败（unknown值）
    df['源端口'] = df['源端口'].fillna(-1)  # 用-1表示未知端口类型
    df['目的端口'] = df['目的端口'].fillna(-1)

    # 应用层协议one-hot编码
    protocol_dummies = pd.get_dummies(df['应用层协议'], prefix='协议')
    df = pd.concat([df, protocol_dummies], axis=1)
    df = df.drop(['应用层协议'], axis=1)

    # 威胁类别one-hot编码（限制高频类别）
    threat_category_counts = df['威胁类别'].value_counts()
    frequent_categories = threat_category_counts[threat_category_counts > 500].index
    df['威胁类别'] = df['威胁类别'].where(df['威胁类别'].isin(frequent_categories), 'other')
    category_dummies = pd.get_dummies(df['威胁类别'], prefix='威胁类别')
    df = pd.concat([df, category_dummies], axis=1)
    df = df.drop(['威胁类别'], axis=1)

    # 威胁名称one-hot编码（限制高频类别）
    threat_name_counts = df['威胁名称'].value_counts()
    frequent_names = threat_name_counts[threat_name_counts > 1000].index
    df['威胁名称'] = df['威胁名称'].where(df['威胁名称'].isin(frequent_names), 'other')
    name_dummies = pd.get_dummies(df['威胁名称'], prefix='威胁名称')
    df = pd.concat([df, name_dummies], axis=1)
    df = df.drop(['威胁名称'], axis=1)

    # 威胁等级数值映射（0-4）
    df['威胁等级数值'] = df['威胁等级'].str.extract(r'(\d)').astype(int) - 1
    df = df.drop(['威胁等级'], axis=1)

    # 文本关键词特征
    df['包含Tor'] = df['详细信息'].str.contains('Tor', na=False).astype(int)
    df['包含VPN'] = df['详细信息'].str.contains('VPN', na=False).astype(int)
    df['包含远程控制'] = df['详细信息'].str.contains('remote-control-tool', na=False).astype(int)
    df = df.drop(['详细信息'], axis=1)

    # 检查剩余列类型
    object_columns = df.select_dtypes(include='object').columns
    if len(object_columns) > 0:
        print(f"警告：剩余object类型列: {object_columns}")
        # 紧急处理：将剩余object列转换为字符串编码
        for col in object_columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # 最终检查缺失值
    final_missing = df.isnull().sum().sum()
    if final_missing > 0:
        print(f"警告：预处理后仍有{final_missing}个缺失值，将使用均值填充")
        # 对所有数值列使用均值填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print(f"处理后列名: {df.columns.tolist()}")
    print(f"处理后数据维度: {df.shape}")
    return df


# 2. 数据划分
def split_data(df):
    """划分训练集和测试集，确保特征均为数值类型"""
    X = df.drop(['威胁等级数值'], axis=1)
    y = df['威胁等级数值']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集特征维度: {X_train.shape}, 测试集特征维度: {X_test.shape}")

    # 检查划分后的数据缺失情况
    print(f"训练集缺失值数量: {X_train.isnull().sum().sum()}")
    print(f"测试集缺失值数量: {X_test.isnull().sum().sum()}")

    return X, y, X_train, X_test, y_train, y_test


# 3. 模型训练 - XGBoost
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """训练XGBoost模型，显式设置特征类型"""
    start_time = time.time()
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        objective='multi:softmax',
        random_state=42,
        eval_metric='merror',
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    train_time = time.time() - start_time
    return xgb_model, train_time


# 4. 模型训练 - 随机森林
def train_random_forest_model(X_train, y_train):
    start_time = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    train_time = time.time() - start_time
    return rf_model, train_time


# 5. 模型训练 - 逻辑回归（使用管道处理缺失值）
def train_logistic_regression_model(X_train, y_train, X_test):
    start_time = time.time()

    # 创建包含填充和标准化的管道
    pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),  # 使用均值填充缺失值
        StandardScaler()  # 数据标准化
    )

    # 对训练数据进行预处理
    X_train_processed = pipeline.fit_transform(X_train)

    # 检查处理后的数据
    nan_rows = np.isnan(X_train_processed).any(axis=1)
    print(f"逻辑回归训练集：处理后包含{np.sum(nan_rows)}行NaN值，总样本数: {len(X_train_processed)}")

    lr_model = LogisticRegression(
        C=1.0,
        solver='lbfgs',
        multi_class='multinomial',
        max_iter=1000,
        random_state=42
    )
    lr_model.fit(X_train_processed, y_train)

    # 对测试数据应用相同的预处理
    X_test_processed = pipeline.transform(X_test)

    train_time = time.time() - start_time
    return lr_model, pipeline, X_test_processed, train_time


# 6. 交叉验证
def perform_cross_validation(model, X, y, model_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if model_name == "LogisticRegression":
        # 为逻辑回归创建专用管道
        pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver='lbfgs',
                multi_class='multinomial',
                max_iter=1000,
                random_state=42
            )
        )
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    else:
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

    print(f"{model_name}交叉验证结果：")
    print(f"准确率均值: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"各折准确率: {scores}")
    return scores.mean(), scores.std()


# 7. 模型评估
def evaluate_model(model, X_test, y_test, model_name, pipeline=None, X_test_processed=None):
    if model_name == "LogisticRegression":
        # 使用预处理后的测试数据
        y_pred = model.predict(X_test_processed)
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    report = classification_report(
        y_test, y_pred,
        target_names=[f'severity_{i + 1}' for i in range(5)]
    )

    print(f"\n{model_name}模型评估：")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n详细分类报告：")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': y_pred
    }


# 8. 特征重要性分析
def analyze_feature_importance(model, X, model_name):
    if model_name == "XGBoost":
        feature_importance = model.feature_importances_
        features = X.columns.tolist()
        importance_df = pd.DataFrame({
            '特征': features,
            '重要性': feature_importance
        }).sort_values('重要性', ascending=False)

        print(f"\n{model_name}特征重要性排名（Top10）：")
        print(importance_df.head(10))

        plt.figure(figsize=(12, 6))
        sns.barplot(x='重要性', y='特征', data=importance_df.head(10))
        plt.title(f'{model_name}特征重要性')
        plt.tight_layout()

        return importance_df
    return None


# 新增：封装执行建模流程并收集结果的函数
def run_modeling():
    file_path = '附件2：IPS日志数据样例.xlsx'
    df = load_and_preprocess_data(file_path)
    X, y, X_train, X_test, y_train, y_test = split_data(df)

    # 训练模型
    xgb_model, xgb_train_time = train_xgboost_model(X_train, y_train, X_test, y_test)
    rf_model, rf_train_time = train_random_forest_model(X_train, y_train)
    lr_model, pipeline, X_test_processed, lr_train_time = train_logistic_regression_model(X_train, y_train, X_test)

    # 交叉验证
    xgb_cv_mean, xgb_cv_std = perform_cross_validation(xgb_model, X, y, "XGBoost")
    rf_cv_mean, rf_cv_std = perform_cross_validation(rf_model, X, y, "RandomForest")
    lr_cv_mean, lr_cv_std = perform_cross_validation(lr_model, X, y, "LogisticRegression")

    # 模型评估
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    rf_results = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    lr_results = evaluate_model(lr_model, X_test, y_test, "LogisticRegression", pipeline, X_test_processed)

    # 特征重要性（这里仅处理 XGBoost 的，若要其他模型可扩展）
    xgb_feature_importance = analyze_feature_importance(xgb_model, X, "XGBoost")

    # 模型性能对比表
    results = {
        'XGBoost': {
            'train_accuracy': xgb_model.score(X_train, y_train),
            'test_accuracy': xgb_results['accuracy'],
            'cv_mean': xgb_cv_mean,
            'cv_std': xgb_cv_std,
            'f1': xgb_results['f1'],
            'train_time': xgb_train_time
        },
        'RandomForest': {
            'train_accuracy': rf_model.score(X_train, y_train),
            'test_accuracy': rf_results['accuracy'],
            'cv_mean': rf_cv_mean,
            'cv_std': rf_cv_std,
            'f1': rf_results['f1'],
            'train_time': rf_train_time
        },
        'LogisticRegression': {
            'train_accuracy': lr_model.score(pipeline.transform(X_train), y_train),
            'test_accuracy': lr_results['accuracy'],
            'cv_mean': lr_cv_mean,
            'cv_std': lr_cv_std,
            'f1': lr_results['f1'],
            'train_time': lr_train_time
        }
    }
    comparison_df = pd.DataFrame({
        '模型': ['XGBoost', '随机森林', '逻辑回归'],
        '训练集准确率': [
            results['XGBoost']['train_accuracy'],
            results['RandomForest']['train_accuracy'],
            results['LogisticRegression']['train_accuracy']
        ],
        '测试集准确率': [
            results['XGBoost']['test_accuracy'],
            results['RandomForest']['test_accuracy'],
            results['LogisticRegression']['test_accuracy']
        ],
        '交叉验证均值': [
            results['XGBoost']['cv_mean'],
            results['RandomForest']['cv_mean'],
            results['LogisticRegression']['cv_mean']
        ],
        '交叉验证标准差': [
            f"{results['XGBoost']['cv_std']:.4f}",
            f"{results['RandomForest']['cv_std']:.4f}",
            f"{results['LogisticRegression']['cv_std']:.4f}"
        ],
        '测试集F1分数': [
            results['XGBoost']['f1'],
            results['RandomForest']['f1'],
            results['LogisticRegression']['f1']
        ],
        '训练时间(秒)': [
            results['XGBoost']['train_time'],
            results['RandomForest']['train_time'],
            results['LogisticRegression']['train_time']
        ]
    })

    # 处理混淆矩阵可视化，转成 base64 给前端显示
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(xgb_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')

    plt.subplot(1, 3, 2)
    sns.heatmap(rf_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('随机森林混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')

    plt.subplot(1, 3, 3)
    sns.heatmap(lr_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('逻辑回归混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')

    plt.tight_layout()
    # 保存到内存
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    confusion_matrix_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # 特征重要性图（XGBoost）
    if xgb_feature_importance is not None:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='重要性', y='特征', data=xgb_feature_importance.head(10))
        plt.title('XGBoost特征重要性')
        plt.tight_layout()
        # 保存到内存
        feat_buf = io.BytesIO()
        plt.savefig(feat_buf, format='png')
        feat_buf.seek(0)
        feat_importance_base64 = base64.b64encode(feat_buf.read()).decode('utf-8')
        plt.close()
    else:
        feat_importance_base64 = None

    return {
        'comparison_df': comparison_df,
        'confusion_matrix_base64': confusion_matrix_base64,
        'feat_importance_base64': feat_importance_base64,
        'xgb_report': classification_report(y_test, xgb_results['predictions'],
                                            target_names=[f'severity_{i + 1}' for i in range(5)]),
        'rf_report': classification_report(y_test, rf_results['predictions'],
                                           target_names=[f'severity_{i + 1}' for i in range(5)]),
        'lr_report': classification_report(y_test, lr_results['predictions'],
                                           target_names=[f'severity_{i + 1}' for i in range(5)]),
    }


# 定义路由，首页展示结果
@app.route('/')
def index():
    modeling_results = run_modeling()
    return render_template(
        'index.html',
        comparison_df=modeling_results['comparison_df'].to_html(index=False),  # 转成 HTML 表格
        confusion_matrix=modeling_results['confusion_matrix_base64'],
        feat_importance=modeling_results['feat_importance_base64'],
        xgb_report=modeling_results['xgb_report'],
        rf_report=modeling_results['rf_report'],
        lr_report=modeling_results['lr_report']
    )


if __name__ == '__main__':
    # 调试模式运行，生产环境记得改 debug=False
    app.run(debug=True)