# -*- coding: utf-8 -*-
"""评估BERT嵌入特征的分类性能"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 全局变量用于可视化
X_scaled_global = None
X_global = None


# ============================================================
# 1. 加载数据
# ============================================================

def load_data():
    """加载BERT嵌入特征和标签"""
    print("正在加载数据...")

    try:
        # 加载BERT嵌入特征
        embeddings = np.loadtxt('ceb_mbert_features.csv', delimiter=',')
        print(f"嵌入特征形状: {embeddings.shape}")
    except FileNotFoundError:
        print("错误: 未找到 'ceb_mbert_features.csv' 文件")
        print("请先运行 extract_embeddings.py 生成特征文件")
        return None, None, None

    try:
        # 加载标题文件
        with open('ceb_mbert_features_titles.txt', 'r', encoding='utf-8') as f:
            titles = [line.strip() for line in f]
        print(f"标题数量: {len(titles)}")
    except FileNotFoundError:
        print("错误: 未找到 'ceb_mbert_features_titles.txt' 文件")
        return None, None, None

    # 从原始数据文件加载年级标签
    try:
        with open('ceb_all_data.txt', 'r', encoding='utf-8', errors='ignore') as file:
            file_contents = file.readlines()
    except FileNotFoundError:
        print("错误: 未找到 'ceb_all_data.txt' 文件")
        # 尝试加载其他特征文件来获取标签
        try:
            trad_df = pd.read_csv('trad.csv')
            grade_labels = trad_df['grade_level'].tolist()
            # 只取前len(titles)个标签
            grade_labels = grade_labels[:len(titles)]
            print(f"从trad.csv加载了 {len(grade_labels)} 个标签")
        except:
            print("错误: 无法加载标签数据")
            return None, None, None
    else:
        # 解析原始文件获取标签
        title_to_grade = {}
        for line in file_contents:
            parts = line.strip().split(',', 2)
            if len(parts) >= 2:
                title = parts[0].strip()
                grade = parts[1].strip()
                title_to_grade[title] = grade

        # 按照标题顺序获取年级标签
        grade_labels = []
        for title in titles:
            if title in title_to_grade:
                grade_labels.append(title_to_grade[title])
            else:
                # 尝试模糊匹配
                found = False
                for key in title_to_grade:
                    if title.lower() in key.lower() or key.lower() in title.lower():
                        grade_labels.append(title_to_grade[key])
                        found = True
                        break
                if not found:
                    # 使用最常见的标签
                    grade_labels.append('1')

        print(f"从ceb_all_data.txt加载了 {len(grade_labels)} 个标签")

    # 确保标签数量与特征数量一致
    if len(grade_labels) != len(titles):
        print(f"警告: 标签数量({len(grade_labels)})与特征数量({len(titles)})不匹配")
        min_len = min(len(grade_labels), len(titles))
        grade_labels = grade_labels[:min_len]
        titles = titles[:min_len]
        embeddings = embeddings[:min_len]
        print(f"调整后: 特征{embeddings.shape}, 标签{len(grade_labels)}, 标题{len(titles)}")

    # 将年级标签转换为数值
    try:
        # 尝试转换为整数
        y = np.array([int(label) for label in grade_labels])
    except ValueError:
        # 如果不是整数，使用标签编码
        le = LabelEncoder()
        y = le.fit_transform(grade_labels)
        print(f"使用标签编码，类别: {le.classes_}")

    # 只保留有有效标签的数据
    X = embeddings
    valid_titles = titles

    print(f"有效样本数: {len(X)}")
    print(f"年级分布: {np.bincount(y)}")
    print(f"年级类别: {np.unique(y)}")

    return X, y, valid_titles


# ============================================================
# 2. 评估函数
# ============================================================

def evaluate_classification(X, y, test_size=0.3, random_state=42):
    """评估分类性能"""
    print("\n正在准备评估...")

    global X_scaled_global, X_global
    X_global = X

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_global = X_scaled

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # 定义多个分类器进行比较
    classifiers = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '支持向量机': SVC(kernel='rbf', random_state=42, probability=True),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}

    print("\n" + "=" * 60)
    print("开始模型训练与评估...")
    print("=" * 60)

    for name, clf in classifiers.items():
        print(f"\n评估 {name}...")

        # 训练模型
        clf.fit(X_train, y_train)

        # 预测
        y_pred = clf.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        # 存储结果
        results[name] = {
            'model': clf,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'y_pred': y_pred,
            'y_test': y_test,
            'scaler': scaler
        }

        print(f"  准确率: {accuracy:.4f}")
        print(f"  宏平均F1: {macro_f1:.4f}")
        print(f"  加权F1: {weighted_f1:.4f}")

    return results, X_test, y_test, X_train, y_train


# ============================================================
# 3. 交叉验证评估
# ============================================================

def cross_validation_evaluation(X, y, cv=5):
    """使用交叉验证评估模型"""
    print("\n" + "=" * 60)
    print("交叉验证评估...")
    print("=" * 60)

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    classifiers = {
        '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
        '支持向量机': SVC(kernel='rbf', random_state=42),
        '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    cv_results = {}

    for name, clf in classifiers.items():
        print(f"\n{name} 交叉验证:")

        # 准确率交叉验证
        accuracy_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

        # F1分数交叉验证
        f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1_macro')

        cv_results[name] = {
            'mean_accuracy': np.mean(accuracy_scores),
            'std_accuracy': np.std(accuracy_scores),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'all_accuracy': accuracy_scores,
            'all_f1': f1_scores
        }

        print(f"  平均准确率: {cv_results[name]['mean_accuracy']:.4f} (±{cv_results[name]['std_accuracy']:.4f})")
        print(f"  平均F1分数: {cv_results[name]['mean_f1']:.4f} (±{cv_results[name]['std_f1']:.4f})")

    return cv_results


# ============================================================
# 4. 可视化结果
# ============================================================

def visualize_results(results, cv_results=None, X_scaled=None, X=None):
    """可视化评估结果"""
    print("\n" + "=" * 60)
    print("生成可视化结果...")
    print("=" * 60)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 模型性能比较（条形图）
    ax1 = axes[0, 0]
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    macro_f1s = [results[m]['macro_f1'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, accuracies, width, label='准确率', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width / 2, macro_f1s, width, label='宏平均F1', alpha=0.8, color='lightcoral')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('模型')
    ax1.set_ylabel('分数')
    ax1.set_title('模型性能比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # 2. 混淆矩阵（最佳模型）
    best_model_name = max(results.keys(), key=lambda x: results[x]['macro_f1'])
    best_result = results[best_model_name]

    ax2 = axes[0, 1]
    cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])

    # 确保混淆矩阵是方阵
    if cm.shape[0] == cm.shape[1]:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=np.unique(best_result['y_test']),
                    yticklabels=np.unique(best_result['y_test']))
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)

    ax2.set_xlabel('预测标签')
    ax2.set_ylabel('真实标签')
    ax2.set_title(f'{best_model_name}混淆矩阵')

    # 3. 交叉验证结果（如果可用）
    if cv_results:
        ax3 = axes[1, 0]

        cv_models = list(cv_results.keys())
        cv_accuracies = [cv_results[m]['mean_accuracy'] for m in cv_models]
        cv_f1s = [cv_results[m]['mean_f1'] for m in cv_models]
        cv_acc_errors = [cv_results[m]['std_accuracy'] for m in cv_models]
        cv_f1_errors = [cv_results[m]['std_f1'] for m in cv_models]

        x = np.arange(len(cv_models))

        ax3.errorbar(x, cv_accuracies, yerr=cv_acc_errors, fmt='o-',
                     label='平均准确率', capsize=5, linewidth=2, markersize=8)
        ax3.errorbar(x, cv_f1s, yerr=cv_f1_errors, fmt='s-',
                     label='平均F1', capsize=5, linewidth=2, markersize=8)

        # 添加数值标签
        for i, (acc, f1) in enumerate(zip(cv_accuracies, cv_f1s)):
            ax3.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)
            ax3.text(i, f1 - 0.03, f'{f1:.3f}', ha='center', fontsize=9)

        ax3.set_xlabel('模型')
        ax3.set_ylabel('分数')
        ax3.set_title('交叉验证性能（平均值±标准差）')
        ax3.set_xticks(x)
        ax3.set_xticklabels(cv_models, rotation=15)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.0)

    # 4. PCA可视化（如果数据可用）
    ax4 = axes[1, 1]

    if X_scaled is not None and X is not None:
        try:
            # 使用PCA降维
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            # 获取前两个主成分的解释方差
            explained_variance = pca.explained_variance_ratio_[:2]

            # 绘制条形图
            bars = ax4.bar([1, 2], explained_variance, color=['skyblue', 'lightcoral'])

            # 在条形上添加数值标签
            for i, (bar, v) in enumerate(zip(bars, explained_variance)):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{v:.3f}', ha='center', va='bottom', fontsize=10)

            ax4.set_xlabel('主成分')
            ax4.set_ylabel('解释方差比例')
            ax4.set_title('前两个主成分解释的方差')
            ax4.set_xticks([1, 2])
            ax4.set_xticklabels(['PC1', 'PC2'])
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, max(explained_variance) * 1.2)

            # 添加总解释方差信息
            total_explained = sum(explained_variance)
            ax4.text(1.5, max(explained_variance) * 1.1,
                     f'总解释方差: {total_explained:.3f}',
                     ha='center', fontsize=10, fontweight='bold')

        except Exception as e:
            ax4.text(0.5, 0.5, f'PCA可视化失败:\n{str(e)}',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('PCA可视化')
    else:
        # 绘制ROC曲线（多分类）
        try:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc

            best_model = best_result['model']
            y_test_bin = label_binarize(best_result['y_test'], classes=np.unique(best_result['y_test']))

            # 获取预测概率
            if hasattr(best_model, 'predict_proba'):
                y_score = best_model.predict_proba(
                    best_result['scaler'].transform(X_global[:len(best_result['y_test'])]))

                # 计算每个类别的ROC曲线
                n_classes = y_test_bin.shape[1]
                colors = ['blue', 'red', 'green', 'orange', 'purple'][:n_classes]

                for i, color in zip(range(n_classes), colors):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax4.plot(fpr, tpr, color=color, lw=2,
                             label=f'类别 {i} (AUC = {roc_auc:.2f})')

                ax4.plot([0, 1], [0, 1], 'k--', lw=2)
                ax4.set_xlim([0.0, 1.0])
                ax4.set_ylim([0.0, 1.05])
                ax4.set_xlabel('假正率')
                ax4.set_ylabel('真正率')
                ax4.set_title(f'{best_model_name} ROC曲线')
                ax4.legend(loc="lower right")
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, '模型不支持概率预测',
                         ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('ROC曲线')
        except:
            ax4.text(0.5, 0.5, '无法生成ROC曲线',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('备选可视化')

    plt.tight_layout()

    # 保存图像
    try:
        plt.savefig('bert_embeddings_evaluation.png', dpi=150, bbox_inches='tight')
        print("可视化结果已保存为 'bert_embeddings_evaluation.png'")
    except:
        print("警告: 无法保存图像文件")

    plt.show()


# ============================================================
# 5. 生成格式化输出
# ============================================================

def generate_formatted_output(results, best_model_name):
    """生成格式化的评估报告"""
    best_result = results[best_model_name]
    y_test = best_result['y_test']
    y_pred = best_result['y_pred']

    # 生成分类报告
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 60)
    print("模型评估结果:")
    print("=" * 60)
    print("\n分类报告:")

    # 获取所有类别
    classes = []
    for key in report.keys():
        if key.isdigit() or (isinstance(key, str) and key.replace('.', '').isdigit()):
            classes.append(key)
    classes = sorted(classes, key=lambda x: int(float(x)))

    # 打印表头
    print(f"{'':<15} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}")
    print("-" * 55)

    # 打印每个类别的指标
    for class_label in classes:
        metrics = report[class_label]
        class_str = str(int(float(class_label))) if '.' in class_label else class_label
        print(f"{class_str:<15} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} "
              f"{metrics['f1-score']:<10.2f} {int(metrics['support']):<10}")

    # 打印平均值
    print("-" * 55)
    accuracy = report.get('accuracy', accuracy_score(y_test, y_pred))
    print(f"{'accuracy':<15} {'':<10} {'':<10} {accuracy:<10.2f} {len(y_test):<10}")

    if 'macro avg' in report:
        print(f"{'macro avg':<15} {report['macro avg']['precision']:<10.2f} "
              f"{report['macro avg']['recall']:<10.2f} {report['macro avg']['f1-score']:<10.2f} "
              f"{int(report['macro avg']['support']):<10}")

    if 'weighted avg' in report:
        print(f"{'weighted avg':<15} {report['weighted avg']['precision']:<10.2f} "
              f"{report['weighted avg']['recall']:<10.2f} {report['weighted avg']['f1-score']:<10.2f} "
              f"{int(report['weighted avg']['support']):<10}")

    # 计算宏平均F1分数
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n宏平均F1分数: {macro_f1:.4f}")

    # 保存详细结果到文件
    save_detailed_results(results, report, best_model_name)


def save_detailed_results(results, report, best_model_name):
    """保存详细结果到文件"""
    with open('evaluation_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("BERT嵌入特征评估结果\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"最佳模型: {best_model_name}\n")
        f.write("-" * 40 + "\n\n")

        f.write("各模型性能比较:\n")
        f.write("-" * 40 + "\n")
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  准确率: {result['accuracy']:.4f}\n")
            f.write(f"  宏平均F1: {result['macro_f1']:.4f}\n")
            f.write(f"  加权F1: {result['weighted_f1']:.4f}\n")
            if name == best_model_name:
                f.write(f"  (最佳模型)\n")
            f.write("\n")

        f.write("\n详细分类报告:\n")
        f.write("-" * 40 + "\n")

        # 获取所有类别
        classes = []
        for key in report.keys():
            if key.isdigit() or (isinstance(key, str) and key.replace('.', '').isdigit()):
                classes.append(key)
        classes = sorted(classes, key=lambda x: int(float(x)))

        # 写入分类报告
        f.write(f"{'类别':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'支持数':<10}\n")
        f.write("-" * 50 + "\n")

        for class_label in classes:
            metrics = report[class_label]
            class_str = str(int(float(class_label))) if '.' in class_label else class_label
            f.write(f"{class_str:<10} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} "
                    f"{metrics['f1-score']:<10.2f} {int(metrics['support']):<10}\n")

        f.write("-" * 50 + "\n")
        accuracy = report.get('accuracy', 0)
        f.write(
            f"{'准确率':<10} {'':<10} {'':<10} {accuracy:<10.2f} {int(report.get('macro avg', {}).get('support', 0)):<10}\n")

        if 'macro avg' in report:
            f.write(f"{'宏平均':<10} {report['macro avg']['precision']:<10.2f} "
                    f"{report['macro avg']['recall']:<10.2f} {report['macro avg']['f1-score']:<10.2f} "
                    f"{int(report['macro avg']['support']):<10}\n")

        f.write("\n评估配置:\n")
        f.write("-" * 40 + "\n")
        f.write(f"测试集比例: 30%\n")
        f.write(f"随机种子: 42\n")
        f.write(f"评估时间: {pd.Timestamp.now()}\n")

    print("\n详细结果已保存到 'evaluation_results.txt'")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("BERT嵌入特征评估系统")
    print("=" * 60)

    # 1. 加载数据
    X, y, titles = load_data()

    if X is None or y is None:
        print("错误: 无法加载数据，请检查文件是否存在")
        return

    # 2. 基本分类评估
    results, X_test, y_test, X_train, y_train = evaluate_classification(X, y)

    # 3. 交叉验证评估
    cv_results = cross_validation_evaluation(X, y, cv=5)

    # 4. 确定最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['macro_f1'])
    print(f"\n最佳模型: {best_model_name}")
    print(f"最佳宏平均F1分数: {results[best_model_name]['macro_f1']:.4f}")

    # 5. 生成格式化输出
    generate_formatted_output(results, best_model_name)

    # 6. 可视化结果（传递必要的参数）
    visualize_results(results, cv_results, X_scaled_global, X_global)

    # 7. 打印重要结论
    print("\n" + "=" * 60)
    print("评估结论:")
    print("=" * 60)
    print(f"1. BERT嵌入特征在文本分级任务上表现{'良好' if results[best_model_name]['macro_f1'] > 0.5 else '一般'}")
    print(f"2. 最佳模型是 {best_model_name}，宏平均F1分数为 {results[best_model_name]['macro_f1']:.4f}")
    print(f"3. 不同年级的分类效果:")

    # 分析每个类别的表现
    best_result = results[best_model_name]
    y_pred = best_result['y_pred']

    for grade in sorted(np.unique(y_test)):
        mask = y_test == grade
        if np.sum(mask) > 0:
            grade_accuracy = np.mean(y_pred[mask] == grade)
            grade_samples = int(np.sum(mask))
            print(f"   年级{grade}: 准确率 {grade_accuracy:.2%} ({grade_samples}个样本)")

    print(f"\n4. 模型稳定性:")
    for name in cv_results:
        print(f"   {name}: {cv_results[name]['mean_f1']:.4f} (±{cv_results[name]['std_f1']:.4f})")


# ============================================================
# 运行主函数
# ============================================================

if __name__ == "__main__":
    main()
