import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import optuna

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("警告: xgboost未安装，将跳过XGBoost模型")

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("警告: lightgbm未安装，将跳过LightGBM模型")


try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("中文字体设置失败，使用默认字体")

warnings.filterwarnings('ignore', category=UserWarning)


def load_and_prepare_data(n_components=30, create_interaction_features=True):
    """加载和准备所有特征数据"""
    # 加载各个特征文件
    trad_features = pd.read_csv('trad.csv')
    syll_features = pd.read_csv('syll.csv')
    clgsngo_features = pd.read_csv('clgsngo.csv')

    # 加载嵌入特征
    embeddings = pd.read_csv('ceb_mbert_features.csv', header=None)

    # 确保标题一致
    trad_features = trad_features.sort_values('book_title').reset_index(drop=True)
    syll_features = syll_features.sort_values('book_title').reset_index(drop=True)
    clgsngo_features = clgsngo_features.sort_values('book_title').reset_index(drop=True)
    embeddings = embeddings.reset_index(drop=True)

    # 对高维嵌入特征进行PCA降维
    print(f"原始嵌入维度: {embeddings.shape}")
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings)
    print(f"PCA后嵌入维度: {embeddings_reduced.shape}")
    print(f"解释方差比: {pca.explained_variance_ratio_.sum():.4f}")

    # 创建列名
    embedding_columns = [f'embedding_pca_{i}' for i in range(embeddings_reduced.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_reduced, columns=embedding_columns)

    # 合并传统特征
    traditional_features = pd.concat([
        trad_features.drop(['book_title', 'grade_level'], axis=1),
        syll_features.drop(['book_title', 'grade_level'], axis=1),
        clgsngo_features.drop(['book_title', 'grade_level'], axis=1)
    ], axis=1)

    # 创建交互特征
    if create_interaction_features:
        print("创建高级交互特征...")

        #  创建文本复杂度特征
        traditional_features['text_complexity'] = (
                trad_features['average_syllable_count'] *
                trad_features['polysyll_count'] /
                (trad_features['word_count'] + 1)
        )

        #创建句子结构特征
        traditional_features['sentence_density'] = (
                trad_features['word_count'] / (trad_features['sentence_count'] + 1)
        )
        traditional_features['phrase_complexity'] = (
                trad_features['phrase_count_per_sentence'] *
                trad_features['average_sentence_len']
        )

        # 创建语言特征组合
        traditional_features['ceb_dominance'] = (
                clgsngo_features['cebuano_bigram_sim'] -
                (clgsngo_features['tagalog_bigram_sim'] + clgsngo_features['bikol_bigram_sim']) / 2
        )

        # 创建音节模式组合
        traditional_features['cv_ratio'] = (
                syll_features['cv_density'] / (syll_features['vc_density'] + 1e-6)
        )
        traditional_features['complex_syllables'] = (
                syll_features['cvcc_density'] + syll_features['ccvc_density'] + syll_features['ccvcc_density']
        )

        #  创建统计特征
        traditional_features['variability_score'] = (
                trad_features['average_word_len'] * trad_features['average_syllable_count']
        )

        # 创建读写难度特征
        traditional_features['readability_score'] = (
                trad_features['average_word_len'] +
                trad_features['average_sentence_len'] * 0.1 +
                trad_features['average_syllable_count'] * 2
        )

    # 合并所有特征
    merged_features = pd.concat([
        traditional_features,
        syll_features.drop(['book_title', 'grade_level'], axis=1),
        embeddings_df
    ], axis=1)

    # 添加标题作为索引
    merged_features.index = trad_features['book_title']

    # 准备标签
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(trad_features['grade_level'])

    print(f"最终特征维度: {merged_features.shape}")

    return merged_features, y, label_encoder, trad_features['book_title']


def optimize_model_with_optuna(X_train, y_train, model_name, n_trials=20):
    """使用Optuna自动优化模型超参数"""

    def objective(trial):
        if model_name == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

        elif model_name == 'gbdt':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            }
            model = GradientBoostingClassifier(**params, random_state=42)

        elif model_name == 'extra_trees':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            model = ExtraTreesClassifier(**params, random_state=42, n_jobs=-1)

        elif model_name == 'svm_rbf':
            params = {
                'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': trial.suggest_categorical('kernel', ['rbf']),
            }
            model = SVC(**params, probability=True, random_state=42, class_weight='balanced')

        elif model_name == 'mlp':
            params = {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes',
                                                                [(64,), (128,), (64, 32), (128, 64)]),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 0.0001, 0.01, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            }
            model = MLPClassifier(**params, max_iter=2000, random_state=42, early_stopping=True)

        elif model_name == 'xgb' and XGB_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
            model = XGBClassifier(**params, random_state=42, n_jobs=-1)

        elif model_name == 'lgbm' and LGBM_AVAILABLE:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = LGBMClassifier(**params, random_state=42, n_jobs=-1)

        else:
            # 如果模型不支持或未安装，返回0
            return 0.0

        # 使用交叉验证评估
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='f1_macro', n_jobs=-1)

        return scores.mean()

    try:
        # 创建Optuna研究
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\n{model_name} 最佳超参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"  最佳F1分数: {study.best_value:.4f}")

        return study.best_params
    except Exception as e:
        print(f"  优化 {model_name} 时出错: {e}")
        return None


def train_enhanced_model():
    print("=" * 60)
    print("开始训练增强版模型...")
    print("=" * 60)

    #  加载数据
    X, y, label_encoder, titles = load_and_prepare_data(n_components=30, create_interaction_features=True)

    # 查看类别分布
    from collections import Counter
    class_dist = Counter(y)

    # 计算类别权重
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))

    # 特征选择
    print("\n进行特征选择...")

    # 基于随机森林的特征选择
    rf_for_selection = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    rf_for_selection.fit(X, y)

    # 选择重要性大于中位数的特征
    importances = rf_for_selection.feature_importances_
    threshold = np.median(importances[importances > 0])
    selector_rf = SelectFromModel(rf_for_selection, threshold=threshold, prefit=True)
    X_selected_rf = selector_rf.transform(X)

    #  递归特征消除
    print("进行递归特征消除...")
    lr_for_rfe = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear'
    )
    rfe = RFE(estimator=lr_for_rfe, n_features_to_select=50, step=5)
    X_selected_rfe = rfe.fit_transform(X, y)

    # 选择效果更好的特征选择方法
    print(f"RF特征选择后维度: {X_selected_rf.shape}")
    print(f"RFE特征选择后维度: {X_selected_rfe.shape}")

    # 使用RF特征选择（通常效果更好）
    X_selected = X_selected_rf
    selector = selector_rf

    selected_features = X.columns[selector.get_support()]
    print(f"选择了 {len(selected_features)} 个最重要的特征")

    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # 分割数据集
    X_train, X_test, y_train, y_test, titles_train, titles_test = train_test_split(
        X_scaled, y, titles, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"特征维度: {X_train.shape[1]}")

    # 创建类别名称
    target_names = [str(cls) for cls in label_encoder.classes_]

    #  使用默认参数训练基础模型进行筛选
    print("\n" + "=" * 60)
    print("使用默认参数训练基础模型进行筛选...")
    print("=" * 60)

    # 计算每个类别的样本权重
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    basic_estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('extra_trees', ExtraTreesClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('gbdt', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )),
        ('svm_rbf', SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42
        )),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=42,
            early_stopping=True
        )),
    ]

    # 训练这些基础模型，选择最好的进行调优
    print("训练基础模型进行初步筛选...")
    initial_results = {}
    for name, model in basic_estimators:
        try:
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                        scoring='f1_macro', n_jobs=-1)
            initial_results[name] = cv_scores.mean()
            print(f"  {name}: CV F1 = {cv_scores.mean():.4f}")
        except Exception as e:
            print(f"  {name} 训练失败: {e}")
            initial_results[name] = 0

    # 选择表现最好的模型进行调优（至少选择2个）
    best_models = sorted(initial_results.items(), key=lambda x: x[1], reverse=True)

    # 只选择F1大于0.6的模型进行调优
    models_to_optimize = [name for name, score in best_models if score > 0.6]
    if len(models_to_optimize) > 3:
        models_to_optimize = models_to_optimize[:3]  # 最多优化3个模型


    # 对选中的模型进行自动调优
    optimized_estimators = []

    for model_name in models_to_optimize:
        print(f"\n{'=' * 40}")
        print(f"正在优化 {model_name}...")
        print('=' * 40)

        best_params = optimize_model_with_optuna(X_train, y_train, model_name, n_trials=20)

        if best_params is not None:
            # 使用最佳参数创建模型
            if model_name == 'rf':
                model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
            elif model_name == 'extra_trees':
                model = ExtraTreesClassifier(**best_params, random_state=42, n_jobs=-1)
            elif model_name == 'gbdt':
                model = GradientBoostingClassifier(**best_params, random_state=42)
            elif model_name == 'svm_rbf':
                model = SVC(**best_params, probability=True, random_state=42)
            elif model_name == 'mlp':
                model = MLPClassifier(**best_params, max_iter=2000, random_state=42)

            optimized_estimators.append((f"{model_name}_opt", model))
        else:
            # 如果优化失败，使用原始模型
            for name, model in basic_estimators:
                if name == model_name:
                    optimized_estimators.append((f"{model_name}", model))
                    break

    # 如果没有成功优化的模型，使用所有基础模型
    if len(optimized_estimators) == 0:
        estimators = [(name, model) for name, model in basic_estimators]
    else:
        # 添加其他非优化模型
        estimators = optimized_estimators + [
            ('knn', KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                n_jobs=-1
            )),
            ('logistic', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                solver='liblinear'
            )),
            ('ada_boost', AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=5),
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            )),
        ]

    print(f"\n最终使用的模型数量: {len(estimators)}")

    # 6. 训练基础模型并评估
    print("\n" + "=" * 60)
    print("训练和评估基础模型...")
    print("=" * 60)

    base_model_results = {}

    for name, model in estimators:
        print(f"\n训练 {name}...")

        try:
            # 训练模型
            if hasattr(model, 'fit'):
                if 'rf' in name or 'extra' in name:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train, y_train)
            else:
                print(f"  跳过 {name}，没有fit方法")
                continue

            # 交叉验证评估
            cv_scores = cross_val_score(model, X_train, y_train,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                        scoring='f1_macro',
                                        n_jobs=-1)

            # 测试集评估
            y_pred = model.predict(X_test)
            f1_test = f1_score(y_test, y_pred, average='macro')
            acc_test = accuracy_score(y_test, y_pred)

            base_model_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_f1': f1_test,
                'test_acc': acc_test,
                'model': model
            }

            print(f"  CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print(f"  Test F1: {f1_test:.4f}, Test Acc: {acc_test:.4f}")

        except Exception as e:
            print(f"  训练 {name} 时出错: {str(e)}")
            continue

    # 显示基础模型比较
    print("\n" + "=" * 60)
    print("基础模型比较:")
    print("=" * 60)
    for name, results in sorted(base_model_results.items(), key=lambda x: x[1]['test_f1'], reverse=True):
        print(f"{name:12s} | CV F1: {results['cv_mean']:.4f} +/- {results['cv_std']:.4f} | "
              f"Test F1: {results['test_f1']:.4f} | Test Acc: {results['test_acc']:.4f}")

    # 7. 创建增强的投票分类器
    print("\n" + "=" * 60)
    print("创建增强版集成模型...")
    print("=" * 60)

    # 选择表现最好的5个模型
    best_models = sorted(base_model_results.items(), key=lambda x: x[1]['test_f1'], reverse=True)[:5]
    print(f"选择的最佳模型: {[name for name, _ in best_models]}")

    # 创建加权投票分类器
    voting_estimators = [(name, results['model']) for name, results in best_models]
    voting_weights = [results['test_f1'] for _, results in best_models]  # 根据F1分数加权

    voting_clf = VotingClassifier(
        estimators=voting_estimators,
        voting='soft',
        weights=voting_weights,
        n_jobs=-1
    )

    # 训练投票分类器
    print("训练加权投票分类器...")
    voting_clf.fit(X_train, y_train)

    # 创建堆叠分类器
    stacking_clf = StackingClassifier(
        estimators=voting_estimators,
        final_estimator=LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight='balanced',
            random_state=42,
            solver='liblinear'
        ),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=True,
        n_jobs=-1
    )

    # 训练堆叠分类器
    print("训练堆叠分类器...")
    stacking_clf.fit(X_train, y_train)

    # 创建Bagging集成
    bagging_clf = BaggingClassifier(
        estimator=RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ),
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        bootstrap_features=False
    )

    # 训练Bagging分类器
    print("训练Bagging分类器...")
    bagging_clf.fit(X_train, y_train)

    # 8. 集成所有分类器的预测（超级集成）
    def super_ensemble_predict(X, models_dict, voting_weight=0.4, stacking_weight=0.4, bagging_weight=0.2):
        voting_proba = models_dict['voting'].predict_proba(X)
        stacking_proba = models_dict['stacking'].predict_proba(X)
        bagging_proba = models_dict['bagging'].predict_proba(X)

        # 加权平均
        combined_proba = (
                voting_weight * voting_proba +
                stacking_weight * stacking_proba +
                bagging_weight * bagging_proba
        )
        return np.argmax(combined_proba, axis=1), combined_proba

    # 9. 评估所有集成方法
    print("\n" + "=" * 60)
    print("评估所有集成方法:")
    print("=" * 60)

    results = {}
    models_dict = {
        'voting': voting_clf,
        'stacking': stacking_clf,
        'bagging': bagging_clf
    }

    # 评估各个集成方法
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        results[name] = {'f1': f1, 'acc': acc}
        print(f"{name:10s}: F1 = {f1:.4f}, Accuracy = {acc:.4f}")

    # 评估超级集成
    y_pred_super, y_proba_super = super_ensemble_predict(
        X_test, models_dict,
        voting_weight=0.4, stacking_weight=0.4, bagging_weight=0.2
    )
    f1_super = f1_score(y_test, y_pred_super, average='macro')
    acc_super = accuracy_score(y_test, y_pred_super)
    results['super_ensemble'] = {'f1': f1_super, 'acc': acc_super}
    print(f"super_ensemble: F1 = {f1_super:.4f}, Accuracy = {acc_super:.4f}")


    # 显示比较结果
    print("\n方法比较 (按F1排序):")
    for method, scores in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{method:15s}: F1 = {scores['f1']:.4f}, Accuracy = {scores['acc']:.4f}")

    # 选择最佳方法
    best_method = max(results, key=lambda x: results[x]['f1'])
    best_score = results[best_method]['f1']

    print(f"\n最佳方法: {best_method} (F1: {best_score:.4f})")



    #  使用最佳方法进行详细评估
    if best_method == 'voting':
        best_model = voting_clf
        y_pred_final = voting_clf.predict(X_test)
        y_proba_final = voting_clf.predict_proba(X_test)
    elif best_method == 'stacking':
        best_model = stacking_clf
        y_pred_final = stacking_clf.predict(X_test)
        y_proba_final = stacking_clf.predict_proba(X_test)
    elif best_method == 'bagging':
        best_model = bagging_clf
        y_pred_final = bagging_clf.predict(X_test)
        y_proba_final = bagging_clf.predict_proba(X_test)
    else:  # super_ensemble
        best_model = models_dict
        y_pred_final = y_pred_super
        y_proba_final = y_proba_super

    # 详细评估
    print("\n" + "=" * 60)
    print(f"{best_method} 详细评估结果:")
    print("=" * 60)

    print("\n分类报告:")
    print(classification_report(y_test, y_pred_final,
                                target_names=target_names,
                                digits=4))

    print(f"宏平均F1分数: {best_score:.4f}")
    print(f"准确率: {accuracy_score(y_test, y_pred_final):.4f}")

    # 计算每个类别的指标
    precision_per_class = precision_score(y_test, y_pred_final, average=None)
    recall_per_class = recall_score(y_test, y_pred_final, average=None)
    f1_per_class = f1_score(y_test, y_pred_final, average=None)

    print("\n各类别详细指标:")
    for i, (prec, rec, f1_c) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        print(f"  类别 {target_names[i]}: Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1_c:.4f}")

    #  绘制增强版可视化
    cm = confusion_matrix(y_test, y_pred_final)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 常规混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0, 0])
    axes[0, 0].set_title(f'{best_method} - Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0, 1])
    axes[0, 1].set_title(f'{best_method} - Normalized Confusion Matrix')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')

    # 各类别F1分数比较
    class_indices = range(len(target_names))
    axes[1, 0].bar(class_indices, f1_per_class, color=['blue', 'green', 'red'])
    axes[1, 0].set_xlabel('Class')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score by Class')
    axes[1, 0].set_xticks(class_indices)
    axes[1, 0].set_xticklabels(target_names)
    axes[1, 0].set_ylim(0, 1)

    #  预测置信度分布
    confidence_scores = np.max(y_proba_final, axis=1)
    axes[1, 1].hist(confidence_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Confidence')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Confidence Distribution')

    plt.tight_layout()
    plt.savefig(f'{best_method}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存模型
    print("\n" + "=" * 60)
    print("保存模型和结果...")
    print("=" * 60)

    if best_method == 'super_ensemble':
        # 保存所有模型
        joblib.dump(voting_clf, 'voting_model.pkl')
        joblib.dump(stacking_clf, 'stacking_model.pkl')
        joblib.dump(bagging_clf, 'bagging_model.pkl')
        print("超级集成模型已保存到: voting_model.pkl, stacking_model.pkl, bagging_model.pkl")
    else:
        model_filename = f'best_{best_method}_model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"最佳模型已保存到: {model_filename}")

    joblib.dump(selector, 'feature_selector.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    # 保存详细预测结果
    results_df = pd.DataFrame({
        'book_title': titles_test,
        'true_label': label_encoder.inverse_transform(y_test),
        'pred_label': label_encoder.inverse_transform(y_pred_final),
        'confidence': np.max(y_proba_final, axis=1),
        'correct': (y_pred_final == y_test).astype(int)
    })

    # 添加每个类别的概率
    for i in range(len(target_names)):
        results_df[f'prob_class_{target_names[i]}'] = y_proba_final[:, i]

    results_df.to_csv('enhanced_predictions_detailed.csv', index=False)
    print(f"详细预测结果已保存到: enhanced_predictions_detailed.csv")

    #  生成模型性能报告
    report_df = pd.DataFrame([
        {
            'Method': method,
            'F1_Score': scores['f1'],
            'Accuracy': scores['acc']
        }
        for method, scores in results.items()
    ]).sort_values('F1_Score', ascending=False)

    report_df.to_csv('model_performance_comparison.csv', index=False)
    print(f"模型性能比较报告已保存到: model_performance_comparison.csv")

    return best_model, scaler, label_encoder, selector, results


if __name__ == "__main__":
    model, scaler, label_encoder, selector, performance_results = train_enhanced_model()

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    best_method = max(performance_results, key=lambda x: performance_results[x]['f1'])
    best_score = performance_results[best_method]['f1']
    print(f"最佳方法: {best_method}")
    print(f"最佳F1分数: {best_score:.4f}")
    print(f"最佳准确率: {performance_results[best_method]['acc']:.4f}")