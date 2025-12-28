import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.base import clone
import joblib


class GradeLevelEnsembleModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stacked_model = None
        self.base_models = None
        self.meta_model = None
        self.feature_names = None

    def create_stacked_ensemble(self):
        """创建堆叠集成模型 - 避免XGBoost兼容性问题"""
        # 使用更兼容的基础模型
        self.base_models = [
            ('random_forest', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )),
            ('logistic_regression', LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                C=1.0,
                penalty='l2',
                solver='liblinear'
            )),
            ('knn', KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )),
            ('svm', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ))
        ]

        # 元学习器
        self.meta_model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            C=0.8,
            penalty='l2',
            solver='liblinear'
        )

        # 创建堆叠模型
        self.stacked_model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=self.meta_model,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            passthrough=True,
            n_jobs=-1
        )

        return self.stacked_model

    def fit(self, X, y, feature_names=None):
        """训练集成模型"""
        # 确保X是numpy数组，避免DataFrame兼容性问题
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        if self.stacked_model is None:
            self.create_stacked_ensemble()

        print("训练堆叠集成模型...")
        self.stacked_model.fit(X_array, y)

        if feature_names is not None:
            self.feature_names = feature_names

        # 保存模型
        self.save_model('grade_level_ensemble_model.pkl')

        return self

    def predict(self, X):
        """预测"""
        if self.stacked_model is None:
            raise ValueError("模型未训练，请先调用fit()方法")

        # 确保X是numpy数组
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        return self.stacked_model.predict(X_array)

    def predict_proba(self, X):
        """预测概率"""
        if self.stacked_model is None:
            raise ValueError("模型未训练，请先调用fit()方法")

        # 确保X是numpy数组
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        return self.stacked_model.predict_proba(X_array)

    def save_model(self, path):
        """保存模型"""
        joblib.dump(self, path)
        print(f"模型已保存到: {path}")

    @classmethod
    def load_model(cls, path):
        """加载模型"""
        return joblib.load(path)