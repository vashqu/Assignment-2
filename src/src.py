from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (f1_score, f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix)
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.linear_model import LogisticRegression


class Preprocessor:
    """
    Preprocessing class for handling missing values and scaling.
    Imputes Nans using median and scales using RobustScaler.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
    
    def fit (self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.pipeline.fit_transform(X,y)

class RepeatedNestedCV:
    """
    Repeated Nested Cross-Validation for model selection and evaluation.
    """
    def __init__(self, X, y, models, param_grids, n_repeats=10, n_outer=5, n_inner=3, random_state=42, scoring='f1'):
        self.X = X
        self.y = y
        self.models = models #dict of model name and scikit model
        self.param_grids = param_grids #dict of model name and param grid
        self.n_repeats = n_repeats
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.random_state = random_state
        self.scoring = scoring
        self.results = {}

    def run(self):
        for model_name, model in self.models.items():
            self.results[model_name] = []
            
            for i in range(self.n_repeats):
                outer_cv = StratifiedKFold(n_splits = self.n_outer, shuffle=True, random_state=self.random_state + i)
                
                for fold_index, (train_index, test_index) in enumerate(outer_cv.split(self.X, self.y)):

                    X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                    y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                    
                    pline = Pipeline([
                        ('preprocessor', Preprocessor().pipeline),
                        ('model', model)
                    ])
                    print(f"Running {model_name} on fold {fold_index + 1} of repeat {i + 1}")
                    inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.random_state + i)
                    grid = GridSearchCV(pline, self.param_grids[model_name], cv=inner_cv, scoring=self.scoring, n_jobs=-1)

                    grid.fit(X_train, y_train)
                    
                    best_model = grid.best_estimator_
                    
                    y_pred = best_model.predict(X_test)
                    y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
                    
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                    metrics = {
                        'repeat': i,
                        'fold': fold_index,
                        'model': model_name,
                        'params': grid.best_params_,
                        'f1_score': f1_score(y_test, y_pred),
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
                        'mcc': matthews_corrcoef(y_test, y_pred),
                        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                        'specificity': tn / (tn + fp) if (tn + fp) > 0 else None,
                        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else None,
                        'npv': tn / (tn + fn) if (tn + fn) > 0 else None,
                        'prauc': average_precision_score(y_test, y_prob) if y_prob is not None else None,
                    }

                    self.results[model_name].append(metrics)

    def get_results(self, summary=True):
        # Flatten all results into one DataFrame
        all_rows = []
        for model_name, scores in self.results.items():
            for row in scores:
                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        if not summary:
            return df  # full results per fold

            # Otherwise, return a grouped summary
        metrics = [col for col in df.columns if col not in ['repeat', 'fold', 'model', 'params']]
        grouped = df.groupby('model')[metrics].agg(['mean', 'median', 'std'])

        return grouped



class Bootstrapper:
    def __init__(self, df, models, metrics, group_col = 'model', n_random_samples = 1000, ci = 95, random_state = 42):
        self.df = df[df[group_col].isin(models)].copy()
        self.models = models
        self.metrics = metrics
        self.group_col = group_col
        self.n_random_samples = n_random_samples
        self.ci = ci
        self.random_state = random_state
        self.bootstrap_results = defaultdict(dict)

    def _bootstrap(self, values):
        medians = [
            np.median(np.random.choice(values, size=len(values), replace=True))
            for _ in range(self.n_random_samples)
        ]
        return np.array(medians)
    
    def compute(self):
        summary_rows = []

        for model, group in self.df.groupby(self.group_col):
            for metric in self.metrics:
                values = group[metric].values
                medians = self._bootstrap(values)
                lower_bound = np.percentile(medians, (100 - self.ci) / 2)
                upper_bound = np.percentile(medians, 100 - (100 - self.ci) / 2)
                self.bootstrap_results[model][metric] = medians

                summary_rows.append({
                    'model': model,
                    'metric': metric,
                    'median': np.median(values),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                })
        return pd.DataFrame(summary_rows)
    

def plot_ci_bars(df, metric):
    # Filter for selected metric
    df_metric = df[df['metric'] == metric].copy()

    df_metric.sort_values(by='median', ascending=True, inplace=True)

    plt.figure(figsize=(8, 4))
    sns.pointplot(
        data=df_metric,
        x='median',
        y='model',
        join=False,
        color='black',
        capsize=0.2,
        errwidth=1.5
    )

    # Add CI manually
    for i, row in df_metric.iterrows():
        plt.plot(
            [row['lower_bound'], row['upper_bound']],
            [row['model'], row['model']],
            color='black',
            lw=2
        )

    plt.title(f'{metric.upper()} with 95% Confidence Intervals')
    plt.xlabel(metric.upper())
    plt.ylabel('Model')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


class FinalModelTrainer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.search = None
        self.final_model = None

        self.param_grid = {
            'model__C': [0.01, 0.1, 1, 10],
            'model__l1_ratio': [0.2, 0.5, 0.8]
        }

    def build_pipeline(self):
        return Pipeline([
            ('preprocess', Preprocessor()),
            ('model', LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                max_iter=10000
            ))
        ])

    def train(self, X, y, scoring='f1'):
        pipeline = self.build_pipeline()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        self.search = GridSearchCV(
            pipeline,
            param_grid=self.param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )

        self.search.fit(X, y)
        self.final_model = self.search.best_estimator_

        print("Best Params:", self.search.best_params_)
        print("Best Score:", self.search.best_score_)

        return self.final_model

    def save_model(self, filename='final_model.pkl'):
        path = f'{self.model_dir}/{filename}'
        joblib.dump(self.final_model, path)
        print(f"Model saved to {path}")
