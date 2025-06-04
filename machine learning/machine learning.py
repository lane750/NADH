import itertools
from sklearn.model_selection import ParameterGrid
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

def train_with_combinations_xgboost(X_selected, y2, selected_features, test_size=3, param_grid=None, output_file='output-xgboost.txt'):
    results = []
    all_combinations = list(itertools.combinations(range(len(X_selected)), test_size))
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    with open(output_file, 'w') as file:
        file.write('TestDataID,MaxDepth,NEstimators,LearningRate,TrainR2,TestR2,Top1Feature,Top2Feature,Top3Feature,Top4Feature,Top5Feature,Top6Feature,Top7Feature,Top8Feature,Top9Feature,Top10Feature\n')
        for combination in all_combinations:
            X_test = X_selected[list(combination), :]
            y_test = y2[list(combination)]
            train_indices = list(set(range(len(X_selected))) - set(combination))
            X_train = X_selected[train_indices, :]
            y_train = y2[train_indices]
            best_model = None
            best_r2 = -float('inf')
            best_params = None
            best_train_r2 = None
            best_test_r2 = None
            best_top_features = None
            for params in ParameterGrid(param_grid):
                max_depth = params['max_depth']
                n_estimators = params['n_estimators']
                learning_rate = params['learning_rate']
                model = xgb.XGBRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    verbosity=0
                )
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                train_r2 = r2_score(y_train, train_preds)
                test_r2 = r2_score(y_test, test_preds)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model
                    best_params = (max_depth, n_estimators, learning_rate)
                    best_train_r2 = train_r2
                    best_test_r2 = test_r2
                    feature_importances = model.feature_importances_
                    feature_names = selected_features  # 使用已选择的特征名称列表
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    })
                    top_k_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
                    best_top_features = top_k_features['Feature'].values
            file.write(f'{combination},{best_params[0]},{best_params[1]},{best_params[2]},{best_train_r2},{best_test_r2},{",".join(best_top_features)}\n')
            results.append({
                'combination': combination,
                'max_depth': best_params[0],
                'n_estimators': best_params[1],
                'learning_rate': best_params[2],
                'train_r2': best_train_r2,
                'test_r2': best_test_r2,
                'top_features': best_top_features
            })
    return results

def train_with_combinations_catboost(X_selected, y2, selected_features, test_size=3, param_grid=None, output_file='output-catboost.txt'):
    results = []
    all_combinations = list(itertools.combinations(range(len(X_selected)), test_size))
    if param_grid is None:
        param_grid = {
            'depth': [3, 5, 7],
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    with open(output_file, 'w') as file:
        file.write('TestDataID,Depth,Iterations,LearningRate,TrainR2,TestR2,Top1Feature,Top2Feature,Top3Feature,Top4Feature,Top5Feature,Top6Feature,Top7Feature,Top8Feature,Top9Feature,Top10Feature\n')
        for combination in all_combinations:
            X_test = X_selected[list(combination), :]
            y_test = y2[list(combination)]
            train_indices = list(set(range(len(X_selected))) - set(combination))
            X_train = X_selected[train_indices, :]
            y_train = y2[train_indices]
            best_r2 = -float('inf')
            best_params = None
            best_train_r2 = None
            best_test_r2 = None
            best_top_features = None
            for params in ParameterGrid(param_grid):
                depth = params['depth']
                iterations = params['iterations']
                learning_rate = params['learning_rate']
                model = CatBoostRegressor(
                    depth=depth,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    border_count=48,
                    l2_leaf_reg=1,
                    subsample=0.8,
                    verbose=0
                )
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                train_r2 = r2_score(y_train, train_preds)
                test_r2 = r2_score(y_test, test_preds)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model
                    best_params = (depth, iterations, learning_rate)
                    best_train_r2 = train_r2
                    best_test_r2 = test_r2
                    feature_importances = model.get_feature_importance()
                    feature_names = selected_features
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    })
                    top_k_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
                    best_top_features = top_k_features['Feature'].values
            file.write(f'{combination},{best_params[0]},{best_params[1]},{best_params[2]},{best_train_r2},{best_test_r2},{",".join(best_top_features)}\n')
            results.append({
                'combination': combination,
                'depth': best_params[0],
                'iterations': best_params[1],
                'learning_rate': best_params[2],
                'train_r2': best_train_r2,
                'test_r2': best_test_r2,
                'top_features': best_top_features
            })
    return results

def train_with_combinations_lightgbm(X_selected, y2, selected_features, test_size=3, param_grid=None, output_file='output-lightgbm.txt'):
    results = []
    all_combinations = list(itertools.combinations(range(len(X_selected)), test_size))  # 获取所有数据组合，每次3个数据作为测试集
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    with open(output_file, 'w') as file:
        file.write('TestDataID,MaxDepth,NEstimators,LearningRate,TrainR2,TestR2,Top1Feature,Top2Feature,Top3Feature,Top4Feature,Top5Feature,Top6Feature,Top7Feature,Top8Feature,Top9Feature,Top10Feature\n')
        for combination in all_combinations:
            X_test = X_selected[list(combination), :]
            y_test = y2[list(combination)]
            train_indices = list(set(range(len(X_selected))) - set(combination))
            X_train = X_selected[train_indices, :]
            y_train = y2[train_indices]
            best_r2 = -float('inf')
            best_params = None
            best_train_r2 = None
            best_test_r2 = None
            best_top_features = None
            for params in ParameterGrid(param_grid):
                max_depth = params['max_depth']
                n_estimators = params['n_estimators']
                learning_rate = params['learning_rate']
                model = lgb.LGBMRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1,
                    verbosity=-1
                )
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                train_r2 = r2_score(y_train, train_preds)
                test_r2 = r2_score(y_test, test_preds)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_params = (max_depth, n_estimators, learning_rate)
                    best_train_r2 = train_r2
                    best_test_r2 = test_r2
                    feature_importances = model.feature_importances_
                    feature_names = selected_features
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    })
                    top_k_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
                    best_top_features = top_k_features['Feature'].values
            file.write(f'{combination},{best_params[0]},{best_params[1]},{best_params[2]},{best_train_r2},{best_test_r2},{",".join(best_top_features)}\n')
            results.append({
                'combination': combination,
                'max_depth': best_params[0],
                'n_estimators': best_params[1],
                'learning_rate': best_params[2],
                'train_r2': best_train_r2,
                'test_r2': best_test_r2,
                'top_features': best_top_features
            })
    return results

def train_with_combinations_random_forest(X_selected, y2, selected_features, test_size=3, param_grid=None, output_file='output-random_forest.txt'):
    results = []
    all_combinations = list(itertools.combinations(range(len(X_selected)), test_size))
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200, 300],
            'min_samples_split': [2, 5, 10]
        }
    with open(output_file, 'w') as file:
        file.write('TestDataID,MaxDepth,NEstimators,MinSamplesSplit,TrainR2,TestR2,Top1Feature,Top2Feature,Top3Feature,Top4Feature,Top5Feature,Top6Feature,Top7Feature,Top8Feature,Top9Feature,Top10Feature\n')
        for combination in all_combinations:
            X_test = X_selected[list(combination), :]
            y_test = y2[list(combination)]
            train_indices = list(set(range(len(X_selected))) - set(combination))
            X_train = X_selected[train_indices, :]
            y_train = y2[train_indices]
            best_r2 = -float('inf')
            best_params = None
            best_train_r2 = None
            best_test_r2 = None
            best_top_features = None
            for params in ParameterGrid(param_grid):
                max_depth = params['max_depth']
                n_estimators = params['n_estimators']
                min_samples_split = params['min_samples_split']
                model = RandomForestRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    min_samples_split=min_samples_split,
                    verbose=0
                )
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                train_r2 = r2_score(y_train, train_preds)
                test_r2 = r2_score(y_test, test_preds)
                if test_r2 > best_r2:
                    best_r2 = test_r2
                    best_model = model
                    best_params = (max_depth, n_estimators, min_samples_split)
                    best_train_r2 = train_r2
                    best_test_r2 = test_r2
                    feature_importances = model.feature_importances_
                    feature_names = selected_features
                    feature_importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': feature_importances
                    })
                    top_k_features = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
                    best_top_features = top_k_features['Feature'].values
            file.write(f'{combination},{best_params[0]},{best_params[1]},{best_params[2]},{best_train_r2},{best_test_r2},{",".join(best_top_features)}\n')
            results.append({
                'combination': combination,
                'max_depth': best_params[0],
                'n_estimators': best_params[1],
                'min_samples_split': best_params[2],
                'train_r2': best_train_r2,
                'test_r2': best_test_r2,
                'top_features': best_top_features
            })

    return results

data = pd.read_excel('train-data.xlsx')
y1,y2,y3= data.iloc[:,-1],data.iloc[:,-2],data.iloc[:,-3]
x = data.iloc[:,2:-3]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
selector = SelectKBest(f_regression, k='all')
X_selected = selector.fit_transform(X_scaled, y2)
selected_features = x.columns[selector.get_support()]

train_with_combinations_xgboost(X_scaled, y2, selected_features, test_size=3, param_grid=None)
train_with_combinations_catboost(X_scaled, y2, selected_features, test_size=3, param_grid=None)
train_with_combinations_lightgbm(X_scaled, y2, selected_features, test_size=3, param_grid=None)
train_with_combinations_random_forest(X_scaled, y2, selected_features, test_size=3, param_grid=None)