# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:03:43 2024

@author: InQ
"""
import pandas as pd
import catboost
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
import joblib
#%%
path = '.'

Train = pd.read_csv(path + '/source/관광지 추천시스템 Trainset_final_total.csv')

Train.drop(['TRAVELER_ID', 'REVISIT_INTENTION', 'RCMDTN_INTENTION','RESIDENCE_TIME_MIN',
            'REVISIT_YN','INCOME'], axis = 1, inplace = True)

Train['VISIT_AREA_TYPE_CD'] = Train['VISIT_AREA_TYPE_CD'].astype('string')

y_train = Train['DGSTFN']
X_train = Train.drop(['DGSTFN','TRAVEL_ID'], axis = 1)

categorical_cols = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD', 'GENDER'
                    , 'TRAVEL_MISSION_PRIORITY', 'AGE_GRP']
#X_train[categorical_cols] = X_train[categorical_cols].astype('category')
# XGBoost 모델 정의
model = catboost.CatBoostRegressor(cat_features=categorical_cols)

#탐색할 하이퍼파라미터 그리드 정의
param_grid = {
    'n_estimators': [400],
    'learning_rate': [0.01, 0.02, 0.05],    
    'depth': [6,8]
    }

# 사용할 평가 지표 정의 (R-squared를 사용하도록 수정)
scorer = make_scorer(r2_score)

# 그리드 서치 객체 생성 (verbose 추가)
grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=2, verbose=2)

# 훈련 데이터에 그리드 서치 수행
grid_result = grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters: ", grid_result.best_params_)

# 최적의 모델 출력
best_model = grid_result.best_estimator_
#%%
joblib.dump(best_model, path + '/ML/bestCatboost_model_D_travel.pkl')
