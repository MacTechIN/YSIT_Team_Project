# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 10:11:05 2024

@author: ham90
"""
import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore') #warning 문구 제거
pd.set_option('display.max_columns', None)

path ='.'
modeld = joblib.load(path + '/catboost_model_D_test.pkl')
traind = pd.read_csv(path + '/관광지 추천시스템 Trainset_D.csv')

st.title('여행지 추천 서비스')

# 사용자 입력 폼
st.header('사용자 정보 입력')
travel_mission_priority = st.slider('여행 목적 우선순위', 1, 5, 3)
gender = st.selectbox('성별', ['남성', '여성'])
age_grp = st.slider('연령대', 1, 100, 30)
sido = st.selectbox('시/도를 선택하세요', ['제주특별자치도', '...'])
gungu = st.selectbox('군/구를 선택하세요', ['서귀포시', '제주시','...'])
eupmyeon = st.selectbox('읍/면을 선택하세요', ['중문동', '일도일동','연동','애월읍','대정읍','성산읍','이도일동','안덕면','애월읍'])
INCOME = st.slider('수입', 1, 12, 1)
TRAVEL_STYL_1 = st.slider('자연 vs 도시', 1, 7, 4)
TRAVEL_STYL_2 = st.slider('숙박 vs 당일', 1, 7, 4)
TRAVEL_STYL_3 = st.slider('새로운 지역 vs 익숙한 지역', 1, 7, 4)
TRAVEL_STYL_4 = st.slider('편하지만 비싼 숙소 vs 불편하지만 저렴한 숙소', 1, 7, 4)
tsy_5 = st.slider('휴양/휴식 vs 체험활동', 1, 7, 4)
tsy_6 = st.slider('잘 알려지지 않은 방문지 vs 알려진 방문지', 1, 7, 4)
tsy_7 = st.slider('계획 vs 즉흥', 1, 7, 4)
tsy_8 = st.slider('사진촬영 안중요 vs 중요', 1, 7, 4)
tmt = st.slider('여행 테마', 1, 28, 12)
tnm = st.slider('여행 횟수', 1, 20, 5)
tcpn = st.slider('동반자 수', 1, 20, 10)
#유저정보
data = pd.DataFrame({
    'TRAVEL_MISSION_PRIORITY': [travel_mission_priority],
    'GENDER': [gender],
    'AGE_GRP': [age_grp],
    'SIDO' : [sido],
    'GUNGU': [gungu],
    'EUPMYEON': [eupmyeon],
    'INCOME': [INCOME],
    'TRAVEL_STYL_1' : [TRAVEL_STYL_1],
    'TRAVEL_STYL_2' : [TRAVEL_STYL_2],
    'TRAVEL_STYL_3' : [TRAVEL_STYL_3],
    'TRAVEL_STYL_4' : [TRAVEL_STYL_4],
    'TRAVEL_STYL_5': [tsy_5],
    'TRAVEL_STYL_6': [tsy_6],
    'TRAVEL_STYL_7': [tsy_7],
    'TRAVEL_STYL_8': [tsy_8],
    'TRAVEL_MOTIVE_1': [tmt],
    'TRAVEL_NUM': [tnm],
    'TRAVEL_COMPANIONS_NUM': [tcpn]
    # ... 추가적인 사용자 정보
})

#여행지 정보
info = pd.read_csv(path + '/관광지 추천시스템 Testset_D- 여행지 정보.csv')

# In[71]:

final_df = pd.DataFrame(columns = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
       'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP', 'INCOME',
       'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
       'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
       'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
       'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean', 'REVISIT_YN_mean',
       'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']) #빈 데이터프레임에 내용 추가
####### 시/도 군/구 별 자료 수집

info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu) & (info['EUPMYEON'] == eupmyeon)] 

# info_df.drop(['SIDO'], inplace = True, axis = 1)
info_df.reset_index(inplace = True, drop = True)
data2 = data.drop(['SIDO','GUNGU', 'EUPMYEON'], axis =1)
user_df = pd.DataFrame([data2.iloc[0].to_list()]*len(info_df), columns = ['TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP', 'INCOME',
                     'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
                     'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
                     'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM'])
df = pd.concat([user_df, info_df], axis = 1)
df = df[['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'EUPMYEON', 'VISIT_AREA_TYPE_CD',
'TRAVEL_MISSION_PRIORITY', 'GENDER', 'AGE_GRP', 'INCOME',
'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
'TRAVEL_MOTIVE_1', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
'RESIDENCE_TIME_MIN_mean', 'RCMDTN_INTENTION_mean', 'REVISIT_YN_mean',
'TRAVEL_COMPANIONS_NUM_mean', 'REVISIT_INTENTION_mean']] # 변수정렬
df['VISIT_AREA_TYPE_CD'] = df['VISIT_AREA_TYPE_CD'].astype('string')
final_df = pd.concat([final_df, df], axis = 0)
final_df.reset_index(drop = True, inplace = True)
final_df.drop_duplicates(['VISIT_AREA_NM'], inplace = True)

#모델 예측
y_pred = modeld.predict(final_df)
y_pred = pd.DataFrame(y_pred, columns = ['y_pred'])
test_df1 = pd.concat([final_df, y_pred], axis = 1)
test_df1.sort_values(by = ['y_pred'], axis = 0, ascending=False, inplace = True) # 예측치가 높은 순대로 정렬

recomand10 = test_df1['VISIT_AREA_NM'].head(10)

if st.button('여행지 추천'):
    # 사용자 정보를 모델에 입력하여 여행지 추천
    # 여기에 모델 추론 및 결과 처리 코드 추가
    # ...

    # 예시: 추천 결과 표시
    st.header('여행지 추천 결과')
    st.write(recomand10)

    # 여행지 추천 결과 예시
    # recommended_places = model.predict(user_data)
    # st.write(recommended_places)