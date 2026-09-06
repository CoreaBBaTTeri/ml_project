import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# 페이지 설정
st.set_page_config(page_title='전력 수요 예측 앱', page_icon='⚡', layout='wide')
st.title('⚡ 계절 맞춤형 일일 최대전력 예측 애플리케이션')
st.markdown('---')

# 1. 모델 및 정보 로드 (캐싱)
@st.cache_resource
def load_models():
    try:
        return joblib.load('./power_models.joblib') # 딕셔너리(4개 모델)가 통째로 로드됨
    except Exception as e:
        st.error(f'모델 로드 실패: {e}')
        return None

def load_info():
    try:
        with open('./power_model_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

models = load_models()
model_info = load_info()

if models is None:
    st.stop()

# 2. 사이드바: 계절 선택 및 성능 표시
st.sidebar.header('⚙️ 설정 및 모델 정보')
selected_season = st.sidebar.selectbox('🌱 예측할 계절을 선택하세요', ['봄', '여름', '가을', '겨울'])

if model_info:
    info = model_info[selected_season]
    st.sidebar.subheader(f'📊 {selected_season}철 모델 성능')
    st.sidebar.info(f"적용 알고리즘: {info['model_type']}")
    st.sidebar.metric('결정계수 (R²)', f"{info['r2_score'] * 100:.1f}%")
    st.sidebar.caption("💡 100%에 가까울수록 예측이 정확함을 의미합니다.")

# 3. 메인 화면: 입력 정보 (계절마다 다르게 보여주기)
st.header(f'🔮 {selected_season}철 최대전력 예측하기')
st.markdown('내일의 기상 예보 정보를 입력해 주세요.')

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('입력 정보')
    
    # [공통 입력 변수]
    is_weekend = st.radio('주말 여부', ['평일', '주말 및 공휴일'], horizontal=True)
    weekend_val = 1 if is_weekend == '주말 및 공휴일' else 0
    
    c1, c2, c3 = st.columns(3)
    with c1:
        humidity = st.number_input('평균 상대습도 (%)', min_value=0.0, max_value=100.0, value=60.0)
    with c2:
        wind_speed = st.number_input('평균 풍속 (m/s)', min_value=0.0, max_value=30.0, value=2.0)
    with c3:
        rain = st.number_input('일강수량 (mm)', min_value=0.0, max_value=500.0, value=0.0)

    # [계절별 맞춤형 입력 변수]
    if selected_season == '여름':
        thi = st.slider('불쾌지수', min_value=50.0, max_value=90.0, value=75.0, help='여름철 전력 수요의 핵심 지표입니다.')
        # DataFrame을 모델 학습 때와 완벽히 동일한 컬럼명으로 생성
        input_df = pd.DataFrame([[humidity, wind_speed, rain, thi, weekend_val]], 
                                columns=['평균 상대습도(%)', '평균 풍속(m/s)', '일강수량(mm)', '불쾌지수', '주말여부'])

    elif selected_season == '겨울':
        wc, snow = st.columns(2)
        with wc:
            wind_chill = st.number_input('체감온도 (°C)', min_value=-30.0, max_value=20.0, value=-5.0)
        with snow:
            snowfall = st.number_input('일 최심신적설 (cm)', min_value=0.0, max_value=100.0, value=0.0)
            
        input_df = pd.DataFrame([[humidity, wind_speed, rain, snowfall, wind_chill, weekend_val]], 
                                columns=['평균 상대습도(%)', '평균 풍속(m/s)', '일강수량(mm)', '일 최심신적설(cm)', '체감온도', '주말여부'])
        
    else: # 봄, 가을
        temp = st.number_input('평균기온 (°C)', min_value=-10.0, max_value=30.0, value=15.0)
        input_df = pd.DataFrame([[temp, humidity, wind_speed, rain, weekend_val]], 
                                columns=['평균기온(°C)', '평균 상대습도(%)', '평균 풍속(m/s)', '일강수량(mm)', '주말여부'])

with col2:
    st.subheader('예측 결과')
    # 예측 버튼
    if st.button('⚡ 예측 실행', type='primary', use_container_width=True):
        # 1. 딕셔너리에서 선택된 계절의 모델만 쏙 뽑아오기
        best_model = models[selected_season]
        
        # 2. 예측 수행
        predicted_power = best_model.predict(input_df)[0]
        
        # 3. 결과 출력
        st.metric(label="내일 예상 최대전력", value=f"{predicted_power:,.0f} MW")
        
        # 결과 해석 팁
        with st.expander('📖 분석 결과 해석'):
            st.write(f"현재 선택된 **{selected_season}철 특화 모델({info['model_type']})**을 통해 도출된 결과입니다.")
            if is_weekend == '평일':
                st.write("- **산업 가동 요인**: 평일이므로 공장 등 산업용 전력 수요가 높게 반영되었습니다.")
            if selected_season == '여름' and thi > 80:
                st.write("- **기상 요인**: 불쾌지수가 매우 높아 에어컨 냉방 부하가 폭증할 것으로 예측되었습니다.")
            elif selected_season == '겨울' and wind_chill < -10:
                st.write("- **기상 요인**: 체감온도가 영하 10도 이하로 떨어져 난방용 전력 소비가 급증할 것으로 예측되었습니다.")