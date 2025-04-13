#streamlit run auto_ml_py310.py
#conda create --name py310 python=3.10
#conda activate py310
#conda install scikit-learn pandas numpy matplotlib tqdm joblib xgboost lightgbm streamlit numba
#conda install pycaret[full]
#conda install -n py310 ipykernel
#pip install ydata_profiling
#pip install streamlit-pandas-profiling
#pip install pandas==2.0.3 -> ydata_profiling 4.16.1 requires pandas 2.0.3


# data source 
# https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data

import streamlit as st
import pandas as pd
#import pandas_profiling

import pycaret.classification as cls 
import pycaret.regression as reg
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import os
import tempfile
import time

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image('mug_obj_202003060908226127.png')
    st.title("AutoML_with_pycaret")
    choice = st.radio("Navigation", 
                      ["Upload","Profiling","Data_Preprocessing", "Modelling", 
                      "Model_Selection","Evaluation","Download","Test Model"])
    st.info("This project application helps you build and explore your Machine Learning Model.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = ProfileReport(df, title="Data Report", explorative=True)
    st_profile_report(profile_df)
    
    #profile_df = df.profile_report()
    #st_profile_report(profile_df)

if choice == "Data_Preprocessing": 
    st.title("Data_Preprocessing")

    st.subheader("Drop & Duplicate")
    
    drop_options = df.columns
    selection = st.pills("삭제할 열을 선택하세요.(중복선택가능)", 
                         drop_options, selection_mode="multi")
    st.markdown(f"삭제하는 열: {selection}.")
    df = df.drop(selection, axis=1)

    duplicate_options = st.radio("중복되는 행 삭제 유무 결정", ("delete", "keep"))
    if duplicate_options == "delete": 
        df = df.drop_duplicates()
    else: 
        pass

    st.subheader("Select the Target Column")

    chosen_target = st.selectbox('Choose the Target Column : ', df.columns)

    y = df[chosen_target]
    X = df.drop(chosen_target, axis=1)

    test_size = st.slider("Select test size(%)?", 0.00, 1.00, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42,stratify=y)

    st.subheader("Train and Test Data")
    st.info("Train Data Length: " + str(len(X_train)))
    
    st.write("X_train")
    X_train = X_train.reset_index(drop=True)
    st.dataframe(X_train.head())

    st.write("y_train")
    y_train = y_train.reset_index(drop=True)
    st.dataframe(y_train.head())

    st.info("Test Data Length: " + str(len(X_test)))

    st.write("X_test")
    X_test = X_test.reset_index(drop=True)
    st.dataframe(X_test.head())

    st.write("y_test")
    y_test = y_test.reset_index(drop=True)
    st.dataframe(y_test.head())
    
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test 
    st.session_state.y_test = y_test
    
if choice == "Modelling": 
    mdl = st.selectbox('Chose Modelling Type : ',['Classification','Regression'])
    
    if st.button('Run Modelling'):
        st.session_state.modelling_start_time = time.time()

        with st.spinner('Modelling in progress...'):
        
            if mdl == 'Classification':
                st.session_state.Model = cls
                #from pycaret.classification import setup, compare_models, pull, save_model 

            elif mdl == 'Regression':
                st.session_state.Model = reg
                #from pycaret.regression import setup, compare_models, pull, save_model

            st.info("Your model of choice is " + mdl) 

            st.session_state.Model.setup(st.session_state.X_train, 
                                         target=st.session_state.y_train)
            st.session_state.Setup_df = st.session_state.Model.pull()
            st.dataframe(st.session_state.Setup_df)
            st.session_state.Model.compare_models()
            st.session_state.Compare_df = st.session_state.Model.pull()
        
        st.session_state.modelling_end_time = time.time()
        st.info("Modelling Time: " + str(st.session_state.modelling_end_time - st.session_state.modelling_start_time))
        st.success('Modelling Done!')

if choice == "Model_Selection":
        
        st.dataframe(st.session_state.Compare_df)
        
        st.session_state.select_model = st.selectbox("Select the best model", st.session_state.Compare_df.index)

        # best 모델 선정
        st.session_state.best_model = st.session_state.Model.create_model(st.session_state.select_model)

        # 모델의 ROC Curves 시각화
        st.subheader("AUC Curve")
        img = st.session_state.Model.plot_model(
            st.session_state.best_model, plot="auc", display_format="streamlit", save=True
        )
        st.image(img)

        st.subheader("Confusion Matrix")
        img2 = st.session_state.Model.plot_model(
            st.session_state.best_model, plot="confusion_matrix", display_format="streamlit", save=True
        )
        st.image(img2)

        st.subheader("Feature Importance")
        img3 = st.session_state.Model.plot_model(
            st.session_state.best_model, plot="feature", display_format="streamlit", save=True
        )
        st.image(img3)
        
        st.session_state.Model.save_model(st.session_state.best_model, 'best_model')

if choice == "Evaluation":

    st.info("Your model of choice is " + st.session_state.select_model)

    if st.button('Evaluation Start'):

        st.session_state.Evaluation_start_time = time.time()

        with st.spinner('Evaluation in progress...'):
        
            y_pred_df = st.session_state.Model.predict_model(st.session_state.best_model, 
                                                          data=st.session_state.X_test)
            y_pred_df['y_true'] = st.session_state.y_test
    
            st.subheader("Predictions and True Values")
            st.info("Test Data Length: " + str(len(y_pred_df)))
            st.dataframe(y_pred_df)
    
            st.session_state.y_pred = y_pred_df["prediction_label"]
    
            # classification_report를 딕셔너리 형태로 변환 후 데이터프레임 생성
            report_dict = classification_report(st.session_state.y_test, st.session_state.y_pred,
                                                output_dict=True)
            df = pd.DataFrame(report_dict).transpose()
    
            # Streamlit에서 출력
            st.subheader("Classification Report")
            st.session_state.Evaluation_end_time = time.time()
            st.info("Evaluation Time: " + str(st.session_state.Evaluation_end_time - st.session_state.Evaluation_start_time) + " seconds")
            st.dataframe(df)  # 인터랙티브 테이블 형태로 출력
        
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")

if choice == "Test Model":

    st.title("Test Model ML Model")

    mdl = st.selectbox('Chose Modelling Type : ',['Classification','Regression'])

    if mdl == 'Classification':
        st.session_state.Test_Model = cls
        #from pycaret.classification import setup, compare_models, pull, save_model
    elif mdl == 'Regression':
        st.session_state.Test_Model = reg
        #from pycaret.regression import setup, compare_models, pull, save_model

    if st.session_state.Test_Model is not None:
        uploaded_model = st.file_uploader("PyCaret 모델 파일(.pkl)을 업로드하세요", type=["pkl"])

    if uploaded_model is not None:
        # 파일 이름 및 타입 정보 표시
        st.write(f"파일 이름: {uploaded_model.name}")
        st.write(f"파일 타입: {uploaded_model.type}")

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            model_path_temp = tmp_file.name
            model_path = model_path_temp.replace('.pkl', '')

        try:
            # 모델 로드
            model = st.session_state.Test_Model.load_model(model_path)
            st.success("모델이 성공적으로 로드되었습니다!")
            # 모델 정보 표시 (선택 사항)
            st.write("로드된 모델 정보:", model)

        except Exception as e:
            st.error(f"모델 로드 중 오류가 발생했습니다: {e}")

    if 'model' in locals():  # 모델이 로드된 경우에만 실행
        st.subheader("예측을 위한 데이터 업로드")
        data_file_y_test = st.file_uploader("예측할 데이터 파일(csv)을 업로드하세요", type=["csv"])
        data_y_test = pd.read_csv(data_file_y_test, index_col=False)

        st.subheader("y_true 데이터 업로드")
        data_file_y_true = st.file_uploader("y_true 데이터 파일(csv)을 업로드하세요", type=["csv"])
        data_y_true = pd.read_csv(data_file_y_true, index_col=False)

        if (data_y_test is not None) and (data_y_true is not None):

            if len(data_y_test) != len(data_y_true):
                st.error("예측 데이터와 y_true 데이터의 길이가 같지 않아 진행이 불가합니다.")
                st.stop()

            st.write("Test 데이터 미리보기:")
            st.dataframe(data_y_test.head())
            #choice_test = st.radio("y test 열 선택하세요.", data_y_test.columns)
            #st.session_state.y_test = data_y_true[choice_test]

            st.write("Y_true 데이터 미리보기:")
            st.dataframe(data_y_true.head())
            choice_true = st.radio("y true 열 선택하세요.", data_y_true.columns)
            st.session_state.y_true = data_y_true[choice_true]

        if (data_y_test is not None) and (st.session_state.y_true is not None):

            if st.button("예측 실행"):

                with st.spinner('Evaluation in progress...'):

                    start_time = time.time()

                    # PyCaret의 predict_model 함수를 사용하여 예측 수행
                    predictions = st.session_state.Test_Model.predict_model(model, data=data_y_test)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    st.info("Evaluation Time: " + str(total_time) + " seconds")
                    
                    st.write("예측 결과:")

                    st.session_state.Prediction_df = predictions
                    st.dataframe(st.session_state.Prediction_df)

                    st.session_state.y_test = predictions["prediction_label"]

                    st.subheader("Predictions and True Values")                  
                    # classification_report를 딕셔너리 형태로 변환 후 데이터프레임 생성
                    report_dict = classification_report(st.session_state.y_true, 
                                                        st.session_state.y_test,
                                                        output_dict=True)
                    result_raw = pd.concat([st.session_state.y_true,st.session_state.y_test],
                                            axis=1)
                    st.dataframe(result_raw)
                    df = pd.DataFrame(report_dict).transpose()                
                    # Streamlit에서 출력
                    st.subheader("Classification Report")
                    st.dataframe(df)  # 인터랙티브 테이블 형태로 출력

                if df is not None:
                    #csv = predictions.to_csv(index=False).encode('utf-8')
                    #st.download_button(
                    #"예측 결과 다운로드",
                    #csv,
                    #"predictions.csv",
                    #"text/csv",
                    #key='download-csv')    

                    # 예측 결과 다운로드 버튼 추가
                    csv = df.to_csv().encode('utf-8')
                    st.download_button(
                        "결과 다운로드",
                        csv,
                        "test_result.csv",
                        "text/csv",
                        key='download-csv'
                    )

    #if model:
    #    # 저장된 모델 불러오기
    #    loaded_model = st.session_state.Model.load_model(model)
#
    #data_set = st.file_uploader("Upload Your Test Dataset")
    #if data_set: 
    #    df_test = pd.read_csv(data_set, index_col=None)
    #    df_test.to_csv('dataset_test.csv', index=None)
    #    st.dataframe(df_test)
    #
    #st.subheader("Select the Target Column")
#
    #st.session_state.predictions = st.session_state.Model.predict_model(loaded_model, 
    #                                                                    data=df_test)
    ## Streamlit에서 출력
    #st.subheader("Predictions")
    #st.dataframe(st.session_state.predictions)




