1. ETRI/data 에 eICU dataset 의 vitalPeriodic.csv 파일을 저장하신 다음,

2. R로 pre-processing.R 로 ETRI/data 에 전처리된 csv 파일을 생성 후,

3. pre-processing.py 로 ETRI/experiments/data에 eicu.data.npy 생성 후,

4-0. 데이터 생성 시, 파라미터를 바꾸려면 ETRI/experiments/settings 에 저장되어있는 세팅 텍스트들을 변경하시면 됩니다.

4-1. sine 파형을 생성하고 싶다면 콘솔창에서 python --settings_file sine 을 입력하시면 됩니다. ETRI/experiments/plots 에 epoch에 따른 파형 그래프가 생성되고, ETRI/experiments/parameters 에 파라미터가 저장됩니다.

4-2. smoothed signal 파형을 생성하고 싶다면 콘솔창에서 python --settings_file gp_rbf 을 입력하시면 됩니다. ETRI/experiments/plots 에 epoch에 따른 파형 그래프가 생성되고, ETRI/experiments/parameters 에 파라미터가 저장됩니다.

4-3. eICU data을 생성하고 싶다면 콘솔창에서 python --settings_file eicu 을 입력하시면 됩니다. ETRI/experiments/plots 에 epoch에 따른 파형 그래프가 생성되고, ETRI/experiments/parameters 에 파라미터가 저장됩니다. ETRI/synthetic_eICU_datasets 에 데이터가 저장됩니다. 

5. eICU data에 대한 TSTR 검정을 실행하고 싶다면 eICU_tstr_evaluation.py 를 실행하면 TSTR 검정 결과가 콘솔에 프린트됩니다.

6. sine 파형이나 smoothed signal 파형에 대한 interpolation 그래프를 그리고 싶다면 interpolation.py 파일을 실행하면 됩니다. 변수 identifier 를 'sine' 으로 하면 생성된 sine parameter를 이용해서, 'gp_rbf' 으로 하면 생성된 smoothed signal parameter를 이용해서, ETRI/experiments/plots 에 epoch에 따른 interpolation 파형 그래프가 생성됩니다.
