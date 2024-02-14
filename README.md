# LG-Aimers
[LG Aimers 해커톤 MQL 분류 챌린지]

## 개요
영업 과정에서 전환 가능성이 높은 고객에게 영업 자원을 집중하기 위해 고객의 전환 여부를 예측하기 위해 기계학습을 도입하고 있다. 이번 경진대회에서는 고객의 다양한 정보를 보고 해당 고객이 전환 고객인지 아닌지를 판단하는 모델을 구현하고 그 성능을 비교하고자 한다.


## 데이터셋

- bant_submit : MQL 구성 요소들 중 [1]Budget(예산), [2]Title(고객의 직책/직급), [3]Needs(요구사항), [4]Timeline(희망 납기일) 4가지 항목에 대해서 작성된 값의 비율
- customer_country : 고객의 국적
- business_unit : MQL 요청 상품에 대응되는 사업부
- com_reg_ver_win_rate : Vertical Level 1, business unit, region을 기준으로 oppty 비율을 계산
- customer_idx : 고객의 회사명
- customer_type : 고객 유형
- enterprise : Global 기업인지, Small/Medium 규모의 기업인지
- historical_existing_cnt : 이전에 Converted(영업 전환) 되었던 횟수
- id_strategic_ver : (도메인 지식) 특정 사업부(Business Unit), 특정 사업 영역(Vertical Level1)에 대해 가중치를 부여
- it_strategic_ver : (도메인 지식) 특정 사업부(Business Unit), 특정 사업 영역(Vertical Level1)에 대해 가중치를 부여
- idit_strategic_ver : id_strategic_ver이나 it_strategic_ver 값 중 하나라도 1의 값을 가지면 1 값으로 표현
- customer_job : 고객의 직업군
- lead_desc_length : 고객이 작성한 Lead Descriptoin 텍스트 총 길이
- inquiry_type : 고객의 문의 유형
- product_category : 요청 제품 카테고리
- product_subcategory : 요청 제품 하위 카테고리
- product_modelname : 요청 제품 모델명
- customer_country.1 : 담당 자사 법인명 기반의 지역 정보(대륙)
- customer_position : 고객의 회사 직책
- response_corporate : 담당 자사 법인명
- expected_timeline : 고객의 요청한 처리 일정
- ver_cus : 특정 Vertical Level 1(사업영역) 이면서 Customer_type(고객 유형)이 소비자(End-user)인 경우에 대한 가중치	
- ver_pro : 특정 Vertical Level 1(사업영역) 이면서 특정 Product Category(제품 유형)인 경우에 대한 가중치
- ver_win_rate_x : 전체 Lead 중에서 Vertical을 기준으로 Vertical 수 비율과 Vertical 별 Lead 수 대비 영업 전환 성공 비율 값을 곱한 값
- ver_win_ratio_per_bu : 특정 Vertical Level1의 Business Unit 별 샘플 수 대비 영업 전환된 샘플 수의 비율을 계산
- business_area : 고객의 사업 영역
- business_subarea : 고객의 세부 사업 영역
- lead_owner : 영업 담당자 이름
- is_converted : 영업 성공 여부. True일 시 성공

## 결과 지표
F1Score 사용. 0.52 이상이 되면 수료

## 기본 코드

### 필수 라이브러리

```
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```

### 데이터 셋 읽어오기

```
df_train = pd.read_csv("train.csv") # 학습용 데이터
df_test = pd.read_csv("submission.csv") # 테스트 데이터(제출파일의 데이터)
```

### 데이터 전처리

```
def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series
```

### 레이블 인코딩
```
 # 레이블 인코딩할 칼럼들
label_columns = [
    "customer_country",
    "business_subarea",
    "business_area",
    "business_unit",
    "customer_type",
    "enterprise",
    "customer_job",
    "inquiry_type",
    "product_category",
    "product_subcategory",
    "product_modelname",
    "customer_country.1",
    "customer_position",
    "response_corporate",
    "expected_timeline",
]

df_all = pd.concat([df_train[label_columns], df_test[label_columns]])

for col in label_columns:
    df_all[col] = label_encoding(df_all[col])
```

### 다시 학습 데이터와 제출 데이터를 분리
```
for col in label_columns:  
    df_train[col] = df_all.iloc[: len(df_train)][col]
    df_test[col] = df_all.iloc[len(df_train) :][col]
```

### 학습, 검증 데이터 분리
```
x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),
    df_train["is_converted"],
    test_size=0.2,
    shuffle=True,
    random_state=400,
)
```

### 모델 학습

```
model = DecisionTreeClassifier()
model.fit(x_train.fillna(0), y_train)
```

### 모델 성능 보기
```
def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))
```

# 오차행렬
```
pred = model.predict(x_val.fillna(0))
get_clf_eval(y_val, pred)
```

### 제출

```
 # 예측에 필요한 데이터 분리
x_test = df_test.drop(["is_converted", "id"], axis=1)

test_pred = model.predict(x_test.fillna(0))
sum(test_pred) # True로 예측된 개수
```

### 제출 파일 작성
```
 # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["is_converted"]

 # 제출 파일 저장
df_sub.to_csv("submission.csv", index=False)
```
