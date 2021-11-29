'''
여러 개의 특성을 사용한 선형 회구를 다중 회귀라고 부릅니다.
특성이 2개면 타깃값과 함께 3차원 공간을 형성하고, 선형 회귀 방정식 '타깃 = a * 특성1 + b* 특성2 + 절편은 평면이 됩니다.
- 이 예제에서는 농어의 길이뿐만 아니라 농어의 높이와 두께도 함께 사용합니다.
'''

import pandas as pd
import numpy as np

df = pd.read_csv('https://bit.ly/perch_csv_data')
# 농어의 길이, 높이, 두께
perch_full = df.to_numpy()
# print(perch_full)

# perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
#        21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
#        23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
#        27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
#        39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
#        44.0])

# 농어의 무게
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# perch_full과 perch_weight를 훈련 세트와 테스트 세트로 나눕니다.
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

'''
사이킷런의 변환기
사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공합니다.
사이킷 런에서는 이런 클래스를 변환기(transformer)라고 부릅니다.
변환기 클래스는 모두 fit(), transform() 메서드를 제공합니다.
'''

# 사용할 변환기는 PolynomialFeatures 클래스입니다.
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures()
# 2개의 특성 `2와 3`으로 이루어진 샘플 하나를 적용합니다.
# 훈련 : fit() 메서드는 새롭게 만들 특성 조합을 찾고
poly.fit([[2, 3]])
# 변환 : transform() 메서드는 실제로 데이터를 변환합니다.
print(poly.transform([[2, 3]]))     # [1. 2. 3. 4. 6. 9.]
# PolynomialFeatures 클래스는 기본적으로 각 특성을 제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가합니다.
# 2와 3을 각기 제곱한 4와 9가 추가되었고, 2와 3을 곱한 6이 추가되었습니다.
# 무게 = a * 길이 + b * 높이 + c * 두께 + d * 1
poly = PolynomialFeatures(include_bias=False)   # 절편을 위한 항을 제거
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))     # [2. 3. 4. 6. 9.]

# train_input에 적용 (훈련 세트 변환)
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)     # (42, 9)
print(poly.get_feature_names())     # 9개의 특성이 각가 어떤 입력의 조건합으로 만들어졌는지 알려줌.
# 테스트 세트 변환
test_poly = poly.transform(test_input)

'''
다중 회귀 모델 훈련하기
: 다중 회귀 모델을 훈련하는 것은 선형 회귀 모델을 훈련하는 것과 같습니다.
  다만 여러 개의 특성을 사용하여 선형 회귀를 수행하는 것뿐이죠.
'''
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)
# 특성이 늘어나면 선형 회귀의 능력은 매우 강하다는 것을 높은 점수 결과로 알 수 있습니다.
print(lr.score(train_poly, train_target))
# 테스트 세트에 대한 점수는 높아지지 않았지만 농어의 길이만 사용했을 때 있던 과소적합 문제는 더이상 나타나지 않았습니다.
print(lr.score(test_poly, test_target))

# 특성을 더많이 추가하면 어떨까요? 5제곱까지 특성을 만들어 출력해 봅니다.
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)     # (42, 55) : 만들어진 특성이 무려 55개

lr.fit(train_poly, train_target)
# 특성 개수를 크게 늘리면 선형 모델은 아주 강력해집니다. 훈련 세트에 대해 거의 완벽하게 학습할 수 있습니다.
print(lr.score(train_poly, train_target))   # 0.999999...
# 하지만 이런 모델은 훈련 세트에 너무 과대적합되므로 테스트 세트에서는 형편없는 점수를 만듭니다.
print(lr.score(test_poly, test_target))     # -144.40226...

'''
규제
'''
