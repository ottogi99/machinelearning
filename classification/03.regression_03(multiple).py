'''
여러 개의 특성을 사용한 선형 회귀를 <다중 회귀>라고 부릅니다.
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
규제 (regularization)
: 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것을 말합니다.
  즉 모델이 훈련 세트에 과대적합되지 않도록 만드는 것입니다. 선형 회귀 모델의 경우 특성에 곱해지는 계수(또는 기울기)의 크기를 작게 만드는 일입니다.
'''
# StatndardScaler 클래스를 사용해 표준점수를 구합니다.
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# 훈련
ss.fit(train_poly)
# 표준점수로 변환한 train_scaled와 test_scaled
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

'''
- 릿지(ridge) 회귀 - 
선형회귀 모델에 규제를 추가한 모델을 릿지와 라쏘라고 부릅니다.
릿지는 계수를 제곱한 값을 기준으로 규제를 적용하고,
라쏘는 계수의 절댓값을 기준으로 규제를 적용합니다.
'''
print('\n - Ridge 회귀 -')
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)
print('ridge train: ', ridge.score(train_scaled, train_target))

# 적절한 alpha 값을 찾는 한 가지 방법은 alpha에 대한 R^2 값의 그래프를 그려 보는 것입니다.
import matplotlib.pyplot as plt
train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       ridge = Ridge(alpha=alpha)
       # 훈련
       ridge.fit(train_scaled, train_target)
       # 훈련 점수와 테스트 점수를 저장
       train_score.append(ridge.score(train_scaled, train_target))
       test_score.append(ridge.score(test_scaled, test_target))

# alpha 값을 0.001 부터 10배씩 늘렸기 때문에 그래프를 그리면 왼쪽이 너무 촘촘해 집니다.
# alpha_list에 있는 6개의 값을 동일한 간격으로 나타내기 위해 로그 함수로 바꾸어 지수로 표현하겠습니다.
# np.log()는 자연로그, np.log10() 상용로그
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 적절한 alpha 값은 두 그래프가 가장 가깝고 테스트 세트의 점수가 가장 높은 -1, 즉 10^-1=0.1입니다.
# alpha 값을 0.1로 하여 최종 모델을 훈련합니다.
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
# 이 모델은 훈련 세트와 테스트 세트의 점수가 비슷하게 모두 높고 과대적합과 과소적합 사이에서 균형을 맞추고 있습니다.
print('ridge train(alpha=0.1): ', ridge.score(train_scaled, train_target))
print('ridge test(alpha=0.1): ', ridge.score(test_scaled, test_target))

'''
라쏘 회귀
'''
print('\n- Lasso 회귀 -')
from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(train_scaled, train_target)
# 라쏘도 과대적합을 잘 억제한 결과를 보여줍니다.
print('lasso train: ', lasso.score(train_scaled, train_target))
# 테스트 세트의 점수도 릿지만큼 아주 좋습니다.
print('lasso test: ', lasso.score(test_scaled, test_target))
# 라소 모델의 alpha값 찾기
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
       lasso = Lasso(alpha=alpha, max_iter=10000)
       # 훈련
       lasso.fit(train_scaled, train_target)
       # 훈련 점수와 테스트 점수를 저장
       train_score.append(lasso.score(train_scaled, train_target))
       test_score.append(lasso.score(test_scaled, test_target))

# 라쏘 모델에서 최적의 alpha 값은 그래프에서 보듯이 1, 즉 10^1=10 입니다.
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 최적의 alpha 값으로 다시 모델을 훈련합니다.
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
# 특성을 많이 사용했지만 릿지와 마찬가지로 라쏘 모델이 과대적합을 잘 억제하고 테스트 세트의 성능을 크게 높였습니다.
print('lasso train(alpha=10): ', lasso.score(train_scaled, train_target))
print('lasso test(alpha=10): ', lasso.score(test_scaled, test_target))

