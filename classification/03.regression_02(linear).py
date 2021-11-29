'''
k-최근접 이웃의 한계 : k-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타깃을 평균합니다.
따라서 새로운 샘플이 훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측할 수 있습니다.
'''

import numpy as np
import matplotlib.pyplot as plt


perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
# k-최근접 이웃 회귀 모델을 훈련합니다.
knr.fit(train_input, train_target)
# 길이가 50cm인 농어의 무계를 예측합니다.
print(knr.predict([[50]]))  # 1033.333333

# 훈련 세트와 50cm 농어 그리고 이 농어의 최근접 이웃을 산점도에 표시합니다.
import matplotlib.pyplot as plt

# 50cm 농어의 이웃을 구합니다.
distances, indexes = knr.kneighbors([[50]])
# # 훈련 세트의 산점도를 그립니다.
# plt.scatter(train_input, train_target)
# # 훈련 세트 중에서 이웃 샘플만 다시 그립니다.
# plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# # 50cm 농어 데이터
# plt.scatter(50, 1033, marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

'''
선형회귀 : 널리 사용되는 대표적인 회귀 알고리즘입니다.
비교적 간단하고 성능이 뛰어나기 때문에 맨 처음 배우는 머신러닝 알고리즘 중 하나입니다.
y = a * x + b
'''
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
# 훈련 (선형 회귀 모델)
lr.fit(train_input, train_target)
# 50cm 농어 대해 예측
print(lr.predict([[50]]))   # 1241.83860323

# 하나의 직선을 그리려면 기울기와 절편이 있어야 합니다.
# y = a * x + b,  농어무게 = a(기울기) * 농어길이 + b(절편)
# LinearRegression 클래스가 찾은 a와 b는 lr 객체의 coef_와 intercept_ 속성에 저장되어 있습니다.
print(lr.coef_, lr.intercept_)     # a: [39.01714496], b: -709.0186449535477

# 훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프를 그립니다.
# 이 직선을 그리려면 앞에서 구한 기울기와 절편을 사용하여 (15, 15*39-709)와 (50, 50*39-709) 두 점을 이으면 됩니다.
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# 이제 훈련 세트와 테스트 세트에 대한 R^2 점수를 확인합니다.
print(lr.score(train_input, train_target))       # 0.939846333997604
print(lr.score(test_input, test_target))         # 0.8247503123313558

print('\n - 다항회귀')
'''
다항회귀
- 선형 회귀가 만든 직선이 왼쪽 아래로 쭉 뻗어 있습니다. 이 직선대로 예측하면 농어의 무게가 0g 이하로 내려갈 텐데 현실에서는 있을수 없는 일입니다.
  농어의 길이와 무게에 대한 산점도를 자세히 보면 일직선이라기보다 왼쪽 위로 조금 구부러진 곡선에 가깝습니다. 
  그렇다면 최적의 직선을 찾기보다 최적의 곡선을 찾으면 어떨까요?
- 2차 방정식 무게 = a * 길이^2 + b * 길이 + c
'''
# column_stack : 리스트를 연결
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# print(train_poly.shape, test_poly.shape)

# train_poly를 사용해 선형 회귀 모델을 다시 훈련합니다. 이 모델이 2차 방정식의 a,b,c를 잘 찾을 것으로 기대합니다.
# 주목할 점은 2차 방정식 그래프를 찾기 위해 훈련 세트에 제곱 항을 추가했지만, 타깃값은 그대로 사용한다는 점입니다.
lr = LinearRegression()
lr.fit(train_poly, train_target)
# 앞에 훈련한 모델보다 더 높을 값을 예측했습니다.
print(lr.predict([[50**2, 50]]))
# 이 모델이 훈련한 계수와 절편을 출력해 봅시다.
print(lr.coef_, lr.intercept_)
# 이 모델은 다음과 같은 그래프를 학습했습니다. 무게 = 1.01 * 길이^2 - 21.6 * 길이 + 116.05
# 이런 방정식을 다항식(polynomial)이라 부르며 다항식을 사용한 선형 회귀를 다항 회귀라고 부릅니다.

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
point = np.arange(15, 50)
# 훈련 세트의 산점도를 그립니다.
plt.scatter(train_input, train_target)
# 15에서 49까지 2차 방정식 그래프를 그립니다.
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# 단순 선형 회귀 모델보다 훨씬 나으 그래프가 그려졌습니다. 훈련 세트의 경향을 잘 따르고 있고 무게가 음수로 나오는 일도 없습니다.
# 그럼, 훈련 세트와 테스트 세트의 R^2 점수를 평가하겠습니다.
print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))
# 훈련 세트와 테스트 세트에 대한 점수가 크게 높여졌지만 여전 테스트 테스트의 점수가 조금 더 높습니다.(과소적합)
