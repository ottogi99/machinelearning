'''
키가 큰 사람의 아이가 부모보다 더 크지 않는다는 사실을 관찰하고 이를 '평균으로 회귀한다'라고 표현하였습니다.
그 후, 두 변수 사이의 상관관계를 분석하는 방법을 회귀라 불렀습니다.

k-최근접 이웃 회귀 : 샘플 X의 가장 가까운 샘플 K를 선택합니다. 이웃 샘플의 수치들의 평균을 구해 예측하는 방법입니다.
'''

import numpy as np
import matplotlib.pyplot as plt


# perch : 농어
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

plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# 1. 훈련 세트와 테스트 세트로 구분
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

# 사이킷런에서 훈련 세트는 2차월 배열이어야 합니다. perch_length가 1차워 배열이기 때문에 2차원 배열로 바꾸어줍니다.
# 크기에 -1을 지정하면 나머지 원소 개수로 모두 채우라는 의미(자동으로 배열의 크기를 지정)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)

'''
결정계수(R^2)
'''
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련합니다.
knr.fit(train_input, train_target)
# 훈련 평가
print('test: ', knr.score(test_input, test_target))

'''
회귀의 경우 결정계수를 통해 평가
R^2 = 1 - (타깃 - 예측)^2 의 합 / (타깃 - 평균)^2 의 합
> 만약 타깃의 평균 정도를 예측하는 수준이라면 (즉 분자와 분모가 비슷해져) R^2는 0에 가까워지고, 
  예측이 타깃에 아주 가까워지면 (분자가 0에 가까워지기 때문에) 1에 가까운 값이 됩니다.
'''
from sklearn.metrics import mean_absolute_error

# 테스트 세트에 대한 예측을 만듭니다.
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절댓값 오차를 계산합니다.
mae = mean_absolute_error(test_target, test_prediction)
# 결과에서 예측이 평균적으로 19g 정도 타깃값과 다르다는 것을 알 수 있습니다.
# print(mae)

'''
- 과대적합 vs 과소적합 -
만약 훈련 세트에서 점수가 굉장히 좋았는데 테스트 세트에서는 점수가 굉장히 나쁘다면 모델이 훈련 세트에 <과대적합> 되었다고 말합니다.
즉 훈련 세트에만 잘 맞는 모델이라 테스트 세트와 나중에 실전에 투입하여 새로운 샘플에 대한 예측을 만들 때 잘 동작하지 않을 것입니다.
반대로 훈련 세트보다 테스트 세트의 점수가 높거나 두 점수가 모두 너무 낮은 경우를 모델이 훈련 세트에 <과소적합> 되었다고 말합니다.
'''
# 훈련 세트보다 테스트 세트의 점수가 더 높으니 과소적합입니다.
print('train: ', knr.score(train_input, train_target))

# 모델을 조금 더 복잡하게 만들면 테스트 점수는 조금 낮아집니다.
# k-최근접 이웃 알고리즘으로 모델을 더 복잡하게 만드는 방법은 이웃의 개수 k를 줄이는 것입니다. k은 기본 값은 5입니다.
knr.n_neighbors = 3
# 모델 다시 훈련
knr.fit(train_input, train_target)
# 평가 (과소적합 문제 해결)
print('train adjustment: ', knr.score(train_input, train_target))
print('test adjustment: ', knr.score(test_input, test_target))


