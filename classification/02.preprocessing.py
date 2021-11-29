import numpy as np

fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# column_stack() 함수를 사용하여 연결
fish_data = np.column_stack((fish_length, fish_weight))
# print(fish_data[:5])

# np.ones() 인자만큼 1을 생성, np.zeros() 인자만큼 0을 생성
# concatenate() 함수를 사용해 타깃 데이터 생성
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# print(fish_target)

'''
사이킷런으로 훈련 세트와 테스트 세트 나누기 (np.random.seed() 함수를 사용할 필요가 없다)
'''
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
# print(train_input.shape, test_input.shape)
# print(train_target.shape, test_target.shape)

# 원래 도미와 방어의 개수가 35개와 14개이므로 두 생선의 비율은 2.5:1 입니다.
# 하지만 이 테스트 세트의 도미와 방어의 비율은 3.3:1 입니다. 샘플링 편향이 여기에서 조금 나타났습니다.
# print(test_target)

# 무작위의 데이터를 나누었을 때 샘플이 골고루 섞이지 않을 수 있습니다.
# train_test_split() 함수는 이런 문제를 간단히 해결할 방법이 있습니다. stratify 매개변수에 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눕니다.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)
# 이제 테스트 세트의 비율이 2.25:1 이 되었습니다.
# print(test_target)

from sklearn.neighbors import KNeighborsClassifier 

kn = KNeighborsClassifier()
# 훈련(학습)
kn.fit(train_input, train_target)
# 평가
# print(kn.score(test_input, test_target))
# 도미 데이터 넣고 결과 확인. 잉? 0. 이 나오네요.
# print(kn.predict([[25, 150]]))

'''
이 샘플을 다른 데이터와 함께 산점도로 그려봅니다.
'''
import matplotlib.pyplot as plt

# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(25, 150, marker='^')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 이웃까지의 거리와 이웃 샘플의 인덱스를 반환, 기본값이 5이므로 5개의 이웃이 반환됩니다.
distance, indexes = kn.kneighbors([[25, 150]])

# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(25, 150, marker='^')
# plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# x축은 범위가 좁고(10~40), y축은 범위가 넓습니다(0~1000).
# 이를 명확히 확인하기 위해 x축의 범위를 동일하게 0~1000으로 맞추어봅니다.
# plt.scatter(train_input[:, 0], train_input[:, 1])
# plt.scatter(25, 150, marker='^')
# plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
# # plt.xlim((0, 1000))  # x축의 범위 지정
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 표준점수는 각 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타냅니다.
# 분산은 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구합니다.
# 표준편차는 분산의 제곱근으로 데이터가 분산된 정도를 나타냅니다.
# 평균을 빼고 표준편차를 나누어줍니다. (axis=0, 행을 따라 각 열의 통계 값을 계산)
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

# 각 특성마다 평균과 표준편차가 구해졌습니다. 이제 원본 데이터에서 평균을 빼고 표준편차로 나누어 표준점수로 변환하겠습니다.
train_scaled = (train_input - mean) / std

'''
전처리 데이터로 모델 훈련하기
샘플[25, 150]을 동일한 비율로 변환해야 함. -> 훈련 세트의 mean, std를 이용해서 변환 
'''
new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
# x축과 y축의 범위가 -1.5~1.5 사이로 바뀜.
plt.show()

# 훈련
kn.fit(train_scaled, train_target)
# 테스트 세트도 훈련 세트의 평균과 표준편차로 변환해야 함.
test_scaled = (test_input - mean) / std
# 드디어 도미(1)로 예측
print(kn.score(test_scaled, test_target))



