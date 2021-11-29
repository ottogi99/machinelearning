'''
도미(bream)와 빙어(smelt)를 구분

* k-최근접 이웃 알고리즘은 데이터가 아주 많은 경우 사용하기 어렵습니다. 데이터가 크기 때문에 메모리가 많이 필요하고, 직선거리를 계산하는 데도 많은 시간이 필요합니다.
'''
import matplotlib.pyplot as plt
import sklearn.neighbors

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# zip() 함수는 나열된 리스트 각각에서 하나씩 원소를 꺼내 반환합니다.
fish_data = [[l, w] for l, w in zip(length, weight)]
# print(fish_data)

fish_target = [1] * 35 + [0] * 14
# print(fish_target)

'''
사이킷런 패키지에서 k-최근접 이웃 알고리즘을 구현 KNeighborsClassifier를 임포트
'''
from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier()
kn = KNeighborsClassifier()
# 훈련(학습)
kn.fit(fish_data, fish_target)
# 훈련 평가
# print(kn.score(fish_data, fish_target))
# 예측
print(kn.predict([[30, 600]]))

kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
# 아래 구문은 print(35/49) 와 동일하다. 49개중에 35개 도미이고, 14개가 빙어이므로
# 그러므로 매개변수를 49로 두는 것은 좋지 않다. 기본값을 5로 하여 도미를 완벽하게 분류한 모델을 사용합니다.
print(kn49.score(fish_data, fish_target))

