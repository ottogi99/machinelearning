'''
로지스틱 회귀 알고리즘을 배우고 이진 분류 문제에서 클래스 확률을 예측합니다.
사이킷런의 k-최근접 이웃 분류기
'''
import pandas as pd


fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())
# 어떤 종류의 생선이 있는지 Species 열에서 고유한 값을 출력
print(pd.unique(fish['Species']))

# 이 데이터프레임에서 Species 열을 타깃으로 만들고 나머지 5개 열은 입력 데이터로 사용합니다.
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()

# 훈련 세트와 테스트 세트로 나눕니다.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
# 훈련 세트와 테스트 세트를 표준화 전처리
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

'''
k-최근접 이웃 분류기의 확률 예측
'''
print('\n- KNeightbors Classfier - ')
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))

print(kn.classes_)
print(kn.predict(test_scaled[:5]))

import numpy as np
proba = kn.predict_proba(test_scaled[:5])
# predict_proba() 메서드의 출력 순서는 앞서 보았던 classes_ 속성과 같습니다. 즉 첫 번째 열이 'Bream'에 대한 확률, 두 번째 열이 'Parkki'에 대한 확률입니다.
print(np.round(proba, decimals=4))
# 이 샘플의 이웃은 'Roach'가 1개이고 세 번째 클래스인 'Perch'가 2개입니다.
# 따라서 다섯 번째 클래스에 대한 확률은 1/3 = 0.3333이고 세 번째 클래스에 대한 확률은 2/3 = 0.6667이 됩니다.
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

'''
3개의 최근접 이웃을 사용하기 때문에 가능한 확률은 0/3, 1/3, 2/3, 3/3이 전부입니다.
뭔가 더 좋은 방법을 찾아야 할 것 같습니다.

- 로지스틱 회귀
: 이름은 회귀이지만 분류 모델입니다. 이 알고리즘은 선형 회귀와 동일하게 선형 방정식을 학습니다.
z = a * (Weight) + b * (Length) + c * (Diagnal) + d * (Height) + e * (Width) + f
'''
print('\n- 로지스틱 회귀 -')