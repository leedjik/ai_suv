import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
train_data = np.load("..\\datasets\\linear_train.npy")

# tf 형식에 맞게 변환
x_data = np.expand_dims(train_data[:,0], axis=1) #train_data의 x값만 따로 저장
y_data = train_data[:,1] #train_data의 y값만 따로 저장

# 결과 시각화
plt.title('Req 1-1')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.scatter(x_data,y_data,s=5)
plt.show()

#load 종료
train_data.close()