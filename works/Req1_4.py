import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models.linear_model import LinearModel


# 데이터 불러오기
train_data = np.load(".\\datasets\\linear_train.npy")
# test_x = np.load(".\\datasets\\linear_test_x.npy")


# tf 형식에 맞게 변환
x_data = np.expand_dims(train_data[:,0], axis=1) #train_data의 x값만 따로 저장
y_data = train_data[:,1] #train_data의 y값만 따로 저장


# 모델 생성
model = LinearModel(num_units=1)

# 최적화 함수, 손실함수와 모델 바인딩
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
			  loss=tf.keras.losses.MSE,
			  metrics=[tf.keras.metrics.MeanSquaredError()])

# SGD(Stochastic Gradient Descent) : 확률적 경사 하강법
#  >> 입력 데이터가 확률적으로 선택된다.
# loss : MSE(Mean Square, Error비용함수)
# MeanSquaredError : 실제결과값과 예상값의 차이(error)

# # 모델 학습
model.fit(x=x_data, 
		  y=y_data, 
		  epochs=10, #데이터 전체에 대한 학습 반복 횟수
		  batch_size=32) #배치(batch=집단)의 크기

# 모델 정리
model.summary()
#load 종료
train_data.close()