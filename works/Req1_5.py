import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Req1_2 import LinearModel


# 데이터 불러오기
train_data = np.load("..\\datasets\\linear_train.npy")
test_x = np.load("..\\datasets\\linear_test_x.npy")


# train_data를 tf 형식에 맞게 변환
x_data = np.expand_dims(train_data[:,0], axis=1) #train_data의 x값만 따로 저장
y_data = train_data[:,1] #train_data의 y값만 따로 저장


# 모델 생성
model = LinearModel(num_units=1)

# 최적화 함수, 손실함수와 모델 바인딩
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
			  loss=tf.keras.losses.MSE,
			  metrics=[tf.keras.metrics.MeanSquaredError()])

# 모델 학습
model.fit(x=x_data, 
		  y=y_data, 
		  epochs=10, 
		  batch_size=32)


# 모델 테스트
prediction = model.predict(x=test_x,
    					   batch_size=None)


# 결과 시각화
plt.title('Req 1-5')
plt.xlabel('X value')
plt.ylabel('Y value')
plt.scatter(x_data,y_data,s=5,label="train data")
plt.scatter(test_x,prediction,s=5,label="prediction data")
plt.legend()
plt.show()


# 모델 정리
model.summary()
#load 종료
train_data.close()