import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# train_images (60000, 28, 28) : (28x28) 크기의 60000개 이미지 데이터
# train_labels (60000, ) : 이미지가 나타내는 품목을 의미하는 60000개의 라벨(스칼라)
# test_images (10000, 28, 28) : (28x28) 크기의 10000개 이미지 데이터
# test_labels (10000,) : 이미지가 나타내는 품목을 의미하는 10000개의 라벨

np.set_printoptions(linewidth=150)
# print(train_images[0]) #사진1, train_labels[0] : 9(=Ankle boot)

class_names = ['T-shirt/top', 'Trouser/pants', 'Pullover shirt', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']#10개의 라벨에 대한 네이밍 배열 저장

# plt.figure()
# plt.imshow(train_images[0], cmap='gray')
# plt.grid(False)
# plt.show()# 사진2 plot 출력

# 이미지의 픽셀 값이 0~1이 되도록 조정한다.
train_images = train_images/255.0
test_images = test_images/255.0

# print(np.round(train_images[0], 2)) #사진3

#########################################################3
# 신경망 모델 생성
# 신경망의 기본 구성은 3 layer
# input layer, hidden layer, output layer

model = tf.keras.Sequential([
    #입력 레이어를 (28,28)을 784개의 노드로 만들어준다.
    # 외부 입력을 신경망으로 가져오는 역할만 한다.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 히든 레이어로 노드 개수가 128개인 Dense 레이어를 사용한다.
    # Dense 레이어는 바로 전(입력) layer 각각의 노드가 현재 레이어의 모든 노드와 연결된다.
    # 노드의 출력을 결정짓는 활성화 함수는 'relu'를 사용한다.
    tf.keras.layers.Dense(128, activation='relu'),
    # 출력 layer는 신경망의 출력을 외부에 전달하는 역할을 한다.
    # 출력 layer로 노드 개수가 10개인 Dense layer로 설정한다.
    # 활성화 함수로는 softmax를 사용한다.
    # softmax를 거치면 10개(label이 10개이므로)의 노드 출력의 합이 1이 되도록 조정한다.
    # 이 10개의 출력 합 중, 가장 큰 값이 신경망이 예측한 값이다.
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# 학습을 시작하기 전에 손실함수, 옵티마이저, 평가 지표를 설정해야 한다.
model.compile(
    # optimizer는 학습 데이터 셋과 손실함수를 사용해, 모델의 가중치를 업데이트하는 방법을 결정한다.
    optimizer='adam',
    # 학습을 하는 동안 손실함수를 최소화하도록 모델의 가중치를 조절한다.
    loss = 'sparse_categorical_crossentropy',
    # 평가지표(metrics)는 학습과 평가시 모델 성능을 측정하기 위해 사용된다.
    # 여기서는 전체 데이터셋에서 올바르게 분류된 이미지 비율을 표시하는 정확도를 사용한다.
    metrics=['accuracy']
)
# 모델을 학습한다
# epochs는 반복 횟수
# 학습은 입력 이미지에 대해 잘못된 판정을 내리는 신경망의 오류가
# 줄어들도록 신경망의 가중치를 조정하여 이루어진다.
# 그 결과 이미지와 라벨간의 관계를 신경망이 학습한다.
model.fit(train_images, train_labels, epochs=10)

# 사진4
# 학습되는 동안 손실(loss)는 줄어들며, 예측 정확도(accuracy)는 상승한다.
# 손실은 위에서 지정한 손실 함수에 의해 계산된다.


# # train_images를 입력으로 학습시, 틀린 예측을 한 건수를 기반으로 예측 정확도를 계산한다.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# # 학습시 사용한 훈련 데이터셋(0.91) 보다 테스트 데이터셋(0.88)에서 정확도가 떨어졌다.
# # 훈련 데이터셋보다 테스트 데이터셋에서 결과가 안좋은 경우 모델이 오버피팅되었다 고 한다.
# # 정규화(regulation) 또는 dropout을 추가하여 이 문제를 해결할 수 있다.

# plt.figure(figsize=(4, 7))
# # # 10개의 임의의 데이터 테스트
# # # 사진6, 임의의 1개 사진 테스트한 결과
# i = 1
# cnt = 10
# while i <= cnt:
#     idx = random.randint(0, len(test_images))

#     # 이 때의 img.shape = (28, 28)
#     img = test_images[idx]
#     label = test_labels[idx]
#     # 신경망 입력에 맞는 사이즈인 (1, 28, 28)로 확장
#     img = np.expand_dims(img, 0)

#     # 테스트 이미지 삽입
#     # 신경망은 활성화 함수로 softmax를 사용한다.
#     # 제일 큰 값이 가장 적합한 노드로 에측된 결과이다.
#     predictions = model.predict(img)
#     result = np.argmax(predictions)
    
#     # Prediction 시각화
#     plt.subplot(2, 5, i)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[idx], cmap='gray')
#     plt.xlabel('Label: '+class_names[label]+'\n'+"Prediction: "+class_names[result])
#     i += 1
# plt.show()