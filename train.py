import config
from data import preprocess
from utils import utils


# config 저장
utils.save_config()


# (Req. 3-1) 이미지 경로 및 캡션 불러오기
dictionary = preprocess.get_path_caption()


# (Req. 3-2) 전체 데이터셋을 train,test 랜덤으로 분리해 저장하기
train_dataset_path, val_dataset_path = preprocess.dataset_split_save(dictionary)


# (Req. 3-3) 저장된 데이터셋 불러오기
# train_data만 원하는 경우
img_caption = preprocess.get_data_file(train_dataset_path)
# test_data만 원하는 경우
img_caption = preprocess.get_data_file(val_dataset_path)


# (Req. 3-4) 데이터 샘플링
# Req.3-1 결과를 파라미터로 넘긴다
if config.do_sampling:
    img_caption = preprocess.sampling_data(dictionary)


# 이미지와 캡션 시각화 하기
#img: 사진파일명, caption 매칭되는 캡션
utils.visualize_img_caption(img, caption)
