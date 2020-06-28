import os
import csv
import numpy as np
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config
import random
import pickle

# Req. 3-1	이미지 경로 및 캡션 불러오기
def get_path_caption():
    # config
    caption_path = config.args.caption_path
    file = open(caption_path, 'r')

    #첫줄 제외하고 line으로 읽기
    data = file.readlines()[1:]
    file.close()

    dictionary = {}
    for line in data:
        token = line.split()
        img = token[0].split('|')[0]
        caption = ' '.join(token[2:]).replace(',', '')

        if img not in dictionary:
            dictionary[img] = [caption]
        else:
            dictionary[img].append(caption)

    return dictionary

##########################################################################
# Req. 3-2	전체 데이터셋을 분리해 저장하기
def dataset_split_save(dictionary):
    # train data : 70%, test data: 30%
    size = len(dictionary)
    train_size = int(size * 0.7)
    test_size = size - train_size

    # random choice train_data and test_data
    # There are two ways to pick random data with dictionary.
    ##########################################################
    # (python 3.5 ver) dictionary는 순서가 보장되어있지 않는다.
    if sys.version_info < (3, 6):
        train_data = dict(list(dictionary.items())[:train_size])
        test_data = dict(list(dictionary.items())[train_size:])

    ##########################################################
    # (python 3.6 ver) dictionary는 순서가 보장되어
    # list로 변환후 shuffle하여 다시 dictionary로 변환해준다.
    else:
        total_keys = list(dictionary.keys())
        random.shuffle(total_keys)

        data_num = 0
        train_data = {}
        test_data = {}
        for key in total_keys:
            train_data[key] = dictionary[key]

            if data_num == train_size:
                test_data[key] = dictionary[key]
            else:
                data_num += 1
                train_data[key] = dictionary[key]

    ##########################################################
    train_dataset_path=config.args.train_data_path+"train_data.pickle"
    test_dataset_path=config.args.test_data_path+"test_data.pickle"

    # pickle 모듈을 활용해 dictionary 자체를 binary로 저장
    with open(train_dataset_path, "wb") as fw:
        pickle.dump(train_data, fw)

    with open(test_dataset_path, "wb") as fw:
        pickle.dump(test_data, fw)

    return train_dataset_path, test_dataset_path

###########################################################################
# Req. 3-3	저장된 데이터셋 불러오기
def get_data_file(data_path):

    # pickle 모듈을 활용해 dictionary 호출
    with open(data_path, "rb") as fr:
        return pickle.load(fr)

###########################################################################
# Req. 3-4	데이터 샘플링
# 전체 데이터에서 do_sampling 비율만큼만 랜덤 샘플링하여 return
def sampling_data(dictionary):

    sample_num = int(config.args.do_sampling * 0.01)

    #########################################################
    # (python 3.5 ver)
    if sys.version_info < (3, 6):
        return dict(list(dictionary.items())[:sample_num])

    #########################################################
    # (python 3.6 ver)
    else:
        keys = list(dictionary.keys())
        random.shuffle(keys)

        data_num = 0
        sample_data = {}

        for key in keys:
            sample_data[key] = dictionary[key]
            data_num += 1
            if data_num == sample_num:
                braek

        return sample_data