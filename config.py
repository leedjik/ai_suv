import argparse

# Req. 2-1	Config.py 파일 생성

parser = argparse.ArgumentParser()
parser.add_argument('--caption_path', type=str, default='.\\datasets\\captions.csv', help='caption path.')
parser.add_argument('--image_path', type=str, default='.\\datasets\\images\\', help='image가 저장된 경로를 입력하세요.')
parser.add_argument('--train_data_path', type=str, default='.\\datasets\\', help='학습에 사용할 데이터가 저장될 경로를 입력하세요.')
parser.add_argument('--test_data_path', type=str, default='.\\datasets\\', help='테스트에 사용할 데이터가 저장될 경로를 입력하세요.')
parser.add_argument('--do_sampling', type=int, default=0, help='표본 데이터를 샘플링할 비율을 입력하세요.(0 ~ 100)%%')
args = parser.parse_args()

do_sampling=args.do_sampling
# print(args)