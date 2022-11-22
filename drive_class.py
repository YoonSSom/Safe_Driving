# Commented out IPython magic to ensure Python compatibility.
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import os
import re
import random
import json
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %matplotlib inline

"""# 전처리를 위한 Drive_data_maker class 생성
- all_data_path : 학습을 위한 이미지의 모든 데이터 경로를 한 txt 파일에 write하여 저장하는 함수
- pick_data_path : 사용할 이미지의 경로만 한 txt파일에 write하여 저장하는 함수
- shuffle_pick : 학습시 데이터를 골고루 사용할 수 있도록 shuffle하는 함수
- resize_img : image resizing 함수
- data_split_txt : train / valid / test로 split하는 함수
- json_path : 사용할 이미지의 json파일 경로만 모아주는 함수
- json_to_txt : json파일을 불러와 모델 학습을 위해 (label, x, y, w, h)로 변환하여 txt파일에 저장 시키는 함수

"""
# 전처리를 위한 Drive_data_maker class 생성
class Drive_data_maker():
    # data_path : 원본 데이터 PATH , make_path : 저장할 PATH 불러오기 및 저장(run)
    def __init__(self, data_path, make_path):
        self.data_path = data_path
        self.make_path = make_path

    # img(jpg) PATH Write.txt
    def all_data_path(self, name='train'):
        print(f'All data path save file name: {name}_img_path.txt')
        f = open(self.make_path + f'{name}_img_path.txt', 'w')
        # train image PATH 의 Folder List 저장
        for folder in tqdm(os.listdir(self.data_path)):
            # data_sub_path = 원본 image PATH + / + Folder(name)
            data_sub_path = self.data_path + '/' + folder
            #각 Folder별 jgp file List Write
            for img in os.listdir(data_sub_path):
                if img[-3:] == 'jpg':
                    f.write(data_sub_path + '/' + img + '\n')
        f.close()

    # ALL img PATH.txt 에서 Name,Label 별 text Write 함수
    def pick_data_path(self, input_txt, name, label):
        print(f'Pick data path save file name: {name}_img_path.txt')
        # ALL img PATH.txt read
        f = open(self.make_path + input_txt, 'r')
        # Name, Label 별 text file Write
        t = open(self.make_path + f'{name}_img_path.txt', 'w')

        # ALL img PATH.txt > all_list List 생성
        all_list = f.readlines()
        # ALL img 중 Label in List > text Write
        for line in all_list:
            if label in line:
                t.write(line)
            else:
                pass

        f.close()
        t.close()
        
    # Text read > shuffle_pick(int-1):run에서 지정 
    def shuffle_pick(self, txt_data, pick_num = -1):
        f = open(self.make_path + txt_data, 'r')
        pick_img_path = f.readlines()
        f.close()
        # PATH List Shuffle
        random.shuffle(pick_img_path)
        # Affer Slicing
        pick_img_path = pick_img_path[:pick_num]
        return pick_img_path
    
    # 각 normal, sleep, phone list split 
    def data_split_txt(self, data1, data2, data3):        
        # mix_data.txt Write (train, valid, test)
        f = open(self.make_path + 'mix_data.txt', 'w')
        for d1, d2, d3 in zip(data1, data2, data3):
            f.write(d1)
            f.write(d2)
            f.write(d3)

        f.close()

        print('Maked "mix_data.txt"')
        # mix_data.txt read > list
        t = open(self.make_path + 'mix_data.txt', 'r')
        mix_data = t.readlines()
        t.close()

        print('test1')
        
        # Data / 0.6 , 0.2 값 
        len_all = len(mix_data)
        a = int(len_all * 0.6)
        b = int(a + (len_all * 0.2))
        # Slicing > train, valid, test
        train_data = mix_data[:a]
        valid_data = mix_data[a:b]
        test_data = mix_data[b:]

        print('Maked [train, vaild, test] txt files')
        
        data_list = [train_data , valid_data, test_data]
        make_list = ['train', 'valid', 'test']

        # 각 Text 별 train, valid, test .txt File Write
        for datas, maker in zip(data_list, make_list):
            f = open(self.make_path + f'{maker}.txt', 'w')
            for data in datas:
                f.write(data)
            
            f.close()

    # img resize + channel*3
    def resize_img(self, data):
        for path in tqdm(data):

            imgs = cv2.imread(path)
            imgs = cv2.resize(imgs,(416, 416))

            gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
            imgs = np.zeros_like(imgs)

            imgs[:,:,0] = gray
            imgs[:,:,1] = gray
            imgs[:,:,2] = gray

            # File Name List
            name = path.split('/')[-1].strip()

            # Save Path + Name > Copy
            save_path = self.make_path + name
            cv2.imwrite(save_path, imgs)

    # json File Write
    def json_path(self):
        # mix_data.txt read > json list,
        f = open(self.make_path + 'mix_data.txt', 'r')
        t = open(self.make_path + 'json_path.txt', 'w')
        
        read_path = f.readlines()
        # img File list Slicing > json List Write
        for path in read_path:
            read_json = path[:-4] + 'json' + '\n'
            t.write(read_json)

        f.close()
        t.close()

    # json load > text (isVisble=T,F)(idx=0, 1, 2, 3)( x, y, w, h )
    def json_to_txt(self):
        f = open(self.make_path + 'json_path.txt', 'r')
        json_path = f.readlines()

        for i in tqdm(range(len(json_path))):
            bass_json = json_path[i].strip()
            j = open(bass_json, 'r')
            read_json = json.load(j)

            json_path[i].split('/')
            t = open(self.make_path + 'json/' + json_path[i].split('/')[-1][:-5] + 'txt', 'w')
            # 각 Name 별 BoundingBox resize % apply, idx , (opened,closed)
            names = ['Leye', 'Reye','Face', 'Phone']
            for name in names : 
                if read_json['ObjectInfo']['BoundingBox'][name]['isVisible'] == True :
                    x1, y1, x2, y2 = read_json['ObjectInfo']['BoundingBox'][name]['Position']
                    
                    x = (x1 + x2) / (2 * 800)
                    y = (y1 + y2) / (2 * 1280)
                    w = (x2 - x1) / 800
                    h = (y2 - y1) / 1280

                    if name == 'Leye':
                        if read_json['ObjectInfo']['BoundingBox'][name]['Opened'] == False :
                          idx = 0
                        else : idx = 1


                    if name == 'Reye' :
                        if read_json['ObjectInfo']['BoundingBox'][name]['Opened'] == False :
                            idx = 0
                        else : idx = 1

                    if name == 'Face' : idx = 2

                    if name == 'Phone' : idx = 3
                
                    t.write(f"{idx} {x:f} {y:f} {w:f} {h:f}\n")
            t.close()
        
        f.close()
        j.close()