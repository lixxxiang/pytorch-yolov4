import os
import shutil


def pretreat(labels, from_, to_):
    count = 0
    for i in os.listdir(labels):
        count += 1
        # print(i)
        # print(os.path.join(from_, i.split('.')[0] + '.jpg'))
        FROM = os.path.join(from_, i.split('.')[0] + '.jpg')
        TO = os.path.join(to_, i.split('.')[0] + '.jpg')
        # shutil.move(FROM, TO)
        print(FROM)
        print(TO)
        # if os.path.join(from_, i).split('.')[-1] == 'txt':
        #     shutil.move(os.path.join(from_, i), os.path.join(to_, i))
    print(count)


pretreat(r'D:\oiltank_data\JL1GF03B01_PMS_20210412000712_200046889\exp8\slice_res_label', r'D:\oiltank_data\JL1GF03B01_PMS_20210412000712_200046889\exp8\slice_res', 'D:\oiltank_data\JL1GF03B01_PMS_20210412000712_200046889\exp8\slice_res_2')
