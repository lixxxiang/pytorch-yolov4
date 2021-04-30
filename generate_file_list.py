import os


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            # print(file.split('')[0])
            # print(file.split('')[1])
            # print(file.split('')[1].split)
            # os.rename('D:/oiltank_data/oiltank_test_img_2/' + file, 'D:/oiltank_data/oiltank_test_img_2/' + file.split('_416_416_0_9338_12316.png')[0] + '.jpg')
            L.append(label_folder + '/' + file)
    return L


label_folder = r'D:\final\images\val'
trainval_file = r'D:\final\images\val.txt'

txt_name = file_name(label_folder)

with open(trainval_file, 'w') as f:
    for i in txt_name:
        f.write('{}\n'.format(i))
f.close()
