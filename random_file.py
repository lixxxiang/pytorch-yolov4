##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil



def moveFile(fileDir, labelDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    sample = random.sample(pathDir, int(len(pathDir) * 0.2))  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(fileDir + name, main_folder + 'final/images/val/' + name)
    for name in sample:
        shutil.move(labelDir + name.replace('jpg', 'txt'), main_folder + 'final/labels/val/' + name.replace('jpg', 'txt'))
    for root, dirs, files in os.walk(fileDir):
        for file in files:
            print(root + file)
            shutil.move(root + file, main_folder + 'final/images/train/' + file)
    for root, dirs, files in os.walk(labelDir):
        for file in files:
            print(root + file)
            shutil.move(root + file, main_folder + 'final/labels/train/' + file)
    return


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


if __name__ == '__main__':
    main_folder = 'D:/'
    mkdir(main_folder + 'final/')
    mkdir(main_folder + 'final/images/')
    mkdir(main_folder + 'final/images/train/')
    mkdir(main_folder + 'final/images/val/')
    mkdir(main_folder + 'final/labels/')
    mkdir(main_folder + 'final/labels/train/')
    mkdir(main_folder + 'final/labels/val/')

    image = 'D:/VOCdevkit/VOCdevkit/VOC2007/JPEGImages/'
    labels = 'D:/VOCdevkit/VOCdevkit/VOC2007/labels/'
    moveFile(image, labels)
    # moveFile2()
