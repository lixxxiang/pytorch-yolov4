import os, random, shutil


def moveFile(fileDir, labelDir, type):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    sample = random.sample(pathDir, int(len(pathDir) * 0.2))  # 随机选取picknumber数量的样本图片
    for name in sample:
        shutil.move(os.path.join(fileDir,name), os.path.join(main_folder, 'final/images/val/' + name))
    for name in sample:
        shutil.move(os.path.join(labelDir, name.replace(type, 'txt')),
                    os.path.join(main_folder, 'final/labels/val/' + name.replace(type, 'txt')))
    # for root, dirs, files in os.walk(fileDir):
    #     for file in files:
    #         print(os.path.join(root, file))
    #         shutil.move(os.path.join(root, file), os.path.join(main_folder, 'final/images/train/' + file))
    # for root, dirs, files in os.walk(labelDir):
    #     for file in files:
    #         print(os.path.join(root, file))
    #         shutil.move(os.path.join(root, file), os.path.join(main_folder, 'final/labels/train/' + file))
    # return


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
    main_folder = 'D:/Cushing'
    mkdir(os.path.join(main_folder, 'final/'))
    mkdir(os.path.join(main_folder, 'final/images/'))
    mkdir(os.path.join(main_folder, 'final/images/train/'))
    mkdir(os.path.join(main_folder, 'final/images/val/'))
    mkdir(os.path.join(main_folder, 'final/labels/'))
    mkdir(os.path.join(main_folder, 'final/labels/train/'))
    mkdir(os.path.join(main_folder, 'final/labels/val/'))

    moveFile("D:\TRAIN\images", "D:\TRAIN\labels", "jpg")
