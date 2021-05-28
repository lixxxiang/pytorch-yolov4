import os
import shutil


def pretreat(labels,dest):
    for root, dirs, files in os.walk(labels):
        for file in files:
            line = open(os.path.join(root, file)).readlines()
            new_file = open(os.path.join(dest, file), "w+")
            for i in line:
                i = i.split(' ')[0:-1]
                for j in i:
                    new_file.write(j)
                    new_file.write(' ')
                new_file.write('\n')

pretreat(r'D:\oiltank_data\VOCdevkit\VOC2007\labels', r'D:\oiltank_data\VOCdevkit\VOC2007\test')
