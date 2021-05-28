import os
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
from matplotlib import pyplot as plt
import json
import skimage.io
from sys import argv
import shutil
import time
import random
import cv2


def tif2jpg(tif, jpg):
    im = Image.open(tif)
    im.save(jpg)
    return jpg


def generate_file(slice_path, file_list):
    L = []
    for root, dirs, files in os.walk(slice_path):
        for file in files:
            L.append(slice_path.split('/')[0] + '\\' + file)
    with open(file_list, 'w') as f:
        for i in L:
            f.write('{}\n'.format(i))


def slice_im_plus_boxes(image_path, out_name, out_dir_images, overlap=0.2):
    imagesize = 416
    image = skimage.io.imread(image_path)
    print("image.shape:", image.shape)
    dx = int((1. - overlap) * imagesize)
    dy = int((1. - overlap) * imagesize)
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):
            n_ims += 1

            if (n_ims % 100) == 0:
                print(n_ims)

            if y0 + imagesize > image.shape[0]:
                y = image.shape[0] - imagesize
            else:
                y = y0
            if x0 + imagesize > image.shape[1]:
                x = image.shape[1] - imagesize
            else:
                x = x0

            window_c = image[y:y + imagesize, x:x + imagesize]
            outpath = os.path.join(
                out_dir_images,
                out_name + '_' + str(y) + '_' + str(x) + '.jpg')
            if not os.path.exists(outpath):
                skimage.io.imsave(outpath, window_c, check_contrast=False)
            else:
                print("outpath {} exists, skipping".format(outpath))

    print("Num slices:", n_ims,
          "sliceHeight", imagesize, "sliceWidth", imagesize)
    return

def rndColor():
    return (random.randint(0,255),random.randint(0,255),random.randint(0,255))


def txt_format(path):
    width = 416
    height = 416
    count = 0
    without_nms_count = 0
    accurate_count = 0
    nms_count = 0
    ls = os.listdir(path)
    XLS = []
    raw_width, raw_height = Image.open(raw_jpg).size
    for i in ls:
        file = open(os.path.join(path, i), "rb")
        H = float(i.split('_')[-2])
        W = float(i.split('_')[-1].split('.')[0])
        xls, xrs, yls, yrs, conf, class_ = restore2(i, file, H, W, raw_width, raw_height, width, height)

        count += len(xls)
        XLS.extend(xls)
        XRS.extend(xrs)
        YLS.extend(yls)
        YRS.extend(yrs)
        CONF.extend(conf)
        CLASS.extend(class_)
    jpg = Image.open(raw_jpg)
    print(len(XLS))
    color = rndColor()
    for i in range(0, len(XLS)):
        without_nms_count += 1
        bbox = np.zeros([len(XLS), 6])
        x1 = XLS[i]
        x2 = XRS[i]
        y1 = YLS[i]
        y2 = YRS[i]

        ImageDraw.Draw(jpg).rectangle([XLS[i], YLS[i], XRS[i], YRS[i]], fill=None, outline=color,
                                      width=2)
    s = 'without nms: ' + str(without_nms_count) + '/' + str(len(XLS)) + str('\n')
    if without_nms_count != 0:
        print(s)
        readme.write(s)
        raw_image_res = jpg.convert('RGB')
        raw_image_res.save(res_path + 'res_' + raw_jpg_filename)
        for i in range(0, len(XLS)):
            bbox[i][0] = XLS[i]
            bbox[i][1] = YLS[i]
            bbox[i][2] = XRS[i]
            bbox[i][3] = YRS[i]
            bbox[i][4] = CONF[i]
            bbox[i][5] = CLASS[i]


        res = nms(bbox, 0.1)
        res = np.sort(res)

        for i in range(0, len(res)):
            bbox_res = np.zeros([len(res), 6])

        jpg2 = Image.open(raw_jpg)
        img = cv2.imread(raw_jpg)
        for i in range(0, len(res)):
            nms_count += 1
            bbox_res[i][0] = bbox[res[i]][0]
            bbox_res[i][1] = bbox[res[i]][1]
            bbox_res[i][2] = bbox[res[i]][2]
            bbox_res[i][3] = bbox[res[i]][3]
            bbox_res[i][4] = bbox[res[i]][4]
            bbox_res[i][5] = bbox[res[i]][5]
            # color = [random.randint(0, 255) for _ in range(3)]

            if bbox_res[i][4] < 0.7:
                label = str(int(bbox_res[i][5])) + '_' + str(bbox_res[i][4] * 100).split('.')[0] + '%'
                ImageDraw.Draw(jpg2).rectangle([bbox_res[i][0], bbox_res[i][1], bbox_res[i][2], bbox_res[i][3]],
                                               fill=None,
                                               outline=(255, 0, 0), width=2)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                ImageDraw.Draw(jpg2).rectangle(
                    [bbox_res[i][0], bbox_res[i][1] - t_size[1], bbox_res[i][0] + 60, bbox_res[i][1]],
                    fill=(255, 0, 0),
                    outline=(255, 0, 0), width=2)
                ImageDraw.Draw(jpg2).text((bbox_res[i][0], bbox_res[i][1] - t_size[1]),
                                          label,
                                          font=ImageFont.truetype(r"C:\Windows\Fonts\Calibrib.ttf",
                                                                  t_size[1]),
                                          fill=(255, 255, 255))
            else:
                accurate_count += 1
                label = str(int(bbox_res[i][5])) + '_' + str(bbox_res[i][4] * 100).split('.')[0] + '%'
                # if (str(int(bbox_res[i][5])) == '1'):
                #     label = 'Floating_Roof_Tanks'
                # else:
                #     label = 'Fixed_Roof_Tanks'

                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=1)[0]

                ImageDraw.Draw(jpg2).rectangle([bbox_res[i][0], bbox_res[i][1], bbox_res[i][2], bbox_res[i][3]],
                                               fill=None,
                                               outline=(1, 129, 74), width=2)
                ImageDraw.Draw(jpg2).rectangle(
                    [bbox_res[i][0], bbox_res[i][1] - t_size[1], bbox_res[i][0] + 70, bbox_res[i][1]],
                    fill=(1, 129, 74),
                    outline=(1, 129, 74), width=2)
                ImageDraw.Draw(jpg2).text((bbox_res[i][0], bbox_res[i][1] - t_size[1]),
                                          label,
                                          font=ImageFont.truetype(r"C:\Windows\Fonts\Calibrib.ttf",
                                                                  t_size[1]),
                                          fill=(255, 255, 255))

        s2 = 'with nms: ' + str(nms_count) + '/' + str(len(res)) + str('\n')
        print(s2)
        readme.write(s2)
        raw_image_res = jpg2.convert('RGB')
        t = time.strftime("%m%d-%H%M", time.localtime())
        res_file = res_path + t + '_nms_res_' + raw_jpg_filename
        raw_image_res.save(res_path + t + '_nms_res_' + raw_jpg_filename)
        s3 = 'accurate: ' + str(accurate_count) + '/ 435' + str('\n')
        print(s3)
        readme.write(s3)
        print('done')
        return res_file
    else:
        return -1


def json_format(jsonfile):
    file = open(jsonfile, "rb")
    without_nms_count = 0
    accurate_count = 0
    nms_count = 0
    fileJson = json.load(file)

    for i in range(0, len(fileJson)):
        filename = fileJson[i]["filename"]
        objects = fileJson[i]["objects"]
        if len(objects) != 0:
            XLS, XRS, YLS, YRS, CONF = restore(filename, objects)
    jpg = Image.open(raw_jpg)

    for i in range(0, len(XLS)):
        without_nms_count += 1
        bbox = np.zeros([len(XLS), 6])
        ImageDraw.Draw(jpg).rectangle([XLS[i], YLS[i], XRS[i], YRS[i]], fill=None, outline=(0, 255, 0),
                                      width=2)
    print('without nms: ', without_nms_count, '/', len(XLS))
    raw_image_res = jpg.convert('RGB')
    raw_image_res.save(res_path + 'res_' + raw_jpg_filename)
    for i in range(0, len(XLS)):
        bbox[i][0] = XLS[i]
        bbox[i][1] = YLS[i]
        bbox[i][2] = XRS[i]
        bbox[i][3] = YRS[i]
        bbox[i][4] = CONF[i]
        bbox[i][5] = CLASS[i]

    res = nms(bbox, 0.1)
    res = np.sort(res)

    for i in range(0, len(res)):
        bbox_res = np.zeros([len(res), 6])

    jpg2 = Image.open(raw_jpg)
    for i in range(0, len(res)):
        nms_count += 1
        bbox_res[i][0] = bbox[res[i]][0]
        bbox_res[i][1] = bbox[res[i]][1]
        bbox_res[i][2] = bbox[res[i]][2]
        bbox_res[i][3] = bbox[res[i]][3]
        bbox_res[i][4] = bbox[res[i]][4]
        bbox_res[i][5] = bbox[res[i]][5]

        color = color or [random.randint(0, 255) for _ in range(3)]

        if bbox_res[i][4] < 0.9:
            cv2.rectangle(jpg2, (bbox_res[i][0], bbox_res[i][1]), (bbox_res[i][2], bbox_res[i][3]), color, -1,
                          cv2.LINE_AA)  # filled
            cv2.putText(jpg2, str(int(bbox_res[i][5])) + '_' + str(bbox_res[i][4] * 100).split('.')[0] + '%',
                        bbox_res[i][0], bbox_res[i][1], 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            # ImageDraw.Draw(jpg2).rectangle([bbox_res[i][0], bbox_res[i][1], bbox_res[i][2], bbox_res[i][3]],
            #                                fill=None,
            #                                outline=(255, 0, 0), width=2)
            # ImageDraw.Draw(jpg2).text((bbox_res[i][0], bbox_res[i][1]),
            #                           str(int(bbox_res[i][5])) + '_' + str(bbox_res[i][4] * 100).split('.')[0] + '%',
            #                           font=ImageFont.truetype(r"D:\darknet-master\build\darknet\x64\TimesNewRoman.ttf",
            #                                                   20),
            #                           fill=(255, 0, 0))
            # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        else:
            accurate_count += 1
            ImageDraw.Draw(jpg2).rectangle([bbox_res[i][0], bbox_res[i][1], bbox_res[i][2], bbox_res[i][3]],
                                           fill=None,
                                           outline=(0, 255, 0), width=2)

    print('with nms: ', nms_count, '/', len(res))
    raw_image_res = jpg2.convert('RGB')
    raw_image_res.save(res_path + 'nms_res_' + raw_jpg_filename)
    print('accurate: ', accurate_count, '/435')
    print('done')
    readme.write('l')
    readme.close()


def nms(bboxs, thresh):
    # bboxs:形似上面设置的boxes，是一组包含了诸多框坐标的数组
    # thresh： IOU阈值

    # 1.获取左上角右下角四个坐标
    x1 = bboxs[:, 0]  # 获取所有框的左上角横坐标
    y1 = bboxs[:, 1]  # 获取所有框的左上角纵坐标
    x2 = bboxs[:, 2]  # 获取所有框的右下角横坐标
    y2 = bboxs[:, 3]  # 获取所有框的右下角纵坐标

    # 2.计算每个框的面积
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)

    # 3.获取得分以排序
    scores = bboxs[:, 4]
    index = scores.argsort()[::-1]  # argsort默认从小到大排序，[::-1]实现翻转

    # 4.保留结果集，返回输出保留下来的Bbox最终结果
    res = []

    while index.size > 0:
        i = index[0]  # index中存储Bbox按分排序后的索引，所以第一个就是得分最高的Bbox索引，直接保留
        res.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])  # 用X11表示重叠区域的左上角横坐标
        y11 = np.maximum(y1[i], y1[index[1:]])  # 用y11表示重叠区域的左上角横坐标
        x22 = np.minimum(x2[i], x2[index[1:]])  # 用X22表示重叠区域的左上角横坐标
        y22 = np.minimum(y2[i], y2[index[1:]])  # 用y221表示重叠区域的左上角横坐标

        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  # index[1:]从下标1开始取到列表结束 最高分的面积加其余的面积

        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return res


def restore(filename, objects):
    width = 416
    height = 416
    XL = 0
    XR = 0
    YL = 0
    YR = 0

    file = filename.split('/')[-1]
    raw_width, raw_height = Image.open(raw_jpg).size
    H = float(file.split('_')[-2])
    W = float(file.split('_')[-1].split('.')[0])
    for i in range(0, len(objects)):
        relative_coordinates = objects[i]["relative_coordinates"]
        confidence = objects[i]["confidence"]
        classid = objects[i]["class_id"]
        xl = (float(relative_coordinates["center_x"]) - 0.5 * float(
            relative_coordinates["width"])) * width - 0.24 * float(relative_coordinates["width"]) * width
        yl = (float(relative_coordinates["center_y"]) - 0.5 * float(
            relative_coordinates["height"])) * height - 0.24 * float(relative_coordinates["height"]) * height
        xr = (float(relative_coordinates["center_x"]) + 0.5 * float(
            relative_coordinates["width"])) * width + 0.24 * float(relative_coordinates["width"]) * width
        yr = (float(relative_coordinates["center_y"]) + 0.5 * float(
            relative_coordinates["height"])) * height + 0.24 * float(relative_coordinates["height"]) * height
        if W <= raw_width:
            XL = xl + W
            XR = xr + W
        if H <= raw_height:
            YL = yl + H
            YR = yr + H
        if (XR - XL) / (YR - YL) < 2 and (YR - YL) / (XR - XL) < 2:
            XLS.append(XL)
            XRS.append(XR)
            YLS.append(YL)
            YRS.append(YR)
            CONF.append(confidence)
            CLASS.append(classid)
    return XLS, XRS, YLS, YRS, CONF


def restore2(filename, file, H, W, raw_width, raw_height, width, height):
    XL = 0
    XR = 0
    YL = 0
    YR = 0
    xls = []
    xrs = []
    yls = []
    yrs = []
    conf_ = []
    class_ = []
    for line in file.readlines():
        line = str(line)
        xl = (float(line.split(' ')[1]) - 0.5 * float(line.split(' ')[3])) * width
        # - 0.24 * float(line.split(' ')[3]) * width
        yl = (float(line.split(' ')[2]) - 0.5 * float(line.split(' ')[4])) * height
        # - 0.24 * float(line.split(' ')[4]) * height
        xr = (float(line.split(' ')[1]) + 0.5 * float(line.split(' ')[3])) * width
        # + 0.24 * float(line.split(' ')[3]) * width
        yr = (float(line.split(' ')[2]) + 0.5 * float(line.split(' ')[4])) * height
        # + 0.24 * float(line.split(' ')[4]) * height
        conf = line.split(' ')[5]
        confidence = float(conf[0:4])
        classid = line.split(' ')[0].split('\'')[-1]

        if W <= raw_width:
            XL = xl + W
            XR = xr + W
        if H <= raw_height:
            YL = yl + H
            YR = yr + H
        if (XR - XL) / (YR - YL) < 2 and (YR - YL) / (XR - XL) < 2:
            xls.append(XL)
            xrs.append(XR)
            yls.append(YL)
            yrs.append(YR)
            conf_.append(confidence)
            class_.append(classid)
    return xls, xrs, yls, yrs, conf_, class_


def change_slash(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(json_file, "w", encoding="utf-8") as f_w:
        for line in lines:
            if "\\" in line:
                line = line.replace("\\", "/")
            f_w.write(line)


def pretreat(from_, to_):
    for i in os.listdir(from_):
        if os.path.join(from_, i).split('.')[-1] == 'txt':
            shutil.move(os.path.join(from_, i), os.path.join(to_, i))


def dirnum(path):
    dirnum = 0
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isdir(sub_path):
            dirnum = dirnum + 1
    return dirnum


def generate_train_folder(labelpath, from_, to_, trainval_file):
    for i in os.listdir(labelpath):
        FROM = os.path.join(from_, i.split('.')[0] + '.jpg')
        TO = os.path.join(to_, i.split('.')[0] + '.jpg')
        shutil.copyfile(FROM, TO)
    L = []
    for root, dirs, files in os.walk(to_):
        for file in files:
            L.append(to_ + '/' + file)
    txt_name = L

    with open(trainval_file, 'w') as f:
        for i in txt_name:
            f.write('{}\n'.format(i))
    f.close()


if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = 2300000000
    raw_dir = argv[1]
    weights = argv[2]
    cfg = argv[3]
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if root == 'D:\OILTANK_TEST':
                XLS = []
                XRS = []
                YLS = []
                YRS = []
                CONF = []
                CLASS = []
                res_dir = os.path.join(raw_dir + '/res/')
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                raw_tif = os.path.join(root, file)
                base_dir = os.path.join(root, file.split('.')[0] + '/')
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)

                raw_jpg = tif2jpg(raw_tif, os.path.join(base_dir, file.split('.')[0] + '.jpg'))
                # json_file = raw_tif.split('.')[0] + '.json'
                # txt_file = raw_tif.split('.')[0] + '.txt'
                raw_jpg_filename = os.path.basename(raw_jpg)
                # slice_res_path = os.path.join(os.path.dirname(raw_jpg),
                #                               os.path.basename(argv[2]).split('.')[0] + time.strftime("----%m%d-%H%M",
                #                                                                                       time.localtime()) + '_slice_res')
                # res_path = os.path.join(os.path.dirname(raw_jpg),
                #                         os.path.basename(argv[2]).split('.')[0] + time.strftime("----%m%d-%H%M",
                #                                                                                 time.localtime()) + '_res/')

                # if not os.path.exists(slice_path):
                #     os.makedirs(slice_path)
                # if not os.path.exists(res_path):
                #     os.makedirs(res_path)
                # if not os.path.exists(slice_res_path):
                #     os.makedirs(slice_res_path)

                # base_dir = os.path.dirname(raw_jpg)
                slice_path = os.path.join(base_dir, 'slice/')
                if not os.path.exists(slice_path):
                    os.makedirs(slice_path)
                index = dirnum(base_dir) - 1
                base_res_dir = os.path.join(base_dir, 'exp' + str(index))
                os.makedirs(base_res_dir)
                readme = open(os.path.join(base_res_dir, 'readme.md'), 'w')
                readme.write(argv[2])
                readme.write('\n')


                slice_res_path = os.path.join(base_res_dir, 'slice_res/')
                os.makedirs(slice_res_path)
                slice_res_label_path = os.path.join(base_res_dir, 'slice_res_label/')
                os.makedirs(slice_res_label_path)
                res_path = os.path.join(base_res_dir, 'result/')
                os.makedirs(res_path)
                train_path = os.path.join(base_res_dir, 'train/')
                os.makedirs(train_path)
                trainval_file = os.path.join(base_res_dir, 'val.txt')
                # for yolov4
                # if os.path.getsize(json_file) == 0:
                #     slice_im_plus_boxes(image_path=raw_jpg, out_name=os.path.splitext(raw_jpg_filename)[0],
                #                         out_dir_images=slice_path, image_size=image_size,
                #                         overlap=0.2)
                #     open(txt_file, 'w+')
                #     generate_file(slice_path, txt_file)
                #     os.system(
                #         r'darknet.exe detector test voc.data yolov4-custom.cfg ' + weights + ' -ext_output -dont_show -out ' + json_file + ' <' + txt_file)
                #     change_slash(json_file)
                # json_format(json_file)

                # for yolov4-pytorch
                # print(slice_res_path.split('\\')[-1].split('----')[0])
                # print(argv[2].split('\\')[-1].split('.')[0])
                # arg1 = slice_res_path.split('\\')[-1].split('----')[0]
                # arg2 = argv[2].split('\\')[-1].split('.')[0]
                # if os.path.dirname(slice_res_path) != 0 and arg1 == arg2 :

                # if os.path.dirname(slice_res_path) == 0:


                if len(os.listdir(slice_path)) == 0:
                # if not os.path.exists(slice_path):
                    slice_im_plus_boxes(image_path=raw_jpg, out_name=os.path.splitext(raw_jpg_filename)[0], out_dir_images=slice_path,
                                        overlap=0.2)
                # print(argv[2].split('\\')[-1])
                os.system(
                    r'python .\detect.py --img 416 --cfg .\\' + argv[3].split('\\')[-1] + ' --weights .\\' + argv[2].split('\\')[
                        -1] + ' --source ' + slice_path + ' --save-txt --output ' + slice_res_path)
                pretreat(slice_res_path, slice_res_label_path)
                raw_image_res = txt_format(slice_res_label_path)
                if raw_image_res != -1:
                    generate_train_folder(slice_res_label_path, slice_path, train_path, trainval_file)
                    # shutil.rmtree(slice_path)
                    print('raw_image_res',raw_image_res)
                    print('-',os.path.join(res_dir, file.split('.')[0] + '.jpg'))
                    shutil.copyfile(raw_image_res, os.path.join(res_dir, file.split('.')[0] + '.jpg'))
