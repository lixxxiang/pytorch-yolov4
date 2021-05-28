import cv2
import os

xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <filename>{}</filename>
    <path>{}</path>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''

labels = ['Fixed_Roof_Tanks', 'Floating_Roof_Tanks']  # label for datasets
cnt = 0
image_folder = r'D:/TRAIN/images'
label_folder = r'D:/TRAIN/labels'
xml_folder = r'D:/TRAIN/xml'
trainval_file = r'D:/TRAIN/list.txt'
txt_name = []
if os.path.getsize(trainval_file) == 0:
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            txt_name.append(os.path.join(root, file))

    with open(trainval_file, 'w') as f:
        for i in txt_name:
            f.write('{}\n'.format(i))
    f.close()

with open(trainval_file, 'r') as train_list:
    for lst in train_list.readlines():
        lst = lst.strip()
        file = os.path.basename(lst)
        jpg = lst  # image path
        txt = label_folder + '/' + file.replace('.jpg', '.txt')  # yolo label txt path
        xml_path = os.path.join(xml_folder, file.replace('.jpg', '.xml'))  # xml save path
        print(xml_path)
        obj = ''

        img = cv2.imread(jpg)
        img_h, img_w = img.shape[0], img.shape[1]
        head = xml_head.format(str(file), str(jpg), str(img_w), str(img_h))

        with open(txt, 'r') as f:
            for line in f.readlines():
                yolo_datas = line.strip().split(' ')
                label = int(float(yolo_datas[0].strip()))
                center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
                center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
                bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
                bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)

                xmin = str(int(center_x - bbox_width / 2))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))

                obj += xml_obj.format(labels[label], xmin, ymin, xmax, ymax)
        with open(xml_path, 'w') as f_xml:
            f_xml.write(head + obj + xml_end)
        cnt += 1
        print(cnt)