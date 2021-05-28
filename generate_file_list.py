import os

label_folder = r'D:\Cushing\Cushing\images\val'
trainval_file = r'D:\Cushing\val.txt'
txt_name = []
for root, dirs, files in os.walk(label_folder):
    for file in files:
        print(os.path.join(root, file))
        txt_name.append(os.path.join(root, file))

with open(trainval_file, 'w') as f:
    for i in txt_name:
        f.write('{}\n'.format(i))
f.close()
