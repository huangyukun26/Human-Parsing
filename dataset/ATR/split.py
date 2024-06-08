import os
import random
import shutil

# 定义文件夹路径
jpeg_images_dir = './image'
segmentation_class_aug_dir = './gt'
train_jpeg_dir = 'train/image'
train_seg_dir = 'train/gt'
test_jpeg_dir = 'val/image'
test_seg_dir = 'val/gt'

# 创建训练集和测试集文件夹
os.makedirs(train_jpeg_dir, exist_ok=True)
os.makedirs(train_seg_dir, exist_ok=True)
os.makedirs(test_jpeg_dir, exist_ok=True)
os.makedirs(test_seg_dir, exist_ok=True)

# 获取JPEGImages文件夹中的所有文件名，并确保是.jpg文件
jpeg_files = [f for f in os.listdir(jpeg_images_dir) if f.endswith('.jpg')]

# 打乱文件名顺序
random.shuffle(jpeg_files)

# 按照8:2的比例划分
split_index = int(len(jpeg_files) * 0.8)
train_files = jpeg_files[:split_index]
test_files = jpeg_files[split_index:]

# 生成训练集和测试集的id文件
with open('id.txt', 'w') as train_id_file, open('id.txt', 'w') as test_id_file:
    # 复制文件到相应的训练集和测试集文件夹中
    for file_name in train_files:
        base_name = os.path.splitext(file_name)[0]
        shutil.copy(os.path.join(jpeg_images_dir, file_name), os.path.join(train_jpeg_dir, file_name))
        shutil.copy(os.path.join(segmentation_class_aug_dir, base_name + '.png'), os.path.join(train_seg_dir, base_name + '.png'))
        train_id_file.write(base_name + '\n')

    for file_name in test_files:
        base_name = os.path.splitext(file_name)[0]
        shutil.copy(os.path.join(jpeg_images_dir, file_name), os.path.join(test_jpeg_dir, file_name))
        shutil.copy(os.path.join(segmentation_class_aug_dir, base_name + '.png'), os.path.join(test_seg_dir, base_name + '.png'))
        test_id_file.write(base_name + '\n')

print(f"训练集和测试集已按8:2的比例划分并复制完成，同时生成了train_ids.txt和test_ids.txt文件。")