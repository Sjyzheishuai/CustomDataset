from torch.utils.data import Dataset
import os
from torchvision.io import image


# 所有数据集都要继承Dataset类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        # self 指定了一个类当中的全局变量，该变量可以让后面的函数使用
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 这里的图片路径不是一张图片的路径，是一个路径数组
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        #读取图片，读取后魏tensor类型，该方法只支持读取png格式图片和jpeg格式图片
        img = image.read_image(img_item_path)
        label = self.label_dir
        if label == "ants":
            label = 0
        else:
            label = 1
        if self.transform is not None:
            img = self.transform(img)  # 对图片进行某些变换
        return img, label

    def __len__(self):
        return len(self.img_path)