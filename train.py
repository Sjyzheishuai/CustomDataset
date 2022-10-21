from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import MyData

transform = transforms.Resize((224, 224))
# 训练集的路径
train_root_dir = "dataset/train"
# 验证集路径
val_root_dir = "dataset/val"
# 分类的label
ants_label_dir = "ants"
bees_label_dir = "bees"

# 创建训练集
train_dataset = train_dataset = MyData(
    train_root_dir, ants_label_dir, transform)+MyData(train_root_dir, bees_label_dir, transform)
# 创建验证集
val_dataset = MyData(val_root_dir, ants_label_dir, transform) + MyData(val_root_dir, bees_label_dir, transform)

# 加载训练集
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
# 加载验证集
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)


if __name__ == "__main__":
    #输出训练集
    for batchidx,(imgs,labels) in enumerate(train_loader):
        print("训练集的第{}个batch,他的shape是:{},他的label是:{}".format((batchidx+1),imgs.shape,labels))
    #输出验证集
    for batchidx,(imgs,labels) in enumerate(val_loader):
        print("验证集的第{}个batch,他的shape是:{},他的label是:{}".format((batchidx+1),imgs.shape,labels))