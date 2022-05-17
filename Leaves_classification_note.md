# kaggle树叶分类竞赛笔记
在这个笔记中，我将记录[kaggle上树叶分类竞赛](https://www.kaggle.com/c/classify-leaves)的实验过程。

我使用的代码是第5名同学提供的代码（训练部分的代码被注释掉了）：
``` python
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math

import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    cross_val_score,
)

# Metric
from sklearn.metrics import f1_score, accuracy_score

# Augmentation
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# 固定随机种子，保证结果可复现。
seed = 415
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# 把数据读进来，并且把label搞定。
path = "/home/users/XXX/leaves_classification"
labels_file_path = os.path.join(path, "train.csv")
sample_submission_path = os.path.join(path, "test.csv")

df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
labels_unique = df["label"].unique()

le = LabelEncoder()
le.fit(df["label"])
df["label"] = le.transform(df["label"])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
label_inv_map = {v: k for k, v in label_map.items()}


# 数据增强
def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            albumentations.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(p=1.0),
        ]
    )


# 定义Dataset，还有准确率之类的函数。
class LeafDataset(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        self.images_filepaths = images_filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average="macro")


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """adjust learning of a given optimizer and return the new learning rate"""
    new_lr = calc_learning_rate(epoch, params["lr"], params["epochs"], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


""" learning rate schedule """


def calc_learning_rate(
    epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type="cosine"
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr


# 参数设置
# 所有模型都用的相同的参数 其实这里应该针对不同模型调整的
params = {
    "model": "seresnext50_32x4d",
    # 'model': 'resnet50d',
    "device": device,
    "lr": 1e-3,
    "batch_size": 64,
    "num_workers": 0,
    "epochs": 50,
    "out_features": df["label"].nunique(),
    "weight_decay": 1e-5,
}


# 训练
class LeafNet(nn.Module):
    def __init__(
        self,
        model_name=params["model"],
        out_features=params["out_features"],
        pretrained=True,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x


def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        f1_macro = calculate_f1_macro(output, target)
        acc = accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("F1", f1_macro)
        metric_monitor.update("Accuracy", acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor
            )
        )
    return metric_monitor.metrics["Accuracy"]["avg"]


def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            f1_macro = calculate_f1_macro(output, target)
            acc = accuracy(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("F1", f1_macro)
            metric_monitor.update("Accuracy", acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor
                )
            )
    return metric_monitor.metrics["Accuracy"]["avg"]


# kf = StratifiedKFold(n_splits=5)
# for k, (train_index, test_index) in enumerate(kf.split(df["image"], df["label"])):
#     train_img, valid_img = df["image"][train_index], df["image"][test_index]
#     train_labels, valid_labels = df["label"][train_index], df["label"][test_index]

#     train_paths = path + "/" + train_img
#     valid_paths = path + "/" + valid_img
#     test_paths = path + "/" + sub_df["image"]

#     train_dataset = LeafDataset(
#         images_filepaths=train_paths.values,
#         labels=train_labels.values,
#         transform=get_train_transforms(),
#     )
#     valid_dataset = LeafDataset(
#         images_filepaths=valid_paths.values,
#         labels=valid_labels.values,
#         transform=get_valid_transforms(),
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=params["batch_size"],
#         shuffle=True,
#         num_workers=params["num_workers"],
#         pin_memory=True,
#     )

#     val_loader = DataLoader(
#         valid_dataset,
#         batch_size=params["batch_size"],
#         shuffle=False,
#         num_workers=params["num_workers"],
#         pin_memory=True,
#     )
#     model = LeafNet()
#     model = nn.DataParallel(model)
#     model = model.to(params["device"])
#     criterion = nn.CrossEntropyLoss().to(params["device"])
#     optimizer = torch.optim.AdamW(
#         model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
#     )

#     for epoch in range(1, params["epochs"] + 1):
#         train(train_loader, model, criterion, optimizer, epoch, params)
#         acc = validate(val_loader, model, criterion, epoch, params)
#         torch.save(
#             model.state_dict(),
#             f"./checkpoints/{params['model']}_{k}flod_{epoch}epochs_accuracy{acc:.5f}_weights.pth",
#         )

# exit()

# 测试和提交
# 提交用的代码，用了两个模型seresnext50_32x4d和resnet50d。
train_img, valid_img = df["image"], df["image"]
train_labels, valid_labels = df["label"], df["label"]

train_paths = path + "/" + train_img
valid_paths = path + "/" + valid_img
test_paths = path + "/" + sub_df["image"]

# model_name = ["seresnext50_32x4d", "resnet50d"]
model_name = ["seresnext50_32x4d"]
model_path_list = [
    "./checkpoints/seresnext50_32x4d_3flod_31epochs_accuracy0.98114_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_35epochs_accuracy0.98006_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_40epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_42epochs_accuracy0.98033_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_43epochs_accuracy0.98033_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_44epochs_accuracy0.98087_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_45epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_46epochs_accuracy0.98114_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_47epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_48epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_49epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_50epochs_accuracy0.98141_weights.pth",
]

model_list = []
for i in range(len(model_path_list)):
    # if i < 5:
    #     model_list.append(LeafNet(model_name[0]))
    # if 5 <= i < 10:
    #     model_list.append(LeafNet(model_name[1]))
    model_list.append(LeafNet(model_name[0]))
    model_list[i] = nn.DataParallel(model_list[i])
    model_list[i] = model_list[i].to(params["device"])
    # print("----------------------开始监视代码----------------------")
    # print("model_path_list[i]: ", model_path_list[i])
    # print("----------------------结束监视代码----------------------")
    # exit()
    init = torch.load(model_path_list[i])
    model_list[i].load_state_dict(init)
    model_list[i].eval()
    model_list[i].cuda()


labels = np.zeros(len(test_paths))  # Fake Labels
test_dataset = LeafDataset(
    images_filepaths=test_paths, labels=labels, transform=get_valid_transforms()
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=10, pin_memory=True
)


predicted_labels = []
pred_string = []
preds = []

with torch.no_grad():
    for (images, target) in test_loader:
        images = images.cuda()
        onehots = sum([model(images) for model in model_list]) / len(model_list)
        for oh, name in zip(onehots, target):
            lbs = label_inv_map[torch.argmax(oh).item()]
            preds.append(dict(image=name, labels=lbs))

df_preds = pd.DataFrame(preds)
sub_df["label"] = df_preds["labels"]
sub_df.to_csv("submission.csv", index=False)
sub_df.head()
```
接下来将对他的代码做一些详细的分析。

## 对上述代码的详细分析
### 导入包
上述代码最开始的部分是导入各种包：
``` python
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
import math

import torch
import torchvision
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    cross_val_score,
)

# Metric
from sklearn.metrics import f1_score, accuracy_score

# Augmentation
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
```
这部分没有什么好说的。主要是一些库我没有用过。之后多学习一些优秀的开源代码，多从优秀的开源代码中学习各种常用库的使用方法。

### 固定随机种子
接下来是固定随机种子，保证结果可复现的代码：
``` python
# 固定随机种子，保证结果可复现。
seed = 415
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
```
这部分代码也没有什么好说的，是一些固定的用法，只需要记住或者直接复用已有的开源代码即可。

### 读取数据并搞定标签
接下来是读取数据并搞定标签部分的代码：
``` python
# 把数据读进来，并且把label搞定。
path = "/home/users/XXX/leaves_classification"
labels_file_path = os.path.join(path, "train.csv")
sample_submission_path = os.path.join(path, "test.csv")

df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
labels_unique = df["label"].unique()

le = LabelEncoder()
le.fit(df["label"])
df["label"] = le.transform(df["label"])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
label_inv_map = {v: k for k, v in label_map.items()}
```
首先，这段代码最开始用了`os.path.join()`函数来拼接路径。我们在空白脚本里测试下述代码：
``` python
import os

a = "aaaaa"
b = "bbbbb"
c = os.path.join(a, b)

print(c)
```
得到的结果为：
```
aaaaa/bbbbb
```
由此就明白了：`os.path.join()`函数实现的是拼接路径的功能。

接下来两行代码在用pandas库读取数据：
``` python
df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
```
我们来测试下述代码：
``` python
df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
print("----------------------开始监视代码----------------------")
print("df: ", df)
print("----------------------我的分割线1----------------------")
print("sub_df: ", sub_df)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
df:                    image                    label
0          images/0.jpg         maclura_pomifera
1          images/1.jpg         maclura_pomifera
2          images/2.jpg         maclura_pomifera
3          images/3.jpg         maclura_pomifera
4          images/4.jpg         maclura_pomifera
...                 ...                      ...
18348  images/18348.jpg          aesculus_glabra
18349  images/18349.jpg  liquidambar_styraciflua
18350  images/18350.jpg            cedrus_libani
18351  images/18351.jpg      prunus_pensylvanica
18352  images/18352.jpg          quercus_montana

[18353 rows x 2 columns]
----------------------我的分割线1----------------------
sub_df:                   image
0     images/18353.jpg
1     images/18354.jpg
2     images/18355.jpg
3     images/18356.jpg
4     images/18357.jpg
...                ...
8795  images/27148.jpg
8796  images/27149.jpg
8797  images/27150.jpg
8798  images/27151.jpg
8799  images/27152.jpg

[8800 rows x 1 columns]
----------------------结束监视代码----------------------
```
由此可知：`pandas`库可以用来读取`.csv`文件。

接下来的一行代码是为了获得唯一的标签：
``` python
labels_unique = df["label"].unique()
```
我们来测试下述代码：
``` python
labels_unique = df["label"].unique()
print("----------------------开始监视代码----------------------")
print('df["label"]: ', df["label"])
print("----------------------我的分割线1----------------------")
print("labels_unique: ", labels_unique)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
df["label"]:  0               maclura_pomifera
1               maclura_pomifera
2               maclura_pomifera
3               maclura_pomifera
4               maclura_pomifera
                  ...           
18348            aesculus_glabra
18349    liquidambar_styraciflua
18350              cedrus_libani
18351        prunus_pensylvanica
18352            quercus_montana
Name: label, Length: 18353, dtype: object
----------------------我的分割线1----------------------
labels_unique:  ['maclura_pomifera' 'ulmus_rubra' 'broussonettia_papyrifera'
 'prunus_virginiana' 'acer_rubrum' 'cryptomeria_japonica'
 'staphylea_trifolia' 'asimina_triloba' 'diospyros_virginiana'
 'tilia_cordata' 'ulmus_pumila' 'quercus_muehlenbergii' 'juglans_cinerea'
 'cercis_canadensis' 'ptelea_trifoliata' 'acer_palmatum'
 'catalpa_speciosa' 'abies_concolor' 'eucommia_ulmoides' 'quercus_montana'
 'koelreuteria_paniculata' 'liriodendron_tulipifera' 'styrax_japonica'
 'malus_pumila' 'prunus_sargentii' 'cornus_mas' 'magnolia_virginiana'
 'ostrya_virginiana' 'magnolia_acuminata' 'ilex_opaca' 'acer_negundo'
 'fraxinus_nigra' 'pyrus_calleryana' 'picea_abies'
 'chionanthus_virginicus' 'carpinus_caroliniana' 'zelkova_serrata'
 'aesculus_pavi' 'taxodium_distichum' 'carya_tomentosa' 'picea_pungens'
 'carya_glabra' 'quercus_macrocarpa' 'carya_cordiformis'
 'catalpa_bignonioides' 'tsuga_canadensis' 'populus_tremuloides'
 'magnolia_denudata' 'crataegus_viridis' 'populus_deltoides'
 'ulmus_americana' 'pinus_bungeana' 'cornus_florida' 'pinus_densiflora'
 'morus_alba' 'quercus_velutina' 'pinus_parviflora' 'salix_caroliniana'
 'platanus_occidentalis' 'acer_saccharum' 'pinus_flexilis'
 'gleditsia_triacanthos' 'quercus_alba' 'prunus_subhirtella'
 'pseudolarix_amabilis' 'stewartia_pseudocamellia' 'quercus_stellata'
 'pinus_rigida' 'salix_nigra' 'quercus_acutissima' 'pinus_virginiana'
 'chamaecyparis_pisifera' 'quercus_michauxii' 'prunus_pensylvanica'
 'amelanchier_canadensis' 'liquidambar_styraciflua' 'pinus_cembra'
 'malus_hupehensis' 'castanea_dentata' 'magnolia_stellata'
 'chionanthus_retusus' 'carya_ovata' 'quercus_marilandica'
 'tilia_americana' 'cedrus_atlantica' 'ulmus_parvifolia' 'nyssa_sylvatica'
 'quercus_virginiana' 'acer_saccharinum' 'magnolia_macrophylla'
 'crataegus_pruinosa' 'pinus_nigra' 'abies_nordmanniana' 'pinus_taeda'
 'ficus_carica' 'pinus_peucea' 'populus_grandidentata' 'acer_platanoides'
 'pinus_resinosa' 'salix_matsudana' 'pinus_sylvestris'
 'albizia_julibrissin' 'salix_babylonica' 'pinus_echinata'
 'magnolia_tripetala' 'larix_decidua' 'pinus_strobus' 'aesculus_glabra'
 'ginkgo_biloba' 'quercus_cerris' 'metasequoia_glyptostroboides'
 'fagus_grandifolia' 'quercus_nigra' 'juglans_nigra' 'pinus_koraiensis'
 'oxydendrum_arboreum' 'morus_rubra' 'crataegus_phaenopyrum'
 'pinus_wallichiana' 'tilia_europaea' 'betula_jacqemontii'
 'chamaecyparis_thyoides' 'acer_ginnala' 'acer_campestre' 'pinus_pungens'
 'malus_floribunda' 'picea_orientalis' 'amelanchier_laevis'
 'celtis_tenuifolia' 'gymnocladus_dioicus' 'quercus_bicolor'
 'malus_coronaria' 'cercidiphyllum_japonicum' 'cedrus_libani'
 'betula_nigra' 'acer_pensylvanicum' 'platanus_acerifolia'
 'robinia_pseudo-acacia' 'ulmus_glabra' 'crataegus_laevigata'
 'quercus_coccinea' 'prunus_serotina' 'tilia_tomentosa'
 'quercus_imbricaria' 'cladrastis_lutea' 'fraxinus_pennsylvanica'
 'phellodendron_amurense' 'betula_lenta' 'quercus_robur' 'aesculus_flava'
 'paulownia_tomentosa' 'amelanchier_arborea' 'quercus_shumardii'
 'magnolia_grandiflora' 'cornus_kousa' 'betula_alleghaniensis'
 'carpinus_betulus' 'aesculus_hippocastamon' 'malus_baccata'
 'acer_pseudoplatanus' 'betula_populifolia' 'prunus_yedoensis'
 'halesia_tetraptera' 'quercus_palustris' 'evodia_daniellii'
 'ulmus_procera' 'prunus_serrulata' 'quercus_phellos' 'cedrus_deodara'
 'celtis_occidentalis' 'sassafras_albidum' 'acer_griseum'
 'ailanthus_altissima' 'pinus_thunbergii' 'crataegus_crus-galli'
 'juniperus_virginiana']
----------------------结束监视代码----------------------
```
这就是说，`labels_unique = df["label"].unique()`这行代码获得了唯一的树叶标签。

接下来的三行代码对树叶的类别进行了重新编码，把树叶的标签编码为了一个唯一的整数。我们来测试下述代码：
``` python
le = LabelEncoder()
le.fit(df["label"])
df["label"] = le.transform(df["label"])
print("----------------------开始监视代码----------------------")
print('df["label"]: ', df["label"])
print("----------------------我的分割线1----------------------")
print('sorted(df["label"].unique()): ', sorted(df["label"].unique()))
print("----------------------结束监视代码----------------------")
exit()
```
得到的结果为：
```
----------------------开始监视代码----------------------
df["label"]:  0         78
1         78
2         78
3         78
4         78
        ... 
18348     14
18349     76
18350     40
18351    125
18352    144
Name: label, Length: 18353, dtype: int64
----------------------我的分割线1----------------------
sorted(df["label"].unique()):  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
----------------------结束监视代码----------------------
```
正好就是本次树叶分类竞赛的176类树叶。

接下来的两行代码得到了标签和序号的对应字典（两种顺序都有）。我们来测试下述代码：
``` python
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
label_inv_map = {v: k for k, v in label_map.items()}
print("----------------------开始监视代码----------------------")
print("label_map: ", label_map)
print("----------------------我的分割线1----------------------")
print("label_inv_map: ", label_inv_map)
print("----------------------结束监视代码----------------------")
exit()
```
结果为：
```
----------------------开始监视代码----------------------
label_map:  {'abies_concolor': 0, 'abies_nordmanniana': 1, 'acer_campestre': 2, 'acer_ginnala': 3, 'acer_griseum': 4, 'acer_negundo': 5, 'acer_palmatum': 6, 'acer_pensylvanicum': 7, 'acer_platanoides': 8, 'acer_pseudoplatanus': 9, 'acer_rubrum': 10, 'acer_saccharinum': 11, 'acer_saccharum': 12, 'aesculus_flava': 13, 'aesculus_glabra': 14, 'aesculus_hippocastamon': 15, 'aesculus_pavi': 16, 'ailanthus_altissima': 17, 'albizia_julibrissin': 18, 'amelanchier_arborea': 19, 'amelanchier_canadensis': 20, 'amelanchier_laevis': 21, 'asimina_triloba': 22, 'betula_alleghaniensis': 23, 'betula_jacqemontii': 24, 'betula_lenta': 25, 'betula_nigra': 26, 'betula_populifolia': 27, 'broussonettia_papyrifera': 28, 'carpinus_betulus': 29, 'carpinus_caroliniana': 30, 'carya_cordiformis': 31, 'carya_glabra': 32, 'carya_ovata': 33, 'carya_tomentosa': 34, 'castanea_dentata': 35, 'catalpa_bignonioides': 36, 'catalpa_speciosa': 37, 'cedrus_atlantica': 38, 'cedrus_deodara': 39, 'cedrus_libani': 40, 'celtis_occidentalis': 41, 'celtis_tenuifolia': 42, 'cercidiphyllum_japonicum': 43, 'cercis_canadensis': 44, 'chamaecyparis_pisifera': 45, 'chamaecyparis_thyoides': 46, 'chionanthus_retusus': 47, 'chionanthus_virginicus': 48, 'cladrastis_lutea': 49, 'cornus_florida': 50, 'cornus_kousa': 51, 'cornus_mas': 52, 'crataegus_crus-galli': 53, 'crataegus_laevigata': 54, 'crataegus_phaenopyrum': 55, 'crataegus_pruinosa': 56, 'crataegus_viridis': 57, 'cryptomeria_japonica': 58, 'diospyros_virginiana': 59, 'eucommia_ulmoides': 60, 'evodia_daniellii': 61, 'fagus_grandifolia': 62, 'ficus_carica': 63, 'fraxinus_nigra': 64, 'fraxinus_pennsylvanica': 65, 'ginkgo_biloba': 66, 'gleditsia_triacanthos': 67, 'gymnocladus_dioicus': 68, 'halesia_tetraptera': 69, 'ilex_opaca': 70, 'juglans_cinerea': 71, 'juglans_nigra': 72, 'juniperus_virginiana': 73, 'koelreuteria_paniculata': 74, 'larix_decidua': 75, 'liquidambar_styraciflua': 76, 'liriodendron_tulipifera': 77, 'maclura_pomifera': 78, 'magnolia_acuminata': 79, 'magnolia_denudata': 80, 'magnolia_grandiflora': 81, 'magnolia_macrophylla': 82, 'magnolia_stellata': 83, 'magnolia_tripetala': 84, 'magnolia_virginiana': 85, 'malus_baccata': 86, 'malus_coronaria': 87, 'malus_floribunda': 88, 'malus_hupehensis': 89, 'malus_pumila': 90, 'metasequoia_glyptostroboides': 91, 'morus_alba': 92, 'morus_rubra': 93, 'nyssa_sylvatica': 94, 'ostrya_virginiana': 95, 'oxydendrum_arboreum': 96, 'paulownia_tomentosa': 97, 'phellodendron_amurense': 98, 'picea_abies': 99, 'picea_orientalis': 100, 'picea_pungens': 101, 'pinus_bungeana': 102, 'pinus_cembra': 103, 'pinus_densiflora': 104, 'pinus_echinata': 105, 'pinus_flexilis': 106, 'pinus_koraiensis': 107, 'pinus_nigra': 108, 'pinus_parviflora': 109, 'pinus_peucea': 110, 'pinus_pungens': 111, 'pinus_resinosa': 112, 'pinus_rigida': 113, 'pinus_strobus': 114, 'pinus_sylvestris': 115, 'pinus_taeda': 116, 'pinus_thunbergii': 117, 'pinus_virginiana': 118, 'pinus_wallichiana': 119, 'platanus_acerifolia': 120, 'platanus_occidentalis': 121, 'populus_deltoides': 122, 'populus_grandidentata': 123, 'populus_tremuloides': 124, 'prunus_pensylvanica': 125, 'prunus_sargentii': 126, 'prunus_serotina': 127, 'prunus_serrulata': 128, 'prunus_subhirtella': 129, 'prunus_virginiana': 130, 'prunus_yedoensis': 131, 'pseudolarix_amabilis': 132, 'ptelea_trifoliata': 133, 'pyrus_calleryana': 134, 'quercus_acutissima': 135, 'quercus_alba': 136, 'quercus_bicolor': 137, 'quercus_cerris': 138, 'quercus_coccinea': 139, 'quercus_imbricaria': 140, 'quercus_macrocarpa': 141, 'quercus_marilandica': 142, 'quercus_michauxii': 143, 'quercus_montana': 144, 'quercus_muehlenbergii': 145, 'quercus_nigra': 146, 'quercus_palustris': 147, 'quercus_phellos': 148, 'quercus_robur': 149, 'quercus_shumardii': 150, 'quercus_stellata': 151, 'quercus_velutina': 152, 'quercus_virginiana': 153, 'robinia_pseudo-acacia': 154, 'salix_babylonica': 155, 'salix_caroliniana': 156, 'salix_matsudana': 157, 'salix_nigra': 158, 'sassafras_albidum': 159, 'staphylea_trifolia': 160, 'stewartia_pseudocamellia': 161, 'styrax_japonica': 162, 'taxodium_distichum': 163, 'tilia_americana': 164, 'tilia_cordata': 165, 'tilia_europaea': 166, 'tilia_tomentosa': 167, 'tsuga_canadensis': 168, 'ulmus_americana': 169, 'ulmus_glabra': 170, 'ulmus_parvifolia': 171, 'ulmus_procera': 172, 'ulmus_pumila': 173, 'ulmus_rubra': 174, 'zelkova_serrata': 175}
----------------------我的分割线1----------------------
label_inv_map:  {0: 'abies_concolor', 1: 'abies_nordmanniana', 2: 'acer_campestre', 3: 'acer_ginnala', 4: 'acer_griseum', 5: 'acer_negundo', 6: 'acer_palmatum', 7: 'acer_pensylvanicum', 8: 'acer_platanoides', 9: 'acer_pseudoplatanus', 10: 'acer_rubrum', 11: 'acer_saccharinum', 12: 'acer_saccharum', 13: 'aesculus_flava', 14: 'aesculus_glabra', 15: 'aesculus_hippocastamon', 16: 'aesculus_pavi', 17: 'ailanthus_altissima', 18: 'albizia_julibrissin', 19: 'amelanchier_arborea', 20: 'amelanchier_canadensis', 21: 'amelanchier_laevis', 22: 'asimina_triloba', 23: 'betula_alleghaniensis', 24: 'betula_jacqemontii', 25: 'betula_lenta', 26: 'betula_nigra', 27: 'betula_populifolia', 28: 'broussonettia_papyrifera', 29: 'carpinus_betulus', 30: 'carpinus_caroliniana', 31: 'carya_cordiformis', 32: 'carya_glabra', 33: 'carya_ovata', 34: 'carya_tomentosa', 35: 'castanea_dentata', 36: 'catalpa_bignonioides', 37: 'catalpa_speciosa', 38: 'cedrus_atlantica', 39: 'cedrus_deodara', 40: 'cedrus_libani', 41: 'celtis_occidentalis', 42: 'celtis_tenuifolia', 43: 'cercidiphyllum_japonicum', 44: 'cercis_canadensis', 45: 'chamaecyparis_pisifera', 46: 'chamaecyparis_thyoides', 47: 'chionanthus_retusus', 48: 'chionanthus_virginicus', 49: 'cladrastis_lutea', 50: 'cornus_florida', 51: 'cornus_kousa', 52: 'cornus_mas', 53: 'crataegus_crus-galli', 54: 'crataegus_laevigata', 55: 'crataegus_phaenopyrum', 56: 'crataegus_pruinosa', 57: 'crataegus_viridis', 58: 'cryptomeria_japonica', 59: 'diospyros_virginiana', 60: 'eucommia_ulmoides', 61: 'evodia_daniellii', 62: 'fagus_grandifolia', 63: 'ficus_carica', 64: 'fraxinus_nigra', 65: 'fraxinus_pennsylvanica', 66: 'ginkgo_biloba', 67: 'gleditsia_triacanthos', 68: 'gymnocladus_dioicus', 69: 'halesia_tetraptera', 70: 'ilex_opaca', 71: 'juglans_cinerea', 72: 'juglans_nigra', 73: 'juniperus_virginiana', 74: 'koelreuteria_paniculata', 75: 'larix_decidua', 76: 'liquidambar_styraciflua', 77: 'liriodendron_tulipifera', 78: 'maclura_pomifera', 79: 'magnolia_acuminata', 80: 'magnolia_denudata', 81: 'magnolia_grandiflora', 82: 'magnolia_macrophylla', 83: 'magnolia_stellata', 84: 'magnolia_tripetala', 85: 'magnolia_virginiana', 86: 'malus_baccata', 87: 'malus_coronaria', 88: 'malus_floribunda', 89: 'malus_hupehensis', 90: 'malus_pumila', 91: 'metasequoia_glyptostroboides', 92: 'morus_alba', 93: 'morus_rubra', 94: 'nyssa_sylvatica', 95: 'ostrya_virginiana', 96: 'oxydendrum_arboreum', 97: 'paulownia_tomentosa', 98: 'phellodendron_amurense', 99: 'picea_abies', 100: 'picea_orientalis', 101: 'picea_pungens', 102: 'pinus_bungeana', 103: 'pinus_cembra', 104: 'pinus_densiflora', 105: 'pinus_echinata', 106: 'pinus_flexilis', 107: 'pinus_koraiensis', 108: 'pinus_nigra', 109: 'pinus_parviflora', 110: 'pinus_peucea', 111: 'pinus_pungens', 112: 'pinus_resinosa', 113: 'pinus_rigida', 114: 'pinus_strobus', 115: 'pinus_sylvestris', 116: 'pinus_taeda', 117: 'pinus_thunbergii', 118: 'pinus_virginiana', 119: 'pinus_wallichiana', 120: 'platanus_acerifolia', 121: 'platanus_occidentalis', 122: 'populus_deltoides', 123: 'populus_grandidentata', 124: 'populus_tremuloides', 125: 'prunus_pensylvanica', 126: 'prunus_sargentii', 127: 'prunus_serotina', 128: 'prunus_serrulata', 129: 'prunus_subhirtella', 130: 'prunus_virginiana', 131: 'prunus_yedoensis', 132: 'pseudolarix_amabilis', 133: 'ptelea_trifoliata', 134: 'pyrus_calleryana', 135: 'quercus_acutissima', 136: 'quercus_alba', 137: 'quercus_bicolor', 138: 'quercus_cerris', 139: 'quercus_coccinea', 140: 'quercus_imbricaria', 141: 'quercus_macrocarpa', 142: 'quercus_marilandica', 143: 'quercus_michauxii', 144: 'quercus_montana', 145: 'quercus_muehlenbergii', 146: 'quercus_nigra', 147: 'quercus_palustris', 148: 'quercus_phellos', 149: 'quercus_robur', 150: 'quercus_shumardii', 151: 'quercus_stellata', 152: 'quercus_velutina', 153: 'quercus_virginiana', 154: 'robinia_pseudo-acacia', 155: 'salix_babylonica', 156: 'salix_caroliniana', 157: 'salix_matsudana', 158: 'salix_nigra', 159: 'sassafras_albidum', 160: 'staphylea_trifolia', 161: 'stewartia_pseudocamellia', 162: 'styrax_japonica', 163: 'taxodium_distichum', 164: 'tilia_americana', 165: 'tilia_cordata', 166: 'tilia_europaea', 167: 'tilia_tomentosa', 168: 'tsuga_canadensis', 169: 'ulmus_americana', 170: 'ulmus_glabra', 171: 'ulmus_parvifolia', 172: 'ulmus_procera', 173: 'ulmus_pumila', 174: 'ulmus_rubra', 175: 'zelkova_serrata'}
----------------------结束监视代码----------------------
```
这就是按两种顺序重新给标签赋值的两个字典，留作以后备用。

### 数据增强
接下来是数据增强部分的代码：
``` python
def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            albumentations.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            ToTensorV2(p=1.0),
        ]
    )
```
这里的数据增强使用了第三方开源库`albumentations`来进行数据增强。

### 数据集、准确率、学习率策略等函数
接下来是跟数据集、准确率和学习率策略有关的代码：
``` python
# 定义Dataset，还有准确率之类的函数。
class LeafDataset(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        self.images_filepaths = images_filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average="macro")


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )


def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """adjust learning of a given optimizer and return the new learning rate"""
    new_lr = calc_learning_rate(epoch, params["lr"], params["epochs"], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


""" learning rate schedule """


def calc_learning_rate(
    epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type="cosine"
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr
```
我们首先看数据集部分的代码：
``` python
class LeafDataset(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        self.images_filepaths = images_filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
```
数据集的代码，整体来看很容易理解，没有特别复杂的用法。

接下来是准确度和`f1`分数的函数：
``` python
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average="macro")
```
这两个函数就是固定的用法，没有什么特别的地方，记住就好，或者，以后直接复用别人的开源代码中的相应函数就好。

接下来定义了一个度量监测器的类：
``` python
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name,
                    avg=metric["avg"],
                    float_precision=self.float_precision,
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
```
关于这个类的具体代码，之后用到的时候再来分析。

接下来是调整学习率的代码：
``` python
def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """adjust learning of a given optimizer and return the new learning rate"""
    new_lr = calc_learning_rate(epoch, params["lr"], params["epochs"], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


""" learning rate schedule """


def calc_learning_rate(
    epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type="cosine"
):
    if lr_schedule_type == "cosine":
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError("do not support: %s" % lr_schedule_type)
    return lr
```
这部分的代码整体来看也没有特别复杂的用法，在实际中我只需要复用开源代码中的这些写法即可。

### 参数设置
接下来是参数设置的代码：
``` python
# 参数设置
# 所有模型都用的相同的参数 其实这里应该针对不同模型调整的
params = {
    "model": "seresnext50_32x4d",
    # 'model': 'resnet50d',
    "device": device,
    "lr": 1e-3,
    "batch_size": 64,
    "num_workers": 0,
    "epochs": 50,
    "out_features": df["label"].nunique(),
    "weight_decay": 1e-5,
}
```
参数是在一个字典里统一设置的。这样很方便。

### 创建模型
接下来是创建模型的代码：
``` python
# 训练
class LeafNet(nn.Module):
    def __init__(
        self,
        model_name=params["model"],
        out_features=params["out_features"],
        pretrained=True,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x
```
模型类是`LeafNet`类。需要注意的是，模型是使用`timm.create_model()`函数直接创建的。一般来说，我不会自己手写神经网络，而是直接使用最新的开源实现来构建神经网络。

### 训练函数
接下来是训练函数的代码：
``` python
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        f1_macro = calculate_f1_macro(output, target)
        acc = accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("F1", f1_macro)
        metric_monitor.update("Accuracy", acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(
                epoch=epoch, metric_monitor=metric_monitor
            )
        )
    return metric_monitor.metrics["Accuracy"]["avg"]
```
训练函数的代码整体上来说，还是比较规范的。基本上就是常规的训练代码的写法。在这里我就不逐行分析了，之后如果需要调整哪里，再详细分析。

### 验证函数
验证函数的代码如下：
``` python
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            f1_macro = calculate_f1_macro(output, target)
            acc = accuracy(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("F1", f1_macro)
            metric_monitor.update("Accuracy", acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor
                )
            )
    return metric_monitor.metrics["Accuracy"]["avg"]
```
验证函数的代码和训练函数的代码非常类似。我也不详细地分析了。

### 训练过程正式开始
接下来是整个的训练过程的代码：
``` python
kf = StratifiedKFold(n_splits=5)
for k, (train_index, test_index) in enumerate(kf.split(df["image"], df["label"])):
    train_img, valid_img = df["image"][train_index], df["image"][test_index]
    train_labels, valid_labels = df["label"][train_index], df["label"][test_index]

    train_paths = path + "/" + train_img
    valid_paths = path + "/" + valid_img
    test_paths = path + "/" + sub_df["image"]

    train_dataset = LeafDataset(
        images_filepaths=train_paths.values,
        labels=train_labels.values,
        transform=get_train_transforms(),
    )
    valid_dataset = LeafDataset(
        images_filepaths=valid_paths.values,
        labels=valid_labels.values,
        transform=get_valid_transforms(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    model = LeafNet()
    model = nn.DataParallel(model)
    model = model.to(params["device"])
    criterion = nn.CrossEntropyLoss().to(params["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )

    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        acc = validate(val_loader, model, criterion, epoch, params)
        torch.save(
            model.state_dict(),
            f"./checkpoints/{params['model']}_{k}flod_{epoch}epochs_accuracy{acc:.5f}_weights.pth",
        )
```
训练代码中我不太熟悉的地方在于那个k折交叉验证。这个部分我暂且先不深入分析了。之后学习一下李沐老师的相关课程，再来深入探讨这个地方。

最后就是测试和提交部分的代码：
``` python
# 测试和提交
# 提交用的代码，用了两个模型seresnext50_32x4d和resnet50d。
train_img, valid_img = df["image"], df["image"]
train_labels, valid_labels = df["label"], df["label"]

train_paths = path + "/" + train_img
valid_paths = path + "/" + valid_img
test_paths = path + "/" + sub_df["image"]

# model_name = ["seresnext50_32x4d", "resnet50d"]
model_name = ["seresnext50_32x4d"]
model_path_list = [
    "./checkpoints/seresnext50_32x4d_3flod_31epochs_accuracy0.98114_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_35epochs_accuracy0.98006_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_40epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_42epochs_accuracy0.98033_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_43epochs_accuracy0.98033_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_44epochs_accuracy0.98087_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_45epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_46epochs_accuracy0.98114_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_47epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_48epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_49epochs_accuracy0.98060_weights.pth",
    "./checkpoints/seresnext50_32x4d_3flod_50epochs_accuracy0.98141_weights.pth",
]

model_list = []
for i in range(len(model_path_list)):
    # if i < 5:
    #     model_list.append(LeafNet(model_name[0]))
    # if 5 <= i < 10:
    #     model_list.append(LeafNet(model_name[1]))
    model_list.append(LeafNet(model_name[0]))
    model_list[i] = nn.DataParallel(model_list[i])
    model_list[i] = model_list[i].to(params["device"])
    # print("----------------------开始监视代码----------------------")
    # print("model_path_list[i]: ", model_path_list[i])
    # print("----------------------结束监视代码----------------------")
    # exit()
    init = torch.load(model_path_list[i])
    model_list[i].load_state_dict(init)
    model_list[i].eval()
    model_list[i].cuda()


labels = np.zeros(len(test_paths))  # Fake Labels
test_dataset = LeafDataset(
    images_filepaths=test_paths, labels=labels, transform=get_valid_transforms()
)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=10, pin_memory=True
)


predicted_labels = []
pred_string = []
preds = []

with torch.no_grad():
    for (images, target) in test_loader:
        images = images.cuda()
        onehots = sum([model(images) for model in model_list]) / len(model_list)
        for oh, name in zip(onehots, target):
            lbs = label_inv_map[torch.argmax(oh).item()]
            preds.append(dict(image=name, labels=lbs))

df_preds = pd.DataFrame(preds)
sub_df["label"] = df_preds["labels"]
sub_df.to_csv("submission.csv", index=False)
sub_df.head()
```
这段代码整体来说也是比较常规的。唯一需要注意的是，多卡训练的模型，评测的时候也必须加上多卡的部分（意即，用`nn.DataParallel(model_list[i])`来包裹模型），否则，会出现模型不一致的错误。

至此，树叶分类竞赛的代码就初步分析完了。以后，这段代码我还可以复用在其他的分类相关任务上。