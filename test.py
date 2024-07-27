import os
import json
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from sklearn.metrics import recall_score, confusion_matrix
import pandas as pd

from models.vanillanet import vanillanet_13 as creatModel

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    imgs_root = r'E:\xwc\pycharm_virtual_cnn\SCUT-FBP5500_1\SCUT-FBP5500\test'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    img_path_list = []  # 存储所有图片的路径
    class_list = {}  # 存储每个类别的图片路径

    # 获取imgs_root下的所有子文件夹
    subdirs = [f.name for f in os.scandir(imgs_root) if f.is_dir()]
    # 遍历子文件夹
    for subdir in subdirs:
        # 构建子文件夹的完整路径
        subdir_path = os.path.join(imgs_root, subdir)
        # 获取子文件夹下的所有.jpg图片路径
        images = [os.path.join(subdir_path, i) for i in os.listdir(subdir_path) if i.endswith(".jpg")]
        # 将图片路径添加到img_path_list
        img_path_list.extend(images)
        # 将图片路径存储在class_list中，以子文件夹名为键
        class_list[subdir] = images

    # read class_indict
    json_path = r'E:\xwc\pycharm_virtual_cnn\deep-learning-for-image-processing-master' \
                r'\pytorch_classification\Test5_resnet\class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    # create model
    model = creatModel(num_classes=5).to(device)
    # model = torchvision.models.resnet34(num_classes=5).to(device)

    # load model weights
    weights_path = r'.\weights\model-300-lr0.0001-bs16-vanillanet13.pth'
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    correct_count = 0  # 用于累计预测正确的数量
    all_preds = []  # 用于存储所有预测结果
    all_labels = []  # 用于存储所有真实标签
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)
            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                img_path = img_path_list[ids * batch_size + idx]
                # 根据图片路径找到其在class_list中的实际类别
                actual_class = None
                for class_name, images in class_list.items():
                    if img_path in images:
                        actual_class = class_name
                        break
                # 比较预测类别和实际类别
                all_preds.append(cla.numpy())
                all_labels.append(int(actual_class))

                if actual_class and class_indict[str(cla.numpy())] == actual_class:
                    correct_count += 1

    # 打印总的预测正确数量
    print("\n总测试数量有：{}，总的预测正确数量: {},准确率为：{}".format(len(img_path_list), correct_count, correct_count/len(img_path_list)))

    # 计算每个类的召回率和宏观平均召回率
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    print("每个类的召回率:", recall_per_class)
    print("宏观平均召回率:", macro_recall)

    # 生成混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("混淆矩阵:\n", conf_matrix)

    # 将测试结果写入Excel文件
    excel_path = 'TestResult.xlsx'
    # 将结果转换为百分比形式，并保留两位小数
    result_data = [f"{x * 100:.2f}" for x in recall_per_class] + [f"{macro_recall * 100:.2f}", f"{(correct_count / len(img_path_list)) * 100:.2f}"]

    if os.path.exists(excel_path):
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            existing_df = pd.read_excel(excel_path)
            columns = existing_df.columns.tolist()
            # 确保 result_data 的长度与现有的列数一致
            if len(result_data) < len(columns):
                result_data.extend([None] * (len(columns) - len(result_data)))
            elif len(result_data) > len(columns):
                raise ValueError("结果数据的列数多于Excel文件中的列数，请检查数据和Excel文件。")
            result_df = pd.DataFrame([result_data], columns=columns)
            result_df.to_excel(writer, index=False, header=False, startrow=len(existing_df) + 1)
    else:
        columns = [f'Recall_Class_{i}' for i in range(len(recall_per_class))] + ['Macro_Recall', 'Accuracy']
        result_df = pd.DataFrame([result_data], columns=columns)
        result_df.to_excel(excel_path, index=False)

    print("测试结果已写入Excel文件")

if __name__ == '__main__':
    main()