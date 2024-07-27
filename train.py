import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from models.vanillanet import vanillanet_5 as creatModel
from torch.utils.tensorboard import SummaryWriter
import torchprofile  # Import torchprofile for FLOPS calculation

# 设置随机种子
seed = 42
torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
random.seed(seed)  # 设置 Python 内置的随机种子
np.random.seed(seed)  # 设置 numpy 的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 设备的随机种子
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(seed)
    random.seed(seed)

def print_model_params(model):
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {num_params:,}")

def print_model_flops(model, input_size=(1, 3, 224, 224)):
    dummy_input = torch.randn(input_size).to(next(model.parameters()).device)
    try:
        flops = torchprofile.profile_macs(model, dummy_input)
        print(f"模型FLOPS: {flops / 1e9:.3f} GFLOPS")
    except Exception as e:
        print(f"计算FLOPS时出错: {e}")

def main():
    global best_loss
    best_loss = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    image_path = os.path.join(data_root, "SCUT-FBP5500_1", "SCUT-FBP5500")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    lr = 0.0001
    note = 'vanillanet5_1e-7'
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw,
                                               worker_init_fn=worker_init_fn)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, worker_init_fn=worker_init_fn)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = creatModel(num_classes=5)
    net.to(device)

    print_model_params(net)
    print_model_flops(net, input_size=(batch_size, 3, 224, 224))  # 这里用批量大小调整

    epochs = 60
    best_acc = 0.0

    total_filename = "{}-lr{}-bs{}-{}".format(epochs, lr, batch_size, note)
    log_dir = "./runs/{}".format(total_filename)  # 自定义日志目录名称
    tb_writer = SummaryWriter(log_dir=log_dir)
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)  # 这里用批量大小调整
    tb_writer.add_graph(net, dummy_input)

    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-7)

    if not os.path.exists('./weights'):
        os.makedirs('./weights')

    train_start_time = time.perf_counter()

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        correct_train = 0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predict_y = torch.max(logits, dim=1)[1]
            correct_train += torch.eq(predict_y, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_acc = correct_train / train_num

        net.eval()
        acc = 0.0
        running_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                running_val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_acc = acc / val_num
        running_loss /= len(train_loader)
        running_val_loss /= len(validate_loader)

        tb_writer.add_scalar('train_loss', running_loss, epoch)
        tb_writer.add_scalar('train_acc', train_acc, epoch)
        tb_writer.add_scalar('val_loss', running_val_loss, epoch)
        tb_writer.add_scalar('val_acc', val_acc, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)

        print('[epoch %d] train_loss: %.3f  val_loss: %.3f  train_accuracy: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss, running_val_loss, train_acc, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            model_filename = "./weights/model-{}-acc.pth".format(total_filename)
            torch.save(net.state_dict(), model_filename)
            print(f"Epoch {epoch}: 保存新最佳模型 {model_filename}，验证准确率: {val_acc:.4f}")

        if running_val_loss < best_loss:
            best_loss = running_val_loss
            model_filename = "./weights/model-{}-loss.pth".format(total_filename)
            torch.save(net.state_dict(), model_filename)
            print(f"Epoch {epoch}: 保存新最佳模型 {model_filename}，验证损失: {running_val_loss:.4f}")

        scheduler.step()

    print('Finished Training')

    train_end_time = time.perf_counter()
    elapsed_time = train_end_time - train_start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")
    print(f"训练完成，最佳验证损失: {best_loss:.4f}")
    print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

    tb_writer.close()

if __name__ == '__main__':
    main()
