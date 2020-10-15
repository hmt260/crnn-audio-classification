import torch
from collections import OrderedDict
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import net as net_module
import data as data_module
from eval import ClassificationEvaluator, AudioInference


# 获取配置中模型参数
def get_model_attr(checkpoint):
    model_name = checkpoint['config']['model']['type']
    state_dict = checkpoint['state_dict']
    classes = checkpoint['classes']
    return model_name, state_dict, classes


# 获取配置中数据变换方式
def get_transform(config, name):
    tsf_name = config['transforms']['type']
    tsf_args = config['transforms']['args']
    return getattr(data_module, tsf_name)(name, tsf_args)


def infer(file_path, checkpoint_path):
    """
    传入
    file_path: 需要推理的文件绝对路径
    checkpoint_path: 推理使用的模型保存点文件绝对路径
    返回
    label: 预测类别
    conf: 预测置信度
    """
    # 异常处理，保证传入路径存在音频文件
    assert os.path.isfile(file_path), "需要推理的文件不存在!"
    assert os.path.isfile(checkpoint_path), "推理使用的模型不存在!"
    assert file_path.endswith(".wav"), "仅支持推理wav文件!"
    # 保存点路径
    checkpoint = torch.load(checkpoint_path)
    # 模型配置
    config = checkpoint["config"]
    # 模型名称，参数，类别
    model_name, state_dict, classes = get_model_attr(checkpoint)
    ordered_classes = OrderedDict(classes)
    class_list = [k for k in ordered_classes.keys()]
    model = getattr(net_module, model_name)(classes, config, state_dict)
    # 加载训练好的模型
    model.load_state_dict(checkpoint['state_dict'])
    # 数据变换方法
    transform = get_transform(config, 'val')
    # 构造推理器
    inference = AudioInference(model, transforms=transform)
    # 执行推理
    label, conf = inference.infer(file_path)
    label = class_list[label]
    print(f"预测类别: {label}, 预测置信度: {conf}")
    return label, conf


# 获取每次训练得到的最佳模型的文件路径
def get_model_paths():
    model_paths = []
    for train_time in os.listdir("./saved_cv"):
        model_paths.append(os.path.abspath(os.path.join("./saved_cv", train_time, "checkpoints", "model_best.pth")))
    return sorted(model_paths)


if __name__ == "__main__":
    # audio_path = "/Users/admin/Public/数据/多分类样本/003/003__03.wav"
    # model_path = "/Users/admin/PycharmProjects/crnn-audio-classification/saved_cv/1015_175032/checkpoints/model_best.pth"
    # infer(audio_path, model_path)

    print(get_model_paths())
