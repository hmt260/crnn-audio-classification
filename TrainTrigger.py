import torch
import json
from run import train_main
from DataProvider import get_transform


# 在模块内初始化为全局对象，在其他模块引入对象，是天然的单例模式
class TrainMonitor:
    def __init__(self, config=None, model_config=None, resume=None):
        self.config = config
        self.model_config = model_config
        self.resume = resume

    def train(self):
        if self.resume:
            checkpoint = torch.load(self.resume)
            self.resume_config = checkpoint['config']
            with open("train_status.log", "w", encoding="utf-8") as f:
                f.write("开始")
            try:
                train_main(self.resume_config, self.resume)
            # 处理一下手动打断训练的情况
            except KeyboardInterrupt:
                pass
            with open("train_status.log", "w", encoding="utf-8") as f:
                    f.write("结束")
        else:
            with open(self.config, "r") as f:
                config = json.load(f)
                config['cfg'] = self.model_config
            with open("train_status.log", "w", encoding="utf-8") as f:
                f.write("开始")
            try:
                train_main(config, None)
            # 处理一下手动打断训练的情况
            except KeyboardInterrupt:
                pass
            with open("train_status.log", "w", encoding="utf-8") as f:
                f.write("结束")
    
    @property
    def train_status(self):
        with open("train_status.log") as f:
            content = f.readlines()
        if "开始" not in content:
            return False
        if "开始" in content:
            return True
        if "结束" in content:
            return False
        else:
            return False

trainMonitor = TrainMonitor("my-config.json", "crnn.cfg")

if __name__ == "__main__":
    """
    触发训练的接口: 
    用法：from TrainTrigger import trainMonitor
    """
    trainMonitor.train()
    """
    查看是否正在训练的接口，通过写日志文件的方式记录，待优化
    """
    status = trainMonitor.train_status
