import logging
import os
import sys
import os.path as osp
logging.getLogger('PIL').setLevel(logging.WARNING) # 设置PIL模块的日志等级为WARNING

def setup_logger(save_dir=None, if_train=False):
    logger = logging.getLogger() #创建日志器
    logger.setLevel(logging.DEBUG)  #设置日志级别

    ch = logging.StreamHandler(stream=sys.stdout) #创建日志处理器，在控制台打印
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s") #创建格式器，指定日志的打印格式

    ch.setFormatter(formatter) #给处理器设置格式
    logger.addHandler(ch) #给日志器添加处理器

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if if_train:
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
        else:
            fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger