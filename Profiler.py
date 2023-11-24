############# 参考链接：https://www.jianshu.com/p/1d09e56abbf8
import os
import numpy as np
import torch
import time
from models.yolo import Model

if __name__ == '__main__':
    model = Model('models\yolov5s_ECAblock.yaml')
    # device = torch.device('cuda')
    device = torch.device('cuda')
    model.eval()
    model.to(device)
    dump_input = torch.ones(1,3,640,640).to(device)

    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(dump_input)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end-start)*1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(dump_input)
    print(prof.table())
    prof.export_chrome_trace('./yolov5_profile.json')#生成可视化文件，可以在：chrome://tracing  打开，导入生成的文件
