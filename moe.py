import torch.nn as nn
import torch
import os
import sys
sys.path.append('/data/ouyuan/yolov5')
from models.yolo import Model
from latency_predictor import LatencyPredictor, Encoder

ModelName = 'yolov5tmp'
ModelSuffix = ['n', 's', 'm', 'l', 'x', 'n6', 's6', 'm6', 'l6', 'x6']
# ModelSuffix = ['n', 's', 'm', 'l', 'x']
Params = torch.Tensor([1.9, 7.2, 21.2, 46.5, 86.7, 3.2, 12.6, 35.7, 76.8, 140.7]).unsqueeze(dim=1)
FLOPs = torch.Tensor([4.5, 16.5, 49.0, 109.1, 206.7, 4.6, 16.8, 50.0, 111.4, 209.8]).unsqueeze(dim=1)
ModelFeature = torch.cat((Params, FLOPs), dim=1)

class MoE(nn.Module):
    def __init__(self, input_dim = 640 * 640 * 3, experts_num = len(ModelSuffix)):
        super(MoE, self).__init__()
        self.experts_num = experts_num
        self.experts = nn.ModuleList([])
        self.latency_predictor = LatencyPredictor()
        self.latency_predictor.load_state_dict(torch.load('latency_predictor.ckpt'))
        self.model_feature = ModelFeature[:self.experts_num]
        for i in range(len(ModelSuffix)):
            model_name = str.replace(ModelName, 'tmp', ModelSuffix[i])
            #model = Model(os.path.join('../models', model_name + '.yaml'), ch=3, nc=80)
            model = torch.load(os.path.join('..', model_name + '.pt'), map_location="cpu")["model"].float()
            self.experts.append(model)
        
        # self.w_gate = nn.Parameter(torch.randn(input_dim, self.experts_num), requires_grad=True)
        self.encoder = Encoder()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, imgs, switch=False):
        # x = imgs.view(imgs.shape[0], -1)
        # gate = x @ self.w_gate
        # value, id = gate.topk(1, dim=1)
        # gate = torch.zeros_like(gate, requires_grad=True).scatter(dim=1, index=id, src=value)
        # gate = self.softmax(gate)
        # input = [torch.tensor([], device=imgs.device) for _ in range(self.experts_num)]
        # model_feature = gate @ self.model_feature.to(device=imgs.device)
        # latency = self.latency_predictor(model_feature, imgs)
        # for i in range(imgs.shape[0]):
        #     input[id[i]] = torch.cat((input[id[i]], imgs[i].unsqueeze(dim=0)), dim=0)

        # output_raw = []
        # for i in range(self.experts_num):
        #     if len(input[i]) > 0:
        #         output_raw.append(self.experts[i](input[i]))
        #     else:
        #         output_raw.append([])
        # output = [torch.tensor([], device=imgs.device) for _ in range(3)]
        # output_id = [0 for _ in range(self.experts_num)]
        # for i in range(imgs.shape[0]):
        #     for j in range(3):
        #         output[j] = torch.cat((output[j], output_raw[id[i]][j][output_id[id[i]]].unsqueeze(dim=0)))
        #     output_id[id[i]] += 1
        gate = self.encoder(imgs)
        gate = self.softmax(gate)
        # print(gate.shape)
        gate = gate.sum(dim=0)
        # print(gate.shape)
        _, id = gate.topk(1)
        if switch:
            return id
        output = self.experts[id](imgs)
        model_feature = self.model_feature.to(device=imgs.device)
        latency = self.latency_predictor(model_feature[id].repeat(imgs.shape[0], 1), imgs)
        return output, id, latency.sum(dim=0)