from latency_predictor import LatencyPredictor
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import math
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"

class MyDataset(Dataset):
    def __init__(self, x, path, latency):
        self.imgsz = 640
        self.x = torch.FloatTensor(x)
        self.path = path
        self.imgs = {}
        self.latency = torch.FloatTensor(latency)
    
    def load_image(self, path):
        img = cv2.imread(path)
        h0, w0 = img.shape[:2]  # orig hw
        r = self.imgsz / max(h0, w0)
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img, (min(self.imgsz, math.ceil(w0 * r)), min(self.imgsz, math.ceil(h0 * r))), interpolation=interp)
        
        h, w = img.shape[:2]
        dh = self.imgsz - h
        dw = self.imgsz - w
        top = math.floor(dh / 2.)
        bottom = dh - top
        left = math.floor(dw / 2.)
        right = dw - left
        #print(top, bottom, left, right)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        path = self.path[index]
        if not path in self.imgs:
            img = self.load_image(self.path[index])
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            img = torch.FloatTensor(img)
            img /= 255
            self.imgs[path] = img
        return self.x[index], self.imgs[path], self.latency[index]
    
def split_data(x, path, latency, ratio=0.8):
    arr = np.arange(len(x))
    np.random.shuffle(arr)
    train_len = int(len(x) * ratio)
    x, latency = x[arr], latency[arr]
    path = [path[arr[i]] for i in range(len(arr))]
    return x[:train_len,:], path[:train_len], latency[:train_len], x[train_len:,:], path[train_len:], latency[train_len:]

def read_data(path):
    data = pd.read_csv(path)
    params = np.array(data['params'])
    flops = np.array(data['flops'])
    params = params[:, np.newaxis]
    flops = flops[:, np.newaxis]
    x = np.concatenate((params, flops), axis=1)
    return x, data['path'], np.array(data['time'])

if __name__ == "__main__":
    x_data, path_data, latency_data = read_data('../latency_data.csv')
    x_train, path_train, latency_train, x_val, path_val, latency_val = split_data(x_data, path_data, latency_data)
    train_dataset = MyDataset(x_train, path_train, latency_train)
    val_dataset = MyDataset(x_val, path_val, latency_val)
    
    batch_size = 64
    workers = 8
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LatencyPredictor().to(device)
    
    criterion = nn.MSELoss()
    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    epoch = 100
    best_loss = math.inf
    early_stop_count = 0
    for _epoch in range(epoch):
        
        model.train()
        loss_record = []
        pbar = tqdm(train_dataloader, desc=f'Epoch {_epoch + 1}/{epoch} Train: ', bar_format=TQDM_BAR_FORMAT)
        for batch_i, (x, img, latency) in enumerate(pbar):
            optimizer.zero_grad()
            x, img, latency = x.to(device), img.to(device), latency.to(device)
            pred = model(x, img).squeeze()
            loss = criterion(pred, latency)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.detach().item()})
            loss_record.append(loss.detach().item())
        mean_train_loss = sum(loss_record)/len(loss_record)
        
        model.eval()
        loss_record = []
        pbar = tqdm(val_dataloader, desc=f'Epoch {_epoch + 1}/{epoch} Eval: ', bar_format=TQDM_BAR_FORMAT)
        for batch_i, (x, img, latency) in enumerate(pbar):
            x, img, latency = x.to(device), img.to(device), latency.to(device)
            with torch.no_grad():
                pred = model(x, img).squeeze()
                loss = criterion(pred, latency)
            pbar.set_postfix({'loss': loss.detach().item()})
            loss_record.append(loss.item())
        
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch {_epoch + 1}/{epoch}: Train loss: {mean_train_loss}, Valid loss: {mean_valid_loss}')
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), 'latency_predictor.ckpt') # Save your best model
            print(f'Saving model with loss {best_loss}...')
            early_stop_count = 0
        else: 
            early_stop_count += 1
            
        early_stop = 10
        if early_stop_count >= early_stop:
            print('\nModel is not improving, so we halt the training session.')
            break