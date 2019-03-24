# Peak Load Forecasting

## 1. Overview
根據台電歷史資料，預測未來七天的"電力尖峰負載"(MW)。

![](https://i.imgur.com/oiFgMtq.png)
[圖片來源](https://www.taipower.com.tw/d006/loadGraph/loadGraph/load_reserve_.html)


## 2. Goal
預測 2019/4/2 ~ 2019/4/8 的每日"電力尖峰負載"(MW)


## 3. 使用資料
 - [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
 - [今日預估尖峰備轉容量率](https://www.taipower.com.tw/d006/loadGraph/loadGraph/load_reserve_.html)
 - [中華民國一百零六年政府行政機關辦公日曆表](https://www.dgpa.gov.tw/information?uid=2&pid=4293)
 - [中華民國一百零七年政府行政機關辦公日曆表](https://www.dgpa.gov.tw/information?uid=83&pid=7473)
 - [中華民國108年（西元2019年）政府行政機關辦公日曆表](https://www.dgpa.gov.tw/information?uid=83&pid=8150)
 
 
## 4. 預測方法
使用一維CNN模型進行訓練後預測
 - Input: 365 天的每日尖峰負載 與 是不是工作日
 - Target: 接下來 7 天的每日尖峰負載
 - training / validation data: 4:1
 - Epoch: 100
 - Batch Size: 4
 - Loss Function: MSELoss
 - Optimizer: Adam
 - LR: 0.0001
 - LR Reduce Strategy: 如果經過 10 次 Epoch ， loss 都沒有下降，則將 LR * 0.1 

CNN模型參數:
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv1d-1              [-1, 32, 359]             480
         LeakyReLU-2              [-1, 32, 359]               0
            Conv1d-3              [-1, 16, 357]           1,552
         LeakyReLU-4              [-1, 16, 357]               0
         MaxPool1d-5              [-1, 16, 178]               0
           Flatten-6                 [-1, 2848]               0
            Linear-7                 [-1, 1000]       2,849,000
         LeakyReLU-8                 [-1, 1000]               0
            Linear-9                    [-1, 7]           7,007
================================================================
Total params: 2,858,039
Trainable params: 2,858,039
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.32
Params size (MB): 10.90
Estimated Total Size (MB): 11.23
----------------------------------------------------------------
```


## 5. 檔案說明
 - data: 所有CSV資料
 - network.py: CNN模型
 - time_series.py: 自定義 PyTorch Dataset，使用 Pandas 讀取時間序列資料後，可方便產生 Training & Validation data
 - trainer.py: Training Script
 - model_best.pth: 預訓練模型


 - prepare_data.py: 將台電歷史資料、行事曆資料進行整合並提取需要的部分
 - train.py: 定義參數並進行訓練
 - inference.py: 讀取預訓練模型並進行特定日期的預測
 - app.py: 讀取預訓練模型並進行 2019/4/2 ~ 2019/4/8 預測，並將結果存到 submission.csv
 
 
 