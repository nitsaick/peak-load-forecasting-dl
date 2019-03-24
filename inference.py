import datetime

import numpy as np
import pandas as pd
import torch

from eval_func import rmse
from network import Net
from prepare_data import prepare_data


def inference(time_series_data, start_date, model_path='./model_best.pth', save_output=False):
    data_channels = 2
    input_time_interval = 365
    output_time_interval = 7
    
    predict_date_start = start_date
    predict_date_end = predict_date_start + datetime.timedelta(days=output_time_interval - 1)
    input_date_start = predict_date_start - datetime.timedelta(days=input_time_interval)
    input_date_end = predict_date_start - datetime.timedelta(days=1)
    
    input = time_series_data[input_date_start:input_date_end].values.transpose(1, 0)
    input = input.reshape((1, data_channels, input_time_interval)).astype(np.float)
    
    target = time_series_data[predict_date_start:predict_date_end]['peak_load'].values
    
    net = Net(in_ch=data_channels, out_ch=output_time_interval)
    
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'])
    
    net.eval()
    torch.set_grad_enabled(False)
    
    input = torch.tensor(input, dtype=torch.float)
    output = net(input).detach().numpy()
    
    print('input: {} ~ {}, output: {} ~ {}'.format(input_date_start.isoformat(), input_date_end.isoformat(),
                                                   predict_date_start.isoformat(), predict_date_end.isoformat()))
    print('output: ', list(map('{:.0f}'.format, output[0])))
    
    if len(target) == output_time_interval:
        target = target.reshape((1, output_time_interval)).astype(np.float)
        score = rmse(target, output)
        print('target: ', list(map('{:.0f}'.format, target[0])))
        print('RMSE:   {}'.format(score))

    if save_output:
        date_list = [(start_date + datetime.timedelta(days=x)).strftime('%Y%m%d') for x in range(0, 7)]
        df_dict = {
            'date': date_list,
            'peak_load(MW)': np.around(output[0]).astype(np.int)
        }
    
        df = pd.DataFrame(df_dict)
        df.to_csv('submission.csv', encoding='UTF-8', index=0)


if __name__ == '__main__':
    time_series_data = prepare_data()
    start_date = datetime.date(2018, 4, 2)
    
    inference(time_series_data, start_date)
