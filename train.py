import torch
import torch.nn as nn

from network import Net
from prepare_data import prepare_data
from time_series import TimeSeries
from trainer import Trainer

if __name__ == '__main__':
    time_series_data = prepare_data()

    epoch_num = 100
    batch_size = 4

    dataset = TimeSeries(time_series_data, input_time_interval=365, output_time_interval=7, output_keyword='peak_load')
    net = Net(in_ch=dataset.data_channels, out_ch=dataset.output_time_interval)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True,
        threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-07
    )

    criterion = nn.MSELoss()

    torch.cuda.empty_cache()
    trainer = Trainer(net, dataset, optimizer, scheduler, criterion, epoch_num, batch_size)
    trainer.run()
