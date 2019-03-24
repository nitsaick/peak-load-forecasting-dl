import numpy as np
import torch
from tqdm import tqdm

from eval_func import rmse


class Trainer:
    def __init__(self, net, dataset, optimizer, scheduler, criterion, epoch_num, batch_size):
        self.net = net
        self.dataset = dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.epoch_num = epoch_num
        self.start_epoch = 0
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        train_loader, valid_loader, _ = self.dataset.get_dataloader(self.batch_size)

        msg = 'Net: {}\n'.format(self.net.__class__.__name__) + \
              'Dataset: {}\n'.format(self.dataset.__class__.__name__) + \
              'Epochs: {}\n'.format(self.epoch_num) + \
              'Learning rate: {}\n'.format(self.optimizer.param_groups[0]['lr']) + \
              'Batch size: {}\n'.format(self.batch_size) + \
              'Training size: {}\n'.format(len(train_loader.sampler)) + \
              'Validation size: {}\n'.format(len(valid_loader.sampler)) + \
              'Device: {}\n'.format(self.device)

        self.net = self.net.to(self.device)
        self.criterion = self.criterion.cuda()

        print('{:-^40s}'.format(' Start training '))
        print(msg)

        best_acc = np.inf

        for epoch in range(self.start_epoch, self.epoch_num):
            epoch_str = ' Epoch {}/{} '.format(epoch + 1, self.epoch_num)
            print('{:-^40s}'.format(epoch_str))

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
                break
            print('Learning rate: {}'.format(lr))

            # Training phase
            self.net.train()
            torch.set_grad_enabled(True)

            try:
                loss = self.training(train_loader)
            except KeyboardInterrupt:
                self.save(epoch, self.net, self.optimizer, 'INTERRUPTED.pth')
                return

            self.net.eval()
            torch.set_grad_enabled(False)

            # Evaluation phase
            train_acc = self.evaluation(train_loader)

            # Validation phase
            valid_acc = self.validation(valid_loader)

            print('Train data RMSE:  {:.5f}'.format(train_acc))
            print('Valid data RMSE:  {:.5f}'.format(valid_acc))

            if valid_acc < best_acc:
                best_acc = valid_acc
                checkpoint_filename = 'best.pth'
                self.save(epoch, self.net, self.optimizer, checkpoint_filename)
                print('Update best acc!')

    def training(self, train_loader):
        tbar = tqdm(train_loader, ascii=True, desc='train')
        for batch_idx, (inputs, targets) in enumerate(tbar):
            self.optimizer.zero_grad()

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            tbar.set_postfix(loss='{:.5f}'.format(loss.item()))
        self.scheduler.step(loss.item())

        return loss.item()

    def evaluation(self, train_loader):
        tbar = tqdm(train_loader, desc='eval ', ascii=True)
        scores = []
        for batch_idx, (inputs, targets) in enumerate(tbar):
            inputs = inputs.to(self.device)
            outputs = self.net(inputs)

            outputs = outputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            scores.append(rmse(targets, outputs))

        train_acc = np.mean(scores)
        return train_acc

    def validation(self, valid_loader):
        tbar = tqdm(valid_loader, desc='valid', ascii=True)
        scores = []
        for batch_idx, (inputs, targets) in enumerate(tbar):
            inputs = inputs.to(self.device)
            outputs = self.net(inputs)

            outputs = outputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            scores.append(rmse(targets, outputs))

        valid_acc = np.mean(scores)
        return valid_acc

    def save(self, epoch, net, optimizer, root):
        torch.save({'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, root)
