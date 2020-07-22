from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import * 

from biRnn import *
from dataset import *
from utils import *

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def train():
    '''训练的同时，使用dev测试评价指标，最终用图可视化'''
    '''训练几个epoch就跑验证集测试model'''

    model = torch.load('not_ignored_model/no_pack_models/model_birnn_900epoch.pth',map_location = device).to(device)
    lr = config.lr * 0.95 ** (900//30)
    optimzer = torch.optim.Adam([{'params' : model.parameters() , 'initial_lr' : lr}], lr=lr)
    lambda1 = lambda epoch: 0.95 ** (epoch // 30)
    scheduler = LambdaLR(optimzer, lr_lambda=lambda1)

    # 加载训练集
    train_data = zip_dataset()#[0:50]
    train_dataset = MyDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  collate_fn=collate_fn)
    writer = SummaryWriter()

    for epoch in range(1000):
	epoch = opoch + 901
        sloss = []
        for words, pos, tags, seq_len in train_dataloader:
            pred_tags = model(words, pos, seq_len)
            optimzer.zero_grad()
            loss = model.get_loss(pred_tags, tags)

            loss.backward(retain_graph=True)
            optimzer.step()

            print('epoch: ', epoch,end=' ')
            print('loss: ', loss.item())
            sloss.append(loss.item())
        
        scheduler.step()

        writer.add_scalar('loss',sum(sloss)/config.batch_size,global_step = epoch)
        if(epoch % config.save_freq == 0):
            torch.save(model, './not_ignored_model/no_pack_models/model_{}_{}epoch.pth'.format(model.mname,epoch) )


if __name__ == '__main__':
    train()
