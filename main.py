import torch
import torchvision
import os
from model import I2V
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image,make_grid
from tensorboardX import SummaryWriter
import random
from dataloader import ImgIter
from config import Config
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from tqdm import tqdm
cfg = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.available_GPUs
cudnn.benchmark = True

i2v = I2V(dropout_rate=cfg.dropout_rate)
for param in i2v.parameters():
    param.requires_grad = True

ct = 0
for child in i2v.feature_extractor:
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False

i2v = torch.nn.DataParallel(i2v).cuda()
# critic = torch.nn.BCEWithLogitsLoss().cuda() # Contains Sigmoid and a BCE Loss
critic = torch.nn.BCELoss().cuda() # Contains Sigmoid and a BCE Loss

#opt = torch.optim.Adam(filter(lambda p: p.requires_grad, i2v.parameters()), lr=cfg.lr,betas=(0.9,0.999))
opt = torch.optim.SGD(filter(lambda p: p.requires_grad, i2v.parameters()), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, cfg.lr_decay_per_epoch)
img_iter = ImgIter(cfg.img_path,cfg.pkl_path,batch_size=cfg.batch_size,img_size=cfg.img_size,n_workers=cfg.n_workers)
tb_logger = SummaryWriter(cfg.log_dir)
global_i = 0
for e in range(cfg.epoch):
    for i in range(img_iter.get_train_n_iters()):
#    for i in range(10):
        opt.zero_grad()
        batch = img_iter.get_train_batch()
        img = batch["img"].cuda()
        attr = batch["attr"].cuda()
        output_attr = i2v(img)
        loss = critic(output_attr, attr)
        loss.backward()
        opt.step()

        attr = attr.view(-1).cpu()
        output_attr = output_attr.view(-1).cpu()
        acc = accuracy_score(attr, output_attr>0.5)
        recall = recall_score(attr, output_attr>0.5)
        precision = precision_score(attr, output_attr>0.5)
        f1 = f1_score(attr, output_attr>0.5)
        print("Epoch %d/%d Iter %d/%d Loss %.2f Acc %.2f Recall %.2f Precison %.2f F1 %.2f" %
              (e, cfg.epoch, i, img_iter.get_train_n_iters(), loss, acc, recall, precision, f1))
        global_i += 1
        if global_i % cfg.print_n_iter == 0:
            tb_logger.add_image("TrainImg", make_grid(img*0.5+0.5))
            tb_logger.add_scalar("Loss", loss, global_i)
            tb_logger.add_scalar("Acc", acc, global_i)
            tb_logger.add_scalar("Recall", recall, global_i)
            tb_logger.add_scalar("Precision", precision, global_i)
            tb_logger.add_scalar("F1", f1, global_i)
        if global_i % cfg.val_n_iter == 0:
        #if global_i % 10 == 0:
            total_loss = 0
            total_acc = 0
            total_recall = 0
            total_precision = 0
            total_f1 = 0
            #n_test_iters = 10
            i2v.eval()
            n_test_iters = img_iter.get_test_n_iters()
            for i in tqdm(range(n_test_iters)):
            # for i in tqdm(range(10)):

                with torch.no_grad():
                    batch = img_iter.get_test_batch()
                    img = batch["img"].cuda()
                    attr = batch["attr"].cuda()
                    output_attr = i2v(img)
                    loss = critic(output_attr, attr)

                attr = attr.view(-1).cpu()
                output_attr = output_attr.view(-1).cpu()
                acc = accuracy_score(attr, output_attr > 0.5)
                recall = recall_score(attr, output_attr > 0.5)
                precision = precision_score(attr, output_attr > 0.5)
                f1 = f1_score(attr, output_attr > 0.5)
                total_loss += loss
                total_acc += acc
                total_recall += recall
                total_precision += precision
                total_f1 += f1
            print("Epoch %d/%d Val Loss %.2f Acc %.2f Recall %.2f Precison %.2f F1 %.2f" %
                  (e, cfg.epoch, total_loss / n_test_iters, total_acc / n_test_iters, total_recall / n_test_iters,
                   total_precision / n_test_iters, total_f1 / n_test_iters))
            tb_logger.add_scalar("Val_Loss", total_loss / n_test_iters, global_i)
            tb_logger.add_scalar("Val_Acc",total_acc / n_test_iters, global_i)
            tb_logger.add_scalar("Val_Recall",total_recall / n_test_iters, global_i)
            tb_logger.add_scalar("Val_Precision",total_precision / n_test_iters, global_i)
            tb_logger.add_scalar("Val_F1",total_f1 / n_test_iters, global_i)
            i2v.train()
        if global_i % cfg.save_n_iter == 0:
            torch.save(i2v.state_dict(), os.path.join(cfg.model_saving_path, "i2v_%d_%d.pth" % (e,global_i)))
            scheduler.step()

