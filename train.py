import torch
import torch.optim as optim

import torchvision
from torchvision import transforms as tf

import os
import time
import math
import argparse

from models.vit import ViT
from utils.modules import ModelEMA
from utils.com_flops_params import FLOPs_and_Params


def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--max_epoch', type=int, default=90, 
                        help='max epoch')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--learning_rate', type=float,
                        default=0.1, help='learning rate for training model')
    parser.add_argument('--path_to_save', type=str, 
                        default='weights/')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='sgd, adam')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    # lr schedule
    parser.add_argument('--wp_iters', type=int, default=100,
                        help='warmup iteration')
    parser.add_argument('--lr_schedule', type=str, default='step',
                        help='step, linear, cos')

    # dataset
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='cifar10, imagenet')
    parser.add_argument('--root', type=str, default='/mnt/share/ssd2/dataset',
                        help='cifar10, imagenet')

    # model config
    parser.add_argument('-size', '--img_size', type=int, default=224,
                        help='input size')
    parser.add_argument('--num_patch', type=int, default=16,
                        help='input size')
    parser.add_argument('--dim', type=int, default=256,
                        help='patch dim')
    parser.add_argument('--depth', type=int, default=4,
                        help='the number of encoder in transformer')
    parser.add_argument('--heads', type=int, default=8,
                        help='the number of multi-head in transformer')
    parser.add_argument('--dim_head', type=int, default=64,
                        help='the number of dim of each head in transformer')
    parser.add_argument('--channels', type=int, default=3,
                        help='the number of image channel')
    parser.add_argument('--mlp_dim', type=int, default=256,
                        help='the number of dim in FFN')
    parser.add_argument('--pool', type=str, default='cls',
                        help='cls or mean')


    return parser.parse_args()

    
def main():
    args = parse_args()

    path_to_save = os.path.join(args.path_to_save, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)
    
    # use cuda
    if args.cuda:
        print("use cuda")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # mosaic ema
    if args.ema:
        print('use EMA ...')

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)

    # dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        data_root = os.path.join(args.root, 'cifar10')
        # train
        train_dataset = torchvision.datasets.CIFAR10(
                            root=data_root, 
                            train=True, 
                            download=True, 
                            transform=tf.Compose([
                                tf.RandomCrop(32, padding=4),
                                tf.RandomHorizontalFlip(),
                                tf.ToTensor(),
                                tf.Normalize((0.4914, 0.4822, 0.4465), 
                                             (0.2023, 0.1994, 0.2010))]))
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=args.num_workers,
                            pin_memory=True)
        # val
        val_dataset = torchvision.datasets.CIFAR10(
                                root=data_root, 
                                train=False, 
                                download=True, 
                                transform=tf.Compose([
                                    tf.ToTensor(),
                                    tf.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010))]))

        val_loader = torch.utils.data.DataLoader(
                                dataset=val_dataset, 
                                batch_size=100, 
                                shuffle=False, 
                                num_workers=8)

    elif args.dataset == 'imagenet':
        num_classes = 1000
        train_data_root = os.path.join(args.root, 'imagenet', 'train')
        val_data_root = os.path.join(args.root, 'imagenet', 'val')
        # train
        train_dataset = torchvision.datasets.ImageFolder(
                            root=train_data_root,
                            transform=tf.Compose([
                                tf.RandomResizedCrop(224),
                                tf.RandomHorizontalFlip(),
                                tf.ToTensor(),
                                tf.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])]))

        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True,
                            num_workers=args.num_workers, 
                            pin_memory=True)
        # val
        val_dataset = torchvision.datasets.ImageFolder(
                            root=val_data_root, 
                            transform=tf.Compose([
                                tf.Resize(256),
                                tf.CenterCrop(224),
                                tf.ToTensor(),
                                tf.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])]))
        val_loader = torch.utils.data.DataLoader(
                            dataset=val_dataset,
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=args.num_workers, 
                            pin_memory=True)
    
    print('total train data size : ', len(train_dataset))

    # build model
    model = ViT(
        img_size=args.img_size,
        num_patch=args.num_patch,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        mlp_dims=args.mlp_dim,
        heads=args.heads,
        dim_head=args.dim_head,
        pool=args.pool,
        channels=args.channels,
        dropout=0.5,
        emb_dropout=0.5
    )

    model.train().to(device)
    ema = ModelEMA(model) if args.ema else None

    # compute FLOPs and Params
    FLOPs_and_Params(model=model, size=args.img_size)

    # basic setup
    best_acc1 = 0.0
    base_lr = args.learning_rate
    tmp_lr = base_lr
    max_epoch = args.max_epoch
    lr_step = [max_epoch//3, max_epoch*2//3]
    epoch_size = len(train_dataset) // args.batch_size
    warmup = True
    print('Max epoch: ', max_epoch)
    print('Lr epoch: ', lr_step)

    # optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), 
                            lr=base_lr, 
                            momentum=0.9, 
                            weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), 
                                lr=base_lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(),
                                )
    else:
        print('Unknow optimizer !!!')

    # define loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    t0 = time.time()
    print("-------------- start training ----------------")
    for epoch in range(max_epoch):

        # use lr step
        if args.lr_schedule == 'step' and epoch in lr_step:
            tmp_lr = tmp_lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = tmp_lr

        # use cos step
        elif args.lr_schedule == 'cos':
            tmp_lr = 1e-5 + 0.5*(base_lr - 1e-5)*(1 + math.cos(math.pi*epoch / max_epoch))
            set_lr(optimizer, tmp_lr)

        # train one epoch
        for iter_i, (images, target) in enumerate(train_loader):
            ni = iter_i + epoch * epoch_size
            # use lr step
            if args.lr_schedule == 'linear':
                tmp_lr = base_lr * (1.0 - ni / (max_epoch * epoch_size))
                set_lr(optimizer, tmp_lr)
                
            # to tensor
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # check NAN
            if torch.isnan(loss):
                continue

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))            

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ema
            if args.ema:
                ema.update(model)

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('loss',  loss.item(),  ni)
                    tblogger.add_scalar('acc1',  acc1.item(),  ni)
                    tblogger.add_scalar('acc5',  acc5.item(),  ni)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: %.2f][acc1 %.2f][acc5 %.2f][time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss.item(), 
                           acc1.item(), 
                           acc5.item(),
                           t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluate
        print('evaluating ...')
        loss, acc1, acc5 = validate(device, val_loader, model, criterion)
        print('On val dataset: [loss: %.2f][acc1: %.2f][acc5: %.2f]' 
                % (loss.item(), 
                   acc1.item(), 
                   acc5.item()),
                flush=True)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            print('saving the model ...')
            torch.save(model.state_dict(), 
                os.path.join(path_to_save, 
                args.version + '_' + str(epoch + 1) + '_' + str(round(acc1.item())) + '.pth'))


def validate(device, val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

    return loss, acc1, acc5


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()