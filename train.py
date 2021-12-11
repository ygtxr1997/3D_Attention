import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

from config import cfg
from datasets.dataset_arcface import DataLoaderX
from datasets.dataset_arcface import MXCifarTrainDataset, MXCifarTestDataset
from datasets.dataset_arcface import MXImageNet1kTrainDataset, MXImageNet1kTestDataset
from utils.utils_callbacks import CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_amp import MaxClipGradScaler

import backbones

def main(args):

    import random
    import numpy as np
    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    import mxnet as mx
    mx.random.seed(1)

    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    cudnn.benchmark = True

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and rank is 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    if cfg.dataset == 'cifar-100':
        trainset = MXCifarTrainDataset(
            root_dir=cfg.rec,
            local_rank=rank,
            re_p=cfg.re_p,
        )
        testset = MXCifarTestDataset(
            root_dir=cfg.rec,
            local_rank=rank,
        )
    elif cfg.dataset == 'imagenet-1k':
        logging.info('train on im1k')
        import torchvision
        from torchvision import transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(cfg.rec, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        )
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(cfg.rec, 'val'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        raise ValueError
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True,
    )
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=cfg.nw, pin_memory=True, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.nw, pin_memory=True)

    backbone = eval("backbones.{}".format(args.network))(
        fp16=cfg.fp16,
        dataset=cfg.dataset,
        num_classes=cfg.num_classes,
        # num_group=cfg.num_deformable_groups,
    ).to(local_rank)

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank is 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank],)
    backbone.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / cfg.base_batch * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank is 0: logging.info("Total Step is: %d" % total_step)

    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()

    global_step = 0
    grad_scaler = MaxClipGradScaler(init_scale=cfg.batch_size,  # cfg.batch_size
                                    max_scale=128 * cfg.batch_size,
                                    growth_interval=100) if cfg.fp16 else None

    cls_criterion = torch.nn.CrossEntropyLoss()

    # TODO: ER LOSS损失
    # ER LOSS 还没想好用什么损失函数
    ER_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        if epoch < args.resume and rank == 0:
            print('=====> skip epoch %d' % (epoch))
            scheduler_backbone.step()
            continue

        for step, (img, label) in enumerate(train_loader):
            global_step += 1

            with torch.cuda.amp.autocast(cfg.fp16):
                final_pred = backbone(img)
                # TODO：how to update the ER LOSS
                """
                # multi-level ER LOSS，3D atten 反卷积后与原图比较(创新点2)
                if not cfg.en_erloss:  # 不使用ER_LOSS
                    final_pred = backbone(img)
                else:  # 使用ER_LOSS
                    final_pred, d1, d2, d3, d4 = model(img)
                    loss1 = ER_criterion(d1, img)
                    loss2 = ER_criterion(d2, img)
                    loss3 = ER_criterion(d3, img)
                    loss4 = ER_criterion(d4, img)
                    ER_loss = loss1 + loss2 + loss3 + loss4  # how to use and update the loss？
                """
                cls_loss = cls_criterion(final_pred, label)

            if cfg.fp16:
                grad_scaler.scale(cls_loss).backward()
                grad_scaler.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt_backbone)
                grad_scaler.update()
            else:
                cls_loss.backward()
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()
            opt_backbone.zero_grad()

            loss_v = cls_loss

            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler)

            if global_step % 1000 == 0:
                for param_group in opt_backbone.param_groups:
                    lr = param_group['lr']
                print(lr)

        if epoch % 10 <= 9 and rank == 0:
            logging.info('======== start evaluate ========')
            backbone.eval()
            top1 = AverageMeter()
            top5 = AverageMeter()

            with torch.no_grad():
                for step, (img, label) in enumerate(test_loader):
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)
                    final_pred = backbone(img)  # (B, num_classes)

                    prec1, prec5 = accuracy(final_pred, label, topk=(1, 5))
                    top1.update(prec1[0], img.size(0))
                    top5.update(prec5[0], img.size(0))

            logging.info('Top 1 acc: %.4f; Top 5 acc: %.4f' % (
                top1.avg,
                top5.avg,
            ))
            backbone.train()

        callback_checkpoint(global_step, backbone, )
        scheduler_backbone.step()

    dist.destroy_process_group()


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 3D_Attention Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='resnet18', help='backbone network')
    parser.add_argument('--loss', type=str, default='Softmax', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)