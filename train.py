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

    trainset = MXCifarTrainDataset(
        root_dir=cfg.rec,
        local_rank=rank,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True,
    )
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=cfg.nw, pin_memory=True, drop_last=True
    )

    testset = MXCifarTestDataset(
        root_dir=cfg.rec,
        local_rank=rank,
    )
    test_loader = DataLoaderX(
        local_rank=local_rank, dataset=testset, batch_size=cfg.batch_size,
        num_workers=cfg.nw, pin_memory=True, drop_last=False
    )

    backbone = eval("backbones.{}".format(args.network))(
        fp16=cfg.fp16,
        num_classes=cfg.num_classes
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
        lr=cfg.lr / 128 * cfg.batch_size * world_size,
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

        if epoch % 10 == 9:
            if rank == 0:
                logging.info('10 epochs finished, start evaluate')
            backbone.eval()
            correct_1 = 0.0
            correct_5 = 0.0

            with torch.no_grad():
                for step, (img, label) in enumerate(test_loader):
                    final_pred = backbone(img)  # (B, num_classes)

                    _, pred = final_pred.topk(5, 1, largest=True, sorted=True)

                    label = label.view(label.size(0), -1).expand_as(pred)
                    correct = pred.eq(label).float()

                    # compute top 5
                    correct_5 += correct[:, :5].sum()

                    # compute top1
                    correct_1 += correct[:, :1].sum()

            if rank == 0:
                logging.info('Top 1 err: %.4f; Top 5 err: %.4f' % (
                    1 - correct_1 / len(testset),
                    1 - correct_5 / len(testset)
                ))
            backbone.train()

        callback_checkpoint(global_step, backbone, )
        scheduler_backbone.step()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 3D_Attention Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='resnet18', help='backbone network')
    parser.add_argument('--loss', type=str, default='Softmax', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)