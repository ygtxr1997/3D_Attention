from easydict import EasyDict as edict

cfg = edict()
cfg.dataset = "imagenet-1k"
cfg.embedding_size = 512
cfg.sample_rate = 1
cfg.fp16 = False
cfg.momentum = 0.9
cfg.weight_decay = 5e-4
cfg.batch_size = 64  # 128
cfg.base_batch = 128
cfg.lr = 0.1  # 0.1 for base batch size

cfg.nw = 20

cfg.re_p = 0.  # Random Erasing p
cfg.en_erloss = False  # ER_LOSS
cfg.num_deformable_groups = 2  # group of deformConv

""" Setting EXP ID """
cfg.exp_id = 5
cfg.output = "res18_im1k" + str(cfg.exp_id)
print('output path: ', cfg.output)

if cfg.dataset == 'cifar-100':
    cfg.rec = '/tmp/train_tmp/cifar-100'
    cfg.nw = 0
    cfg.num_classes = 100
    cfg.num_epoch = 300
    cfg.warmup_epoch = -1
    cfg.val_targets = []

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [151, 226, ] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001
    cfg.lr_func = lr_step_func

elif cfg.dataset == 'imagenet-1k':
    cfg.rec = '/tmp/train_tmp/imagenet_1k'
    cfg.nw = 4
    cfg.num_classes = 1000
    cfg.num_epoch = 90
    cfg.warmup_epoch = -1
    cfg.val_targets = []

    cfg.weight_decay = 1e-4
    cfg.base_batch = 256
    cfg.batch_size = 64

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [31, 61, ] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001
    cfg.lr_func = lr_step_func

elif cfg.dataset == "emore":
    cfg.rec = "/train_tmp/faces_emore"
    cfg.num_classes = 85742
    cfg.num_image = 5822653
    cfg.num_epoch = 16
    cfg.warmup_epoch = -1
    cfg.val_targets = ["lfw", ]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14] if m - 1 <= epoch])
    cfg.lr_func = lr_step_func

elif cfg.dataset == "ms1m-retinaface-t2":
    cfg.rec = "/home/yuange/dataset/ms1m-retinaface"
    import os
    if os.path.exists("/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/datasets/ms1m-retinaface"):
        cfg.rec = "/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/datasets/ms1m-retinaface"
        cfg.nw = 14
    elif os.path.exists("/tmp/train_tmp"):
        cfg.rec = "/tmp/train_tmp/ms1m-retinaface"  # mount on RAM
        cfg.nw = 0
    cfg.num_classes = 93431  # 91180
    cfg.num_epoch = 25
    cfg.warmup_epoch = -10  # -1
    cfg.val_targets = ["lfw", "cfp_fp", ]  # ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [11, 17, 22] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001

    import numpy as np
    cfg.min_lr = 0
    def lr_fun_cos(cur_epoch):
        """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
        lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.num_epoch))
        return (1.0 - cfg.min_lr) * lr + cfg.min_lr

    cfg.warmup_factor = 0.3
    def lr_step_func_cos(epoch):
        cur_lr = lr_fun_cos(cur_epoch=epoch) * cfg.lr
        if epoch < cfg.warmup_epoch:
            alpha = epoch / cfg.warmup_epoch
            warmup_factor = cfg.warmup_factor * (1.0 - alpha) + alpha
            cur_lr *= warmup_factor
        return lr_fun_cos(cur_epoch=epoch)
        # return cur_lr / cfg.lr

    cfg.lr_func = lr_step_func_cos

elif cfg.dataset == "glint360k":
    # make training faster
    # our RAM is 256G
    # mount -t tmpfs -o size=140G  tmpfs /train_tmp
    cfg.rec = "/train_tmp/glint360k"
    cfg.num_classes = 360232
    cfg.num_image = 17091657
    cfg.num_epoch = 20
    cfg.warmup_epoch = -1
    cfg.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [8, 12, 15, 18] if m - 1 <= epoch])
    cfg.lr_func = lr_step_func

elif cfg.dataset == "webface":
    cfg.rec = "/train_tmp/faces_webface_112x112"
    cfg.num_classes = 10572
    cfg.num_image = "forget"
    cfg.num_epoch = 34
    cfg.warmup_epoch = -1
    cfg.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    cfg.lr_func = lr_step_func

