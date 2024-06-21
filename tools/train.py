# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import logging
import os
import pprint
import timeit

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from tensorboardX import SummaryWriter

from model.config import configs
from model.config import update_config
from model.core.criterion import CrossEntropy, OhemCrossEntropy
from model.core.function import train, validate
from model.utils.utils import create_logger, PhysicsFormer


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--logits_shape", type=tuple, default=(769,769))
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(configs, args)

    return args


def get_sampler(dataset):
    from lib.utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        configs, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(configs)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = configs.CUDNN.BENCHMARK
    cudnn.deterministic = configs.CUDNN.DETERMINISTIC
    cudnn.enabled = configs.CUDNN.ENABLED
    gpus = list(configs.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

        # build model
    model = eval('models.' + configs.MODEL.NAME +
                 '.get_seg_model')(config)

    # dump_input = torch.rand(
    #     (1, 3, configs.TRAIN.IMAGE_SIZE[1], configs.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # copy model file
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        # if os.path.exists(models_dst_dir):
        #     shutil.rmtree(models_dst_dir)
        # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        batch_size = configs.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = configs.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (configs.TRAIN.IMAGE_SIZE[1], configs.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.' + configs.DATASET.DATASET)(
        root=configs.DATASET.ROOT,
        list_path=configs.DATASET.TRAIN_SET,
        num_samples=None,
        num_classes=configs.DATASET.NUM_CLASSES,
        multi_scale=configs.TRAIN.MULTI_SCALE,
        flip=configs.TRAIN.FLIP,
        ignore_label=configs.TRAIN.IGNORE_LABEL,
        base_size=configs.TRAIN.BASE_SIZE,
        crop_size=crop_size,
        downsample_rate=configs.TRAIN.DOWNSAMPLERATE,
        scale_factor=configs.TRAIN.SCALE_FACTOR)

    train_sampler = get_sampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=configs.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=configs.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler)

    extra_epoch_iters = 0
    if configs.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('datasets.' + configs.DATASET.DATASET)(
            root=configs.DATASET.ROOT,
            list_path=configs.DATASET.EXTRA_TRAIN_SET,
            num_samples=None,
            num_classes=configs.DATASET.NUM_CLASSES,
            multi_scale=configs.TRAIN.MULTI_SCALE,
            flip=configs.TRAIN.FLIP,
            ignore_label=configs.TRAIN.IGNORE_LABEL,
            base_size=configs.TRAIN.BASE_SIZE,
            crop_size=crop_size,
            downsample_rate=configs.TRAIN.DOWNSAMPLERATE,
            scale_factor=configs.TRAIN.SCALE_FACTOR)
        extra_train_sampler = get_sampler(extra_train_dataset)
        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=batch_size,
            shuffle=configs.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=configs.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)
        extra_epoch_iters = np.int(extra_train_dataset.__len__() /
                                   configs.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    test_size = (configs.TEST.IMAGE_SIZE[1], configs.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.' + configs.DATASET.DATASET)(
        root=configs.DATASET.ROOT,
        list_path=configs.DATASET.TEST_SET,
        num_samples=configs.TEST.NUM_SAMPLES,
        num_classes=configs.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=configs.TRAIN.IGNORE_LABEL,
        base_size=configs.TEST.BASE_SIZE,
        crop_size=test_size,
        downsample_rate=1)

    test_sampler = get_sampler(test_dataset)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=configs.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    # criterion
    if configs.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=configs.TRAIN.IGNORE_LABEL,
                                     thres=configs.LOSS.OHEMTHRES,
                                     min_kept=configs.LOSS.OHEMKEEP,
                                     weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=configs.TRAIN.IGNORE_LABEL,
                                 weight=train_dataset.class_weights)

    model = PhysicsFormer(model, criterion,num_classes=args.num_classes,iterations=args.inference,image_size=args.logits_shape)
    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if configs.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if configs.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in configs.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': configs.TRAIN.LR},
                      {'params': nbb_lr, 'lr': configs.TRAIN.LR * configs.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': configs.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=configs.TRAIN.LR,
                                    momentum=configs.TRAIN.MOMENTUM,
                                    weight_decay=configs.TRAIN.WD,
                                    nesterov=configs.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                         configs.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0
    if configs.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = configs.TRAIN.END_EPOCH + configs.TRAIN.EXTRA_EPOCH
    num_iters = configs.TRAIN.END_EPOCH * epoch_iters
    extra_iters = configs.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    for epoch in range(last_epoch, end_epoch):

        current_trainloader = extra_trainloader if epoch >= configs.TRAIN.END_EPOCH else trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        # valid_loss, mean_IoU, IoU_array = validate(configs, 
        #             testloader, model, writer_dict)

        if epoch >= configs.TRAIN.END_EPOCH:
            train(configs, epoch - configs.TRAIN.END_EPOCH,
                  configs.TRAIN.EXTRA_EPOCH, extra_epoch_iters,
                  configs.TRAIN.EXTRA_LR, extra_iters,
                  extra_trainloader, optimizer, model, writer_dict)
        else:
            train(configs, epoch, configs.TRAIN.END_EPOCH,
                  epoch_iters, configs.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict)

        valid_loss, mean_IoU, IoU_array = validate(configs,
                                                   testloader, model, writer_dict)

        if args.local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                           os.path.join(final_output_dir, 'best.pth'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

    if args.local_rank <= 0:
        torch.save(model.module.state_dict(),
                   os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end - start) / 3600))
        logger.info('Done')


if __name__ == '__main__':
    main()
