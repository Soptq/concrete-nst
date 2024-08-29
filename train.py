import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import save_image

import quant_model as net
from datasets import FlatFolderDataset
from config import get_train_config
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def load_pretrained_vgg(vgg_path):
    vgg = net.vgg
    vgg.load_state_dict(torch.load(vgg_path))
    return nn.Sequential(*list(vgg.children())[:31])


if __name__ == '__main__':
    args = get_train_config()

    if args.quality == "high":
        image_size = 192
    elif args.quality == "mid":
        image_size = 128
    elif args.quality == "low":
        image_size = 64
    else:
        raise ValueError(f"Unknown quality size: {args.quality}")

    def train_transform():
        transform_list = [
            transforms.Resize(size=(image_size * 2, image_size * 2)),
            transforms.RandomCrop(image_size),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = Path(args.checkpoints)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    vgg = load_pretrained_vgg(args.vgg)

    content_encoder = net.Encoder()
    style_encoder = net.Encoder()
    modulator = net.Modulator()
    decoder = net.Decoder()

    network = net.Net(vgg, content_encoder, style_encoder, modulator, decoder)

    # continue training from the checkpoint
    if args.resume:
        checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
        network.load_state_dict(checkpoints['net'], strict=False)

    network.train()
    network.to(args.device)

    content_tf = train_transform()
    style_tf = train_transform()
    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.style_dir, style_tf)
    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam([
        {'params': network.content_encoder.parameters()},
        {'params': network.style_encoder.parameters()},
        {'params': network.modulator.parameters()},
        {'params': network.decoder.parameters()}
    ], lr=args.lr)

    start_iter = -1

    # continue training from the checkpoint
    if args.resume:
        checkpoints = torch.load(args.checkpoints + '/checkpoints.pth.tar')
        optimizer.load_state_dict(checkpoints['optimizer'])
        start_iter = checkpoints['epoch']

    # training
    pbar = tqdm(range(start_iter + 1, args.max_iter))
    for i in pbar:
        adjust_learning_rate(optimizer, iteration_count=i, lr=args.lr, lr_decay=args.lr_decay)
        content_images = next(content_iter).to(args.device)
        style_images = next(style_iter).to(args.device)
        stylized_results, loss_c, loss_s, loss_contrastive = network(content_images, style_images)
        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss_contrastive = args.SSC_weight * loss_contrastive
        loss = loss_c + loss_s + loss_contrastive

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            f"loss_content={loss_c.item():.4f}, loss_style={loss_s.item():.4f}, loss_contrastive={loss_contrastive.item():.4f}")

        # save intermediate samples
        output_dir = Path(args.sample_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        if (i + 1) % 500 == 0:
            visualized_imgs = torch.cat([content_images, style_images, stylized_results])
            output_name = output_dir / 'output{:d}.jpg'.format(i + 1)
            save_image(visualized_imgs, str(output_name), nrow=args.batch_size)

        # save checkpoint models
        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.content_encoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'content_encoder_iter_{:d}.pth.tar'.format(i + 1))

            state_dict = network.style_encoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'style_encoder_iter_{:d}.pth.tar'.format(i + 1))

            state_dict = network.modulator.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'modulator_iter_{:d}.pth.tar'.format(i + 1))

            state_dict = network.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'decoder_iter_{:d}.pth.tar'.format(i + 1))

            checkpoints = {
                "net": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i
            }
            torch.save(checkpoints, checkpoints_dir / 'checkpoints.pth.tar')
