import os
import time

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import ImageDataset
from model import Generator


def main():
    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    #load dataset
    train_datasets = ImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "train")
    valid_datasets = ImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "valid")

    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=False)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  persistent_workers=False)
    
    # build model
    model = Generator().to(config.device)

    # optimizer
    psnr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.MSELoss().to(config.device)
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)

    if(True):
        state_dict = torch.load(config.continue_model_path, map_location=config.device)
        model.load_state_dict(state_dict)
    
    scaler = amp.GradScaler()
    best_psnr = 0.0

    print("Start train SRResNet model.")

    for epoch in range(config.start_epoch, config.epochs):
        train(model, train_dataloader, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)

        psnr = validate(model, valid_dataloader, psnr_criterion, epoch, writer)
        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save(model.state_dict(), os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth"))
        print(f"epoch: {epoch} * PSNR: {psnr:4.2f}.\n")
        if is_best:
            torch.save(model.state_dict(), os.path.join(results_dir, "g-best.pth"))

    # Save the generator weight under the last Epoch in this stage
    torch.save(model.state_dict(), os.path.join(results_dir, "g-last.pth"))
    print("End train SRResNet model.")

def train(model, train_dataloader, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_dataloader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in train mode.
    model.train()

    end = time.time()
    for index, (lr, hr) in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        lr = lr.to(config.device, non_blocking=True)
        hr = hr.to(config.device, non_blocking=True)

        # Initialize the generator gradient
        model.zero_grad()

        # Mixed precision training
        if(config.train_mode == 'mixed_precision'):
            with amp.autocast():
                sr = model(lr)
                loss = pixel_criterion(sr, hr)
        else:
            sr = model(lr)
            loss = pixel_criterion(sr, hr)
        # Gradient zoom
        scaler.scale(loss).backward()
        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Writer Loss to file
        writer.add_scalar("Train/Loss", loss.item(), index + epoch * batches + 1)
        if index % config.print_frequency == 0 and index != 0:
            progress.display(index)

def validate(model, valid_dataloader, psnr_criterion, epoch, writer) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(valid_dataloader), [batch_time, psnres], prefix="Valid: ")

    # Put the generator in verification mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for index, (lr, hr) in enumerate(valid_dataloader):
            lr = lr.to(config.device, non_blocking=True)
            hr = hr.to(config.device, non_blocking=True)

            # Mixed precision
            if(config.train_mode == 'mixed_precision'):
                with amp.autocast():
                    sr = model(lr)
            else:
                sr = model(lr)
            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), hr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if index % config.print_frequency == 0:
                #progress.display(index)

        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
        # Print evaluation indicators.
        

    return psnres.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()