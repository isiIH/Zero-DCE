from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
import os

from model import DCE_Net
from dataLoader import LowLightLoader
from loss import SPA_Loss, EXP_Loss, COL_Loss, ILLUM_Loss

def weights_init(m):
  if isinstance(m, nn.Conv2d):
    # mean 0.0 / std 0.02
    nn.init.normal_(m.weight, mean=0.0, std=0.02)
    if m.bias is not None:
      nn.init.constant_(m.bias, 0.0) # Constant

def train_dce_net():
  args = get_args()
  torch.backends.cudnn.benchmark = True
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dir_checkpoint = Path(args.checkpoint_folder)
  dir_checkpoint.mkdir(parents=True, exist_ok=True)

  # Load Model
  model = DCE_Net().to(device)
  model.apply(weights_init)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  start_epoch = 1
  train_losses = []
  val_losses = []
  best_val_loss = float('inf')
  patience_counter = 0

  if args.load:
    checkpoint = torch.load(args.pretrain_dir, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        if val_losses:
            best_val_loss = min(val_losses)
    else:
        model.load_state_dict(checkpoint)
    print(f'Model loaded from {args.pretrain_dir} -> Resuming at epoch {start_epoch}')

  # Load Dataset
  dataset = LowLightLoader(args.image_path)

  # Create data loaders
  n_val = int(len(dataset) * args.val)
  n_train = len(dataset) - n_val
  train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

  loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
  train_loader = DataLoader(train_set, shuffle=True, **loader_args)
  val_loader = DataLoader(val_set, shuffle=False, **loader_args)

  scaler = torch.amp.GradScaler('cuda')

  # Init Losses with weights
  L_spa = SPA_Loss().to(device)
  L_exp = EXP_Loss(10).to(device)
  L_col = COL_Loss(5).to(device)
  L_illum = ILLUM_Loss(200).to(device)

  end_epoch = start_epoch + args.epochs
  for epoch in range(start_epoch, end_epoch):
    with tqdm(train_loader, desc=f'Epoch {epoch}/{end_epoch - 1}', unit='batch') as pbar:
      #### Training ####
      model.train()
      train_loss = 0.0
      for batch in pbar:
        assert batch.shape[1] == 3, f'Expected 3 channels, got {batch.shape[1]}'

        image = batch.to(device)

        with torch.amp.autocast('cuda'):
          enhanced_image, A = model(image)
          loss = L_spa(image, enhanced_image) + L_exp(enhanced_image) + L_col(enhanced_image) + L_illum(A)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    #### Validation ####
    model.eval()
    with torch.no_grad():
      val_loss = 0.0
      for batch in val_loader:
        assert batch.shape[1] == 3, f'Expected 3 channels, got {batch.shape[1]}'

        image = batch.to(device)
        enhanced_image, A = model(image)

        loss = L_spa(image, enhanced_image) + L_exp(enhanced_image) + L_col(enhanced_image) + L_illum(A)
        val_loss += loss.item()

    epoch_train_loss = train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f'Epoch {epoch} -> Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')

    # Save checkpoint
    if epoch_val_loss < best_val_loss:
      best_val_loss = epoch_val_loss
      patience_counter = 0

      checkpoint = {
          'epoch': epoch,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'train_losses': train_losses,
          'val_losses': val_losses,
          'best_val_loss': best_val_loss
      }
      torch.save(checkpoint, os.path.join(dir_checkpoint, "best_model.pth"))
      print(f'--> Val loss improved to {best_val_loss:.4f}. Saved best_model.pth')
    else:
      patience_counter += 1
      print(f'--> No improvement in val loss. Patience: {patience_counter}/{args.patience}')
      if patience_counter >= args.patience:
        print(f'Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}')
        break

def get_args():
    parser = ArgumentParser(description='Train the DCE-Net on the SICE dataset')
    parser.epochs = 20
    parser.batch_size = 8
    parser.lr = 1e-4
    parser.load = False
    parser.pretrain_dir = "checkpoints/best_model.pth"
    parser.val = 0.2
    parser.image_path = "../data/train_data/"
    parser.checkpoint_folder = "checkpoints/"
    parser.num_workers = 2
    parser.weight_decay = 1e-4
    parser.grad_clip_norm = 0.1
    parser.patience = 5

    return parser

if __name__ == "__main__":
    train_dce_net()