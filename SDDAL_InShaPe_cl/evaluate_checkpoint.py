import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from QuantUNetT_model import QuantUNetT as PImodel
from utils.loader import get_validation_data


def calculate_frcm(img1, img2):
    """Fourier Ring Correlation Metric — PyTorch 2 version (torch.fft.fft2)."""
    batch_size = 1
    device = img1.device
    nz, nx, ny = img1.shape
    rnyquist = nx // 2

    half = nx // 2
    x = torch.cat((torch.arange(0, half), torch.arange(-half, 0))).to(device)
    y = x

    X, Y = torch.meshgrid(x, y, indexing='ij')
    r_map = X ** 2 + Y ** 2
    index = torch.round(torch.sqrt(r_map.float()))

    r = torch.arange(0, rnyquist + 1, device=device).float()

    F1 = torch.fft.fft2(img1).permute(1, 2, 0)
    F2 = torch.fft.fft2(img2).permute(1, 2, 0)

    C_r = torch.empty(rnyquist + 1, batch_size, device=device)
    C_i = torch.empty_like(C_r)
    C1  = torch.empty_like(C_r)
    C2  = torch.empty_like(C_r)

    for ii in r:
        auxF1 = F1[torch.where(index == ii)]
        auxF2 = F2[torch.where(index == ii)]
        ii = ii.int()

        real1, imag1 = auxF1.real, auxF1.imag
        real2, imag2 = auxF2.real, auxF2.imag

        C_r[ii] = torch.sum(real1 * real2 + imag1 * imag2, dim=0)
        C_i[ii] = torch.sum(imag1 * real2 - real1 * imag2, dim=0)
        C1[ii]  = torch.sum(real1 ** 2 + imag1 ** 2, dim=0)
        C2[ii]  = torch.sum(real2 ** 2 + imag2 ** 2, dim=0)

    FRC  = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
    FRCm = 1 - torch.where(FRC != FRC, torch.tensor(1.0, device=device), FRC)
    return torch.mean(FRCm ** 2).item()


def main():
    parser = argparse.ArgumentParser(description='Inline evaluation: SSIM, MAE, FRCM + timing log')
    parser.add_argument('--checkpoint',      required=True,  type=str,
                        help='Path to .pth.tar model checkpoint')
    parser.add_argument('--test_data',       required=True,  type=str,
                        help='Dir containing test_set/intensity/npy and test_set/phase/npy')
    parser.add_argument('--gpu',             default=0,      type=int)
    parser.add_argument('--log_file',        required=True,  type=str,
                        help='Path to CSV log file (created if not exists, appended otherwise)')
    parser.add_argument('--round',           required=True,  type=int,
                        help='Current SDDAL round number')
    parser.add_argument('--dataset_size',    required=True,  type=int,
                        help='Current training set size (number of samples)')
    parser.add_argument('--wall_clock_s',    required=True,  type=float,
                        help='Total elapsed seconds since SDDAL loop started')
    parser.add_argument('--cumul_scanner_s', required=True,  type=float,
                        help='Cumulative seconds spent in Scanner.py across all rounds so far')
    parser.add_argument('--cumul_trainer_s', required=True,  type=float,
                        help='Cumulative seconds spent in Trainer.py across all rounds so far')
    parser.add_argument('--cumul_eval_s',   required=True,  type=float,
                        help='Cumulative seconds spent in evaluate_checkpoint.py across all previous rounds')
    parser.add_argument('--save_preds',      action='store_true',
                        help='Save predicted and GT phase maps for visual inspection. '
                             'Saved to <log_dir>/preds/round_<N>/')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- Optional prediction output directories ---
    if args.save_preds:
        pred_root = os.path.join(os.path.dirname(args.log_file), 'preds', f'round_{args.round}')
        for sub in ['Phi_pred/npy', 'Phi_pred/img', 'Phi_gt/npy', 'Phi_gt/img']:
            os.makedirs(os.path.join(pred_root, sub), exist_ok=True)

    # --- Load model ---
    model = PImodel().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print(f"[eval] Loaded checkpoint (epoch {checkpoint['epoch']})")

    # --- Load test set (expects test_data/test_set/intensity/npy + .../phase/npy) ---
    val_dataset = get_validation_data(args.test_data)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[eval] Test set size: {len(val_dataset)}")

    mae_list, ssim_list, frcm_list = [], [], []

    with torch.no_grad():
        for idx, (I, Phi) in enumerate(val_loader):
            I   = I.to(device)
            Phi = Phi.to(device)

            _, pred, _ = model(I)

            pred_np = pred.squeeze().cpu().numpy()
            gt_np   = Phi.squeeze().cpu().numpy()

            # MAE
            mae_list.append(float(np.mean(np.abs(pred_np - gt_np))))

            # SSIM
            ssim_list.append(ssim(gt_np, pred_np, data_range=gt_np.max() - gt_np.min()))

            # FRCM — computed on CPU to avoid GPU memory pressure during long runs
            frcm_list.append(calculate_frcm(
                Phi.squeeze(0).cpu(), pred.squeeze(0).cpu()))

            if args.save_preds:
                fname = f"{idx:04d}"
                np.save(os.path.join(pred_root, 'Phi_pred', 'npy', f"{fname}.npy"), pred_np)
                np.save(os.path.join(pred_root, 'Phi_gt',   'npy', f"{fname}.npy"), gt_np)
                plt.imsave(os.path.join(pred_root, 'Phi_pred', 'img', f"{fname}.png"), pred_np, cmap='gray')
                plt.imsave(os.path.join(pred_root, 'Phi_gt',   'img', f"{fname}.png"), gt_np,   cmap='gray')

    mean_mae  = float(np.mean(mae_list))
    mean_ssim = float(np.mean(ssim_list))
    mean_frcm = float(np.mean(frcm_list))

    print(f"[eval] round={args.round}  dataset_size={args.dataset_size}")
    print(f"[eval] MAE={mean_mae:.6f}  SSIM={mean_ssim:.6f}  FRCM={mean_frcm:.6f}")
    print(f"[eval] wall_clock={args.wall_clock_s:.1f}s  "
          f"cumul_trainer={args.cumul_trainer_s:.1f}s  "
          f"cumul_scanner={args.cumul_scanner_s:.1f}s  "
          f"cumul_eval={args.cumul_eval_s:.1f}s")

    # --- Append to CSV log ---
    write_header = not os.path.exists(args.log_file)
    with open(args.log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['round', 'dataset_size', 'wall_clock_s',
                             'cumul_trainer_s', 'cumul_scanner_s', 'cumul_eval_s',
                             'mae', 'ssim', 'frcm'])
        writer.writerow([
            args.round,
            args.dataset_size,
            f"{args.wall_clock_s:.1f}",
            f"{args.cumul_trainer_s:.1f}",
            f"{args.cumul_scanner_s:.1f}",
            f"{args.cumul_eval_s:.1f}",
            f"{mean_mae:.6f}",
            f"{mean_ssim:.6f}",
            f"{mean_frcm:.6f}",
        ])

    print(f"[eval] Results appended to {args.log_file}")
    if args.save_preds:
        print(f"[eval] Predictions saved to {pred_root}/")


if __name__ == '__main__':
    main()
