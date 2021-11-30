import argparse
import os

import lpips
import numpy as np
import torchmetrics
import torchvision

from gradattack.metrics.pixelwise import MeanPixelwiseError

PSNR = torchmetrics.functional.psnr
lpips_fn = lpips.LPIPS(net="vgg")


def parse_args():
    parser = argparse.ArgumentParser(description="gradattack training")
    parser.add_argument("--dir", default=None, type=str)
    parser.add_argument("--decode_defense", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()
rootdir = args.dir

rmse, psnr, lpips = [], [], []

if not args.decode_defense:
    for subdir, dirs, files in os.walk(rootdir):
        for file in (f for f in files
                     if f.startswith("reconstructed") and f.endswith(".npy")):
            recovered = np.load(os.path.join(subdir, "reconstructed.npy"))
            transform = np.load(os.path.join(subdir, "transformed.npy"))

            for r in recovered:
                tmp_rmse, tmp_psnr, tmp_lpips = [], [], []
                for t in transform:
                    recovered_tensor = torchvision.transforms.functional.to_tensor(
                        r)
                    transform_tensor = torchvision.transforms.functional.to_tensor(
                        t)
                    tmp_rmse.append(
                        MeanPixelwiseError(recovered_tensor, transform_tensor))
                    tmp_psnr.append(PSNR(recovered_tensor, transform_tensor))
                    tmp_lpips.append(
                        lpips_fn(recovered_tensor,
                                 transform_tensor).detach().numpy())
                rmse.append(min(tmp_rmse))
                psnr.append(max(tmp_psnr))
                lpips.append(min(tmp_lpips))
else:
    originals = np.load(os.path.join(rootdir, "originals.npy"))
    recovereds = np.load(os.path.join(rootdir, "grad_decode.npy"))

    for idx, original in enumerate(originals):
        original = torchvision.transforms.functional.to_tensor(original)
        recovered = torchvision.transforms.functional.to_tensor(
            recovereds[idx])

        rmse.append(MeanPixelwiseError(original, recovered))
        psnr.append(PSNR(recovered, original))
        lpips.append(lpips_fn(original, recovered).detach().numpy())

print(
    f"##### Avg. PSNR: {np.mean(psnr)} \t Avg. RMSE: {np.mean(rmse)} \t Avg. LPIPS: {np.mean(lpips)} #####"
)
print(
    f"##### Best PSNR: {np.max(psnr)} \t Best RMSE: {np.min(rmse)} \t Best LPIPS: {np.min(lpips)} #####"
)
print(
    f"##### PSNR std: {np.std(psnr)} \t RMSE std: {np.std(rmse)} \t LPIPS std: {np.std(lpips)} #####"
)
