import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



def hybrid_loss(pred_raster, gt_raster,
                pred_svg_coords, gt_svg_coords,
                pred_mask_logits, gt_mask,
                lambda_svg=1.0, lambda_mask=1.0, lambda_raster = 1.0):
    """
    pred_svg_coords: (B, N_Paths, N_Curves, 4, 2)
    pred_mask_logits: (B, N_Paths, N_Curves)
    gt_mask: (B, N_Paths, N_Curves)
    """

    # 1. Raster Loss
    loss_r = F.mse_loss(pred_raster,gt_raster)

    # 2. Mask Loss (BCE)
    loss_m = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask)

    # 3. SVG Coordinate Loss (Masked)
    # Expand mask to (B, N_Paths, N_Curves, 1, 1) for broadcasting over (4, 2)
    mask_expanded = gt_mask.view(*gt_mask.shape, 1, 1)

    # Compute Smooth L1 for all params
    coord_loss_raw = F.smooth_l1_loss(pred_svg_coords, gt_svg_coords, reduction='none')

    # Mask out padding
    coord_loss_masked = coord_loss_raw * mask_expanded

    # Normalize by sum of valid points (avoid div by zero)
    num_valid_elements = mask_expanded.sum() * 8  # 8 coords per valid curve
    loss_v = coord_loss_masked.sum() / (num_valid_elements + 1e-6)

    total_loss = (loss_r * lambda_raster) + (lambda_mask * loss_m) + (lambda_svg * loss_v)

    return total_loss, loss_r.detach(), loss_m.detach(), loss_v.detach()

def train_one_epoch(model, dataloader, optimizer, device, lambda_svg=0.5, lambda_mask=0.5, lambda_raster=0.5, verbose=False):
    model.train()
    total_loss = 0.0
    lr, lm, lv = 0.0, 0.0, 0.0

    # pbar = tqdm(dataloader, desc="train")
    for batch in dataloader:
        inp = batch["image"].to(device)
        clean = batch["ground_truth_image"].to(device)
        gt_svg = batch["svg_params"].to(device)
        gt_mask = batch["svg_mask"].to(device)

        optimizer.zero_grad()

        # Forward
        pred_raster, pred_svg, pred_mask_logits = model(inp)

        # Loss
        loss, lr_, lm_, lv_ = hybrid_loss(
            pred_raster, clean,
            pred_svg, gt_svg,
            pred_mask_logits, gt_mask,
            lambda_svg=lambda_svg, lambda_mask=lambda_mask,
            lambda_raster = lambda_raster
        )

        lr += lr_.item()
        lm += lm_.item()
        lv += lv_.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
      
    # pbar.set_postfix({
    #     "loss": f"{total_loss / (1 + pbar.n):.4f}",
    #     "r": f"{lr:.3f}",
    #     "m": f"{lm:.3f}",
    #     "v": f"{lv:.3f}"
    # })
    if verbose:
      print(f"""
          epoch: loss: f{total_loss / len(dataloader):.4f}, r: f{lr:.4f}, m: {lm:4f}, v: f{lv:.4f}
            """
        )

    return total_loss / len(dataloader)

@torch.no_grad()
def validate_one_epoch(model, dataloader, device,
                       lambda_svg=0.5, lambda_mask=0.5, lambda_raster=0.5):
    model.eval()
    total_loss = 0.0
    lr, lm, lv = 0.0, 0.0, 0.0

    for batch in dataloader:
        inp = batch["image"].to(device)
        clean = batch["ground_truth_image"].to(device)
        gt_svg = batch["svg_params"].to(device)
        gt_mask = batch["svg_mask"].to(device)

        # forward only
        pred_raster, pred_svg, pred_mask_logits = model(inp)

        loss, lr_, lm_, lv_ = hybrid_loss(
            pred_raster, clean,
            pred_svg, gt_svg,
            pred_mask_logits, gt_mask,
            lambda_svg=lambda_svg, lambda_mask=lambda_mask,
            lambda_raster=lambda_raster,
        )

        lr += lr_.item()
        lm += lm_.item()
        lv += lv_.item()
        total_loss += loss.item()

    return {
        "loss": total_loss / len(dataloader),
        "r": lr / len(dataloader),
        "m": lm / len(dataloader),
        "v": lv / len(dataloader),
    }
