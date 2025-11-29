import torch
from tqdm import tqdm
import wandb

from dual_head_model import TransformerSVGNetSmall
from train_utils import validate_one_epoch, train_one_epoch
from svg_load import tensor_to_svg_string



def initialize_model_and_optimizer(device):
    model = TransformerSVGNetSmall(
        max_paths=4,
        max_curves_per_path=4+1, # Increased capacity per path
        image_size=128,
        latent_dim=2048
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return model, optimizer


def intitate_wandb_run(config):
    num_epochs = config["num_epochs"]
    lambda_svg_start = config["lambda_svg_start"]
    lambda_mask_start = config["lambda_mask_start"]
    lambda_raster_start = config["lambda_raster_start"]
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="yimoninguoft-university-of-toronto",
        # Set the wandb project where this run will be logged.
        project="shared_encoder",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.001,
            "architecture": "TransformerSVGNetSmall",
            "epochs": num_epochs,
            "lambda_svg": lambda_svg_start,
            "lambda_mask": lambda_mask_start,
            "lambda_raster": lambda_raster_start,
            "dataset": "EmojySvg",
            "decay": False
        },
    )
    return run

def execute_run(model, optimizer, train_loader, test_loader, run, config, device):
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(config["num_epochs"])):

        # --- Linear Schedule ---
        lambda_svg = config["lambda_svg_start"]
        lambda_mask = config["lambda_mask_start"]
        lambda_raster = config["lambda_raster_start"]
        # ------------------------
        train_loss, lraster, lm, lv = train_one_epoch(
                model, train_loader, optimizer, device,
                lambda_svg=lambda_svg, lambda_mask=lambda_mask, lambda_raster=lambda_raster
            )
        val_stats = validate_one_epoch(
            model, test_loader, device,
            lambda_svg=lambda_svg, lambda_mask=lambda_mask, lambda_raster=lambda_raster
        )

        if epoch == 0:
            continue

        if epoch % 50 == 1:
            print(f"""
            epoch: loss: f{train_loss / len(train_loader):.4f}, r: f{lraster:.4f}, m: {lm:4f}, v: f{lv:.4f}
                """)
            print(f"Validatinon val={val_stats['loss']:.4f}, r={val_stats['r']:.4f}, fm={val_stats['m']:.4f}, fv={val_stats['v']:.4f}")  

        run.log({"train_loss": train_loss / len(train_loader), 
                "train_l_raster": lraster, 
                "train_l_mask": lm, 
                "train_l_svg": lv,
                "val_loss": val_stats['loss'],
                "val_l_raster": val_stats['r'],
                "val_l_mask": val_stats['m'],
                "val_l_svg": val_stats['v'],
                })

        train_losses.append(train_loss)
        val_losses.append(val_stats['loss'])

    return train_losses, val_losses

def log_artifacts(model, run, test_set):
    imgs = []
    true_images = []
    for idx in range(0, 10):
        img = test_set[idx]["ground_truth_image"]                    # (C, H, W)
        img_copy = img.permute(1, 2, 0).detach().cpu().numpy()
        true_images.append(img_copy)
        imgs.append(img)

    batch = torch.stack(imgs, dim=0).to(device)       

    with torch.no_grad():
        pred_raster, pred_svg, pred_mask_logits = model(batch)

    pred_imgs = []

    for i in range(0, 10):
        img = pred_raster[i].permute(1, 2, 0).detach().cpu().numpy()
        pred_imgs.append(img)


    pred_mask_probs = torch.sigmoid(pred_mask_logits)

    artifact = wandb.Artifact("final_Svgs", type="results")
    # --- Just iterate and render each SVG separately ---
    for i in range(10):
        svg_np = pred_svg[i].detach().cpu().numpy()
        mask_np = pred_mask_probs[i].detach().cpu().numpy()

        svg_string = tensor_to_svg_string(svg_np, mask_np, stroke_width=0.1)

        with open(f"sample{i}.svg", "w") as f:
            f.write(svg_string)
        artifact.add_file(f"sample{i}.svg")
        

    run.log({
        "true_images": [wandb.Image(img) for img in true_images],
        "predicted_images": [wandb.Image(img) for img in pred_imgs]
    })

    run.log_artifact(artifact)

    return