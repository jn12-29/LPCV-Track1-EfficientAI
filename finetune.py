import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import open_clip

from dataset import ImageTextRetrievalDataset


def _load_clip(model_name: str, device: torch.device):
    print(f"Loading CLIP model '{model_name}'...")
    available_models_tuple = open_clip.list_pretrained()
    available_models_dict = {}
    for temp in available_models_tuple:
        name, ckpt = temp
        if name not in available_models_dict:
            available_models_dict[name] = ckpt
    if model_name in available_models_dict.keys():
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=available_models_dict[model_name]
        )
    else:
        raise ValueError(f"Available models: {open_clip.list_pretrained()}")
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer


def create_collate_fn(tokenizer):
    def collate_fn(batch):
        images = torch.stack([item["image"] for item in batch])
        texts = [item["text"] for item in batch]
        text_tokens = tokenizer(texts)
        return images, text_tokens

    return collate_fn


def train_epoch(model, dataloader, optimizer, scaler, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, texts) in enumerate(dataloader):
        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()

        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            loss = loss_fn(image_features, text_features, logit_scale)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch: {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}"
            )

    return total_loss / len(dataloader)


def run_clip_finetune(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, preprocess, tokenizer = _load_clip(args.model_name, device)

    train_dataset = ImageTextRetrievalDataset(
        root_dir=args.root_dir,
        image_to_text_csv=args.image_to_text_csv,
        textnums_to_texts_csv=args.textnums_to_texts_csv,
        image_transform=preprocess,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=create_collate_fn(tokenizer),
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = open_clip.ClipLoss()
    scaler = GradScaler()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(
            model, train_dataloader, optimizer, scaler, loss_fn, device, epoch
        )
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        save_path = (
            output_dir
            / f"clip_finetuned_{args.model_name.replace('/', '-')}_epoch_{epoch}.pt"
        )
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCLIP Fine-tuning Pipeline")
    parser.add_argument("--root-dir", type=str, default="./sample_data")
    parser.add_argument(
        "--image-to-text-csv", type=str, default="./sample_data/img_list.csv"
    )
    parser.add_argument(
        "--textnums-to-texts-csv", type=str, default="./sample_data/txt_list.csv"
    )
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--model-name", type=str, default="MobileCLIP2-S0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_clip_finetune(args)
