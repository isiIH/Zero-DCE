import pyiqa
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm

from model import DCE_Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
niqe_metric = pyiqa.create_metric('niqe', device=device)

def test_wo_ref(checkpoint_path, image_dir, save_path=None):

    model = DCE_Net().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    model.eval()

    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = sorted([f for f in Path(image_dir).iterdir() if f.is_file() and f.suffix.lower() in valid_ext])

    total_niqe = 0.0
    with torch.no_grad():
        for f in tqdm(files, desc="Evaluando No-Ref"):
            low_tensor = TF.to_tensor(Image.open(f).convert('RGB')).unsqueeze(0).to(device)
            enh_tensor, _ = model(low_tensor)
            enh_tensor = torch.clamp(enh_tensor, 0.0, 1.0)

            enhanced_img = TF.to_pil_image(enh_tensor.squeeze(0).cpu())

            if save_path:
                enhanced_img.save(save_path)

            total_niqe += niqe_metric(enh_tensor).item()

    n = len(files)
    print(f"\n NIQE Score with {n} images ({Path(image_dir).name}):")
    print(f"  NIQE (↓) : {total_niqe / n:.4f}")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/best_model_dce_net.pth"
    dataset = "real"
    image_dir = "../data/test_data/" + dataset + "/"
    save_path = "results/" + dataset + "/"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    test_wo_ref(checkpoint_path, image_dir, save_path)