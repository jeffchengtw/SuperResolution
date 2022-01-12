import os

import torch
from PIL import Image
from natsort import natsorted

import config
import preprocess
from model import Generator

def main():
    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the super-resolution model
    print("Build SR model...")
    model = Generator().to(config.device)
    print("Build SR model successfully.")

    # Load the super-resolution model weights
    print(f"Load SR model weights `{os.path.abspath(config.model_path)}`...")
    state_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    print(f"Load SR model weights `{os.path.abspath(config.model_path)}` successfully.")

    model.eval()
    model.half()
        # Initialize the image evaluation index.
    total_psnr = 0.0

    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        (filename,extension) = os.path.splitext(file_names[index])
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, filename + '_pred' + extension)
        #hr_image_path = os.path.join(config.hr_dir, file_names[index])

        
        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_image = Image.open(lr_image_path).convert("RGB")
        #hr_image = Image.open(hr_image_path).convert("RGB")

        # Extract RGB channel image data
        lr_tensor = preprocess.image2tensor(lr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
        #hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp_(0, 1)

        # Cal PSNR
        sr_y_tensor = preprocess.convert_rgb_to_y(sr_tensor)
        #hr_y_tensor = imgproc.convert_rgb_to_y(hr_tensor)
        #total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))

        sr_image = preprocess.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = Image.fromarray(sr_image)
        sr_image.save(sr_image_path)

    print(f"PSNR: {total_psnr / total_files:.2f} dB.\n")


if __name__ == "__main__":
    main()