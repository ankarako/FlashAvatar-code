import argparse
import torch
import numpy as np
import tloaders
from tqdm import tqdm
from face_parsing.face_parsing_pytorch.model import BiSeNet
from face_parsing.face_parsing_pytorch.test import vis_parsing_maps
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import os

k_chkp_path = "./face_parsing/chkp/79999_iter.pth"
k_face_parsing_dirname = 'face_parsing'

k_mouth_cls = 11
k_up_lip_cls = 12
k_lo_lip_cls = 13

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Preprocess GaussianAvatars dataset with face parsing")
    parser.add_argument("--data_root", type=str, help="The root directory with the dataset's files")
    args = parser.parse_args()

    # load dataset
    dataset = tloaders.DatasetRegistry.get_dataset(
        "GaussianAvatars", 
        data_root=args.data_root, 
        transforms_filename="transforms_train.json",
        prefetch=False
    )

    # load face parsing model
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(k_chkp_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        loop = tqdm(
            enumerate(dataset),
            desc="Parsing dataset",
            total=len(dataset)
        )
        sample: tloaders.FlameSample
        for idx, sample in loop:
            # convert the image back to PIL so we
            # can use face parsing's transform
            image = sample['rgb']
            image = (image.numpy() * 255.0).astype(np.uint8)
            image = Image.fromarray(image)

            img = to_tensor(image)
            img = torch.unsqueeze(img, dim=0)
            img = img.cuda()
            out = net(img)[0]

            # get lips mask
            mouth = out[:, k_mouth_cls, ...]
            ulip = out[:, k_up_lip_cls, ...]
            llip = out[:, k_lo_lip_cls, ...]
            lip_msk = torch.clip(ulip + llip + mouth, 0, 1)


            # get save path
            image_path = dataset.cam_infos[idx].image_path
            path_parts = image_path.parts

            # find where the parent directory is
            idx = path_parts.index('..')
            tail = path_parts[idx+1:]

            output_dir = os.path.join(args.data_root, '..', tail[0], 'parsing')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            filename = f"{tail[-1].replace('.png', '')}_mouth.png"
            filepath = os.path.join(output_dir, filename)

            # save parsing image
            lip_msk_out = (lip_msk.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            lip_msk = Image.fromarray(lip_msk_out)
            lip_msk.save(filepath)