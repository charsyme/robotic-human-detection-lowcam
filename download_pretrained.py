import gdown
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract COCO dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["YOLO11n", "YOLO11s"],
        default="YOLO11n",
        help="YOLO model."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        required=False,
        default='./out/models/',
        help="COCO years."
    )
    args = parser.parse_args()

    save_dir=os.path.join(args.save_dir, args.model)
    file_ids = {
        'YOLO11s': '1AWXD7abu0v_5UAVNop_X-vP_cEFdRZmj',
        'YOLO11n': '1IOtWK8f5xOICfALLE_qdJyH_SP9NWZTJ'
    }
    os.makedirs(save_dir, exist_ok=True)
    output = os.path.join(save_dir, f'{args.model}_pretrained.pth')

    # Build the download URL
    url = f'https://drive.google.com/uc?id={file_ids[args.model]}'

    # Download the file
    gdown.download(url=url, output=output, quiet=False)