from torchvision.datasets.utils import download_url
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./data/train")
    parser.add_argument("--test_dir", type=str, default="./data/test")
    args = parser.parse_args()

    # Download train.mp4 from github.com/commaai/speedchallenge
    download_url(url="https://github.com/commaai/speedchallenge/blob/master/data/train.mp4?raw=True",
                 root=args.train_dir,
                 filename="train.mp4")
    # Download train.txt from github.com/commaai/speedchallenge
    download_url(url="https://github.com/commaai/speedchallenge/blob/master/data/train.txt?raw=True",
                 root=args.train_dir,
                 filename="train.txt")
    # Download test.mp4 from github.com/commaai/speedchallenge
    download_url(url="https://github.com/commaai/speedchallenge/blob/master/data/test.mp4?raw=True",
                 root=args.test_dir,
                 filename="test.mp4")