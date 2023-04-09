from pathlib import Path
import os
import cv2
import argparse


def extract_frames(data_path: str, fname: str):
    data_path = Path(data_path)
    frames_path = data_path / fname
    

    cap = cv2.VideoCapture(frames_path.as_posix())
    
    ret, frame = cap.read()
    count = 0
    
    while ret:
        cv2.imwrite(data_path.as_posix() + f"/frame{count}.png", frame)

        ret, frame = cap.read()
        count += 1
    
    cap.release()
    os.remove(data_path / fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/train")
    parser.add_argument("--fname", type=str, default="train.mp4")
    args = parser.parse_args()

    extract_frames(data_path=args.data_path, fname=args.fname)