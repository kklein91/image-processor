import os
import time
from typing import List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_files_sorted_by_ctime(folder: str) -> List[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder) if is_image_file(f)]
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: os.path.getctime(p))
    return files


def move_file_to_folder(src: str, dst_folder: str) -> str:
    ensure_dir(dst_folder)
    dst = os.path.join(dst_folder, os.path.basename(src))
    # If destination exists, append timestamp
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        dst = f"{base}_{int(time.time())}{ext}"
    os.replace(src, dst)
    return dst
