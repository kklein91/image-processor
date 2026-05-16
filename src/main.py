import time
import argparse
import threading
import os
import logging

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import Config
from .processor import BowlingProcessor
from .utils import is_image_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class NewImageHandler(FileSystemEventHandler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_created(self, event):
        if event.is_directory:
            return
        if is_image_file(event.src_path):
            # small delay to allow file to finish writing
            time.sleep(0.2)
            logging.info(f"New image detected: {event.src_path}")
            self.queue.append(event.src_path)


def run_watcher(cfg: Config):
    processor = BowlingProcessor(cfg)
    q = []

    handler = NewImageHandler(q)
    observer = Observer()
    observer.schedule(handler, cfg.input_folder, recursive=False)
    observer.start()

    logging.info(f"Watching folder: {cfg.input_folder}")
    try:
        while not processor.is_game_over():
            if q:
                path = q.pop(0)
                try:
                    res = processor.process_image(path)
                    current = res.get('current_score')
                    logging.info(f"Processed: frame={res['frame']} ball={res['ball']} image={res['image']} knocked={res['knocked']} current_score={current}")
                except Exception as e:
                    logging.exception(f"Error processing {path}: {e}")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        observer.stop()
        observer.join()

    # Print summary
    logging.info("Game finished — summary:")
    total = 0
    for fr in processor.frames_results:
        logging.info(f"Frame {fr['frame']}: score={fr['score']} balls={fr['balls']}")
        total += fr['score']
    logging.info(f"Total score: {total}")
    # Cleanup processed frame files when the game is over
    try:
        _cleanup_processed_frames(cfg.processed_root)
    except Exception:
        logging.exception("Error during cleanup of processed frames")


def ensure_input_folder(path: str):
    os.makedirs(path, exist_ok=True)


def _cleanup_processed_frames(processed_root: str) -> None:
    """Remove all files inside processed frame folders (frame_*) and remove
    the empty frame directories.
    """
    if not os.path.isdir(processed_root):
        return

    for name in os.listdir(processed_root):
        if not name.startswith("frame_"):
            continue
        frame_dir = os.path.join(processed_root, name)
        if not os.path.isdir(frame_dir):
            continue
        for root, _, files in os.walk(frame_dir):
            for fname in files:
                try:
                    os.remove(os.path.join(root, fname))
                except Exception:
                    logging.exception(f"Failed to remove file: {os.path.join(root, fname)}")
        # attempt to remove the now-empty directory
        try:
            os.rmdir(frame_dir)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Input folder to watch", default="bowling_frames")
    parser.add_argument("--processed", help="Processed output root folder", default="processed_frames")
    args = parser.parse_args()

    cfg = Config(input_folder=args.input, processed_root=args.processed)
    ensure_input_folder(cfg.input_folder)
    run_watcher(cfg)


if __name__ == "__main__":
    main()
