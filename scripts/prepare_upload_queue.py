import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "output" / "upload_queue.csv"
UPLOAD_READY = ROOT / "output" / "upload_ready"


def safe_name(value):
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def main():
    if not QUEUE_PATH.exists():
        raise FileNotFoundError(f"Queue file not found: {QUEUE_PATH}")

    UPLOAD_READY.mkdir(parents=True, exist_ok=True)
    captions_path = UPLOAD_READY / "captions.txt"

    copied = 0
    with QUEUE_PATH.open("r", newline="", encoding="utf-8") as queue_file, captions_path.open(
        "w", encoding="utf-8"
    ) as captions_file:
        reader = csv.DictReader(queue_file)
        for index, row in enumerate(reader, start=1):
            source = Path(row["file_path"])
            if row.get("status") != "ready" or not source.exists():
                continue

            theme = safe_name(row["theme"])
            resolution = safe_name(row["resolution"])
            target_name = f"{index:03d}_{theme}_{resolution}_{source.name}"
            target = UPLOAD_READY / target_name
            shutil.copy2(source, target)
            captions_file.write(f"{target_name}\n")
            captions_file.write(f"{row['caption']}\n\n")
            copied += 1

    print(f"Prepared {copied} files in {UPLOAD_READY}")
    print(f"Captions: {captions_path}")


if __name__ == "__main__":
    main()
