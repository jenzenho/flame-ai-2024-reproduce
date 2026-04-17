from pathlib import Path
import tarfile
import urllib.request

OWNER = "jenzenho"
REPO = "flame-ai-2024-reproduce"
ASSET = "model_outputs_v1.tar.gz"
URL = f"https://github.com/{OWNER}/{REPO}/releases/latest/download/{ASSET}"

archive_path = Path(ASSET)

print(f"Downloading {URL}")
urllib.request.urlretrieve(URL, archive_path)

print(f"Extracting {archive_path}")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(".")

print("Done. model_outputs/ is now available.")
