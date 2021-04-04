import requests
import logging
from zipfile import ZipFile

logger = logging.getLogger(__name__)


def download_flickr8k(dest_path):
    URL = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/'
    FILENAMES = ['Flickr8k_text.zip', 'Flickr8k_Dataset.zip']

    for fn in FILENAMES:
        if not (dest_path/fn).exists():  # do not download if the dataset has already been downloaded
            logger.info(f"Downloading {fn}")
            r = requests.get(URL + fn)
            (dest_path/fn).write_bytes(r.content)
            logger.info(f"Extracting files...")
            with ZipFile(dest_path/fn) as zipf:
                zipf.extractall(dest_path)
