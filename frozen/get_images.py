import os
import csv
import requests
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Paths
DATA_DIR = "datasets/DialogCC"
IMAGE_NAMES_DIR = os.path.join(DATA_DIR, "image_names")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_FILES = ["train.csv", "test.csv"]

# Ensure directories exist
os.makedirs(IMAGE_NAMES_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# Function to download a single image
def download_image(image):
    url, image_name = image
    try:
        session = requests.Session()
        retries = requests.adapters.Retry(total=1, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
        response = session.get(url, timeout=(1, 2))
        if response.status_code == 200:
            save_path = os.path.join(IMAGE_DIR, image_name)
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

# Function to extract image URLs from a CSV file and save them in txt files
def extract_images(csv_file):
    images = []
    with open(os.path.join(DATA_DIR, csv_file), 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogue = eval(row["dialogue"])  # Safely parse the dialogue
            for utterance in dialogue:
                for image in utterance.get("shared_image", []):
                    url = image.get("image_url")
                    if url:
                        image_name = f"{csv_file.split('.')[0]}_{len(images)}.jpg"
                        images.append((url, image_name))

    # Save URLs to txt file
    txt_file_path = os.path.join(IMAGE_NAMES_DIR, f"{csv_file.split('.')[0]}_image_urls.txt")
    with open(txt_file_path, 'w') as txt_file:
        for url, image_name in images:
            txt_file.write(f"{url}\t{image_name}\n")
    
    print(f"Extracted {len(images)} image URLs from {csv_file}.")
    return images

# Main function to process datasets and download images
def process_and_download():
    all_images = []
    for csv_file in CSV_FILES:
        images = extract_images(csv_file)
        all_images.extend(images)
    
    # Download images using multiprocessing
    print(f"Starting download of {len(all_images)} images...")
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(download_image, all_images), total=len(all_images), desc="Downloading images"))

if __name__ == '__main__':
    process_and_download()
    print("Image extraction and download complete.")
