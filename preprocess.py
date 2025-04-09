import os
import shutil
import zipfile
import pandas as pd
import datetime
import json

# Path to the network directory (read-only)
NETWORK_DIR = r'\\your_network_path'  # Example: r'\\server\folder'

# Local directory to copy the latest file to
LOCAL_DIR = r'C:\local_folder'

# Function to get the latest zip file by modification date
def get_latest_zip_file(network_dir):
    latest_file = None
    latest_time = datetime.datetime.min
    
    # Traverse all files in the network directory
    for root, dirs, files in os.walk(network_dir):
        for file in files:
            if file.endswith('.zip'):
                file_path = os.path.join(root, file)
                # Check the file's last modification date
                file_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_path

    return latest_file

# Function to copy the latest file from the network directory to the local directory
def copy_latest_file_to_local(latest_file, local_dir):
    if latest_file:
        shutil.copy(latest_file, local_dir)
        print(f"File {latest_file} copied to {local_dir}")
        return os.path.join(local_dir, os.path.basename(latest_file))
    else:
        print("No zip files found.")
        return None

# Function to extract a zip file to a local directory
def extract_zip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files from {zip_path} extracted to {extract_to}")

# Function to collect metadata from CSV files
def collect_metadata_from_csv(df):
    metadata = {
        "columns": [],
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_types": {},
        "missing_values": {}
    }

    for col in df.columns:
        metadata["columns"].append(col)
        metadata["column_types"][col] = str(df[col].dtype)
        missing = df[col].isnull().sum()
        metadata["missing_values"][col] = missing
    
    return metadata

# Function to process CSV files from the extracted directory
def process_csv_files(extracted_dir):
    metadata_all = {}

    for root, dirs, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                
                df = pd.read_csv(file_path)
                file_metadata = collect_metadata_from_csv(df)
                metadata_all[file] = file_metadata

    return metadata_all

# Function to perform all steps
def process_latest_zip_and_extract_metadata(network_dir, local_dir):
    # Step 1: Get the latest zip file
    latest_zip_file = get_latest_zip_file(network_dir)
    
    if not latest_zip_file:
        print("No zip file found.")
        return

    # Step 2: Copy the latest zip file to the local directory
    local_zip_path = copy_latest_file_to_local(latest_zip_file, local_dir)
    
    if not local_zip_path:
        return

    # Step 3: Extract the zip file
    extract_dir = os.path.join(local_dir, os.path.splitext(os.path.basename(local_zip_path))[0])
    os.makedirs(extract_dir, exist_ok=True)
    extract_zip_file(local_zip_path, extract_dir)

    # Step 4: Extract metadata from CSV files
    metadata = process_csv_files(extract_dir)

    # Save the metadata to a JSON file
    with open(os.path.join(local_dir, 'metadata.json'), 'w', encoding='utf-8') as json_file:
        json.dump(metadata, json_file, ensure_ascii=False, indent=4)

    print(f"Metadata saved to 'metadata.json'.")

# Example usage:
if __name__ == "__main__":
    process_latest_zip_and_extract_metadata(NETWORK_DIR, LOCAL_DIR)
