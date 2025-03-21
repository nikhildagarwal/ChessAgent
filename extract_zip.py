import os
import zipfile


def extract_zip_files(directory):
    # Loop through all files in the specified directory
    for filename in os.listdir(directory):
        # Check if the file is a ZIP file by extension
        if filename.lower().endswith('.zip'):
            file_path = os.path.join(directory, filename)
            # Define a folder name for extraction (same as ZIP name without extension)
            extract_folder = os.path.join(directory, filename[:-4])
            os.makedirs(extract_folder, exist_ok=True)

            # Open the ZIP file and extract its contents
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)
                print(f'Extracted "{filename}" to folder: "{extract_folder}"')

extract_zip_files("./extractor")