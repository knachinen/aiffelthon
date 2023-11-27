import pickle
import logging
import os
import datetime
import shutil
import joblib
import json

def ensure_directory_exists(directory: str) -> None:
    """
    Ensures a directory exists, creates it if it doesn't.

    Parameters:
    - directory: The directory path.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_file_path(name: str) -> str:
    """
    Generates a file path for saving a variable.

    Parameters:
    - name: The name of the variable.

    Returns:
    - str: The generated file path.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{name}_{timestamp}.pkl"
    return os.path.join('save', file_name)

def rename_existing_file(file_path: str) -> None:
    """
    Renames an existing file by appending the current timestamp.

    Parameters:
    - file_path: The path of the existing file.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    new_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_{timestamp}.pkl"
    new_file_path = os.path.join('save', new_name)
    os.rename(file_path, new_file_path)
    logging.info(f'Renamed existing file to: {new_file_path}')

def save_var(var, name: str = "var") -> None:
    """
    Saves a variable to a pickle file.

    Parameters:
    - var: The variable to be saved.
    - name (optional): The name of the pickle file (without extension). Defaults to "var".
    """
    ensure_directory_exists('save')
    file_path = os.path.join('save', f'{name}.pkl')

    if os.path.exists(file_path):
        rename_existing_file(file_path)

    try:
        with open(file_path, 'wb') as f:
            joblib.dump(var, f)

        file_size = os.path.getsize(file_path) / (1024 ** 2)  # Get size in MB
        logging.info(f'Saved: {file_path} (Size: {file_size:.2f} MB)')
    except Exception as e:
        logging.error(f"Error saving variable to {file_path}: {e}")

def load_var(name: str = "var"):
    """
    Loads a variable from a pickle file.
    
    Parameters:
    - name (optional): The name of the pickle file (without extension). Defaults to "var".
    
    Returns:
    - Loaded variable.
    """
    ensure_directory_exists('save')
    file_path = os.path.join('save', f'{name}.pkl')
    
    try:
        with open(file_path, 'rb') as f:
            # var = pickle.load(f)
            var = joblib.load(f)
        
        file_size = os.path.getsize(file_path) / (1024 ** 2)  # Get size in MB
        logging.info(f'Loaded: {file_path} (Size: {file_size:.2f} MB)')
        return var
    except Exception as e:
        logging.error(f"Error loading variable from {file_path}: {e}")
        return None

def save_text(text, name: str = "./noname.txt") -> None:
    """
    Saves a text to a text file.

    Parameters:
    - text: The text to be saved.
    - name (optional): The name of the text file. Defaults to "./noname.txt".
    """

    file_name = name.split('/')[-1]
    file_path = '/'.join(name.split('/')[:-1])
    ensure_directory_exists(file_path)
    save_path = os.path.join(file_path, f'{file_path}')

    if os.path.exists(file_path):
        rename_existing_file(file_path)

    try:
        with open(file_path, 'w') as f:
            f.write(text)

        file_size = os.path.getsize(file_path) / (1024 ** 2)  # Get size in MB
        logging.info(f'Saved: {file_path} (Size: {file_size:.2f} MB)')
        
    except Exception as e:
        logging.error(f"Error saving variable to {file_path}: {e}")

def load_text(name: str = "./noname.txt") -> str:
    """
    Loads text from a text file.

    Parameters:
    - name (optional): The name of the text file. Defaults to "./noname.txt".
    """

    try:
        with open(name, 'r') as f:
            text = f.read()
        
        file_size = os.path.getsize(name) / (1024 ** 2)  # Get size in MB
        logging.info(f'Loaded: {name} (Size: {file_size:.2f} MB)')
        return text
        
    except Exception as e:
        logging.error(f"Error loading text from {name}: {e}")
        return ""

def zip_directories(path="."):
    """
    Zip directories in the given path into separate zip files.
    
    Args:
        path (str): The path where the directories to be zipped are located. Defaults to the current directory.
    """
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.make_archive(item_path, 'zip', item_path)
            print(f"Zipped directory: {item}")

            
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(file_path):
    with open(file_path) as f:
        tmp_json = json.load(f)
        return tmp_json
