import os

def save_unique_file(upload_dir, filename, file_bytes):
    """
    Сохраняет файл с уникальным именем, если такое уже существует.
    """
    os.makedirs(upload_dir, exist_ok=True)
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(upload_dir, new_filename)):
        new_filename = f"{base}({counter}){ext}"
        counter += 1

    file_path = os.path.join(upload_dir, new_filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path
