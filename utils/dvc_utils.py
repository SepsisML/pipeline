def get_dvc_hash(file_path):
    # Este comando obtiene el hash del archivo .dvc o la URL remota si estás usando DVC remote
    result = subprocess.run(
        ["dvc", "get", ".", file_path, "--show-url"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )