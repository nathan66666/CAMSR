class FileClient:
    """Read LR/HR image bytes from disk."""

    def __init__(self, backend="disk", **kwargs):
        if backend != "disk":
            raise ValueError(f"Unsupported backend: {backend}. CAMSR uses disk images.")
        if kwargs:
            raise TypeError(f"Unexpected disk backend options: {sorted(kwargs)}")

    def get(self, filepath, client_key="default"):
        with open(filepath, "rb") as file:
            return file.read()
