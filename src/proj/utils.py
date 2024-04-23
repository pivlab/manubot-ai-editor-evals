def simplify_filename(filename: str) -> str:
    """
    Given a filename, it returns a simplified version of it by removing
    special characters and spaces, and converting it to lowercase.
    """
    return "".join(
        c if c.isalnum() or c in ("-", ".") else "_" for c in filename.lower()
    )
