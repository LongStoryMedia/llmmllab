"""File extension constants and utilities - centralized source of truth for file type classification."""

from typing import Set, Dict, Any

# Core text file extensions (basic text/markup files)
TEXT_EXTENSIONS: Set[str] = {
    ".txt",
    ".md",
    ".rst",
    ".rtf",
}

# Programming language extensions
CODE_EXTENSIONS: Set[str] = {
    # Web technologies
    ".html",
    ".css",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    # Python and related
    ".py",
    ".pyw",
    ".pyx",
    ".ipynb",
    # Systems programming
    ".c",
    ".cpp",
    ".cxx",
    ".cc",
    ".h",
    ".hpp",
    ".hxx",
    ".go",
    ".rs",
    # JVM languages
    ".java",
    ".kt",
    ".scala",
    # Other compiled languages
    ".swift",
    ".dart",
    ".cs",
    # Scripting languages
    ".php",
    ".rb",
    ".r",
    ".m",
    ".pl",
    ".lua",
    # Database
    ".sql",
    # Shell scripts
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".cmd",
}

# Data/configuration file extensions
DATA_CONFIG_EXTENSIONS: Set[str] = {
    # Structured data
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    # Spreadsheet data
    ".csv",
    ".tsv",
    # Environment and docker
    ".env",
    ".dockerfile",
    ".gitignore",
    ".gitattributes",
}

# All text-based extensions (combination of above categories)
ALL_TEXT_EXTENSIONS: Set[str] = (
    TEXT_EXTENSIONS | CODE_EXTENSIONS | DATA_CONFIG_EXTENSIONS
)

# Image file extensions
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".svg",
    ".ico",
    ".avif",
    ".heic",
    ".heif",
}

# Document file extensions (non-text binary formats)
DOCUMENT_EXTENSIONS: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
}

# Archive file extensions
ARCHIVE_EXTENSIONS: Set[str] = {
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
}

# Audio/video file extensions
MEDIA_EXTENSIONS: Set[str] = {
    # Audio
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".m4a",
    # Video
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
}


def get_file_extension(filename: str) -> str:
    """
    Extract file extension from filename, normalized to lowercase.

    Args:
        filename: The filename to extract extension from

    Returns:
        File extension including the dot (e.g., ".py") or empty string if no extension
    """
    if "." not in filename:
        return ""
    extension = filename.split(".")[-1].lower()
    return f".{extension}"


def is_text_file(filename: str) -> bool:
    """Check if filename has a text-based extension."""
    ext = get_file_extension(filename)
    return ext in ALL_TEXT_EXTENSIONS


def is_code_file(filename: str) -> bool:
    """Check if filename has a programming language extension."""
    ext = get_file_extension(filename)
    return ext in CODE_EXTENSIONS


def is_image_file(filename: str) -> bool:
    """Check if filename has an image extension."""
    ext = get_file_extension(filename)
    return ext in IMAGE_EXTENSIONS


def is_document_file(filename: str) -> bool:
    """Check if filename has a document extension."""
    ext = get_file_extension(filename)
    return ext in DOCUMENT_EXTENSIONS


def get_file_category(filename: str) -> str:
    """
    Get the broad category of a file based on its extension.

    Args:
        filename: The filename to categorize

    Returns:
        Category string: "code", "text", "image", "document", "archive", "media", or "unknown"
    """
    ext = get_file_extension(filename)

    if ext in CODE_EXTENSIONS:
        return "code"
    elif ext in TEXT_EXTENSIONS:
        return "text"
    elif ext in DATA_CONFIG_EXTENSIONS:
        return "config"
    elif ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in DOCUMENT_EXTENSIONS:
        return "document"
    elif ext in ARCHIVE_EXTENSIONS:
        return "archive"
    elif ext in MEDIA_EXTENSIONS:
        return "media"
    else:
        return "unknown"


def get_file_metadata(
    filename: str, content_type: str = "", file_size: int = 0
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from file information.

    Args:
        filename: Original filename
        content_type: MIME type (optional)
        file_size: Size in bytes (optional)

    Returns:
        Dictionary with file metadata
    """
    extension = get_file_extension(filename)
    category = get_file_category(filename)

    return {
        "filename": filename,
        "extension": extension,
        "content_type": content_type,
        "file_size": file_size,
        "category": category,
        "is_text": is_text_file(filename) or content_type.startswith("text/"),
        "is_code": is_code_file(filename),
        "is_image": is_image_file(filename) or content_type.startswith("image/"),
        "is_document": is_document_file(filename),
    }
