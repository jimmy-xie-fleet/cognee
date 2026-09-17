import io
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, BinaryIO, Optional

import filetype
from filetype.types.base import Type


class FileTypeException(Exception):
    """
    Represents an exception for file type errors.

    This exception is raised when an invalid file type is encountered. It includes a custom
    message to describe the error.

    Parameters:
    -----------

        - message (str): The message describing the exception error.
    """

    message: str

    def __init__(self, message: str):
        self.message = message


def guess_file_type(file: BinaryIO, name: str | None = None) -> filetype.Type:
    """
    Guess the file type from the given binary file stream.

    If the file type cannot be determined from content, attempts to infer from extension.
    If still unable to determine, raise a FileTypeException with an appropriate message.

    Parameters:
    -----------

        - file (BinaryIO): A binary file stream to analyze for determining the file type.

    Returns:
    --------

        - filetype.Type: The guessed file type, represented as filetype.Type.
    """

    # Note: If file has .txt or .text extension, consider it a plain text file as filetype.guess may not detect it properly
    # as it contains no magic number encoding
    ext = None
    if isinstance(file, str):
        ext = Path(file).suffix
    elif name is not None:
        ext = Path(name).suffix

    # File extensions are case-insensitive: a file named "data.CSV" must be
    # detected the same as "data.csv". Without this, uppercase extensions for
    # types that have no magic number (csv/md/json/xml/yaml) fall through to
    # filetype.guess, return None, and get mislabeled as text/plain — so e.g. a
    # ".CSV" is classified as a plain TextDocument instead of a CsvDocument.
    if ext is not None:
        ext = ext.lower()

    if ext in [".txt", ".text"]:
        file_type = Type("text/plain", "txt")
        return file_type

    if ext in [".csv"]:
        return Type("text/csv", "csv")

    if ext in [".md", ".markdown"]:
        return Type("text/markdown", "md")

    if ext in [".json"]:
        return Type("application/json", "json")

    if ext in [".xml"]:
        return Type("application/xml", "xml")

    if ext in [".yaml", ".yml"]:
        return Type("application/yaml", "yaml")

    file_type = filetype.guess(file)

    # ``filetype`` identifies an Office Open XML document by scanning the first four zip
    # entries after ``[Content_Types].xml`` for ``word/``, ``ppt/`` or ``xl/``. A .docx
    # written by python-docx (and by some Word builds) puts ``docProps/`` and a second
    # ``_rels`` entry first, so ``word/document.xml`` is the sixth entry and the file is
    # reported as a plain ``application/zip`` -- for which no loader is registered, so
    # ingestion of the document fails outright. The zip's own directory is the
    # authoritative answer, and reading it costs one pass over the central directory.
    if file_type is None or file_type.mime == "application/zip":
        zipped_type = _guess_zipped_document_type(file)
        if zipped_type is not None:
            return zipped_type

    # If file type could not be determined consider it a plain text file as they don't have magic number encoding
    if file_type is None:
        file_type = Type("text/plain", "txt")

    return file_type


# A zip entry prefix (or, for OpenDocument, the ``mimetype`` entry's content) that
# identifies the document format the archive holds.
_ZIPPED_DOCUMENT_TYPES = (
    ("word/", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    ("ppt/", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
    ("xl/", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
)
_OPENDOCUMENT_TYPES = {
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
}


def _guess_zipped_document_type(file: Any) -> Optional[Type]:
    """The document type a zip archive holds, read from its entry names, or ``None``.

    Reads the central directory only. The stream position is restored, so the
    caller's later reads (content hash, loader) see the whole file.
    """
    import zipfile

    if isinstance(file, str):
        source: Any = file
        position = None
    else:
        if not hasattr(file, "seek") or not hasattr(file, "read"):
            return None
        try:
            position = file.tell()
            file.seek(0)
        except (OSError, io.UnsupportedOperation):
            return None
        source = file
    try:
        with zipfile.ZipFile(source) as archive:
            names = archive.namelist()
            for prefix, mime, extension in _ZIPPED_DOCUMENT_TYPES:
                if any(name.startswith(prefix) for name in names):
                    return Type(mime, extension)
            if "mimetype" in names:
                declared = archive.read("mimetype").decode("ascii", "ignore").strip()
                extension = _OPENDOCUMENT_TYPES.get(declared)
                if extension:
                    return Type(declared, extension)
    except (zipfile.BadZipFile, OSError, ValueError):
        return None
    finally:
        if position is not None:
            try:
                file.seek(position)
            except (OSError, io.UnsupportedOperation):
                pass
    return None
