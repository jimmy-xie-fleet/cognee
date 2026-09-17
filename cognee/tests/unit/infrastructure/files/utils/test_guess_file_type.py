"""``guess_file_type`` must detect extensions case-insensitively.

Extensions are case-insensitive by convention, but the extension-based mapping
only matched lowercase. For types with no magic-number signature
(csv/md/json/xml/yaml), an uppercase extension fell through to
``filetype.guess``, returned ``None``, and was mislabeled ``text/plain`` — so a
``data.CSV`` was classified as a plain ``TextDocument`` instead of a
``CsvDocument`` (the crash for these was fixed in #3662, but the wrong type
survived). PDFs and other magic-number formats are unaffected either way.
"""

import io

import pytest

from cognee.infrastructure.files.utils.guess_file_type import guess_file_type

# Content with no magic number, so detection depends on the extension.
_CSV_BYTES = b"name,age\nJohn,30\nJane,25\n"


@pytest.mark.parametrize(
    "name, expected_mime, expected_ext",
    [
        ("data.csv", "text/csv", "csv"),
        ("data.CSV", "text/csv", "csv"),
        ("notes.md", "text/markdown", "md"),
        ("notes.MD", "text/markdown", "md"),
        ("conf.json", "application/json", "json"),
        ("conf.JSON", "application/json", "json"),
        ("doc.XML", "application/xml", "xml"),
        ("conf.YAML", "application/yaml", "yaml"),
    ],
)
def test_extension_detection_is_case_insensitive(name, expected_mime, expected_ext):
    file_type = guess_file_type(io.BytesIO(_CSV_BYTES), name=name)
    assert file_type.mime == expected_mime
    assert file_type.extension == expected_ext


def test_uppercase_txt_is_still_plain_text():
    file_type = guess_file_type(io.BytesIO(b"just some text"), name="README.TXT")
    assert file_type.mime == "text/plain"


def _office_archive(document_prefix: str, office_entry_index: int) -> io.BytesIO:
    """A minimal OOXML-shaped zip whose office directory is the N-th entry.

    ``filetype`` only looks at the first four entries after ``[Content_Types].xml``;
    python-docx writes ``word/document.xml`` sixth, which is the layout this builds.
    """
    import zipfile

    filler = [
        "[Content_Types].xml",
        "_rels/.rels",
        "docProps/app.xml",
        "docProps/core.xml",
        "docProps/custom.xml",
        "customXml/item1.xml",
    ]
    names = filler[:office_entry_index] + [f"{document_prefix}document.xml"] + ["misc/x.xml"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in names:
            archive.writestr(name, "<x/>")
    buffer.seek(0)
    return buffer


@pytest.mark.parametrize(
    "prefix, expected_ext",
    [("word/", "docx"), ("ppt/", "pptx"), ("xl/", "xlsx")],
)
def test_an_office_document_with_late_office_entries_is_not_a_plain_zip(prefix, expected_ext):
    """The docx layout python-docx writes: the office directory is the sixth entry."""
    archive = _office_archive(prefix, office_entry_index=5)

    file_type = guess_file_type(archive, name=f"Answer.{expected_ext}")

    assert file_type.extension == expected_ext
    assert file_type.mime.startswith("application/vnd.openxmlformats-officedocument")
    # the stream is left where the caller can hash and load it
    assert archive.tell() == 0


def test_an_opendocument_archive_is_detected_from_its_mimetype_entry():
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        archive.writestr("content.xml", "<x/>")
    buffer.seek(0)

    file_type = guess_file_type(buffer, name="notes.odt")

    assert (file_type.mime, file_type.extension) == (
        "application/vnd.oasis.opendocument.text",
        "odt",
    )


def test_a_zip_that_holds_no_document_stays_a_zip():
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hello")
    buffer.seek(0)

    file_type = guess_file_type(buffer, name="bundle.zip")

    assert (file_type.mime, file_type.extension) == ("application/zip", "zip")
