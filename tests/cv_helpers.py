"""Helpers for CV tests: a tiny PDF writer, so no PDF file ever has to be committed."""

from pathlib import Path

CV_FIXTURES = Path(__file__).parent / "fixtures" / "cv"


def synthetic_cv_text() -> str:
    """The invented CV of "Alex Example" (see tests/fixtures/cv/README.md)."""
    return (CV_FIXTURES / "alex_example.txt").read_text(encoding="utf-8")


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """A valid PDF with one page per inner list, one text line per string (Helvetica, Latin-1).

    A page with no lines is a blank page, like a scanned PDF that has no text layer.
    Object numbers: 1 catalog, 2 page tree, 3 font, then (page, content) pairs.
    """
    objects: list[bytes] = [b""] * (3 + 2 * len(pages))
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()
    objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    for i, lines in enumerate(pages):
        body = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({_escape(x)}) Tj T*" for x in lines)
        stream = (body + " ET").encode("latin-1")
        objects[3 + 2 * i] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
        objects[4 + 2 * i] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>"
    out += f"{trailer}\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)
