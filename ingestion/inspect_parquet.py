import requests
import struct

URL = (
    "https://huggingface.co/datasets/"
    "ai4bharat/MSMARCO-XI/resolve/main/"
    "train/asmtrain.parquet"
)

session = requests.Session()

# ------------------------------------------------------------
# Get file size
# ------------------------------------------------------------

r = session.get(
    URL,
    headers={"Range": "bytes=0-0"},
    allow_redirects=True,
    timeout=30,
)

r.raise_for_status()

size = int(
    r.headers["Content-Range"].split("/")[-1]
)

url = r.url

print(f"File size: {size:,} bytes")
print(f"URL: {url[:100]}...")

# ------------------------------------------------------------
# Read last 8 bytes
# ------------------------------------------------------------

start = size - 8
end = size - 1

r = session.get(
    url,
    headers={
        "Range": f"bytes={start}-{end}"
    },
    timeout=30,
)

r.raise_for_status()

tail = r.content

print(
    f"Last 8 bytes: {tail!r}"
)

# ------------------------------------------------------------
# Parquet footer format
#
# Last 8 bytes:
#
# 4 bytes = footer length
# 4 bytes = PAR1
# ------------------------------------------------------------

if tail[-4:] != b"PAR1":

    raise RuntimeError(
        "This does not look like a valid "
        "Parquet file."
    )

footer_size = struct.unpack(
    "<I",
    tail[:4]
)[0]

print(
    f"Footer size: {footer_size:,} bytes"
)

footer_start = (
    size
    - 8
    - footer_size
)

footer_end = (
    size - 9
)

print(
    f"Footer range: "
    f"{footer_start:,} - "
    f"{footer_end:,}"
)

print(
    f"Downloading only "
    f"{footer_size:,} bytes of metadata..."
)

r = session.get(
    url,
    headers={
        "Range":
        f"bytes={footer_start}-{footer_end}"
    },
    timeout=60,
)

r.raise_for_status()

footer = r.content

print(
    f"Footer downloaded: "
    f"{len(footer):,} bytes"
)

print()
print("SUCCESS")