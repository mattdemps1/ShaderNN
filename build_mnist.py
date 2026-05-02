"""
Download MNIST and pack a small subset to data/mnist_subset.bin.

Format (little-endian):
  uint32 magic = 0x4D4E4953   ("MNIS")
  uint32 numTrain
  uint32 numTest
  uint32 imageSize = 784
  uint8  trainImages[numTrain * 784]    (row-major, top-left origin, 0..255)
  uint8  trainLabels[numTrain]          (0..9)
  uint8  testImages[numTest * 784]
  uint8  testLabels[numTest]
"""
import gzip
import struct
import urllib.request
from pathlib import Path

NUM_TRAIN = 20000
NUM_TEST = 5000

MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def fetch(name):
    last_err = None
    for mirror in MIRRORS:
        url = mirror + name
        print(f"  trying {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "shadernn/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as e:
            print(f"    failed: {e}")
            last_err = e
    raise RuntimeError(f"all mirrors failed for {name}: {last_err}")


def parse_images(gz_bytes):
    raw = gzip.decompress(gz_bytes)
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, f"bad image magic {magic}"
    assert rows == 28 and cols == 28
    return raw[16:16 + n * 784], n


def parse_labels(gz_bytes):
    raw = gzip.decompress(gz_bytes)
    magic, n = struct.unpack(">II", raw[:8])
    assert magic == 2049, f"bad label magic {magic}"
    return raw[8:8 + n], n


def main():
    here = Path(__file__).parent
    out = here / "data" / "mnist_subset.bin"
    out.parent.mkdir(exist_ok=True)

    print("Downloading MNIST...")
    raw = {k: fetch(v) for k, v in FILES.items()}

    train_imgs, n_train = parse_images(raw["train_images"])
    train_lbls, _       = parse_labels(raw["train_labels"])
    test_imgs,  n_test  = parse_images(raw["test_images"])
    test_lbls,  _       = parse_labels(raw["test_labels"])

    n_train = min(NUM_TRAIN, n_train)
    n_test  = min(NUM_TEST,  n_test)

    header = struct.pack("<IIII", 0x4D4E4953, n_train, n_test, 784)
    with open(out, "wb") as f:
        f.write(header)
        f.write(train_imgs[:n_train * 784])
        f.write(train_lbls[:n_train])
        f.write(test_imgs[:n_test * 784])
        f.write(test_lbls[:n_test])

    size = out.stat().st_size
    print(f"wrote {out}  ({size:,} bytes — {n_train} train + {n_test} test)")


if __name__ == "__main__":
    main()
