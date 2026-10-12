"""Embed compressed dataset payload into the training notebook for 100% standalone portability."""

import base64
import gzip
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

train_p = PROJECT_ROOT / "data" / "train" / "train.jsonl"
val_p = PROJECT_ROOT / "data" / "validation" / "validation.jsonl"
test_p = PROJECT_ROOT / "data" / "test" / "test.jsonl"
nb_p = PROJECT_ROOT / "notebooks" / "nexus_qwen25_05b_lora_training.ipynb"

train_b64 = base64.b64encode(gzip.compress(open(train_p, "rb").read())).decode("ascii")
val_b64 = base64.b64encode(gzip.compress(open(val_p, "rb").read())).decode("ascii")
test_b64 = base64.b64encode(gzip.compress(open(test_p, "rb").read())).decode("ascii")

with open(nb_p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 3 (Index 4 in cells: code cell after markdown cell 3)
setup_cell_code = [
    "import hashlib\n",
    "import json\n",
    "import gzip\n",
    "import base64\n",
    "from pathlib import Path\n",
    "\n",
    "# Determine workspace root\n",
    "WORK_DIR = Path('/kaggle/working/nexus_training') if Path('/kaggle/working').exists() else Path('./nexus_training')\n",
    "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
    "DATA_DIR = WORK_DIR / 'data'\n",
    "DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
    "(DATA_DIR / 'train').mkdir(parents=True, exist_ok=True)\n",
    "(DATA_DIR / 'validation').mkdir(parents=True, exist_ok=True)\n",
    "(DATA_DIR / 'test').mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "def compute_sha256(p: Path) -> str:\n",
    "    if not p.exists(): return 'NOT_FOUND'\n",
    "    h = hashlib.sha256()\n",
    "    with open(p, 'rb') as f:\n",
    "        while chunk := f.read(8192): h.update(chunk)\n",
    "    return h.hexdigest()\n",
    "\n",
    "train_file = DATA_DIR / 'train/train.jsonl'\n",
    "val_file = DATA_DIR / 'validation/validation.jsonl'\n",
    "test_file = DATA_DIR / 'test/test.jsonl'\n",
    "\n",
    "# 1. If running in cloned repository, copy from local files\n",
    "repo_train = Path('data/train/train.jsonl')\n",
    "repo_val = Path('data/validation/validation.jsonl')\n",
    "repo_test = Path('data/test/test.jsonl')\n",
    "\n",
    "if repo_train.exists():\n",
    "    import shutil\n",
    "    shutil.copy(repo_train, train_file)\n",
    "    shutil.copy(repo_val, val_file)\n",
    "    shutil.copy(repo_test, test_file)\n",
    "    print('Using local dataset files from cloned repository.')\n",
    "else:\n",
    "    print('Standalone cloud runtime detected. Self-extracting canonical verified datasets...')\n",
    f"    EMBEDDED_TRAIN = '{train_b64}'\n",
    f"    EMBEDDED_VAL = '{val_b64}'\n",
    f"    EMBEDDED_TEST = '{test_b64}'\n",
    "    train_file.write_bytes(gzip.decompress(base64.b64decode(EMBEDDED_TRAIN)))\n",
    "    val_file.write_bytes(gzip.decompress(base64.b64decode(EMBEDDED_VAL)))\n",
    "    test_file.write_bytes(gzip.decompress(base64.b64decode(EMBEDDED_TEST)))\n",
    "    print('Self-extraction complete!')\n",
    "\n",
    "# 2. Verify Cryptographic SHA-256 Hashes\n",
    "EXPECTED_TRAIN_HASH = '11df262072d7243f623bfd829b407a8dbd2c5a3087ed697a836a7152c7688967'\n",
    "EXPECTED_VAL_HASH = '2e349ad3b1330aec224c266460d2e4189b94138a1d63146a10bd8f45ea65508e'\n",
    "EXPECTED_TEST_HASH = 'edf3fa5cd949a149ad5d759d81553d9704467491877d0f3c8b7cc5a887a9eda4'\n",
    "\n",
    "act_train = compute_sha256(train_file)\n",
    "act_val = compute_sha256(val_file)\n",
    "act_test = compute_sha256(test_file)\n",
    "\n",
    "print(f'Train hash:      {act_train} -> {\"VERIFIED\" if act_train == EXPECTED_TRAIN_HASH else \"MISMATCH\"}')\n",
    "print(f'Validation hash: {act_val} -> {\"VERIFIED\" if act_val == EXPECTED_VAL_HASH else \"MISMATCH\"}')\n",
    "print(f'Test hash:       {act_test} -> {\"VERIFIED\" if act_test == EXPECTED_TEST_HASH else \"MISMATCH\"}')\n",
    "assert act_train == EXPECTED_TRAIN_HASH, 'Train dataset hash mismatch!'\n",
    "assert act_val == EXPECTED_VAL_HASH, 'Validation dataset hash mismatch!'\n",
    "assert act_test == EXPECTED_TEST_HASH, 'Test dataset hash mismatch!'\n",
]

# Find the cell index with compute_sha256
for cell in nb["cells"]:
    if cell.get("cell_type") == "code" and any("compute_sha256" in line for line in cell.get("source", [])):
        cell["source"] = setup_cell_code
        print("Updated dataset setup cell in notebook.")
        break

with open(nb_p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook successfully updated at: {nb_p}")
