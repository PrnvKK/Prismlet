"""Model constants and default paths."""

import math
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights"
DEFAULT_PRUNED = REPO_ROOT / "pruned"
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_RESULTS = REPO_ROOT / "results"

HID = 4096
NH = 64
QK_NOPE = 128
QK_ROPE = 64
Q_HEAD = QK_NOPE + QK_ROPE          # 192
V_DIM = 128
KV_LORA = 512
KV_B_OUT = NH * (QK_NOPE + V_DIM)   # 16384
E = 128
TOPK = 8
MOE_I = 2048
DENSE_I = 16384
ROUTED_SCALE = 2.5
EPS = 1e-6
NLAYERS = 32

SOFTMAX_SCALE = 192 ** -0.5 * (0.1 * math.log(40.0) + 1.0) ** 2   # 0.1352337788608801
assert abs(SOFTMAX_SCALE - 0.1352337788608801) < 1e-15

ROPE = dict(dim=64, base=10000.0, factor=40.0, orig=4096, beta_fast=32,
            beta_slow=1, mscale=1.0, mscale_all=1.0)
MAX_POS = 4096

TYPES = {"choice": 0, "yes_no": 1, "score": 2}

# Tensor sizes in bytes, from the published weight index.
SIZE_EMBED = 2147491840
SIZE_HEAD = 117837824
SIZE_LAYER0 = 296258816
SIZE_LAYER = 3347500800
SIZE_EXPERT = 25198592
SIZE_ROUTER_ROW = 16388
SIZE_SHARED = 25198592
SIZE_ATTN = 94784768


def layer_bytes(n_experts: int) -> int:
    """Bytes of one routed-expert layer with n_experts kept."""
    return n_experts * SIZE_EXPERT + n_experts * SIZE_ROUTER_ROW + SIZE_SHARED + SIZE_ATTN


def total_backbone(n_experts: int) -> int:
    """Bytes of the 32 layer files for n_experts kept per routed layer."""
    return 31 * layer_bytes(n_experts) + SIZE_LAYER0


def file_for_layer(i: int) -> str:
    return f"model-layer-{i:02d}.safetensors"
