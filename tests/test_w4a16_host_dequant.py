"""Regression tests for the W4A16 host-dequant path (the Inf2 correctness fix).

Context: on Inferentia2 the *in-graph* W4A16 dequant miscomputes — greedy decode
emits one token repeated forever (`Atha Atha Atha`) with 0.0 agreement against a
CPU oracle — while the identical arithmetic performed on the host is correct.
`--dequant-at-load` is the fix, and it lets the 65x
`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1` workaround be turned off.
See benchmarks/runs/2026-08-02-inf2-latest-stack-e2b/BISECT.md.

The device-level defect cannot be reproduced off Neuron, so these tests guard the
things that CAN be checked anywhere:

  1. the numpy twin agrees with the JAX implementation (if it drifts, the fix
     silently starts serving different weights than it was validated with);
  2. `on_host=True` and `on_host=False` agree on a whole params tree, so the flag
     is a placement choice and never a semantic one;
  3. the deployment does not reinstate the workaround by default.

Nothing in tests/ covered this defect when it shipped — the suite passed 175/175
on the Inf2 host while the served model produced garbage. That is the failure
mode CLAUDE.md warns about, and it is why (3) asserts on the deployment files
rather than only on library code.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp  # noqa: E402

from ports.gemma4.jax_e_model import (  # noqa: E402
    dequantize_params_to_dense,
    qat_w4a16_unpack_dequant_jax,
    qat_w4a16_unpack_dequant_numpy,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _packed_and_scale(out_features=64, packed_k=16, seed=0, group_size=32):
    rng = np.random.default_rng(seed)
    # Full int32 range: negative words exercise sign extension in `>>`, which is
    # where a logical-vs-arithmetic shift bug would hide.
    packed = rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max,
        size=(out_features, packed_k), dtype=np.int32,
    )
    in_features = packed_k * 8
    scale = rng.uniform(
        0.001, 0.05, size=(out_features, in_features // group_size)
    ).astype(np.float32)
    return packed, scale


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_numpy_twin_matches_jax(seed):
    """The host dequant must compute what the JAX path computes.

    Tolerance is not zero because the JAX path rounds `scale` to bf16 while the
    numpy path keeps float32. That is the only permitted difference; anything
    structural (bad shift, wrong nibble order, missing zero point) moves the
    result by whole quantization steps and blows past this bound.
    """
    packed, scale = _packed_and_scale(seed=seed)

    got = qat_w4a16_unpack_dequant_numpy(packed, scale).astype(np.float32)
    want = np.asarray(
        qat_w4a16_unpack_dequant_jax(
            jnp.asarray(packed), jnp.asarray(scale, dtype=jnp.bfloat16)
        ),
        dtype=np.float32,
    )

    assert got.shape == want.shape
    denom = max(float(np.abs(want).max()), 1e-6)
    assert float(np.abs(got - want).max()) / denom < 0.02


def test_nibble_decoding_is_exact():
    """Decode a hand-built word: the zero point and nibble order must be right.

    Independent of both implementations — if the layout convention is wrong, this
    fails even when numpy and JAX agree with each other.
    """
    # Nibble i of word j is input column 8*j+i and stores q + 8.
    nibbles = [0, 1, 2, 8, 15, 7, 3, 12]
    word = 0
    for i, n in enumerate(nibbles):
        word |= n << (4 * i)
    word = np.int32(np.uint32(word).astype(np.int64) - (1 << 32)
                    if word >= (1 << 31) else word)

    packed = np.array([[word]], dtype=np.int32)
    scale = np.ones((1, 8 // 8 * 1), dtype=np.float32)  # group_size must divide 8
    got = qat_w4a16_unpack_dequant_numpy(packed, scale, group_size=8)

    assert got.shape == (1, 8)
    np.testing.assert_array_equal(got[0], np.array(nibbles, dtype=np.float32) - 8.0)


def test_on_host_matches_in_graph_for_a_params_tree():
    """`on_host` selects WHERE the dequant runs, never WHAT it computes."""
    packed, scale = _packed_and_scale(seed=3)
    params = {
        "layer": {
            "proj_packed": jnp.asarray(packed),
            "proj_scale": jnp.asarray(scale, dtype=jnp.bfloat16),
            "untouched": jnp.asarray(np.ones((2, 2), dtype=np.float32)),
        }
    }

    host = dequantize_params_to_dense(params, on_host=True)
    graph = dequantize_params_to_dense(params, on_host=False)

    # Packed/scale replaced by a single dense weight, other entries preserved.
    assert set(host["layer"]) == {"proj", "untouched"}
    assert set(graph["layer"]) == {"proj", "untouched"}

    h = np.asarray(host["layer"]["proj"], dtype=np.float32)
    g = np.asarray(graph["layer"]["proj"], dtype=np.float32)
    assert h.shape == g.shape, "host and in-graph dequant disagree on orientation"
    denom = max(float(np.abs(g).max()), 1e-6)
    assert float(np.abs(h - g).max()) / denom < 0.05


def test_deployment_does_not_reinstate_the_cpu_workaround():
    """`NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1` costs ~65x and is no longer needed.

    It was a correctness workaround for the in-graph dequant. With
    `--dequant-at-load` the output is correct without it, so a default of "1"
    would be a silent 65x regression that no output check would catch.
    """
    entrypoint = os.path.join(_ROOT, "deployments", "aws-inf2", "neuron_entrypoint.py")
    with open(entrypoint) as fh:
        src = fh.read()

    m = re.search(
        r'setdefault\(\s*["\']NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU["\']\s*,\s*["\'](\d)["\']',
        src,
    )
    assert m, "entrypoint no longer sets NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU"
    assert m.group(1) == "0", (
        "the 65x CPU-dispatch workaround is enabled by default again; "
        "--dequant-at-load makes it unnecessary (see BISECT.md)"
    )


def test_service_passes_dequant_at_load():
    """The serving unit must use the fix, or Inf2 serves garbage."""
    user_data = os.path.join(_ROOT, "deployments", "aws-inf2", "user_data.sh")
    with open(user_data) as fh:
        src = fh.read()

    assert "--dequant-at-load" in src, (
        "the Inf2 service no longer passes --dequant-at-load; W4A16 output on "
        "the NeuronCore is wrong without it"
    )
