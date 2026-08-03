"""Bisect the W4A16 miscomputation on Neuron down to a single primitive.

Parity established (2026-08-02) that fp16 weights and int8 KV are token-exact on
NeuronCore, while w4a16 diverges at token 0. The only path w4a16 adds is
`qat_w4a16_unpack_dequant_jax`, whose first step is signed-int32 bit
manipulation:

    q = ((words >> shifts) & 0xF)

This script runs that decomposition op-by-op on the NeuronCore and on CPU with
identical inputs, and reports the first stage that diverges. It needs no model
weights and compiles in seconds, so it isolates the defect without a 160 s
parameter load.

Run with NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=0 to expose the fault; run with
=1 to confirm the workaround is what masks it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _stages():
    """Each stage: (name, fn(packed, scale) -> array, description).

    Ordered from the rawest integer op to the full dequant, so the first
    failing stage names the primitive.
    """
    import jax
    import jax.numpy as jnp

    def s_identity(packed, scale):
        # Control: does an int32 array survive a round trip at all?
        return packed

    def s_shift(packed, scale):
        shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
        return (packed[:, :, None] >> shifts).reshape(packed.shape[0], -1)

    def s_shift_mask(packed, scale):
        shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
        q = (packed[:, :, None] >> shifts) & jnp.int32(0xF)
        return q.reshape(packed.shape[0], -1)

    def s_to_bf16(packed, scale):
        shifts = (jnp.arange(8, dtype=jnp.int32) * 4)[None, None, :]
        q = (packed[:, :, None] >> shifts) & jnp.int32(0xF)
        q = q.reshape(packed.shape[0], -1)
        return (q.astype(jnp.bfloat16) - jnp.bfloat16(8)).astype(jnp.float32)

    def s_full(packed, scale):
        from ports.gemma4.jax_e_model import qat_w4a16_unpack_dequant_jax
        return qat_w4a16_unpack_dequant_jax(packed, scale).astype(jnp.float32)

    return [
        ("identity", s_identity, "int32 round trip (control)"),
        ("shift", s_shift, "signed int32 arithmetic right shift"),
        ("shift_mask", s_shift_mask, "shift then & 0xF -> nibble value"),
        ("to_bf16", s_to_bf16, "nibble -> bfloat16, minus 8 (zero point)"),
        ("full", s_full, "qat_w4a16_unpack_dequant_jax (with group scale)"),
    ]


def _numpy_reference(packed_np, scale_np, group_size=32):
    """Ground truth in plain numpy on the host — no JAX, no device."""
    out_features, packed_k = packed_np.shape
    in_features = packed_k * 8
    shifts = (np.arange(8, dtype=np.int32) * 4)[None, None, :]
    words = packed_np[:, :, None].astype(np.int32)
    q = (words >> shifts) & np.int32(0xF)
    ref = {
        "identity": packed_np.astype(np.int64),
        "shift": (words >> shifts).reshape(out_features, in_features).astype(np.int64),
        "shift_mask": q.reshape(out_features, in_features).astype(np.int64),
    }
    qf = q.reshape(out_features, in_features).astype(np.float32) - 8.0
    ref["to_bf16"] = qf
    grouped = qf.reshape(out_features, in_features // group_size, group_size)
    ref["full"] = (grouped * scale_np[:, :, None].astype(np.float32)).reshape(
        out_features, in_features
    )
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-features", type=int, default=256)
    ap.add_argument("--packed-k", type=int, default=32, help="in_features = packed_k*8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import jax
    import jax.numpy as jnp

    group_size = 32
    in_features = args.packed_k * 8
    assert in_features % group_size == 0

    rng = np.random.default_rng(args.seed)
    # Full int32 range, so negative words exercise sign extension in `>>`.
    packed_np = rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max,
        size=(args.out_features, args.packed_k), dtype=np.int32,
    )
    scale_np = rng.uniform(0.001, 0.05, size=(args.out_features, in_features // group_size)).astype(np.float32)

    ref = _numpy_reference(packed_np, scale_np, group_size)

    devices = {d.platform: d for d in jax.devices()}
    try:
        cpu_dev = jax.devices("cpu")[0]
    except Exception:
        cpu_dev = None
    neuron_dev = jax.devices()[0]

    print(f"jax {jax.__version__}  default device: {neuron_dev} "
          f"(platform={neuron_dev.platform})")
    print(f"NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU="
          f"{os.environ.get('NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU', '<unset>')}")
    print(f"shape: packed{packed_np.shape} int32 -> dense[{args.out_features},{in_features}]\n")

    results = []
    first_bad = None
    for name, fn, desc in _stages():
        row = {"stage": name, "description": desc}
        try:
            packed_d = jax.device_put(jnp.asarray(packed_np), neuron_dev)
            scale_d = jax.device_put(jnp.asarray(scale_np, dtype=jnp.bfloat16), neuron_dev)
            got = np.asarray(jax.jit(fn)(packed_d, scale_d), dtype=np.float64)
            want = np.asarray(ref[name], dtype=np.float64)

            if name == "full":
                # bf16 scale round trip means exact equality is not the bar.
                tol = 0.05 * np.maximum(np.abs(want), 1e-6)
                bad = np.abs(got - want) > tol
            else:
                bad = got != want

            n_bad = int(bad.sum())
            row.update({
                "elements": int(want.size),
                "mismatched": n_bad,
                "match_rate": float(1.0 - n_bad / want.size),
                "max_abs_err": float(np.abs(got - want).max()),
                "passed": n_bad == 0,
            })
            if n_bad:
                idx = np.argwhere(bad)[0]
                row["first_mismatch"] = {
                    "index": [int(i) for i in idx],
                    "got": float(got[tuple(idx)]),
                    "want": float(want[tuple(idx)]),
                }
                if first_bad is None:
                    first_bad = name
        except Exception as exc:  # compile or runtime failure is itself a result
            row.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
            if first_bad is None:
                first_bad = name

        results.append(row)
        mark = "PASS" if row.get("passed") else "FAIL"
        extra = ""
        if not row.get("passed"):
            if "error" in row:
                extra = f"  {row['error'][:100]}"
            else:
                extra = (f"  {row['mismatched']}/{row['elements']} wrong, "
                         f"max|err|={row['max_abs_err']:.4g}, "
                         f"first {row['first_mismatch']['got']} != {row['first_mismatch']['want']}")
        print(f"[{mark}] {name:<12} {desc}{extra}")

    print()
    if first_bad is None:
        print("All stages match the host reference. The unpack path is NOT the defect;")
        print("look at the matmul consuming these weights, or the loader.")
    else:
        print(f"FIRST DIVERGING STAGE: {first_bad}")

    out = {
        "jax_version": jax.__version__,
        "device": str(neuron_dev),
        "platform": neuron_dev.platform,
        "trivial_on_cpu": os.environ.get("NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU", None),
        "shape": {"out_features": args.out_features, "packed_k": args.packed_k,
                  "in_features": in_features},
        "first_diverging_stage": first_bad,
        "stages": results,
    }
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
