#!/usr/bin/env python3
"""Configure JAX NeuronX and run the shared Gemma 4 OpenAI server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

# The repo root, so `ports.gemma4` resolves in --check-only too, not just once
# main() is about to hand off to the server.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def configure_neuron() -> None:
    # All of these must be set before JAX is imported.
    #
    # "neuron,cpu" and not "neuron": the loader quantizes the 4.70 GB per-layer
    # embedding table on the HOST in row chunks, which needs a CPU device. See
    # quantize_ple_table in ports/gemma4/jax_e_model.py.
    os.environ.setdefault("JAX_PLATFORMS", "neuron,cpu")
    # RBG is AWS's documented recommendation and is the cheaper counter-based
    # generator. It is a preference, NOT a compile requirement: threefry2x32,
    # rbg, and unsafe_rbg all compiled a categorical sample over the 262144-token
    # vocabulary for inf2 under neuronx-cc 2.26. Kept as a setdefault so a caller
    # comparing sampled output against the TPU path can pin threefry instead.
    os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "rbg")
    # This one variable costs ~2700x, and it is load-bearing for CORRECTNESS.
    # Do not remove it without first fixing what it is hiding.
    #
    # MEASURED on inf2.xlarge, same engine, weights, device and checkpoint,
    # varying only this variable (2026-07-31, see
    # benchmarks/runs/2026-07-31-inf2-serving-perf/):
    #
    #   =1     : "The capital of France is" -> 'Paris.', 5 tokens, ~126 s
    #   unset  : same prompt -> '' , 0 completion tokens, ~0.01 s
    #
    # Unset, generation emits EOS on the first sampled token every time. So some
    # operation in the decode graph computes the wrong thing on the NeuronCore
    # and is only right when dispatched to the host. Setting this is a
    # workaround, not a configuration preference.
    #
    # The cost of the workaround: every decode step ships parameter-sized
    # buffers through host memory -- 13.71 GB of host churn per request against
    # 0.00 GB without it, 6.89 s/token against 0.002 s/token. On a 16 GB host
    # that exhausts RAM and the box swaps (116 MB/s out, 55% iowait). Device HBM
    # never moves throughout, so the accelerator looks idle and healthy while
    # this happens.
    #
    # The real fix is to find the miscomputing op -- jax_neuron/parity.py is the
    # tool, and it must be run with this variable unset to reproduce the fault.
    os.environ.setdefault("NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU", "1")
    os.environ.setdefault("NEURON_CC_FLAGS", "--model-type=transformer")
    # No JAX_E_PALLAS_INTERPRET override here any more. The engine reads
    # ports/gemma4/backend.py, sees that Neuron has no Pallas backend, and
    # refuses `set_w4a16_impl("fused")` outright rather than silently routing it
    # through an interpreter that would unroll the kernel into the graph. The
    # default impl is "reference", which is also what measured fastest on TPU.


def verify_neuron() -> list[object]:
    import jax

    devices = list(jax.devices())
    neuron = [device for device in devices if device.platform == "neuron"]
    if not neuron:
        found = ", ".join(f"{d.platform}:{d}" for d in devices) or "none"
        raise RuntimeError(f"No JAX Neuron device found; discovered {found}")

    # The device being present is not the same as the engine having noticed. If
    # detection disagrees, every Neuron branch in the model is wrong (Pallas,
    # fp8 KV, the scatter-free sampler) and the failure would surface much later
    # as a compiler error inside the first request.
    from ports.gemma4 import backend

    caps = backend.caps()
    if not caps.is_neuron:
        raise RuntimeError(
            f"JAX sees a Neuron device but the engine detected platform "
            f"{caps.platform!r}. Unset JAX_E_PLATFORM."
        )

    print(f"JAX Neuron ready: {len(neuron)} device(s): {neuron}", flush=True)
    print(f"engine caps: pallas={caps.pallas} device_top_k={caps.device_top_k} "
          f"float8_kv={caps.float8_kv} donation={caps.buffer_donation}", flush=True)
    return neuron


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--check-only", action="store_true")
    known, server_args = parser.parse_known_args()

    configure_neuron()
    verify_neuron()
    if known.check_only:
        return

    server = _ROOT / "jax_openai_server.py"
    if not server.exists():
        raise FileNotFoundError(f"Shared server not found at {server}")

    # Keep the shared server CLI authoritative. The reference W4A16 path is its
    # default and is required because the optional fused kernel targets TPU.
    sys.argv = [str(server), *server_args]
    runpy.run_path(str(server), run_name="__main__")


if __name__ == "__main__":
    main()
