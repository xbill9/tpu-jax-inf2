"""Accelerator capability detection shared by the TPU and Inferentia2 paths.

The Gemma 4 E-series engine in this package was written against Cloud TPU and
hard-coded several TPU-only assumptions: Pallas/Mosaic kernels, float8 KV
storage, buffer donation, and a data-dependent scatter in the sampler. AWS
Inferentia2 reaches JAX through a different PJRT plugin (`jax-neuronx`) and an
ahead-of-time compiler (`neuronx-cc`) that does not accept all of those.

Rather than fork the model, every such site now asks this module what the
current backend can do. The defaults keep the TPU path byte-identical to what
was measured in `benchmarks/runs/`; only non-TPU platforms take a different
branch.

Detection is by PJRT platform string, with `JAX_E_PLATFORM` as an override so
the Neuron branches can be exercised on a CPU host (that is how the code paths
are unit-tested off-device; it does NOT make the CPU compile like Neuron).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_PLATFORM_ENV = "JAX_E_PLATFORM"

# Platform strings JAX reports through `Device.platform`.
TPU = "tpu"
NEURON = "neuron"
CPU = "cpu"
GPU = "gpu"

_KNOWN_PLATFORMS = (TPU, NEURON, CPU, GPU)


@dataclasses.dataclass(frozen=True)
class BackendCaps:
    """What the active accelerator can actually compile and run.

    Each field is consumed at exactly one kind of site in the engine; the
    comment names the failure that flipping it to False avoids.
    """

    platform: str

    # Whether the fused W4A16 Pallas kernel can be selected at all. Pallas lowers
    # through Mosaic, which has TPU and GPU backends; on CPU it stays reachable
    # through the Pallas *interpreter*, which is how this repo tests the kernel's
    # numerics off-device. Neuron is the one platform where it is unreachable:
    # neuronx-cc compiles ahead of time and Mosaic never enters the picture, and
    # the interpreter is not a substitute in production because it traces the
    # kernel body into the enclosing graph, unrolling the K loop over every tile.
    pallas: bool = False

    # Whether that kernel must run under the interpreter rather than lowering
    # natively. Only meaningful when `pallas` is True.
    pallas_interpret: bool = False

    # `jnp.float8_e4m3fn` / `e5m2` KV storage. MEASURED: compiling the decode
    # step with an fp8 cache for inf2 fails in the type verifier, before any
    # question of the surrounding pattern arises —
    #   [NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2. Target
    #   TRN3 or later hardware, or use the --experimental-unsafe-fp8e4m3fn-as-fp8e4m3
    #   flag to cast F8E4M3FN to F8E4M3.
    # inf2 carries NeuronCore-v2, which is TRN1-class. The same decode step with
    # an int8 cache compiles, and int8 carries the identical per-token scales for
    # the identical one-byte capacity, so it is the portable choice — and it was
    # the more accurate of the two on TPU anyway.
    float8_kv: bool = False

    # `jax.jit(..., donate_argnums=...)`. Worth 1.62x on the TPU decode step,
    # but a plugin that ignores donation silently copies instead, and one that
    # rejects it raises at compile time.
    buffer_donation: bool = False

    # `jax.lax.top_k`. MEASURED against neuronx-cc 2.26 targeting inf2: the HLO
    # `topk` operator is rejected outright —
    #   [NCC_EVRF001] Operator topk is not supported. Locate the operator in
    #   source or libraries and replace it with an alternate implementation via
    #   Neuron Kernel Interface (NKI).
    # `argmax`, `sort`, and `jax.random.categorical` all compile, so the sampler
    # substitutes an iterative masked-max for the threshold. See `_kth_largest`.
    #
    # There is deliberately NO `scatter` capability next to this one. A sibling
    # Gemma 4 port to the same hardware reported data-dependent scatter as a risk
    # site, and this table briefly encoded that as fact. Compiled in isolation
    # against neuronx-cc 2.26 — supplying indices and values as inputs so `topk`
    # could not fail first — the sampler's vector-indexed
    # `mask.at[arange(B)[:, None], idx].set(vals)` PASSES, as does the scalar
    # `valid.at[:, slot].set(True)` inside the compiled decode step. Inherited
    # risk, not measured risk; the field was removed rather than left asserting
    # something untrue.
    device_top_k: bool = False

    # Setting `jax_default_matmul_precision="bfloat16"` globally. This is an MXU
    # concept; on other backends it either does nothing or degrades the CPU
    # reference that Inf2 debugging depends on.
    default_bf16_matmul: bool = False

    # JAX's persistent XLA compilation cache. Neuron has its own on-disk cache
    # keyed by the NEFF; layering JAX's on top is harmless but unproven, and on
    # a read-only or missing cache dir it raises at import.
    persistent_compilation_cache: bool = True

    @property
    def is_tpu(self) -> bool:
        return self.platform == TPU

    @property
    def is_neuron(self) -> bool:
        return self.platform == NEURON


_CAPS_BY_PLATFORM = {
    TPU: BackendCaps(
        platform=TPU,
        pallas=True,
        float8_kv=True,
        buffer_donation=True,
        device_top_k=True,
        default_bf16_matmul=True,
        persistent_compilation_cache=True,
    ),
    NEURON: BackendCaps(
        platform=NEURON,
        pallas=False,
        float8_kv=False,
        buffer_donation=False,     # MEASURED: runtime fails with must-alias donation error
        device_top_k=False,        # MEASURED: neuronx-cc 2.26 rejects HLO topk
        default_bf16_matmul=False,
        persistent_compilation_cache=False,
    ),
    GPU: BackendCaps(
        platform=GPU,
        pallas=True,
        # Mosaic has a GPU backend, but this repo has never run on one. Keeping
        # the interpreter preserves the pre-port rule exactly ("interpret unless
        # TPU") rather than making an untested claim about native lowering.
        pallas_interpret=True,
        float8_kv=True,
        buffer_donation=True,
        device_top_k=True,
        default_bf16_matmul=False,
        persistent_compilation_cache=True,
    ),
    CPU: BackendCaps(
        platform=CPU,
        # Reachable, but only through the interpreter: that is how the fused
        # kernel's numerics are tested off-device (tests/test_perf_optimizations.py).
        pallas=True,
        pallas_interpret=True,
        float8_kv=True,
        buffer_donation=True,
        device_top_k=True,
        default_bf16_matmul=True,   # unchanged from the pre-port behaviour
        persistent_compilation_cache=True,
    ),
}


_cached: Optional[BackendCaps] = None


def _detect_platform() -> str:
    override = os.environ.get(_PLATFORM_ENV, "").strip().lower()
    if override:
        if override not in _KNOWN_PLATFORMS:
            raise ValueError(
                f"{_PLATFORM_ENV}={override!r} is not one of {_KNOWN_PLATFORMS}"
            )
        return override

    # Imported lazily so that reading the module does not force backend
    # initialization in tools that only want the capability table.
    import jax

    try:
        devices = jax.devices()
    except Exception as exc:                     # no backend at all
        logger.warning("JAX reported no devices (%s); assuming CPU semantics", exc)
        return CPU

    # An accelerator is authoritative even when a CPU device is also present:
    # `JAX_PLATFORMS=neuron,cpu` is the documented Inf2 setting and lists both.
    for preferred in (TPU, NEURON, GPU):
        if any(getattr(d, "platform", None) == preferred for d in devices):
            return preferred
    return CPU


def caps() -> BackendCaps:
    """Capabilities of the active backend. Detected once, then cached."""
    global _cached
    if _cached is None:
        platform = _detect_platform()
        _cached = _CAPS_BY_PLATFORM.get(platform, _CAPS_BY_PLATFORM[CPU])
        if platform not in _CAPS_BY_PLATFORM:
            logger.warning("Unknown JAX platform %r; using CPU capabilities", platform)
        else:
            _cached = _CAPS_BY_PLATFORM[platform]
    return _cached


def reset_caps_cache() -> None:
    """Forget the detected backend. For tests that flip `JAX_E_PLATFORM`."""
    global _cached
    _cached = None


def active_platform() -> str:
    return caps().platform


def is_neuron() -> bool:
    return caps().is_neuron


def is_tpu() -> bool:
    return caps().is_tpu
