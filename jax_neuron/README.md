# Gemma 4 E2B QAT on JAX-Neuron

This directory is the experimental JAX backend. It is intentionally separate
from the proven `torch_neuronx` Option-B serving path under `quant/`.

## Milestones

1. ~~Prove that the installed JAX PJRT plugin compiles and executes the decoder
   primitives on both NeuronCores of one `inf2.xlarge`.~~ **Done** 2026-07-27 —
   `probe.py`, [report](../benchmarks/runs/2026-07-27-inf2-jax-probe/REPORT.md).
2. ~~Prove that `neuronx-cc` accepts the *real* engine graphs, not a
   hand-written stand-in.~~ **Done** 2026-07-30 — `compile_probe.py`,
   [report](../benchmarks/runs/2026-07-30-neuron-compile-e2b/REPORT.md).
   Decode, prefill, and sampling all compile to NEFFs for `inf2`.
3. ~~Load `google/gemma-4-E2B-it-qat-q4_0-unquantized` into the pure-JAX Gemma 4
   model and establish greedy token parity with the existing PyTorch reference.~~
   **Done** 2026-07-31 — `parity.py`,
   [report](../benchmarks/runs/2026-07-31-inf2-jax-parity-e2b/REPORT.md).
   4/4 prompts token-identical on Neuron against a PyTorch float32 CPU
   reference. Found and fixed two correctness bugs in the *shared* engine's
   windowed-KV path that affect TPU equally.
4. Wrap the compiled functions with the existing OpenAI-compatible HTTP
   contract and benchmark them against the Option-B baseline.
5. Only after BF16 parity, evaluate the compressed W4A16 checkpoint. Loading a
   QAT checkpoint is not evidence that compressed weights remain compressed in
   the compiled graph.

## Platform quirks

`../docs/neuron-jax-quirks.md` collects what this stack does differently from
TPU, measured on device. Read it before debugging anything here — the headline
is that a gather over the 4.70 GB PLE table returns **zeros rather than an
error**, which surfaces as a clean `200 OK` with zero completion tokens, and
that the workaround for it currently costs a factor of 2700.

## Three tools, three different questions

`probe.py` asks whether the runtime works, `compile_probe.py` whether the
compiler accepts this engine's graphs, and `parity.py` whether the accepted
graphs compute the right thing. Passing the first two says nothing about the
third: milestone 3 found the engine compiling and running happily while
producing fluent, wrong output.

### `parity.py` — does it compute the right thing?

Needs the real checkpoint and a PyTorch that can run it; the *subject* runs on
whatever backend you point it at.

```bash
# oracle once, anywhere with torch + the checkpoint
python3 jax_neuron/parity.py --local-dir "$CKPT" --skip-subject --save-reference ref.json

# subject, on the Inf2 host
python3 jax_neuron/parity.py --local-dir "$CKPT" --reference ref.json \
  --subject-platform neuron
```

Greedy-decodes the same prompts through `JaxGemmaEngine` — the class the server
loads — and through Hugging Face `transformers` in float32 on the CPU, then
compares token ids. Both sides get byte-identical inputs. A divergence reports
the index it broke at and the reference's top1−top2 margin there, which
separates a numerical tie from a real bug.

**Run `--subject-platform cpu` before suspecting the device.** If the engine
diverges on CPU too, the accelerator is exonerated and the bug is in this
repository. That is exactly how milestone 3's windowed-KV bug was caught.

## Two probes, two different questions

### `probe.py` — does the runtime work?

Run inside the JAX virtual environment **on an Inf2 host**:

```bash
source /opt/aws_neuronx_venv_jax/bin/activate
python jax_neuron/probe.py
```

Checks device discovery and executes a hand-written Gemma-shaped decoder block:
RMS normalization, grouped-query attention, a gated MLP, and a functional
static-KV update. No model weights, no Hugging Face token. Success is reported
as JSON; `platform` must be `neuron` and `device_count` must be `2` on
`inf2.xlarge`.

### `compile_probe.py` — does the compiler accept *this* engine?

Runs **anywhere x86-64 Linux**, no Inferentia device and no weights:

```bash
pip install "neuronx-cc==2.26.*" --extra-index-url=https://pip.repos.neuron.amazonaws.com
python3 jax_neuron/compile_probe.py --tiny        # ~3 min, all stages
python3 jax_neuron/compile_probe.py --stage decode  # full E2B geometry
```

Lowers the shared engine's actual `prefill_with_kv_cache`,
`make_cached_decode_step`, and `onchip_sample_tpu_v6e_jax` to HLO and hands each
to `neuronx-cc --target inf2`. Hardware is unnecessary because the compiler is
ahead-of-time — a NeuronCore executes a NEFF, it does not produce one — and
weights are unnecessary because `jax.eval_shape` describes the 7.29 GB parameter
tree without allocating it.

Exit status is non-zero if any stage fails to lower or compile. A failure names
the offending HLO instruction, which maps back to a line in
`ports/gemma4/jax_e_model.py`.

This is a **compilability** gate. A PASS means the graph is expressible on
Neuron. It says nothing about numerics or speed; those need the device, the real
checkpoint, and the parity tests in `deployments/aws-inf2/README.md`.

## Where the platform differences live

`ports/gemma4/backend.py` holds one capability table, and every platform-specific
branch in the engine reads it instead of testing for TPU inline. `JAX_E_PLATFORM`
overrides detection so the Neuron branches can be exercised on a CPU host —
that is how `tests/test_backend_caps.py` runs without hardware. It does not make
CPU compile like Neuron; only `compile_probe.py` answers that.

Every `False` in the Neuron row is backed by a compiler error quoted in the
capability comment, with one exception, marked as such: buffer donation, which
is AWS documentation and still unverified on device.
