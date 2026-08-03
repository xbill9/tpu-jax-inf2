# Gemma 4 pure-JAX on AWS Inferentia2

This is the AWS deployment scaffold for the same pure-JAX engine used on TPU.
It targets an `inf2.xlarge` through the JAX NeuronX PJRT plugin; it does not use
vLLM, PyTorch, `optimum-neuron`, or NxD Inference.

The model math, safetensors loader, cached decode, OpenAI API, and corrected
benchmark methodology remain shared with the TPU path. The platform-specific
pieces are isolated here:

- `neuron_entrypoint.py` configures and verifies the Neuron JAX backend before
  importing the server.
- `user_data.sh` mounts the cache volume, installs the Inf2-compatible JAX
  NeuronX stack, and creates a systemd service.
- `deploy.py` plans, launches, or terminates one tagged EC2 host with SSM access
  and a persistent EBS cache volume.

> **Start here to bring a host up: [RESUME.md](RESUME.md).** It carries the
> current launch command, the verification order, and the traps. Sections below
> that predate 2026-08-03 describe older pins — RESUME.md and
> [BISECT.md](../../benchmarks/runs/2026-08-02-inf2-latest-stack-e2b/BISECT.md)
> win where they disagree.

## Support boundary

AWS documents JAX NeuronX on Inf2 as beta. This scaffold is therefore a porting
target, not a measured-performance claim. In particular:

- W4A16 uses the JAX reference dequantize-and-matmul path, which is the engine
  default. The optional fused kernel lowers through Mosaic and cannot compile on
  Neuron at all, so `set_w4a16_impl` refuses it outright on this platform.
  **That reference path miscomputes on the NeuronCore when it runs in-graph**
  (greedy decode emits one token repeated, 0.0 agreement against a CPU oracle).
  The service therefore runs `--dequant-at-load`, which performs the identical
  arithmetic on the host and is correct at ~43 tok/s settled. Removing that flag
  brings the garbage back. Localized 2026-08-03; `neuronx-cc`'s handling of the
  fused dequant-and-matmul at real shapes is still open upstream. See
  [BISECT.md](../../benchmarks/runs/2026-08-02-inf2-latest-stack-e2b/BISECT.md).
- The complete Gemma decode, prefill, and sampling graphs compile to NEFFs for
  `inf2` (see the compatibility section below) **and produce correct tokens**:
  greedy output is identical to a PyTorch float32 CPU reference on the real
  checkpoint — [2026-07-31 parity
  report](../../benchmarks/runs/2026-07-31-inf2-jax-parity-e2b/REPORT.md).
  Throughput is still unmeasured; do not quote that run's timings.
- Neuron is documented to support JAX buffer donation. The parity run had
  `donate_cache=True` and was token-exact, so donation does not break
  correctness there — but whether the plugin donates or silently copies is a
  separate measurement and is still open.
- `jax.debug` callbacks/checkify, dynamic `while_loop`, integer dot products,
  and several other JAX features are unsupported on Neuron. The serving path
  must remain static-shaped.
- **SUPERSEDED 2026-08-02 — the two paragraphs below describe the old pinning and
  are kept only for the reasoning.** The bootstrap now takes
  `jax-neuronx[stable]==0.10.0.1.0.*` on `ami-09e1477ba5140fe3e` (Ubuntu 24.04,
  **SDK 2.31.0**, NRT 2.33.10.0) with jax/jaxlib 0.9.2, and `libneuronxla` is
  **not** pinned — the metapackage resolves 3.0.3854.0, the exact build the old
  pin existed to avoid, and it works because this AMI ships an NRT it matches.
  The rule that survives: move the jax-neuronx pin and the AMI together, and
  re-run `jax_neuron/probe.py` (a one-minute check) when you do.

  The bootstrap pins the JAX NeuronX component to `0.6.2.1.0.*`, which is an
  SDK-2.28 build. This pin is about the JAX PJRT plugin only — Inf2 itself is not
  deprecated on newer lines; a sibling deployment in this org runs an SDK-**2.30**
  vLLM container on the same `inf2.xlarge`. If you move the pin, move the AMI with
  it — and re-run `jax_neuron/probe.py`, which is a one-minute check.

  **Measured 2026-07-31:** no SDK-2.28 DLAMI is offered in us-east-2; the oldest
  images that state a version are 2.29.0. What was verified to work is
  `ami-05235a8b272ee7f7e` (Base Neuron AMI 20260511, **SDK 2.29.1**) with the
  0.6.2.1.0 pin **and `libneuronxla` pinned to the 2.2 line**. That second pin is
  not optional and is not implied by the first: `jax-neuronx==0.6.2.1.0.*`
  constrains only `libneuronxla>=2.2.12677.0`, so pip takes 3.0.x, which targets
  an NRT 3.0 runtime this AMI does not ship. The install succeeds and dies much
  later at PJRT load with
  `undefined symbol: nrta_event_register_xu_completion, version NRT_3.0.0`.

## Neuron compatibility of the shared engine — measured

`jax_neuron/compile_probe.py` compiles the engine's real graphs
(`prefill_with_kv_cache`, `make_cached_decode_step`, the sampler) for `inf2`
with `neuronx-cc`. It needs neither an Inf2 instance nor model weights: the
compiler is ahead-of-time and an x86-64 pip wheel, and `jax.eval_shape` supplies
the 7.29 GB parameter tree as shapes. Full results:
[`benchmarks/runs/2026-07-30-neuron-compile-e2b/REPORT.md`](../../benchmarks/runs/2026-07-30-neuron-compile-e2b/REPORT.md).

Run it before blaming the hardware for anything:

```bash
pip install "neuronx-cc==2.26.*" --extra-index-url=https://pip.repos.neuron.amazonaws.com
python3 jax_neuron/compile_probe.py --tiny
```

### Confirmed against neuronx-cc 2.26 / target inf2

| Finding | Where | Status |
|---|---|---|
| `lax.top_k` rejected (`NCC_EVRF001 Operator topk is not supported`) | top-k sampler | **FIXED** — `_kth_largest` uses an iterative masked-max threshold when `caps.device_top_k` is False. `argmax`, `sort`, and `jax.random.categorical` all compile; the masked-max was picked over `sort` on NEFF size (0.1 MB vs 2.5 MB). |
| fp8 KV cache rejected (`NCC_EVRF051 Data type F8E4M3FN is not supported on TRN1/TRN2`) | `jax_engine.resolve_cache_dtype` | **FIXED** — fp8 names raise on Neuron and point at `int8`, which compiles and carries the same per-token scales for the same one-byte capacity. |
| jaxlib's HLO packs instruction ids as `(computation_id << 32) \| index`; neuronx-cc's older embedded XLA CHECKs them against int32 and aborts | any JAX → neuronx-cc lowering | **WORKED AROUND** — `compile_probe.normalize_instruction_ids` renumbers densely. Toolchain skew, not a model issue. |
| Pallas/Mosaic has no Neuron backend | fused W4A16 kernel | **BY DESIGN** — `set_w4a16_impl` refuses `"fused"` on Neuron rather than routing it to the Pallas interpreter, which would unroll the kernel's K loop into the graph. `"reference"` is the default and measured fastest on TPU anyway. |
| `JAX_DEFAULT_PRNG_IMPL=rbg` required for sampling | entrypoint | **FALSE** — `threefry2x32`, `rbg`, and `unsafe_rbg` all compile a categorical over the 262144 vocab. `rbg` is kept as AWS guidance and a cost preference, not a requirement. |

### Corrected — inherited from the sibling port, measured false here

The rows below came from a PyTorch/NxD port of the same model family to the same
hardware. That is a different stack, and on this one they do not hold. They are
kept so the claims are not re-imported.

| Inherited claim | What the compiler actually says |
|---|---|
| Data-dependent scatter must be replaced with one-hot arithmetic | The sampler's vector-indexed `mask.at[arange(B)[:, None], idx].set(vals)` **compiles** (0.02 MB NEFF) when given indices as inputs — on neuronx-cc 2.26 and 2.23 alike. The scalar `valid.at[:, slot].set(True)` and the KV `dynamic_update_slice` compile inside the decode step that passed. No rewrite needed. |
| `tanh` logit softcap over the 262144-token vocab overflows the 196608 B/partition SBUF (`NCC_INLA001`) | **Not reproduced at full size.** The softcap over the real 262144 vocab sits inside the full-E2B decode module that compiled to a 44.3 MB NEFF. If it ever does bite, the mitigation still stands: softcap is monotonic, so `argmax(softcap(x)) == argmax(x)` and greedy decode is unaffected by dropping it. |

### Bootstrap defects fixed 2026-07-31

`user_data.sh` had never completed a launch end to end. Each of these aborted it
under `set -euo pipefail`, leaving a host that boots, reports healthy, and serves
nothing. All four are fixed; they are recorded so they are not reintroduced.

| Defect | How it presented |
|---|---|
| `install -d /opt/gemma4` left the parent root-owned | `Permission denied: '/opt/gemma4/venv'` — the venv is built as `ubuntu` |
| `--index-url` instead of `--extra-index-url` | The Neuron repo holds no `jax` wheel, so `jax<=0.6.2,>=0.4.30` resolved against nothing: `No matching distribution found for jax` |
| `libneuronxla` unpinned | pip took 3.0.x against an NRT 2.31 AMI; `undefined symbol: nrta_event_register_xu_completion` at PJRT load |
| systemd unit set no `PATH` | `libneuronxla` shells out to `neuronx-cc` by bare name; the unit's default PATH excludes the venv, so the first compile died as `XlaRuntimeError: UNKNOWN: sh: 1: neuronx-cc: not found` |

Run `jax_neuron/probe.py` on a fresh host before anything else. It exercises
driver, plugin, PATH, and device discovery in about a minute, and every one of
the failures above surfaces there rather than twenty minutes into a model load.

### Still open

| Risk | Where | Note |
|---|---|---|
| Fused/SDPA attention overflowed SBUF at larger sliding windows | not applicable — `eager_attention_jax` is already eager | Already the portable choice; keep it. |
| Param-only checkpoint loads silently skip the `layer_scalar` **buffer**, over-scaling every layer ~16x into `cos ≈ 0` garbage with no error | handled at `ports/gemma4/jax_e_model.py` (`layer_scalar`) | Covered; the safetensors loader reads it explicitly. Do not regress this. The compile probe supplies it too, so the compiled graph matches the served one. |
| Host RAM, not the accelerator, is the binding constraint on `inf2.xlarge` | `user_data.sh` swapfile | Already provisioned (`--swap-gib`, default 32). |
| Buffer donation on the real plugin | `jax_engine.donate_cache` | The one capability in the table not backed by a compile — it is AWS documentation. Worth 1.62x on TPU; verify on device before assuming it here. |

Two capacity notes carried over from that port: `inf2.xlarge` and `inf2.8xlarge`
carry the *identical* 2-core / 32 GB-HBM accelerator and differ only in host
vCPU and RAM, so the cheap box is cheaper per token once swap is in place; and
"E2B" is a MatFormer *effective* parameter count — the real device footprint is
~5B, which is why capacity must be planned from real parameters. This scaffold
serves the W4A16 QAT checkpoint, which is smaller again.

### Measured device occupancy

`neuronx-cc` reports **8.16 GB** for the full-E2B decode step at batch 1,
context 512, against a stated budget of 16 GiB (`DRAM size: 17179869184`).

That budget is 16 GiB and not 32 because the graph compiles
`--logical-nc-config=1` — one NeuronCore. The chip's other core is idle;
splitting the model across both is unexplored. Plan capacity per core.

The 44.3 MB NEFF is the executable, not the model: 127.5 MB of instructions and
11.2M DMA descriptors for a fully unrolled 35-layer graph, compressed. Weights
are graph inputs and are not in it. Of the 8.16 GB, most is the 7.29 GB
parameter tree, and 4.70 GB of *that* is the BF16 per-layer-embedding table —
so `ple_bits=8` (−2.35 GB) is the first lever if a longer context does not fit.

The same log shows **19.04 GB of DMA traffic per decode step, ~45% of it
spill/reload**, which would make the port bandwidth-bound (~46 ms/token at
Inferentia2's ~410 GB/s per core). Two things to know before acting on it:

- It is a static estimate from a graph that has never run. Confirm with
  `neuron-monitor` on a device before treating it as throughput.
- **`neuronx-cc --optlevel` will not improve it.** Levels 1, 2, and 3 all hand
  the backend `--optlevel 2 --policy 3` and produce byte-identical DMA figures,
  so this is already the optimized schedule. The levers are in the graph —
  `dequant_at_load`, `ple_bits=8`, windowed KV, batching — not in the flag.

## Prerequisites

1. An AWS account with Inf2 quota, an existing VPC/subnet/security group, and
   an EC2 instance profile that includes `AmazonSSMManagedInstanceCore`.
2. The instance profile may read the Hugging Face token from Secrets Manager
   secret `hf-token`. Store it as a plain string, not JSON — the bootstrap
   passes `SecretString` straight through as `HF_TOKEN`.
3. A source bundle uploaded to S3. It should unpack with this repository at its
   root. The instance role needs `s3:GetObject` for that object.
4. A Deep Learning AMI Neuron image pinned to SDK 2.28. Pass its AMI ID with
   `--ami-id`; automatic name discovery is convenient for development but does
   not guarantee the SDK line.

No inbound SSH is required. The API binds to `127.0.0.1:8000`; reach it through
SSM port forwarding or put a private load balancer in front of it.

## Two ways to launch

`deploy.py` (below) is the plan/apply CLI: explicit, reviewable, and the only
path that attaches and reports the persistent compile-cache volume.

The `inf2-devops` MCP server is the conversational path — `create_inf2_instance(
serving="jax", source_uri=...)`. It renders **this same `user_data.sh`**, read at
call time rather than copied, so the two cannot drift; `tests/test_inf2_mcp_jax.py`
asserts the rendered bytes are identical. Its `verify_neuron_health`,
`get_vllm_logs`, and `get_endpoint` tools all take `serving="jax"` and must be
given it — their default targets a docker container this host does not run.

The MCP path does not manage the cache volume. Use `deploy.py` when that matters.

## Plan and launch

Install only the local control-plane dependency:

```bash
python3 -m pip install boto3
```

Generate a launch plan (read-only; this is the default):

```bash
python3 deployments/aws-inf2/deploy.py plan \
  --region us-east-1 \
  --subnet-id subnet-... \
  --security-group-id sg-... \
  --instance-profile-name gemma4-inf2 \
  --source-uri s3://my-bucket/tpu-jax-inf2.tar.gz
```

Launch only after inspecting the plan:

```bash
python3 deployments/aws-inf2/deploy.py launch --apply \
  --region us-east-1 \
  --subnet-id subnet-... \
  --security-group-id sg-... \
  --instance-profile-name gemma4-inf2 \
  --source-uri s3://my-bucket/tpu-jax-inf2.tar.gz
```

The launcher refuses to create a second pending or running instance with the
same `Project` tag in the region. Spot is opt-in with `--market-type spot`.

## Storage and teardown

Two volumes are attached. The root volume is disposable and deletes on
termination. A second gp3 volume on `/dev/sdf` holds `/opt/gemma4/cache`
(Hugging Face weights, the XLA compilation cache, and the Neuron compile cache)
and is **retained** on termination, because recompiling the Gemma graph from
cold costs far more than the idle volume does.

Teardown plans first, like launch:

```bash
python3 deployments/aws-inf2/deploy.py terminate --region us-east-2   # plan
python3 deployments/aws-inf2/deploy.py terminate --apply --region us-east-2

# ...or terminate and stop the volume charge in one step
python3 deployments/aws-inf2/deploy.py terminate --apply --delete-cache --region us-east-2
```

Without `--delete-cache` the volume is retained and keeps billing (~$0.08/GiB-month),
which is what you want between runs and not what you want when you are done.

## Fast relaunch: reuse the cache volume

A cold start is roughly 15–25 minutes, and two terms dominate it: the ~9.6 GB
checkpoint download and the Neuron graph compile (measured at 1173 s for the
full-E2B decode step on 4 cores). Both land on the cache volume, so reattaching
it skips both.

```bash
python3 deployments/aws-inf2/deploy.py launch --apply \
  --region us-east-2 --reuse-cache \
  --subnet-id subnet-... --security-group-id sg-... \
  --instance-profile-name gemma4-inf2 \
  --source-uri s3://my-bucket/tpu-jax-inf2.tar.gz
```

`--reuse-cache` attaches an available volume tagged `Project=<project>` in the
launch AZ, and falls back to creating one when there is nothing to reuse.
`--cache-volume-id vol-...` names one explicitly. Ambiguity is an error rather
than a guess: attaching the wrong cache is worse than a cold start because it
looks like it worked.

Two constraints worth knowing before you plan a launch:

- **EBS is AZ-locked.** The volume's Availability Zone dictates the subnet you
  can launch into. `deploy.py` refuses a mismatch and names both zones. This
  interacts with spot capacity — on 2026-07-31 `us-east-2c` had no
  `inf2.8xlarge` spot capacity at all while `2a` and `2b` did, so the cheapest
  zone and the zone holding your cache are not always the same one.
- **The volume is attached after the instance launches**, because `RunInstances`
  can only *create* volumes, never attach existing ones. `user_data.sh` waits up
  to `CACHE_WAIT_SECS` (180 s) for the device to appear. If it gives up, it says
  so loudly and falls back to the root volume — where the caches are destroyed at
  termination.

## When a bootstrap fails, retry it in place

The bootstrap records a marker per completed phase under
`/var/lib/gemma4-bootstrap`, so it is re-runnable:

```bash
sudo bash /var/lib/cloud/instance/user-data.txt
```

Completed phases are skipped, so a failure in the last step costs a retry rather
than a fresh 15-minute launch. The source bundle is deliberately *not* phase
marked — it is the one thing that changes between runs, and skipping it would
silently serve stale code.

`jax_neuron/probe.py` runs as a gate immediately after the Python dependencies
and before the checkpoint download. It exercises driver, PJRT plugin, `PATH`,
and `neuronx-cc` together in about a minute. Every bootstrap defect listed below
surfaces there; without the gate the first thing to touch the accelerator is the
model load, ~20 minutes in, and it reports as an XLA error rather than as a setup
problem.

## Validate on the host

```bash
sudo journalctl -u gemma4-jax-inf2 -f
sudo -u ubuntu /opt/gemma4/venv/bin/python \
  /opt/gemma4/app/deployments/aws-inf2/neuron_entrypoint.py --check-only
curl http://127.0.0.1:8000/health
```

Before publishing any Inf2 result, run cached-decode parity, HTTP smoke tests,
and a corrected v2 sweep on the device. Do not reuse TPU throughput or memory
numbers as Inf2 claims.

### If the output is garbage

Two rules from the sibling port, both earned by losing hours to them:

**Run the CPU oracle before you suspect the device.** Load the same checkpoint
on the same box with `JAX_PLATFORMS=cpu` and run one greedy forward. If the CPU
reference emits the *same* garbage, the accelerator is exonerated and the bug is
upstream — tokenizer, inputs, or weights. In that port every single garbage-output
incident was innocent silicon: a missing `tokenizer.json` that mapped every prompt
to `<unk>`, a mis-restored weight reload, an unloaded scalar buffer, and a
driver/SDK mismatch. Sanity-check `tok("hello world").input_ids` first; it is a
two-minute check that has replaced multi-hour compiler hunts.

**Validate the serving path, not the trace.** Parity that passes in-process on a
freshly built model does not exercise the code path a fresh server process uses.
Check the exact artifact, loaded the exact way the service loads it, against an
*independent* float reference. A green test against the wrong oracle — that port
had an auto-port report "100% PASS" against a golden built from a PLE-stripped
checkpoint — is worse than a red one.
