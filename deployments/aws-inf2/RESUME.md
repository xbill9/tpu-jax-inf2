# Resuming Gemma 4 E2B on Inferentia2

State as of **2026-08-03**. Everything below was measured, not assumed.
Background and evidence: `benchmarks/runs/2026-08-02-inf2-latest-stack-e2b/BISECT.md`.

## Where things stand

**It works.** Gemma 4 E2B serves correct output on Inf2 at ~43 tok/s settled,
with the 65x `NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU` workaround **off**. The
long-standing W4A16 miscomputation was localized to the *in-graph* dequant and is
routed around by `--dequant-at-load`, which the service now passes by default.

No instance is running — the host was terminated after validation. Nothing is
left to clean up in AWS.

## Bring it back up

One command. ~7 minutes to serving-ready from nothing.

```bash
python3 deployments/aws-inf2/deploy.py launch --apply \
  --region us-east-2 \
  --instance-type inf2.8xlarge \
  --market-type spot \
  --source-uri s3://xbill-gemma4-inf2-use2/tpu-jax-inf2-latest.tar.gz \
  --subnet-id subnet-0b0f0d29473b0a624 \
  --security-group-id sg-037ebb092f5a90464 \
  --instance-profile-name gemma-build-s3 \
  --ami-id ami-09e1477ba5140fe3e
```

**`--source-uri` is the one that will bite you.** `deploy.py` does *not* build or
upload the bundle; it downloads whatever is already in S3. The app code on the
host comes from that tarball, so a stale URI silently serves pre-fix code and
reproduces the garbage output. `tpu-jax-inf2-latest.tar.gz` currently points at
commit `c42aadd` (the fix). **After any change to the engine, rebuild it:**

```bash
SHA=$(git rev-parse --short HEAD)
git archive --format=tar.gz --prefix=tpu-jax-inf2/ HEAD -o /tmp/tpu-jax-inf2-$SHA.tar.gz
aws s3 cp /tmp/tpu-jax-inf2-$SHA.tar.gz s3://xbill-gemma4-inf2-use2/tpu-jax-inf2-$SHA.tar.gz --region us-east-2
aws s3 cp /tmp/tpu-jax-inf2-$SHA.tar.gz s3://xbill-gemma4-inf2-use2/tpu-jax-inf2-latest.tar.gz --region us-east-2
```

(`user_data.sh` is read from your *local* working tree by `deploy.py` at launch
time, so edits there take effect without a rebuild. Everything else does not.
That asymmetry is easy to trip over.)

### Then verify, in this order

```bash
INST=<new-instance-id>
aws ssm start-session --target $INST --region us-east-2   # or use send-command

systemctl is-active gemma4-jax-inf2
curl -s localhost:8000/health
```

`/health` must report **`"weights":"bf16"`**. If it says `w4_int4`, the
`--dequant-at-load` flag did not reach the engine and **output will be wrong** —
check `/usr/local/bin/gemma4-jax-inf2-run`.

Then a smoke test. `The capital of France is` must return `Paris.`, not a
repeated token.

## Traps, all of which cost time on 2026-08-02/03

**A slow first request is the compiler, not a hang.** `max_new_tokens` is a
`static_argname` (`jax_engine.py:353`) fed from HTTP `max_tokens`, so a novel
value triggers a fresh `neuronx-cc` compile. First touch of a bucket measured
1.74 tok/s against 43 warm, and a cold compile can exceed 900 s while `curl`
returns nothing. Before concluding anything:

```bash
curl -s localhost:8000/metrics | grep requests_total   # in-flight shows 0 success AND 0 failed
pgrep -f neuronx-cc                                     # still compiling?
```

**Only one process can hold the NeuronCores.** Stop the service before any parity
or bisect run, or JAX dies at backend init with `NRT_FAILURE status_code=1`:

```bash
systemctl stop gemma4-jax-inf2
```

**Quote only settled numbers.** Three warmth states are easy to confuse: cold
process, first-touch-of-bucket, and settled. They measured 0.82, 1.74, and
43 tok/s for the same work. The settled figure reproduces within 2% across
independent processes — and is still batch 1, greedy, 16 tokens, one bucket, so
it is not engine capability (`CLAUDE.md` measurement rule).

**Parity against `ref.json` is cross-checkpoint for w4a16 subjects.** The oracle
is built from `-q4_0-unquantized`; a `-w4a16-ct` subject legitimately drifts.
Judge by agreement rate, not exact match — correct w4a16 looks like 92–94%
agreement diverging late, not `passed: true`.

**Spot: retry per-AZ, not per-region.** Placement scores of 1/10 across
us-east-1 and us-east-2 still yielded an `inf2.8xlarge` in **us-east-2a** at
$0.3872/hr (80% off) after us-east-2c failed with `InsufficientInstanceCapacity`.
On-demand is $1.9679/hr if spot is dry.

**A retained cache volume is usually not worth it.** EBS is AZ-locked, so it pins
the AZ and costs you the spot retry. A cold rebuild is ~7 min (7.8 GB checkpoint
in ~3 min); the volume costs $16/mo. The 2026-08-03 run deleted its volumes.

**Set a cost cap on the host itself**, not just in your session — it then holds
regardless of what the controlling session does:

```bash
shutdown -h +180   # 3 hours
```

## The open defect

`neuronx-cc` miscompiles the fused dequant-and-matmul at real shapes. This is
**not fixed** — `--dequant-at-load` routes around it by doing the dequant on the
host. Drop the flag and the garbage returns.

Already ruled out, so don't re-tread (evidence in `BISECT.md`):

- the int4 unpack primitive — bit-exact on device, workaround off, all four
  integer stages, full int32 range (`jax_neuron/bisect_w4a16.py` reproduces this
  in seconds, no weights needed);
- the dequant math and the loader — coherent text on CPU from the same
  checkpoint;
- windowed KV, int8 KV, the sampler, the logits path, and the HTTP layer — all
  passed parity earlier.

Next step if you want the real fix: dump the HLO for the fused
dequant-and-matmul and compare against the unfused form, or file it upstream with
`bisect_w4a16.py` extended to full projection shapes inside one jitted graph
(the current version uses a small isolated graph, which passes).

## Guard rails now in the repo

`tests/test_w4a16_host_dequant.py` fails if the workaround default returns to
`1`, or if the service stops passing `--dequant-at-load`. Both regressions are
silent otherwise: the first costs 65x while output stays *correct*; the second
makes output wrong while every other test still passes. The suite went 175/175 on
the very host that was serving garbage — do not treat a green suite as evidence
the Neuron path works.
