#!/bin/bash
set -euo pipefail

# Substituted by deploy.py. Values are shell-quoted before insertion.
SOURCE_URI=__SOURCE_URI__
MODEL_ID=__MODEL_ID__
HF_SECRET_ID=__HF_SECRET_ID__
AWS_DEPLOY_REGION=__AWS_REGION__
MAX_MODEL_LEN=__MAX_MODEL_LEN__
SWAP_GIB=__SWAP_GIB__
NEURON_CC_FLAGS_VALUE=__NEURON_CC_FLAGS__

# How long to wait for a REUSED cache volume to appear. deploy.py attaches it
# after run_instances returns, so on a fast boot this script can genuinely get
# here first. Cheap to wait, expensive to miss: missing it silently costs a
# 9.6 GB re-download and a full NEFF recompile.
CACHE_WAIT_SECS=180

# neuronx-cc is a console script and libneuronxla shells out to it by bare name.
# Defined once here and reused for both the fail-fast probe and the systemd unit
# so the two cannot drift. No venv, per this repo's standard (CLAUDE.md Coding
# Standards): packages install into the DLAMI's own interpreter, so the console
# scripts land in /usr/local/bin, which is already on this PATH.
SERVICE_PATH=/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

exec > >(tee /var/log/gemma4-jax-inf2-bootstrap.log | logger -t gemma4-inf2 -s 2>/dev/console) 2>&1

# Phase markers make this script re-runnable. Every phase below is skipped if it
# already completed, so a bootstrap that dies partway can be retried in place
# with `bash /var/lib/cloud/instance/user-data.txt` instead of costing a full
# relaunch — which, on a ~15 minute cold start, is the difference between a
# one-minute retry and starting over.
PHASE_DIR=/var/lib/gemma4-bootstrap
install -d "$PHASE_DIR"
phase_done() { [ -f "$PHASE_DIR/$1" ]; }
phase_mark() { touch "$PHASE_DIR/$1"; }

export DEBIAN_FRONTEND=noninteractive
if ! phase_done os-packages; then
  apt-get update
  # The Neuron DLAMI already ships AWS CLI v2 and the Python the Neuron wheels
  # are built against; do not install a second interpreter or a v1 CLI over them.
  # python3-pip, not python3-venv: this deployment installs into that shipped
  # interpreter directly rather than building a venv on top of it.
  apt-get install -y python3-pip
  phase_mark os-packages
fi
command -v aws >/dev/null || { echo "FATAL: AWS CLI missing from the AMI" >&2; exit 1; }

# inf2.xlarge is 4 vCPU / 16 GB host. Field-measured on the sibling NxD port:
# the one-time Neuron graph load peaks ~14.5 GB, and a stock DLAMI with no swap
# OOM-kills the serving process AND the SSM agent -- which also costs you the
# only way back into a host with no inbound SSH. Swap first, install second.
if [ "$SWAP_GIB" -gt 0 ] && ! swapon --show=NAME --noheadings | grep -q '^/swapfile$'; then
  fallocate -l "${SWAP_GIB}G" /swapfile ||
    dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GIB * 1024))
  chmod 0600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >>/etc/fstab
fi

CACHE_ROOT=/opt/gemma4/cache
# Owned by ubuntu, not root: the app tree and caches below are written as ubuntu
# *inside* this directory, and a root-owned parent fails them with EACCES. Under
# `set -e` that aborts the bootstrap before pip and systemd ever run, leaving a
# host that looks booted and serves nothing. (This mattered doubly when a venv
# was created here; the ownership requirement outlives it.)
install -d -o ubuntu -g ubuntu /opt/gemma4

# Find the separate EBS cache volume: on Nitro the attachment point is remapped
# to an NVMe name, so identify it as the one disk the root filesystem is not on.
# inf2 has no instance store, so any other disk is ours.
find_cache_dev() {
  local root_source root_disk name kind
  root_source="$(findmnt -no SOURCE /)"
  root_disk="$(lsblk -no PKNAME "$root_source" 2>/dev/null || true)"
  [ -n "$root_disk" ] || root_disk="$(basename "$root_source")"
  while read -r name kind; do
    [ "$kind" = disk ] || continue
    [ "$name" = "$root_disk" ] && continue
    echo "/dev/$name"
    return 0
  done < <(lsblk -dno NAME,TYPE)
  return 1
}

# A REUSED cache volume is attached by deploy.py *after* run_instances returns,
# because RunInstances cannot take an existing VolumeId — BlockDeviceMappings
# only creates new volumes. So the device may legitimately not exist yet when
# this script runs, and the old code raced it: it looked once, found nothing,
# and silently put a 9.6 GB checkpoint plus the Neuron cache on the root volume
# that is destroyed at termination. That is the whole saving, lost to a warning
# nobody reads. Wait for it, then fall back.
cache_dev=""
deadline=$(( SECONDS + CACHE_WAIT_SECS ))
while [ "$SECONDS" -lt "$deadline" ]; do
  cache_dev="$(find_cache_dev || true)"
  [ -n "$cache_dev" ] && break
  sleep 3
done

if [ -n "$cache_dev" ]; then
  # Only format a blank volume; a reattached one already holds the checkpoint
  # and the Neuron/XLA compile caches, which are the whole reason it survives
  # termination. `blkid` succeeding is the guard — never mkfs on a hit.
  blkid "$cache_dev" >/dev/null 2>&1 || mkfs.ext4 -L gemma4cache "$cache_dev"
  install -d "$CACHE_ROOT"
  grep -q '^LABEL=gemma4cache' /etc/fstab ||
    echo "LABEL=gemma4cache $CACHE_ROOT ext4 defaults,nofail 0 2" >>/etc/fstab
  # Idempotent: a bare `mount` of an already-mounted path exits non-zero, and
  # under `set -e` that aborted the whole script — which is what made this
  # bootstrap impossible to re-run after a partial failure.
  mountpoint -q "$CACHE_ROOT" || mount "$CACHE_ROOT"
  echo "cache volume ready at $cache_dev ($(df -h --output=size "$CACHE_ROOT" | tail -1 | tr -d ' '))"
else
  echo "WARNING: no cache volume appeared within ${CACHE_WAIT_SECS}s; caches land" \
       "on the root volume and are LOST at termination (expect a full" \
       "re-download and recompile next launch)" >&2
fi

install -d -o ubuntu -g ubuntu /opt/gemma4/app
install -d -o ubuntu -g ubuntu "$CACHE_ROOT/huggingface"
install -d -o ubuntu -g ubuntu "$CACHE_ROOT/jax"
install -d -o ubuntu -g ubuntu "$CACHE_ROOT/neuron"
chown ubuntu:ubuntu "$CACHE_ROOT"

# Always refreshed, never phase-marked: the source bundle is the one thing that
# changes between launches of the same host, and skipping it on a re-run would
# silently serve stale code.
aws s3 cp "$SOURCE_URI" /tmp/gemma4-source.tar.gz --region "$AWS_DEPLOY_REGION"
tar -xzf /tmp/gemma4-source.tar.gz -C /opt/gemma4/app --strip-components=1
chown -R ubuntu:ubuntu /opt/gemma4/app

# No venv, per this repo's standard: install into the interpreter the Neuron
# wheels were built against. Ubuntu 24.04 marks it externally-managed (PEP 668),
# so every pip call here needs --break-system-packages; without it pip refuses
# with "error: externally-managed-environment" and the bootstrap dies under
# `set -e`. Installing system-wide as root (not --user) puts the neuronx-cc
# console script in /usr/local/bin, which SERVICE_PATH above already carries.
# --break-system-packages lets pip WRITE to the distro interpreter's
# site-packages. It does NOT let pip REPLACE a package that apt installed:
# Debian wheels ship no RECORD file, so any dependency resolution that wants to
# upgrade one dies with
#   ERROR: Cannot uninstall typing_extensions 4.10.0, RECORD file not found.
# and, under `set -e`, takes the whole bootstrap with it — leaving a host with
# os-packages marked, no python-deps, and cloud-init in `error`. --ignore-installed
# is the fix: install over the distro copy rather than trying to remove it.
#
# Also do NOT add `--upgrade pip`. That is reflexive inside a venv and hits the
# identical wall here (`Cannot uninstall pip 24.0`); the shipped pip is new
# enough for everything below.
#
# MEASURED 2026-08-02 on i-0d00da0fd6274952d, the first launch after the venv was
# removed — both failures above are from that host, in that order.
PIP="python3 -m pip install --break-system-packages --ignore-installed"
# Pair this with the newest Neuron DLAMI for the OS line (currently
# ami-09e1477ba5140fe3e, Ubuntu 24.04, Neuron SDK 2.31.0, aws-neuronx-runtime-lib
# 2.33.10.0). The stable metapackage then selects the tested
# JAX/jaxlib/libneuronxla combination from that Neuron package repository.
# --extra-index-url, NOT --index-url. The Neuron repository carries the Neuron
# wheels only; `jax` itself lives on PyPI. `--index-url` *replaces* the default
# index, so the jax-neuronx jax dependency resolves against a repository that has
# never held a jax wheel and the install dies with "No matching distribution
# found for jax" after downloading ~100 MB of libneuronxla. `--extra-index-url`
# is also the form AWS documents.
#
# libneuronxla is deliberately NOT pinned here, which reverses an earlier
# decision — read this before re-adding a pin. Under jax-neuronx 0.6.2.1.0 the
# dependency was `libneuronxla>=2.2.12677.0`, an unbounded lower bound, so pip
# took the newest (3.0.3854.0). That build targets an NRT 3.0 runtime while the
# SDK-2.29.1 AMI line shipped NRT 2.31, the install still succeeded, and the
# failure surfaced much later at PJRT load:
#
#   libneuronpjrt.so: undefined symbol: nrta_event_register_xu_completion,
#   version NRT_3.0.0
#
# `libneuronxla==2.2.*` was the fix FOR THAT PAIRING and is wrong for this one.
# Pinning the current jax-neuronx line lets the metapackage resolve the
# libneuronxla built against the runtime this AMI actually ships, which is what
# the pin was approximating by hand. The tripwire is unchanged: the symbol error
# above means the libneuronxla/NRT pairing is wrong. Move the jax-neuronx pin and
# the AMI together, and re-run jax_neuron/probe.py when you do — the FAIL FAST
# gate below catches it in about a minute.
if ! phase_done python-deps; then
  $PIP 'jax-neuronx[stable]==0.10.0.1.0.*' \
    --extra-index-url https://pip.repos.neuron.amazonaws.com
  # deployments/aws-inf2/requirements-serving.txt, NOT the repo-root
  # requirements.txt — that one lists the MCP server's dependencies and none of
  # the serving stack, so a host built from it installs cleanly and then dies on
  # `import fastapi`.
  $PIP -r /opt/gemma4/app/deployments/aws-inf2/requirements-serving.txt
  phase_mark python-deps
fi

# FAIL FAST. jax_neuron/probe.py discovers both NeuronCores and executes a
# Gemma-shaped decoder block in about a minute, exercising driver, PJRT plugin,
# PATH, and neuronx-cc together. Every bootstrap defect found on 2026-07-31 --
# the root-owned venv parent, the --index-url resolution failure, the
# libneuronxla/NRT symbol mismatch, and the missing PATH -- surfaces HERE.
#
# Without this gate the first thing that touches the accelerator is the model
# load, which happens after a ~9.6 GB checkpoint download and minutes of NEFF
# compilation. A stack that was never going to work then takes ~20 minutes to
# say so, and says it as an XLA error rather than as a setup problem.
if ! phase_done neuron-probe; then
  if sudo -u ubuntu env PATH="$SERVICE_PATH" NEURON_RT_NUM_CORES=2 \
       python3 /opt/gemma4/app/jax_neuron/probe.py; then
    phase_mark neuron-probe
  else
    echo "FATAL: the JAX Neuron stack does not work on this host. Fix it before" \
         "the model download; nothing downstream can succeed." >&2
    exit 1
  fi
fi

cat >/usr/local/bin/gemma4-fetch-hf-token <<'SCRIPT'
#!/bin/bash
set -euo pipefail
umask 077
tmp="$(mktemp /run/gemma4-hf-token.XXXXXX)"
trap 'rm -f "$tmp"' EXIT
aws secretsmanager get-secret-value \
  --region "$AWS_DEPLOY_REGION" \
  --secret-id "$HF_SECRET_ID" \
  --query SecretString \
  --output text > "$tmp"
# This runs as root (ExecStartPre=+) but the service runs as ubuntu, so hand the
# file over explicitly; a root-owned 0600 token makes the unit crash-loop.
chown ubuntu:ubuntu "$tmp"
chmod 0400 "$tmp"
mv -f "$tmp" /run/gemma4-hf-token
trap - EXIT
SCRIPT
chmod 0755 /usr/local/bin/gemma4-fetch-hf-token

cat >/usr/local/bin/gemma4-jax-inf2-run <<'SCRIPT'
#!/bin/bash
set -euo pipefail
export HF_TOKEN
HF_TOKEN="$(cat /run/gemma4-hf-token)"
exec python3 \
  /opt/gemma4/app/deployments/aws-inf2/neuron_entrypoint.py \
  --model "$MODEL_ID" \
  --kv-cache-dtype int8 \
  --quant-mode w4a16 \
  --dequant-at-load \
  --max-model-len "$MAX_MODEL_LEN" \
  --host 127.0.0.1 \
  --port 8000
SCRIPT
chmod 0755 /usr/local/bin/gemma4-jax-inf2-run

cat >/etc/gemma4-inf2.env <<EOF
# neuronx-cc is a console script in /usr/local/bin, and libneuronxla shells out to it
# by bare name to compile every graph. systemd hands the unit a default PATH of
# /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin, which
# does not include it, so the first compile dies inside XLA as
#   XlaRuntimeError: UNKNOWN: sh: 1: neuronx-cc: not found
# — a message that points at the compiler rather than at PATH. /opt/aws/neuron/bin
# carries neuron-ls and friends for diagnostics.
PATH=$SERVICE_PATH
AWS_DEPLOY_REGION=$AWS_DEPLOY_REGION
HF_SECRET_ID=$HF_SECRET_ID
MODEL_ID=$MODEL_ID
MAX_MODEL_LEN=$MAX_MODEL_LEN
NEURON_CC_FLAGS=$NEURON_CC_FLAGS_VALUE
HF_HOME=$CACHE_ROOT/huggingface
# JAX_COMPILATION_CACHE_DIR is deliberately NOT set. ports/gemma4/backend.py:134
# declares persistent_compilation_cache=False for Neuron, and it is right: the
# Neuron PJRT plugin cannot serialize a module for JAX's persistent cache. On
# jax 0.6.2 setting it anyway was silently tolerated (3.9 MB of entries were
# written and nobody noticed the contradiction). On jax 0.9.2 it is a hard
# crash loop at startup, before the model ever loads:
#   RET_CHECK failure (xla/hlo/ir/hlo_module.cc:822)
#   proto.has_host_program_shape()  No program shape found in the proto
# MEASURED 2026-08-02 on ami-09e1477ba5140fe3e / jax 0.9.2: unsetting this is
# the whole fix; the service then starts clean and serves.
#
# Graph caching still happens, via NEURON_COMPILE_CACHE_URL below, which does
# populate $CACHE_ROOT/neuron (model.hlo_module.pb + compile_flags.json per
# MODULE_*). Do NOT add --cache_dir to NEURON_CC_FLAGS to "fix" that: neuronx-cc
# 2.26 rejects it outright and every compile dies as
#   [NCC_EARG002] Illegal argument(s) ... unrecognized: --cache_dir=...
NEURON_COMPILE_CACHE_URL=$CACHE_ROOT/neuron
# RESOLVED 2026-08-03. This was "1" as a correctness workaround costing ~65x
# (5 tokens in 77-84 s). The fault was localized to the IN-GRAPH W4A16 dequant,
# and the service now passes --dequant-at-load, which does the same arithmetic on
# the host and is correct on the NeuronCore with this OFF. Set explicitly rather
# than left unset: the entrypoint only setdefault()s it, so an env file that
# still said "1" would silently reimpose the cost.
# See benchmarks/runs/2026-08-02-inf2-latest-stack-e2b/BISECT.md.
NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=0
EOF
chmod 0600 /etc/gemma4-inf2.env

cat >/etc/systemd/system/gemma4-jax-inf2.service <<'UNIT'
[Unit]
Description=Gemma 4 pure-JAX server on AWS Inferentia2
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
EnvironmentFile=/etc/gemma4-inf2.env
ExecStartPre=+/usr/local/bin/gemma4-fetch-hf-token
ExecStart=/usr/local/bin/gemma4-jax-inf2-run
Restart=on-failure
RestartSec=15
TimeoutStartSec=3600
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now gemma4-jax-inf2.service
