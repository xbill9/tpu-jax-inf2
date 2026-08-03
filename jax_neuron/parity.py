"""Greedy-token parity between this repo's JAX engine and an independent PyTorch reference.

Milestone 3 of `jax_neuron/README.md`: the compile probe proved `neuronx-cc`
accepts the engine's graphs, but a NEFF that exists is not a NEFF that computes
the right thing. This is the correctness gate.

Two rules from `deployments/aws-inf2/README.md`, both earned by losing hours,
shape the design:

**Run the CPU oracle before you suspect the device.** The reference is Hugging
Face `transformers` running the same checkpoint in PyTorch on CPU — a different
implementation of the same math, not a rearrangement of this one. If the JAX
engine diverges from it on CPU too, the accelerator is exonerated and the bug is
in this repository. `--subject-platform` picks which backend the *engine* runs
on; the reference is always CPU/PyTorch.

**Validate the serving path, not the trace.** The subject is `JaxGemmaEngine`
loaded exactly as `jax_openai_server.py` loads it — same class, same loader,
same cached decode step, same sampler. A parity test against a hand-built model
proves the hand-built model works.

The oracle can be computed once and reused: `--save-reference` writes the
reference tokens to JSON, `--reference` reads them back. That is how the device
run works at all, since an Inf2 host need not have a PyTorch that can run Gemma.

    # on any box with torch + the checkpoint
    python3 jax_neuron/parity.py --save-reference ref.json --skip-subject

    # on the Inf2 host
    python3 jax_neuron/parity.py --reference ref.json --subject-platform neuron

Exit status is non-zero if any prompt diverges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# The QAT checkpoint with weights stored *unquantized*. Both sides then read the
# same bf16 numbers, so any divergence is a porting bug rather than a
# quantization difference. Comparing the W4A16 export against a bf16 PyTorch
# reference would diverge for a legitimate reason and prove nothing, which is
# why `jax_neuron/README.md` puts the compressed checkpoint at milestone 5.
DEFAULT_MODEL_ID = "google/gemma-4-E2B-it-qat-q4_0-unquantized"

# Short, deterministic, and chosen to exercise different paths: factual recall,
# a chat-formatted turn, code, and a long-ish prompt that crosses the 128-token
# prefill bucket. Divergence usually appears first on whichever prompt has the
# flattest logit distribution, so variety matters more than length.
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Q: What is 17 * 23?\nA:",
    "def fibonacci(n):",
    "Explain in one sentence why the sky is blue.",
]


@dataclass
class PromptResult:
    prompt: str
    prompt_token_ids: list[int]
    reference_tokens: list[int] = field(default_factory=list)
    subject_tokens: list[int] = field(default_factory=list)
    # Reference logit margin (top1 - top2) at each step, in the reference's own
    # float32. A divergence where this is ~0 is a numerically-tied coin flip; a
    # divergence where it is large is a real bug. Distinguishing the two is the
    # difference between "ship it" and "keep looking".
    reference_margins: list[float] = field(default_factory=list)
    matched: bool | None = None
    first_divergence: int | None = None
    divergence_margin: float | None = None
    agreement: float | None = None
    reference_text: str = ""
    subject_text: str = ""


@dataclass
class ParityReport:
    model_id: str
    max_new_tokens: int
    subject_platform: str = ""
    subject_devices: str = ""
    reference_source: str = ""
    tokenizer_check: dict[str, Any] = field(default_factory=dict)
    results: list[PromptResult] = field(default_factory=list)
    passed: bool = False
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- stage 0


def tokenizer_sanity(tok, model_id: str) -> dict[str, Any]:
    """Cheap checks for the failure that has caused every garbage-output incident.

    A missing or wrong `tokenizer.json` maps every prompt to `<unk>`, and the
    model then dutifully generates garbage from garbage while every layer of the
    stack reports success. Two minutes here has repeatedly replaced multi-hour
    compiler hunts, so it runs before anything is loaded.
    """
    probe = "hello world"
    ids = tok(probe, add_special_tokens=False).input_ids
    decoded = tok.decode(ids)
    unk_id = getattr(tok, "unk_token_id", None)

    check = {
        "model_id": model_id,
        "probe": probe,
        "input_ids": list(map(int, ids)),
        "decoded": decoded,
        "vocab_size": int(getattr(tok, "vocab_size", 0)),
        "bos_token_id": _maybe_int(getattr(tok, "bos_token_id", None)),
        "eos_token_id": _maybe_int(getattr(tok, "eos_token_id", None)),
        "unk_token_id": _maybe_int(unk_id),
        "problems": [],
    }

    if not ids:
        check["problems"].append("tokenizer produced no ids for a non-empty prompt")
    if unk_id is not None and all(int(i) == int(unk_id) for i in ids):
        check["problems"].append(
            f"every token is <unk> ({unk_id}) — the tokenizer files are wrong or missing"
        )
    # Round-trip need not be byte-exact (sentencepiece normalizes whitespace),
    # but it must contain the words. All-<unk> and empty both fail this too;
    # it is kept as a second, independent signal.
    if "hello" not in decoded.lower():
        check["problems"].append(f"round-trip lost the input: {decoded!r}")
    if check["bos_token_id"] is None:
        check["problems"].append(
            "tokenizer reports no bos_token_id; Gemma echoes the prompt without <bos>"
        )
    return check


def _maybe_int(v):
    return None if v is None else int(v)


# --------------------------------------------------------------------- stage 1


def _load_reference_model(path: str):
    """Load the checkpoint in PyTorch, float32, on CPU.

    E2B ships as a *multimodal* checkpoint whose top-level config is a
    conditional-generation config, so `AutoModelForCausalLM` does not always
    match it — `Gemma4ForConditionalGeneration` is what this repo's own PyTorch
    scripts under `quant/` use. Both expose `logits` over the text vocabulary
    for a text-only forward, which is all the oracle needs, so try the specific
    class first and fall back to the generic one.

    float32 is deliberate: the oracle spends accuracy freely so that a
    divergence can be attributed to the port rather than to the reference.
    """
    import torch

    kwargs = dict(dtype=torch.float32, device_map="cpu",
                  token=os.environ.get("HF_TOKEN"))
    errors = []
    try:
        from transformers import Gemma4ForConditionalGeneration
        return Gemma4ForConditionalGeneration.from_pretrained(path, **kwargs)
    except Exception as exc:                       # wrong class for this export
        errors.append(f"Gemma4ForConditionalGeneration: {exc}")
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    except Exception as exc:
        errors.append(f"AutoModelForCausalLM: {exc}")
    raise RuntimeError(
        "Could not load a PyTorch reference for this checkpoint:\n  "
        + "\n  ".join(errors)
    )


def reference_greedy(
    model_id: str,
    prompt_ids_batch: list[list[int]],
    max_new_tokens: int,
    eos_ids: set[int],
    local_dir: str | None = None,
    use_cache: bool = True,
) -> list[tuple[list[int], list[float]]]:
    """Greedy decode with Hugging Face `transformers` in PyTorch on CPU.

    float32 throughout: this is the oracle, so it spends accuracy freely. The
    loop is written out rather than delegated to `generate()` so that the
    per-step top1-top2 margin is available — without it a divergence cannot be
    told apart from a numerical tie.

    `use_cache=False` re-forwards the whole prefix every step. It is O(n^2) and
    slow, but it removes HF's KV cache from the oracle, which matters if the
    thing under suspicion is *this* repo's KV cache.
    """
    import torch

    model = _load_reference_model(local_dir or model_id)
    model.eval()

    out: list[tuple[list[int], list[float]]] = []
    for prompt_ids in prompt_ids_batch:
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        generated: list[int] = []
        margins: list[float] = []
        past = None
        cur = ids

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if use_cache:
                    res = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = res.past_key_values
                else:
                    res = model(input_ids=ids, use_cache=False)
                logits = res.logits[0, -1].float()

                top2 = torch.topk(logits, 2)
                nxt = int(top2.indices[0])
                margins.append(float(top2.values[0] - top2.values[1]))

                if nxt in eos_ids:
                    break
                generated.append(nxt)

                nxt_t = torch.tensor([[nxt]], dtype=torch.long)
                ids = torch.cat([ids, nxt_t], dim=1)
                cur = nxt_t if use_cache else ids

        out.append((generated, margins))

    del model
    return out


# --------------------------------------------------------------------- stage 2


def build_engine(
    model_id: str,
    max_model_len: int,
    local_dir: str | None,
    bos_token_id: int | None,
    kv_cache_dtype: str,
    quant_mode: str,
    window_kv: bool | None = None,
    dequant_at_load: bool = False,
):
    """Load the engine the way `jax_openai_server.py` loads it."""
    from jax_engine import JaxGemmaEngine

    engine = JaxGemmaEngine(
        model_id=model_id,
        kv_cache_dtype=kv_cache_dtype,
        quant_mode=quant_mode,
        max_model_len=max_model_len,
        window_kv=window_kv,
        dequant_at_load=dequant_at_load,
    )
    engine.load(local_dir=local_dir)
    engine.bos_token_id = bos_token_id
    return engine


def subject_greedy(engine, prompt_ids: list[int], max_new_tokens: int,
                   eos_ids: set[int]) -> list[int]:
    """Greedy decode through the engine's real generate path.

    `temperature=0.0` takes the `argmax` branch of `onchip_sample_tpu_v6e_jax`,
    which is deterministic and identical on every backend — so a divergence here
    is the model, never the sampler.
    """
    tokens, _stats = engine.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        eos_token_ids=sorted(eos_ids),
    )
    return [int(t) for t in tokens]


# --------------------------------------------------------------------- stage 3


def compare(reference: list[int], subject: list[int],
            margins: list[float] | None = None) -> dict[str, Any]:
    """Token-level comparison, with the reference's confidence at the break."""
    n = min(len(reference), len(subject))
    first = None
    for i in range(n):
        if reference[i] != subject[i]:
            first = i
            break
    if first is None and len(reference) != len(subject):
        # Identical prefix but one side stopped early — still a divergence, at
        # the index where the shorter sequence ran out.
        first = n

    agree = sum(1 for i in range(n) if reference[i] == subject[i])
    margin = None
    if first is not None and margins and first < len(margins):
        margin = float(margins[first])

    return {
        "matched": first is None and len(reference) == len(subject),
        "first_divergence": first,
        "divergence_margin": margin,
        "agreement": (agree / n) if n else 0.0,
    }


# ---------------------------------------------------------------------- driver


def effective_prompt_ids(tok, prompt: str, bos_token_id: int | None,
                         chat_template: bool) -> list[int]:
    """Token ids exactly as the engine will see them.

    The engine prepends `<bos>` itself when the first id is not already one
    (`JaxGemmaEngine.generate_stream`). Prepending here makes that a no-op, so
    both sides are handed byte-identical inputs while the engine still runs its
    real code path. Without this the reference and the subject would silently be
    answering different questions.
    """
    if chat_template and hasattr(tok, "apply_chat_template"):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = list(map(int, ids))
    else:
        ids = list(map(int, tok(prompt, add_special_tokens=False).input_ids))

    if bos_token_id is not None and (not ids or ids[0] != bos_token_id):
        ids = [int(bos_token_id)] + ids
    return ids


def resolve_eos_ids(tok) -> set[int]:
    eos: set[int] = set()
    raw = getattr(tok, "eos_token_id", None)
    if isinstance(raw, (list, tuple)):
        eos.update(int(x) for x in raw)
    elif raw is not None:
        eos.add(int(raw))
    # Gemma's instruction format ends a turn with <end_of_turn>, which is not
    # always the tokenizer's eos_token_id but does end generation.
    for name in ("<end_of_turn>", "<eos>"):
        tid = tok.convert_tokens_to_ids(name)
        if tid is not None and tid >= 0 and tid != getattr(tok, "unk_token_id", -1):
            eos.add(int(tid))
    return eos


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--local-dir", default=None,
                    help="Read the checkpoint from a directory instead of downloading.")
    ap.add_argument("--prompt", action="append", dest="prompts", default=None,
                    help="Repeatable. Defaults to a built-in set.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--chat-template", action="store_true",
                    help="Wrap prompts in the instruction template, as the server does.")
    ap.add_argument("--window-kv", choices=("auto", "on", "off"), default="auto",
                    help="Ring-buffer the sliding layers' KV. 'auto' leaves the "
                         "engine's own max_model_len-based rule alone.")
    ap.add_argument("--kv-cache-dtype", default="bf16")
    ap.add_argument("--quant-mode", default="fp16",
                    help="fp16 = dense path, correct for the -unquantized checkpoint.")
    ap.add_argument("--dequant-at-load", action="store_true",
                    help="Materialize packed weights to dense BF16 on the HOST at load, "
                         "before device_put, and run the dense path. Splits an in-graph "
                         "dequant defect from a dequant-math defect: the arithmetic is "
                         "identical, only where it executes changes.")
    ap.add_argument("--subject-platform", default=None,
                    help="Sets JAX_PLATFORMS for the engine (e.g. cpu, neuron, tpu).")
    ap.add_argument("--reference", default=None,
                    help="Read reference tokens from this JSON instead of running PyTorch.")
    ap.add_argument("--save-reference", default=None,
                    help="Write the computed reference tokens here for later reuse.")
    ap.add_argument("--skip-subject", action="store_true",
                    help="Only build the reference. Use on a box with torch but no accelerator.")
    ap.add_argument("--no-reference-cache", action="store_true",
                    help="Reference re-forwards the full prefix each step (slow, fewer assumptions).")
    ap.add_argument("--json", dest="json_out", default=None, help="Write the full report here.")
    args = ap.parse_args(argv)

    # Must precede any JAX import.
    if args.subject_platform:
        os.environ["JAX_PLATFORMS"] = args.subject_platform

    from transformers import AutoTokenizer

    prompts = args.prompts or DEFAULT_PROMPTS
    report = ParityReport(model_id=args.model_id, max_new_tokens=args.max_new_tokens)

    print(f"[0/3] tokenizer sanity — {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.local_dir or args.model_id,
                                        token=os.environ.get("HF_TOKEN"))
    check = tokenizer_sanity(tok, args.model_id)
    report.tokenizer_check = check
    print(f"      {check['probe']!r} -> {check['input_ids']} -> {check['decoded']!r}")
    if check["problems"]:
        for p in check["problems"]:
            print(f"      FAIL: {p}")
        report.notes.append("tokenizer check failed; nothing downstream is meaningful")
        _emit(report, args.json_out)
        return 2
    print(f"      ok (bos={check['bos_token_id']} eos={check['eos_token_id']})")

    eos_ids = resolve_eos_ids(tok)
    bos = check["bos_token_id"]
    prompt_ids = [effective_prompt_ids(tok, p, bos, args.chat_template) for p in prompts]
    for p, ids in zip(prompts, prompt_ids):
        report.results.append(PromptResult(prompt=p, prompt_token_ids=ids))

    # ---- reference
    if args.reference:
        with open(args.reference) as fh:
            saved = json.load(fh)
        report.reference_source = args.reference
        by_prompt = {r["prompt"]: r for r in saved.get("results", [])}
        mismatched = [p for p in prompts if p not in by_prompt]
        if mismatched:
            print(f"[1/3] FAIL: saved reference has no entry for {mismatched}")
            return 2
        if saved.get("model_id") != args.model_id:
            report.notes.append(
                f"reference was built from {saved.get('model_id')!r}, "
                f"subject is {args.model_id!r}"
            )
        for res in report.results:
            src = by_prompt[res.prompt]
            if src.get("prompt_token_ids") != res.prompt_token_ids:
                print(f"[1/3] FAIL: saved reference tokenized {res.prompt!r} differently — "
                      "different tokenizer files. Refusing to compare.")
                return 2
            res.reference_tokens = list(map(int, src["reference_tokens"]))
            res.reference_margins = list(map(float, src.get("reference_margins", [])))
            res.reference_text = src.get("reference_text", "")
        print(f"[1/3] reference loaded from {args.reference}")
    else:
        print(f"[1/3] reference — PyTorch float32 on CPU"
              f"{' (no kv cache)' if args.no_reference_cache else ''}")
        t0 = time.perf_counter()
        ref = reference_greedy(args.model_id, prompt_ids, args.max_new_tokens, eos_ids,
                               local_dir=args.local_dir,
                               use_cache=not args.no_reference_cache)
        report.reference_source = "transformers/pytorch/cpu/float32"
        for res, (toks, margins) in zip(report.results, ref):
            res.reference_tokens = toks
            res.reference_margins = margins
            res.reference_text = tok.decode(toks)
        print(f"      {len(prompts)} prompts in {time.perf_counter() - t0:.1f}s")

    if args.save_reference:
        _emit(report, args.save_reference)
        print(f"      reference written to {args.save_reference}")

    if args.skip_subject:
        report.passed = True
        report.notes.append("subject skipped; reference only, no parity claim")
        _emit(report, args.json_out)
        print("[done] reference built. No parity was checked.")
        return 0

    # ---- subject
    print(f"[2/3] subject — JaxGemmaEngine ({args.quant_mode}, kv={args.kv_cache_dtype})")
    import jax
    from ports.gemma4 import backend

    report.subject_platform = backend.active_platform()
    report.subject_devices = ", ".join(str(d) for d in jax.devices())
    print(f"      platform={report.subject_platform} devices=[{report.subject_devices}]")

    t0 = time.perf_counter()
    window_kv = {"auto": None, "on": True, "off": False}[args.window_kv]
    engine = build_engine(args.model_id, args.max_model_len, args.local_dir, bos,
                          args.kv_cache_dtype, args.quant_mode, window_kv,
                          args.dequant_at_load)
    print(f"      window_kv={engine.window_kv} donate_cache={engine.donate_cache} "
          f"dequant_at_load={args.dequant_at_load}")
    print(f"      loaded {engine.weight_bytes / 1e9:.2f} GB in {time.perf_counter() - t0:.1f}s")

    for res in report.results:
        t0 = time.perf_counter()
        res.subject_tokens = subject_greedy(engine, res.prompt_token_ids,
                                            args.max_new_tokens, eos_ids)
        res.subject_text = tok.decode(res.subject_tokens)
        print(f"      {res.prompt!r}: {len(res.subject_tokens)} tokens "
              f"in {time.perf_counter() - t0:.1f}s")

    # ---- compare
    print("[3/3] comparison")
    all_ok = True
    for res in report.results:
        verdict = compare(res.reference_tokens, res.subject_tokens, res.reference_margins)
        res.matched = verdict["matched"]
        res.first_divergence = verdict["first_divergence"]
        res.divergence_margin = verdict["divergence_margin"]
        res.agreement = verdict["agreement"]
        all_ok &= bool(res.matched)

        status = "PASS" if res.matched else "FAIL"
        print(f"      [{status}] {res.prompt!r}")
        if not res.matched:
            i = res.first_divergence
            print(f"             diverged at token {i} "
                  f"(agreement {res.agreement:.1%} over the common prefix)")
            if res.divergence_margin is not None:
                # A margin near zero means the reference itself was nearly tied
                # at this token, so bf16 rounding is enough to flip it. That is
                # not the same finding as a confident reference being contradicted.
                hint = ("numerically tied — bf16 rounding is sufficient to explain this"
                        if res.divergence_margin < 1e-2
                        else "reference was confident — this is a real divergence")
                print(f"             reference top1-top2 margin {res.divergence_margin:.4g} ({hint})")
            print(f"             ref: {res.reference_text!r}")
            print(f"             sub: {res.subject_text!r}")

    report.passed = all_ok
    _emit(report, args.json_out)
    print(f"\n{'PASS' if all_ok else 'FAIL'}: "
          f"{sum(1 for r in report.results if r.matched)}/{len(report.results)} prompts matched "
          f"on {report.subject_platform}")
    return 0 if all_ok else 1


def _emit(report: ParityReport, path: str | None) -> None:
    if not path:
        return
    with open(path, "w") as fh:
        json.dump(asdict(report), fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
