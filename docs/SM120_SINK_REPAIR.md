# SM12x learned-sink repair

Base recipe: `8233ee9effa45ff6ea133fd75432969dfea4f670` (0.6.0).
Runtime dependencies: vLLM 0.32.0, vllm-ernie
`ba322a5634f91d3e6f96ae711df72ddaf3c38b6e`, FlashInfer 0.7.0.post1.

## Scope

`spark_ernie_runtime/sm120_attention.py` supplies the missing bindings. The
normal overlay installer calls it before the other ERNIE patches. No installed
vLLM, plugin, or FlashInfer source file needs an edit.

- SM12x HCA uses the real `DeepseekV4FlashInferSM120Attention` with ERNIE's
  existing gate and VHA mixin. The vLLM implementation owns packed FP8 pages,
  compressed two-row pages, sparse metadata, prefill, decode, and sink delivery.
- SM12x DSA uses a subclass of the real FlashInfer sparse implementation. It
  retains the learned sink Parameter and passes it to the actual GPU call.
- H=64 reuses the recipe's calibrated DSA helper. H=32 uses FlashInfer's public
  planner with `sinks=`. The helper's forced `PREFILL_SWAPAB` variant has no
  H=32 specialization. A real TP=2-shape test exposed this failure.
- The existing ERNIE selector and constructor resolve their backend globals at
  call time. The overlay changes those globals only on a real SM12x device.
  Validation still uses `has_sink=True` and the actual device capability.
- Non-SM12x selection stays unchanged. This hardware run does not validate
  SM90/SM100 execution.

## Numerical contract

For each query/head, the reference computes `softmax([QK^T * scale, sink])`.
The sink contributes zero to the numerator and `exp(sink)` to the denominator.
The test does not add the sink to every key logit. It independently decodes the
packed test data and computes FP32 PyTorch softmax and matmul.

`tests/test_sm120_attention.py` uses real vLLM classes, distributed state,
metadata, GPU kernels, BF16 queries, and packed FP8 caches. It covers H=32/64,
decode and prefill token counts, invalid sparse entries, variable context sizes,
compressed two-row HCA pages with padded strides, and nonzero learned sinks.
It changes the Parameter after construction and checks an explicit no-sink
negative control. No mocks or test doubles enter the attention path.

Plain BF16 KV is not an SM120 sparse-MLA format in this dependency build. The
validated combination is BF16 queries plus packed FP8 KV, not BF16 KV.

## Isolated build headers

If `Python.h` is absent, do not upgrade the system Python for this repair.
Extract the matching official ARM64 header package into the ERNIE venv:

```sh
apt-get download libpython3.12-dev=<matching-package-version>
dpkg-deb -x libpython3.12-dev_<matching-package-version>_arm64.deb \
  "$HOME/.venv-ernie/ernie-build-headers"
export CPATH="$HOME/.venv-ernie/ernie-build-headers/usr/include/python3.12:$HOME/.venv-ernie/ernie-build-headers/usr/include"
```

The second path must be `usr/include`, not its architecture subdirectory. The
Ubuntu `pyconfig.h` includes `aarch64-linux-gnu/python3.12/pyconfig.h` relative
to that path. Preserve the old ERNIE user config before adding the absolute
`CPATH` value to `extra_env`. Keep both hosts at the same paths.

## Verification commands

Run the focused suite before any full model load, on each host:

```sh
python -m pytest tests/test_sm120_attention.py -q
python -m pytest tests kernels -q
python -m mutmut run --max-children 1
python -m mutmut results
```

The mutmut configuration limits mutation to the new adapter. It uses a clean
forkserver without warmup imports because CUDA is not fork-safe. Run mutations
serially on one GPU. Do not run them beside a serving process. Record survivors
and hardware limits rather than claiming an adjusted or perfect score.

The verified run killed **65 of 75 mutants (86.7%)**. Ten survivors remain:

- `_run_mqa_kernel` mutants 1 and 3 change the H=64 dispatch predicate or route
  H=64 through FlashInfer. Both paths pass the numerical contract. The calibrated
  helper remains selected for its intended SM121 H=64 execution path.
- `_run_mqa_kernel` mutants 26, 30, 31, 46, 48, and 50 omit or relax optional
  FlashInfer arguments: `qk_nope_head_dim`, `max_seq_len`, `out`, `bmm2_scale`,
  and `kv_scale_format`. FlashInfer accepts its defaults or infers values for the
  tested packed-FP8 shapes. The adapter keeps explicit values to preserve the
  intended execution contract.
- `install` mutant 2 changes the `None` capability guard to `and`. It is equivalent
  on the real SM12.1 device because the capability is not `None`.
- `install` mutant 6 returns `True` on non-SM12x devices. That branch is not
  exercised by this SM12.1-only hardware run.

No survivor removes the learned-sink tensor, changes its values, or bypasses the
real SM12x backend validation.

Before a launch, check memory and compute processes on both nodes. Keep the
checkpoint snapshot, topology, automatic RoCE GIDs, and port unchanged. Launch
both real `spark-ernie serve --profile 2x-1m-fast --rank <rank>` commands. Preserve
their logs and native exit codes if they exit. An active rank has no success
exit code. Require actual endpoint completions, streaming, a tool call, and the
supplied `spark-ernie test` before calling the deployment complete.

Keep validation logs and machine-specific evidence outside the source tree.
