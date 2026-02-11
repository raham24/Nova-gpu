# Nova-GPU: GPU-Accelerated Nova Recursive SNARKs

This is a fork of [Microsoft Nova](https://github.com/microsoft/Nova) that adds **GPU-accelerated multi-scalar multiplication (MSM)** for the Pallas curve via CUDA. The GPU backend replaces the CPU MSM in Nova's Pedersen commitment engine, accelerating the most compute-intensive operation in IVC proof generation.

## What Changed From Upstream Nova

This fork introduces a `pallas-gpu` feature flag that offloads Pallas curve MSM to the GPU using a custom CUDA library ([pallas-gpu](https://github.com/raham24/pallas-gpu)). When enabled, all MSM operations on the Pallas curve are dispatched to the GPU automatically. When disabled, the original CPU codepath is used unchanged.

### Modified Files

| File | Change |
|------|--------|
| `Cargo.toml` | Added `pallas-gpu` optional dependency and feature flag |
| `src/provider/mod.rs` | Conditionally includes `pallas_gpu` module |
| `src/provider/pasta.rs` | Manual `DlogGroupExt` impl for Pallas with GPU dispatch |
| `src/provider/pallas_gpu.rs` | **New** -- GPU MSM wrapper that bridges Nova types to the CUDA library |

### How GPU Dispatch Works

In `src/provider/pasta.rs`, the `DlogGroupExt::vartime_multiscalar_mul` implementation for `pallas::Point` is conditionally compiled:

```rust
#[cfg(feature = "pallas-gpu")]
fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    super::pallas_gpu::vartime_multiscalar_mul(scalars, bases)
}

#[cfg(not(feature = "pallas-gpu"))]
fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    msm(scalars, bases)  // original CPU Pippenger
}
```

### Data Flow

1. **Rust side** (`pallas_gpu.rs`): Converts `halo2curves` field elements to standard (non-Montgomery) form via `to_repr()`, packs them as `[u64; 4]` limbs
2. **GPU side** (`pallas-gpu` crate): Receives standard-form points, converts to Montgomery form on-device, runs Pippenger MSM with CUDA kernels, converts result back to standard form
3. **Rust side**: Reconstructs `pallas::Point` from the GPU result via `from_repr()`

### The pallas-gpu CUDA Library

The GPU MSM implementation lives in a separate crate ([pallas-gpu](https://github.com/raham24/pallas-gpu)) with the following architecture:

```
pallas-gpu/
├── cuda/
│   ├── pallas_field.cuh/.cu   -- Montgomery field arithmetic (CIOS method, 8x32-bit limbs)
│   ├── pallas_curve.cuh/.cu   -- Point operations (homogeneous projective coordinates)
│   └── pallas_msm.cu          -- Pippenger MSM kernels (window=8, parallel bucket accumulation)
├── src/lib.rs                 -- Rust FFI bindings
└── build.rs                   -- CUDA compilation via nvcc
```

Key implementation details:
- **Coordinates**: Homogeneous projective (affine = X/Z, Y/Z) using EFD add-1998-cmo-2 formulas
- **Field arithmetic**: Montgomery multiplication via CIOS with 8x32-bit limbs
- **MSM algorithm**: Pippenger's with 8-bit windows, 255 buckets per window, summation by parts
- **Parallelization**: 4-phase pipeline -- shared-memory bucket accumulation (per-block spinlocks) → cross-chunk reduction → bucket reduction → window combination. Adaptive chunk sizes (4K/16K/32K) and persistent GPU memory pool. Serial fallback for < 64 points.
- **Performance**: ~2x speedup over CPU at 100K--1M points (benchmarked on RTX 4080)

## Requirements

### Standard (CPU only)
Same as upstream Nova -- Rust 1.79+.

### GPU acceleration
- **NVIDIA GPU** with Compute Capability 8.0+ (Ampere, Ada Lovelace, or newer)
- **CUDA Toolkit** 11.0+
- **Linux** (tested on Ubuntu with NVIDIA GeForce RTX 4080)

## Building and Testing

### CPU only (default, same as upstream)

```bash
cargo test --release
```

### With GPU acceleration

```bash
cargo test --features pallas-gpu --release
```

To run only the GPU MSM tests:

```bash
cargo test --features pallas-gpu --release -- test_gpu_msm --nocapture
```

### Running an example

```bash
cargo run --release --example minroot
```

## About Nova

Nova is a high-speed recursive SNARK that achieves [incrementally verifiable computation (IVC)](https://iacr.org/archive/tcc2008/49480001/49480001.pdf). It allows a prover to produce a proof of correct execution of a long-running sequential computation in an incremental fashion, where the prover's work to update the proof does not depend on the number of steps executed thus far. Nova is constructed from a simple primitive called a *folding scheme*, which reduces the task of checking two NP statements into the task of checking a single NP statement.

This repository provides `nova-snark`, a Rust library implementation of Nova over a cycle of elliptic curves. The code supports three curve cycles: (1) Pallas/Vesta, (2) BN254/Grumpkin, and (3) secp/secq.

At its core, Nova relies on a commitment scheme for vectors. The code implements two commitment schemes and evaluation arguments:
1. Pedersen commitments with IPA-based evaluation argument (supported on all three curve cycles)
2. HyperKZG commitments and evaluation argument (supported on curves with pairings, e.g., BN254)

A SNARK based on [Spartan](https://eprint.iacr.org/2019/550.pdf) is used to compress IVC proofs. There are two variants: one without preprocessing and MicroSpartan (described in the [MicroNova](https://eprint.iacr.org/2024/2099) paper) which uses preprocessing to make verifier runtime independent of circuit size.

## Supported Front-ends

1. **Bellman-style circuits** (native): See [minroot.rs](https://github.com/microsoft/Nova/blob/main/examples/minroot.rs) or [sha256.rs](https://github.com/microsoft/Nova/blob/main/benches/sha256.rs)
2. **Circom**: Via [Nova Scotia](https://github.com/nalinbhardwaj/Nova-Scotia) and [Circom Scotia](https://github.com/lurk-lab/circom-scotia)

## References

[Nova: Recursive Zero-Knowledge Arguments from Folding Schemes](https://eprint.iacr.org/2021/370) \
Abhiram Kothapalli, Srinath Setty, and Ioanna Tzialla \
CRYPTO 2022

[Revisiting the Nova Proof System on a Cycle of Curves](https://eprint.iacr.org/2023/969) \
Wilson Nguyen, Dan Boneh, and Srinath Setty \
IACR ePrint 2023/969

[HyperNova: Recursive arguments for customizable constraint systems](https://eprint.iacr.org/2023/573) \
Abhiram Kothapalli and Srinath Setty \
CRYPTO 2024

[MicroNova: Folding-based arguments with efficient (on-chain) verification](https://eprint.iacr.org/2024/2099) \
Jiaxing Zhao, Srinath Setty, Weidong Cui, and Greg Zaverucha \
IEEE S&P 2025

## Acknowledgments

- [Microsoft Nova](https://github.com/microsoft/Nova) -- the upstream project this fork is based on
- [halo2curves](https://github.com/privacy-scaling-explorations/halo2curves) -- Pallas/Vesta curve implementations
- [ICICLE](https://github.com/ingonyama-zk/icicle) -- reference for GPU cryptography patterns
