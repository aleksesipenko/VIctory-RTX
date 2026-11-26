# VIctory RTX (NVIDIA Ada / Ampere / Blackwell Fork)

A performance- and visuals-focused NVIDIA RTX fork of [VIctory](https://github.com/Marvisak/VIctory)
built on top of [avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX),
targeting **16 GB VRAM NVIDIA GPUs and 32 GB system RAM** with **DLSS 4 / 4.5 Super Resolution**,
opt-in **Hardware DXR 1.1 Ray Tracing** (SM 6.6 `RayQuery` on RT Cores), and verified
community VRAM, frame-pacing, visual, and stability patches.

**This repository contains no game files, code, or assets.** You must provide your own
copy of the title outside this repository. This project is not affiliated with or endorsed
by the game's developer or publisher.

---

## Highlights & Current Status

| Milestone | Status | Summary |
|---|---|---|
| **Stage 1: NVIDIA Wave32 / 1M Heap / 32 GB RAM** | Verified live | Clean visuals, working facial animation, SH1 indirect lighting + reflection clamp, `0` TDRs |
| **DLSS 4 / 4.5 Super Resolution (`--features dlss`)** | Verified live | Intercepts `ffxCreateContext`/`DestroyContext`/`Query` for arbitrary render/output scaling (1080p / 2K / 4K, Preset `K` & `M`, zero framegen) |
| **Phase 0–1: DXIL SM 6.6 `RayQuery` Emitter & Ada Oracle** | Verified on HW | `f64` CPU oracle + SW BVH + Ada DXR 3-way parity on 65,536 rays; **`3.75×` Ada RT-Core speedup** (`32.77 µs` → `8.70 µs`), `0` D3D12 Debug Layer errors |
| **Phase 2: Hardened DXR 1.1 Sidecar (`XboxBvhBuilder.dll`)** | Verified on HW | Submission-ordered BLAS/TLAS/refit/compaction (`VICTORY_THIN=1` safe), 7-export ABI verification, `12.5 GiB` VRAM ceiling, `0` Debug Layer errors |
| **Phase 2.5–2.6: Community Optimization & Stability Pack** | Integrated | 4 KiB small-texture placement, 512 MiB scratch reuse, `LOOSE.read()` lock, async texback, `TEXTURE_NO_WAIT=2`, Intel `CLZERO` guard fault, 2K UI `display_size()`, thin LTO |
| **Phase 3: DXR 1.1 `RayQuery` Lifter & `#854` Paired Wave64** | Merged (`opt-in`) | All 64 corpus RT shaders pass `dxil.dll` + Ada PSO (`0` Debug Layer errors); paired Wave64 hardware-verified on 64/64 & 45/45 lanes; expanding conservative `Bvh64` admission coverage before live DXR default |

### Reference Hardware & Verified Baseline

- **Test rig:** Intel Core i5-14600K, **NVIDIA GeForce RTX 4070 Ti SUPER 16 GB** (Ada `sm_89`, Wave32, DXR 1.1), 32 GB RAM, Windows 11 x64.
- **2K DLSS 4.5 Preset M (`1920×1080 → 2560×1440`, RT-off baseline, pre-Phase 2.5/2.6):**
  `21.8 avg / 30.3 max FPS` | `10.09 GB VRAM` | `23.81 GB Private RAM` | `0 TDRs` *(fresh Run 24/25 measurements post-Phase 2.5–3 in progress)*.
- **Ada RT-Core microbenchmark (Phase 1, 65,536 rays):**
  `32.768 µs` software BVH → `8.704 µs` hardware DXR 1.1 (**`3.751×` speedup**, 100% 3-way parity, `0` D3D12 Debug Layer errors).

---

## Integrated Features & Fixes

### Upscaling & Display
- **NVIDIA DLSS 4 / 4.5 (`--features dlss`):** Set `VICTORY_UPSCALER=7`, `VICTORY_UPSCALE=dlss`, `VICTORY_DLSS_DIR=<sdk-bin>`, `VICTORY_DLSS_PRESET=M` (or `K`), `VICTORY_OUTPUT=2560x1440`, `VICTORY_FFX_SCALE=0.75`.
- **2K / 1080p UI Layout Fix (`dxgi.rs`):** `display_size()` falls back to `VICTORY_OUTPUT` when `VICTORY_DISPLAY_MODE` is unset so Settings menus do not clip at 4K layout (`Rubenpannen/VIctory-RTX5070 9f2fd8c`).

### Hardware DXR 1.1 Ray Tracing (Opt-In)
- **Sidecar (`crates/victory-gpu/src/dxr.rs`, `crates/victory-hle/src/modules/bvh_builder.rs`):** Intercepts `XboxBvhBuilder.dll` batched builds, compactions, and rehydrations; maintains host BLAS/TLAS on `DIRECT` with strict VRAM capping (`VICTORY_DXR_VRAM_CAP_MB=12800`), `NaN` active↔inactive refit safety, `missing_blas` and `empty_tlas_fallbacks` telemetry, and `D3D12_OPTIONS19` 1M-heap probing.
- **Shader Lifter (`crates/victory-shader/src/emit/rayquery_lift.rs`, `rayquery_lower.rs`):** Lifts XDXR GFX10 BVH traversal loops (`Bvh64` & `Bvh32SharedTlas`) into SM 6.6 `RayQuery` with `#854` paired-lane Wave64 isolation (`VICTORY_NATIVE_RT=1`, optional `VICTORY_NATIVE_RT_ANYHIT=1`, `VICTORY_NATIVE_RT_FILTER=<hash>`, `VICTORY_NATIVE_RT_TWO_PASS=1`, `VICTORY_NATIVE_RT_FBT_TWOPASS=1`).
- **RT-Off Fallback:** Defaults to `0` overhead (`VICTORY_NATIVE_RT` unset) with bit-identical RT-off shader output.

### VRAM, RAM, Frame Pacing & Stability
| Area | What's Integrated | Provenance (GPL-3.0) |
|---|---|---|
| **VRAM & RAM Reduction** | 4 KiB small-texture placement default on non-AMD GPUs (`-3+ GB` VRAM/RAM); idle/pressured shadow snapshot pool trimming (`let_go_idle_snapshots`); `VICTORY_RESIDENCY=1` 32 GB RAM budgets & 8-worker cap | `TaynanGT/Vice-Tay` (`7d62e53`, `#41`), `codec-d/cool_fox` (`8cc31ab`, `f4b48ad`) |
| **Frame Pacing & CPU Overhead** | `SCRATCH_FIRST` 4 MiB spare reuse + 512 MiB pool cap; `LOOSE.read()` lock in `shadow.rs`; async `VICTORY_TEXBACK_ASYNC=1` writer + pooled readback buffers; `VICTORY_BURY_ASIDE=1` off-thread COM drop; `VICTORY_WIDE_PRIORITY=1` helper priority; `VICTORY_TEXTURE_NO_WAIT=2` anti-stall streaming; cached `HostHeap::gpu` start pointer; `lto = "thin"` release profile | `fxceless/victory-fork` (`#1224`, `#1367`, `#1359`), `Peppacodes/playablegames-peppa` (`#6`), `kitty-core/Kitty6` (`42046f5`), `trustno1-coder` (`#7`), `voidopcode/vilocity` (`6804c16`) |
| **Visual & D3D12 Correctness** | `ExecuteIndirect` reflection clamp (`CS_TemporalResolve2_Opaque` / `CS_TemporalResolve1_Alpha`, fixes white eyes/glass); SH1 ambient (`VICTORY_AMBIENT`, `VICTORY_AMBIENT_SETTLE`, `VICTORY_AMBIENT_MOST`, `VICTORY_AMBIENT_AIR`, `VICTORY_AMBIENT_RT` band handoff); `VICTORY_NAN_SCRUB=1`; separate `card_compare` submit & strict SRV/UAV depth match (fixes D3D12 errors 538 & 646); HUD `COPY_SOURCE -> COPY_DEST` barrier; `SLOT_TABLE_STRIDES` facial fix; intro video flicker fix | `avatarmanz/VIctory-7900XTX` (`77137b9`), `kitty-core/Kitty6` (`1f95b3a`), `trustno1-coder/VIctory-NVIDIA` (`#1`, `#2`, `77e13a9`, `24a479d`), `Belobug`, `codec-d/cool_fox` (`f4b48ad`) |
| **NVIDIA 1M Heap & Intel CPU Safety** | `D3D12_OPTIONS19` unconditional cap query; `VICTORY_DYNAMIC_SLOTS` (`null / 2`); monotonic `HostHeap::id` guard; `catch_unwind` pipeline worker; `submit_split` slot return; Intel `CLZERO` `zero_through_fault` guard & VEH `MEM_FREE` safety; strict `VICTORY_WAVE_LOCAL` default | `fxceless/victory-fork` (`#1380`, `#1324`, `#1353`, `#1320`), `trustno1-coder/VIctory-NVIDIA` (`a9632bc`), `Rubenpannen/VIctory-RTX5070` (`b696f73`) |

For the full catalog of NVIDIA/Wave32/32 GB RAM notes, see [docs/NVIDIA-KNOWN-ISSUES.md](docs/NVIDIA-KNOWN-ISSUES.md).

---

## How to Build and Run

### Requirements
- Windows 10/11 x64, AVX2 CPU, Direct3D 12 GPU with **Shader Model 6.6 + Mesh Shaders** (DXR 1.1 required for `VICTORY_NATIVE_RT=1`).
- Rust toolchain (pinned in [rust-toolchain.toml](rust-toolchain.toml)) and MSVC Build Tools.
- Optional: NVIDIA DLSS SDK (for `--features dlss`, pointed to by `VICTORY_DLSS_SDK`).
- Your own title files stored **outside** this repository.

### Build
```powershell
copy victory.example.toml victory.toml   # set game_dir / cache_dir / data_dir outside this repo

# Standard release build
cargo build --locked --release -p victory

# Release build with NVIDIA DLSS 4 / 4.5 support
$env:VICTORY_DLSS_SDK = '<path-to-dlss-sdk>'
cargo build --locked --release -p victory --features dlss

# Diagnostic build (frame timing telemetry + DLSS)
cargo build --locked --profile diag -p victory --features diagnostics,dlss
```

### Precompile Shaders (Recommended)
Set the exact shader-affecting switches you will run with so the `VSH2` cache key matches:

```powershell
$env:VICTORY_D3D12_ADAPTER = 'NVIDIA'
$env:VICTORY_NAN_SCRUB     = '1'
$env:VICTORY_BVH_MISS      = '1'
target\release\victory.exe shaders <path-to-game>\shaders.rpf
```

### Launch
Use the upstream launcher [scripts/play.ps1](scripts/play.ps1) (supports `-Safe`, `-Build release|diag`, `-More @{...}`), or keep machine-specific presets inside gitignored `local/` and `victory.toml`.

Recommended baseline switches for **NVIDIA RTX 16 GB / 32 GB RAM (2K DLSS 4.5, RT-off)**:
```powershell
$env:VICTORY_D3D12_ADAPTER       = 'NVIDIA'
$env:VICTORY_RESIDENCY           = '1'
$env:VICTORY_DYNAMIC_SLOTS       = '524288'
$env:VICTORY_DSTORAGE_STAGING_MB = '256'
$env:VICTORY_OUTPUT              = '2560x1440'
$env:VICTORY_UPSCALER            = '7'
$env:VICTORY_UPSCALE             = 'dlss'
$env:VICTORY_FFX_SCALE           = '0.75'
$env:VICTORY_DLSS_DIR            = '<path-to-dlss-sdk>\bin'
$env:VICTORY_DLSS_PRESET         = 'M'
$env:VICTORY_AMBIENT             = '0.8'
$env:VICTORY_AMBIENT_SETTLE      = '250,3'
$env:VICTORY_AMBIENT_AIR         = '1.5'
$env:VICTORY_AMBIENT_MOST        = '2'
$env:VICTORY_NAN_SCRUB           = '1'
$env:VICTORY_BVH_MISS            = '1'
$env:VICTORY_TEXTURE_NO_WAIT     = '2'
$env:VICTORY_TEXBACK_ASYNC       = '1'
```

Opt-in **Hardware DXR 1.1** switches (experimental, requires DXR 1.1 GPU):
```powershell
$env:VICTORY_NATIVE_RT           = '1'
$env:VICTORY_RT_BACKEND          = 'dxr'
$env:VICTORY_DXR_VRAM_CAP_MB     = '12800'
```

---

## Documentation
- [docs/NVIDIA-KNOWN-ISSUES.md](docs/NVIDIA-KNOWN-ISSUES.md)
- [docs/findings.md](docs/findings.md)
- [docs/README-fxceless.md](docs/README-fxceless.md)
- [docs/replay.md](docs/replay.md)
- [docs/asset-policy.md](docs/asset-policy.md)

---

## Credits & Provenance

This fork builds on a deep chain of GPL-3.0 community engineering; per-commit provenance is preserved in git history and source headers:

- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)**: original loader, HLE runtime, Direct3D 12 translation layer, and GFX10-to-DXIL shader recompiler.
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)**: base upstream repository (renderer port patches, SH1 ambient adaptation, Series X resolution/scaling levers, `VICTORY_THIN` / async submit / copy-queue architecture).
- **[maren-halv/VIctory-DXR](https://github.com/maren-halv/VIctory-DXR)**: DXR 1.1 `XboxBvhBuilder.dll` ABI & instance patching (`e1740b3`) and SM 6.6 `RayQuery` traversal lifter architecture (`cd87189`), hardened and adapted here for NVIDIA Ada/Wave32 and paired Wave64 safety.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**: community fork & PRs (`#1224`, `#1320`, `#1324`, `#1353`, `#1359`, `#1367`, `#1380` — `VICTORY_BURY_ASIDE`, scratch reuse, readback pool, Intel `CLZERO` guard fault, Wave32 & facial view fixes, heap/pipeline hardening).
- **[Peppacodes/playablegames-peppa](https://github.com/Peppacodes/playablegames-peppa)** (`#6` / `6f713ab`): `VICTORY_WIDE_PRIORITY` worker priority and NVIDIA CPU prep findings.
- **[TaynanGT/Vice-Tay](https://github.com/TaynanGT/Vice-Tay)** (`7d62e53`, `#41`): 4 KiB small-texture placement on non-AMD GPUs and idle snapshot pool trimming.
- **[kitty-core/Kitty6](https://github.com/kitty-core/Kitty6)** (`42046f5`, `1f95b3a`): `LOOSE.read()` shadow lock and `ExecuteIndirect` reflection clamp.
- **[voidopcode/vilocity](https://github.com/voidopcode/vilocity)** (`6804c16`): `VICTORY_TEXTURE_NO_WAIT=2` non-blocking streaming texture upload.
- **[codec-d/cool_fox](https://github.com/codec-d/cool_fox)** (`8cc31ab`, `f4b48ad`): NVIDIA Wave32 fixes, `SLOT_TABLE_STRIDES`, `Detached::release_tables()`, 32 GB RAM residency budgets, RT resolve skipping, and bake shader list.
- **[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)** (`a9632bc`, `24a479d`, `77e13a9`, `#1`, `#2`, `#7`): capped-heap `VICTORY_DYNAMIC_SLOTS`, `VICTORY_NAN_SCRUB`, `VICTORY_BVH_MISS`, `VICTORY_AMBIENT_MOST`, async texback writer, D3D12 errors 538/646 fixes, and intro video flicker fix.
- **[Rubenpannen/VIctory-RTX5070](https://github.com/Rubenpannen/VIctory-RTX5070)** (`b696f73`, `9f2fd8c`): strict `VICTORY_WAVE_LOCAL` default and 2K/1080p `display_size()` fallback.
- **[miaohaixiang1-netizen/cool-fox-12gb](https://github.com/miaohaixiang1-netizen/cool-fox-12gb)** (`db556df`, `6d28e9b`), **Belobug/VIctory-Belobug-AMD**, **ZigZagBoi1/ScarlettPort**, **DexNom/VIctory**, **FQY55kai/ZJOU-VICTORY**, **myokex/victory-amd**, **Geckomacabre**, and **cxy0072019-sketch**: upstream renderer, ambient, HUD barrier, watchdog, guard, and readback contributions.

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE) and [docs/asset-policy.md](docs/asset-policy.md).
