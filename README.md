# VIctory RTX (NVIDIA 16 GB / 32 GB RAM Fork)

A fork of [VIctory](https://github.com/Marvisak/VIctory) based on
[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX) with
NVIDIA Ada/Wave32, 1M descriptor-heap, lighting, and 32 GB host RAM fixes
ported upfront from the community ecosystem.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's developer
or publisher.

## The Goal

1. **Stage 1 (current):** Stable **30+ FPS** with clean graphics and adequate
   SH1 indirect lighting (no white-out RT miss rooms, no NaN black squares,
   working facial animation, clean intro video) on an NVIDIA RTX 16 GB GPU and
   32 GB system RAM.
2. **Stage 2 (planned):** Hardware-appropriate ray-traced indirect lighting
   (paired-lane Wave64 BVH builder or native DXR 1.1 on RT cores) once Stage 1
   frame pacing and visuals are solid.

## Known Initial Problems & Mitigations

Before launching, see [docs/NVIDIA-KNOWN-ISSUES.md](docs/NVIDIA-KNOWN-ISSUES.md)
for the full catalog of NVIDIA/Wave32/32 GB RAM issues documented upfront and
the pre-launch fixes already integrated here:

| Area | Fix integrated | Source |
|---|---|---|
| 1M view heap cap | `VICTORY_DYNAMIC_SLOTS` on capped heaps (`null / 2`), `Detached::release_tables()`, `settle_for_fence()` retry | `trustno1-coder/VIctory-NVIDIA` (`a9632bc`), `codec-d/cool_fox` (`f4b48ad`) |
| Wave32 TDR & RT | Refuse unsafe approximate wave64 compute (`VICTORY_WAVE_LOCAL`), skip `CS_RaytracingResolve` / `CS_ScreenProbes_ActivateRays` without BVH (`VICTORY_RT_RESOLVE`, `VICTORY_ACTIVATE_RAYS`), per-site resource guard tracing | `Rubenpannen/VIctory-RTX5070` (`b696f73`), `codec-d/cool_fox` (`8cc31ab`, `f4b48ad`) |
| Stage 1 lighting & NaNs | SH1 ambient + `VICTORY_AMBIENT_SETTLE`, `VICTORY_AMBIENT_MOST` + `clear_air` NaN guard, `VICTORY_NAN_SCRUB=1`, `VICTORY_BVH_MISS=1` | `avatarmanz/VIctory-7900XTX`, `trustno1-coder/VIctory-NVIDIA` (`77e13a9`, `24a479d`), `cool-fox-12gb` (`db556df`) |
| Faces & indirect draws | `SLOT_TABLE_STRIDES` + dynamic-slot stride mirroring (preserving `VICTORY_PORT_CBV`), opt-in `VICTORY_INDIRECT_REST` | `codec-d/cool_fox` (`f4b48ad`) |
| 32 GB RAM & cold compiles | `VICTORY_RESIDENCY=1`, 8-worker cap on <=32 GB RAM, `VICTORY_PIPELINE_WAIT_ONCE=1`, `scene_input` + 7 bake shader hashes, opt-in `VICTORY_NO_WATCHDOG=1` | `codec-d/cool_fox` (`8cc31ab`, `f4b48ad`), `trustno1-coder/VIctory-NVIDIA` (`77e13a9`), `cool-fox-12gb` (`6d28e9b`) |
| Intro video flicker | Always-copy linear 1-mip R8 video planes (`VICTORY_VIDEO_PLANES`, `VICTORY_VIDEO_HOT_FRAMES`) | `trustno1-coder/VIctory-NVIDIA` (`77e13a9`) |
| Renderer port patches | `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_JOURNAL=1`, `VICTORY_OUTPUT`, `VICTORY_RENDER_SCALE`, `VICTORY_UPSCALER` | `avatarmanz/VIctory-7900XTX` (`c9c932c`) |

## How to Build and Run

Requirements: Windows 10/11 x64, AVX2 CPU, Direct3D 12 GPU with Shader Model
6.6 and Mesh Shaders, Rust (pinned in `rust-toolchain.toml`), MSVC build tools,
and your own title dump stored **outside** this repository.

```powershell
copy victory.example.toml victory.toml   # point game_dir / cache_dir / data_dir outside this repo
cargo build --locked --release -p victory
cargo build --locked --profile diag -p victory --features diagnostics
```

Ahead-of-time shader translation (set the same shader switches you will run
with so the cache key matches):

```powershell
$env:VICTORY_D3D12_ADAPTER = 'NVIDIA'
$env:VICTORY_NAN_SCRUB = '1'
$env:VICTORY_BVH_MISS = '1'
$env:VICTORY_PORT_CBV = '1'
target\release\victory.exe shaders <path-to-game>\shaders.rpf
```

Standard upstream launcher: `scripts\play.ps1 -Exe <path-to-exe>` (pass `-Safe`
for conservative settings, `-Build release` or `-Build diag`, `-More @{...}` for
overrides). Machine-specific presets and notes belong in gitignored `local/` and
`victory.toml`.

Upstream documentation and switch reference:
- [docs/NVIDIA-KNOWN-ISSUES.md](docs/NVIDIA-KNOWN-ISSUES.md)
- [docs/findings.md](docs/findings.md)
- [docs/README-fxceless.md](docs/README-fxceless.md)
- [docs/replay.md](docs/replay.md)
- [docs/asset-policy.md](docs/asset-policy.md)

## Credits

This fork stands on a chain of community work; commit provenance is preserved in
git history:

- **[Marvisak/VIctory](https://github.com/Marvisak/VIctory)**: original loader,
  HLE runtime, Direct3D 12 translation layer, and shader recompiler.
- **[fxceless/victory-fork](https://github.com/fxceless/victory-fork)**:
  community fork and pull requests (including Wave32 and facial integer view
  fixes).
- **[avatarmanz/VIctory-7900XTX](https://github.com/avatarmanz/VIctory-7900XTX)**
  (`0dec5dc`): base repository for this fork (renderer port patches, SH1
  ambient adaptation, Series X resolution/render-scale levers, `VICTORY_THIN` /
  async submit / copy-queue architecture).
- **[codec-d/cool_fox](https://github.com/codec-d/cool_fox)** (`8cc31ab`,
  `f4b48ad`): NVIDIA Wave32 fixes, `SLOT_TABLE_STRIDES`,
  `Detached::release_tables()`, `residency.rs` 32 GB RAM budgets, RT resolve
  skipping, bake shader list, and `TablePlan`.
- **[trustno1-coder/VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA)**
  (`a9632bc`, `24a479d`, `77e13a9`): capped-heap `VICTORY_DYNAMIC_SLOTS`,
  `VICTORY_NAN_SCRUB`, `VICTORY_BVH_MISS`, `VICTORY_AMBIENT_MOST` + `clear_air`
  NaN guard, `VICTORY_PIPELINE_WAIT_ONCE`, and intro video flicker fix.
- **[Rubenpannen/VIctory-RTX5070](https://github.com/Rubenpannen/VIctory-RTX5070)**
  (`b696f73`): strict default `VICTORY_WAVE_LOCAL` refusing TDR-inducing
  approximate wave64 compute shaders.
- **[miaohaixiang1-netizen/cool-fox-12gb](https://github.com/miaohaixiang1-netizen/cool-fox-12gb)**
  (`db556df`, `6d28e9b`) and **ZigZagBoi1/ScarlettPort**: NaN scrubbing and
  opt-in byte-checked `VICTORY_NO_WATCHDOG=1` hook.
- **DexNom/VIctory**, **Belobug/VIctory-Belobug-AMD**,
  **FQY55kai/ZJOU-VICTORY**, **myokex/victory-amd**, **Geckomacabre**, and
  **cxy0072019-sketch**: upstream renderer, ambient light, guard, and readback
  contributions carried via `VIctory-7900XTX`.

## License

GPL-3.0-or-later, as VIctory. See [LICENSE](LICENSE) and
[docs/asset-policy.md](docs/asset-policy.md).
