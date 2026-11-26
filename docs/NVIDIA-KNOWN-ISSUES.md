# Known Initial Issues & NVIDIA Architecture Notes

This document records the known engineering problems, hardware constraints, and
initial mitigations carried in this fork **before first launch** on an NVIDIA
Ada Lovelace GPU (16 GB VRAM, Wave32, SM 6.6 + Mesh Shaders) with 32 GB host RAM.

It follows the upstream fork's practice of stating clearly what is solved, what
is worked around, and what remains open. Machine-local paths, launcher presets,
and run notes live outside git in `victory.toml` and `local/`.

---

## 1. Descriptor Heap Cap on NVIDIA (1,000,000 vs 1,245,184 views)

- **Problem:** The title requests a shader-visible CBV/SRV/UAV descriptor heap
  of 1,245,184 views. AMD RDNA cards grant that size, whereas NVIDIA D3D12
  drivers cap shader-visible view heaps at 1,000,000 (`PORTABLE_VIEWS` tier).
  With the default 262,144 (`DYNAMIC`) dynamic table slots, heavy world scenes
  exhaust the pool and drop 1,100-4,800 draws per frame.
- **Mitigation ported upfront:**
  - `descriptor/host.rs`: `dynamic_slots()` honors `VICTORY_DYNAMIC_SLOTS` on
    capped heaps (`null < PORTABLE_VIEWS`), allowing up to `null / 2` (~500,000)
    dynamic slots by borrowing from the title's kept-in-place range
    (`trustno1-coder/VIctory-NVIDIA` `a9632bc`).
  - `list.rs` & `host.rs`: `Detached::release_tables()` returns dynamic
    descriptor slots as soon as a submission lands rather than waiting for
    command-list reset (`codec-d/cool_fox` `f4b48ad`).
  - `draw.rs`: `host.dynamic(...)` retries once after `settle_for_fence()` if
    in-flight lists are holding the last slots (`codec-d/cool_fox` `f4b48ad`).
- **Expected setting:** `VICTORY_DYNAMIC_SLOTS=524288` (clamped cleanly to
  `null / 2` on a 1,000,000-view heap).

---

## 2. Wave32 (NVIDIA) vs Wave64 (Xbox RDNA2) & GPU Hangs (TDR)

- **Problem:** Xbox Series X|S runs RDNA2 shaders in Wave64. NVIDIA Ada executes
  compute and graphics waves in Wave32. Two specific hazards follow:
  1. **Approximate per-half wave64 compute causes GPU TDR:** Lowering
     `wave64` compute shaders that use cross-half wave-local ops approximately
     (such as `2561af9577c47289`) causes a ~15 s `DXGI_ERROR_DEVICE_HUNG` on
     NVIDIA RTX cards.
  2. **Console BVH ray-tracing builder does not run on Wave32 yet:** The title's
     53 BVH build compute shaders rely on Wave64 semantics and stay disabled
     (`VICTORY_BVH_BUILDER` off) until Stage 2 (paired-lane Wave64 compute or
     native DXR 1.1).
  3. **Resource tracing under Wave32:** Tracing descriptor guards at block
     granularity missed `CS_AlphTransmissionPrefilterMip`.
- **Mitigations ported upfront:**
  - `victory-shader/src/emit/mod.rs`: refuses unsafe approximate per-half
    wave64 compute by default (`VICTORY_WAVE_LOCAL=any` opts back in for
    testing; `Rubenpannen/VIctory-RTX5070` `b696f73`).
  - `victory-shader/src/resource.rs`: tracks active guards per `(block, inst)`
    site (`codec-d/cool_fox` `f4b48ad`).
  - `victory-gpu/src/pipeline/make.rs`: skips `CS_RaytracingResolve` (~1,000
    shaders) and `CS_ScreenProbes_ActivateRays` when `VICTORY_BVH_BUILDER` is
    off (overridable with `VICTORY_RT_RESOLVE=1` / `VICTORY_ACTIVATE_RAYS=1`),
    preventing cold compile stalls and empty-BVH miss artifacts
    (`codec-d/cool_fox` `8cc31ab`).

---

## 3. Lighting & Shading Without Live BVH Ray Tracing (Stage 1)

- **Problem:** With the console BVH builder disabled on Wave32, ray-tracing
  queries return misses. Unmitigated, this causes three visual defects:
  - dark or under-lit interiors and slow exposure recovery when crossing
    doorways;
  - white/blown-out surfaces or shimmering highlights from missed ray hits;
  - black square artifacts / NaN explosions propagating through post-processing
    and volumetric scattering (`clear_air`).
- **Stage 1 mitigations ported upfront:**
  - **Screen-space SH1 indirect light + temporal adaptation** (inherited from
    `avatarmanz/VIctory-7900XTX`): `VICTORY_AMBIENT=0.8`,
    `VICTORY_AMBIENT_SETTLE=250,3`, `VICTORY_AMBIENT_AIR=1.5`.
  - **Sky-scatter cap & NaN guard in `ambient.rs`:** `VICTORY_AMBIENT_MOST=2`
    caps air/scatter blow-out and `clear_air` zeroes NaN/Inf cells
    (`trustno1-coder/VIctory-NVIDIA` `77e13a9`).
  - **Shader-level NaN/Inf scrubbing (`VICTORY_NAN_SCRUB=1`):** scrubs
    non-finite values in `textureStore.f32` and pixel `Target::Color(_)` outputs
    (`trustno1-coder/VIctory-NVIDIA` `77e13a9` / `cool-fox-12gb` `db556df`).
    Keyed into `shader_cache.rs`.
  - **Explicit BVH miss lowering (`VICTORY_BVH_MISS=1`):** makes ray-BVH
    intersection intrinsics return a clean miss (`trustno1-coder/VIctory-NVIDIA`
    `24a479d`). Keyed into `shader_cache.rs`.
- **Stage 2 (future work, not yet enabled):** Port paired-lane Wave64 compute
  for the console BVH builder or replace the title's BVH queries with native
  DXR 1.1 hardware ray tracing on Ada RT cores.

---

## 4. Facial Animation (`d16` Integer Views) & Indirect Draw Safety

- **Problem:** The facial deformation network reads 16-bit integer typed buffer
  views through descriptor tables. Without per-slot stride/kind metadata for
  dynamic table slots, NVIDIA reads them as zeroed floats and faces freeze.
  Separately, `VICTORY_INDIRECT_REST` caused instability when left on by
  default on Windows NVIDIA drivers.
- **Mitigations ported upfront:**
  - `SLOT_TABLE_STRIDES` root SRV (`emit/shader.rs`, `emit/lower.rs`,
    `pipeline/host.rs`, `draw.rs`) paired with dynamic-slot stride writes in
    `descriptor/host.rs` (`make_at`, `make_listed`, `make_null`, `raw_at`,
    `raw_uav_at`, `constant_at`, `constant_raw_at`) while preserving
    `write_view_address` for `VICTORY_PORT_CBV=1` (`codec-d/cool_fox`
    `f4b48ad`).
  - `indirect.rs`: `VICTORY_INDIRECT_REST` changed to opt-in (`=1`), matching
    `codec-d/cool_fox` `f4b48ad`.

---

## 5. 32 GB Host RAM & Commit Headroom

- **Problem:** Upstream `VIctory-7900XTX` targets 64 GB RAM and can consume
  35-46 GB of private commit during world streaming, risking OOM aborts or
  severe paging stutter on 32 GB PCs.
- **Mitigations active:**
  - `VICTORY_RESIDENCY=1` (`residency.rs`): monitors available RAM and Windows
    commit headroom, throttling speculative texture prefetch (64 MiB -> 16 MiB)
    and blocking speculative staging growth when commit drops below 1 GiB.
  - `resource.rs` & `descriptor.rs` (`avatarmanz` PR #206): evicts idle host
    textures after 5 s (instead of 10 min) while `residency::pressured()` is
    true, zeroes stack-junk padding words in placed-view descriptions, and keys
    `shape_index` by hash to prevent unbounded shape-table growth.
  - `descriptor/host.rs`, `pool.rs`, `texture.rs` (`avatarmanz` PR #237):
    replaces linear `Vec<bool>` dynamic-slot scans with a 64-bit word bitmask
    (`Taken`) and guards aliasing barriers with `pool::pooled(texture)` so
    committed/stand-in textures do not trigger `E_INVALIDARG`.
  - `fence.rs` (`codec-d/cool_fox`): uses `WaitOnAddress` / `WakeByAddressAll`
    for blocking fence waits instead of returning `E_NOTIMPL`.
  - `pipeline/make.rs`: caps automatic pipeline worker threads to 8 on hosts
    with <= 32 GB RAM (`worker_count_for` + `residency::limited_host_ram()`),
    avoiding parallel NVIDIA driver compiler RAM spikes (`codec-d/cool_fox`
    `8cc31ab`).
  - `VICTORY_PORT_CONSTANTS=1`, `VICTORY_PORT_DESCRIPTORS=1`,
    `VICTORY_PORT_CBV=1`, `VICTORY_PORT_GUARD=1`, `VICTORY_PORT_JOURNAL=1`
    (from `avatarmanz/VIctory-7900XTX` `c9c932c`): cut per-frame CPU copies,
    mirroring, guard lock contention, and journal lock contention.
  - `VICTORY_CARD_BUFFERS_MIN_KB=64` + `VICTORY_CARD_BUFFERS_MB=2048`: offloads
    shader-written buffer comparisons from CPU to GPU (+43% avg / +58% median
    FPS measured on RTX 4070 Ti SUPER 16 GB).
  - **Warning on `VICTORY_LEVELS_ON_DEMAND=1`:** On NVIDIA + Series X, enabling
    tiled on-demand mip uploads increased VRAM (`9.6 -> 14.2 GB`), private RAM
    (`23.1 -> 29.2 GB`), and `heap buffers` CPU time (`1.1 -> 24.5 ms/frame`).
    Leave `VICTORY_LEVELS_ON_DEMAND` off on Series X + NVIDIA.

---

## 6. Cold Shader/Pipeline Compilation & Title 30 s Watchdog

- **Problem:** The first world frame requests over 1,200 host pipelines. On
  NVIDIA, a single complex pipeline compile can exceed the per-frame wait budget
  and re-stall every subsequent frame, or trigger the title's 30-second GPU/main
  thread watchdog fault.
- **Mitigations active:**
  - Pre-translate archives ahead of time with `victory.exe shaders` using the
    exact same shader-affecting switches (`VICTORY_NAN_SCRUB=1`,
    `VICTORY_BVH_MISS=1`, `VICTORY_PORT_CBV=1`, `VICTORY_D3D12_ADAPTER=NVIDIA`).
  - `pipeline/make.rs`: `VICTORY_PIPELINE_WAIT_ONCE=1` (on by default) caps how
    long the game waits for any single slow pipeline across the run
    (`trustno1-coder/VIctory-NVIDIA` `77e13a9`); `scene_input` prioritizes
    first-scene lighting and `BAKE_PIXEL_SHADERS` (`codec-d/cool_fox` `8cc31ab`
    & `f4b48ad`).
  - `victory-loader/src/watchdog.rs`: opt-in `VICTORY_NO_WATCHDOG=1` hook for
    build `1.0.2.1` (byte-checked at `+0x2215c27` and `+0x23cf683`, ported from
    `ZigZagBoi1/ScarlettPort` / `cool-fox-12gb` `6d28e9b`).

---

## 7. Console Profile & Resolution Caveats on NVIDIA

- **Series S 1080p breakpoint:** Running the Series S profile at native 1080p
  hits an early title breakpoint on NVIDIA; use Series S at its default lower
  render scale or prefer Series X with `VICTORY_OUTPUT=1920x1080`,
  `VICTORY_RENDER_SCALE=0.67` (1280x720 internal), and `VICTORY_UPSCALER=6`
  (the title's own temporal upscaler).
- **FidelityFX render scale warning:** `VICTORY_FFX_SCALE=0.5` on Series S can
  trigger a divide-by-zero in the title's jitter phase calculation; prefer
  `VICTORY_UPSCALER=6` + `VICTORY_RENDER_SCALE`.

---

## 8. Intro Video Flicker & Speed-vs-Quality Tradeoffs

- **Intro video ring-buffer flicker:** Software-decoded Bink video planes are
  linear single-level R8 textures written through an alternate mapping that the
  write-watch guard cannot see. Fixed via `VICTORY_VIDEO_PLANES` (on by default)
  and `VICTORY_VIDEO_HOT_FRAMES` (default 1024) in `texture.rs`
  (`trustno1-coder/VIctory-NVIDIA` `77e13a9`).
- **Speed vs visual stability tradeoff:**
  - **Fast / Balanced presets** enable `VICTORY_RELAY=2`, `VICTORY_CARD_STEPS=2`,
    and `VICTORY_TEXTURE_NO_WAIT=1` for ~5-7 extra FPS, at the risk of
    occasional stretched meshes or unbaked/rainbow ped skin during heavy
    streaming.
  - **Quality preset** dials back to `VICTORY_RELAY=1`, `VICTORY_CARD_STEPS=0`,
    and `VICTORY_TEXTURE_NO_WAIT=0` when diagnosing visual glitches.

---

## 9. Still Open (Observed in Live Runs)

- **Stage 1 RT reflection speckles:** Uninitialized RT reflection targets when
  `CS_RaytracingResolve` is skipped can show RGB speckles on car glass and
  magenta tint on foliage/white cloth; real ray-traced reflections & GI wait
  for Stage 2.
- **Series S dark hair/eye blocks:** Series S produces 32x32 dark blocks around
  character hair/eyes on NVIDIA Wave32 even with `VICTORY_UPSCALER=6`; Series X
  (`1920x1080` @ `0.67`, `VICTORY_UPSCALER=6`) renders faces and hair cleanly.
- **12 refused compute host pipelines (`E_INVALIDARG` / `0x80070057`):**
  Identical across Series X and Series S (`02384a60fda6f2d0`,
  `0a5d4f9047b3dce4`, `0a5ecfee6b79e13b`, `369cbe906ea72764`,
  `3c196d81eca7ff15`, `5dc3b25986622949`, `6b82bc7d22016cb5`,
  `6f225b641c5b0708`, `71883e40620a567a`, `acf14a70f37fb012`,
  `f0aa2b1e79275592`, `f79824724d8e2daf`).
- **Remaining CPU bottleneck for native 30 FPS:** On RTX 4070 Ti SUPER 16 GB,
  GPU wait is only ~1-6 ms (`~66%` util, `57 C`, `~97 W`) while CPU command
  translation (`VICTORY_THIN`), `preparing` (~22 ms), and non-twin table CBV
  copies (~6,200-8,600/frame on the 1M NVIDIA heap) set the ~20-25 FPS native
  world ceiling. Native DLSS (`avatarmanz` PR #203) and NVIDIA Smooth Motion
  (driver 621.40+ on RTX 40-series) are the next high-leverage steps.
