use super::capture::Checked;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use windows::Win32::Graphics::{
    Direct3D::ID3DDestructionNotifier, Direct3D12::*, Dxgi::Common::*, Dxgi::*,
};
use windows::core::Interface;

const MIB: u64 = 1 << 20;
const SLAB: u64 = 64 * MIB;
static SLAB_BYTES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn slab_bytes() -> u64 {
    SLAB_BYTES.load(Ordering::Relaxed)
}

pub(crate) fn slab_headroom() -> u64 {
    if !super::requested() {
        return 0;
    }
    static TARGET: OnceLock<u64> = OnceLock::new();
    TARGET
        .get_or_init(|| {
            std::env::var("VICTORY_CARD_ROOM_FOR_DXR_MB")
                .ok()
                .and_then(|s| s.parse::<u64>().ok()?.checked_mul(MIB))
                .unwrap_or(1024 * MIB)
        })
        .saturating_sub(slab_bytes())
}

pub(super) fn limit(cap: u64, budget: u64) -> u64 {
    cap.min(budget.saturating_sub((budget / 10).max(1024 * MIB)))
}

pub(super) fn groups(adapter: &IDXGIAdapter3) -> Checked<[u64; 4]> {
    let mut local = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    let mut other = local;
    // SAFETY: both destinations match the queried DXGI segment on node 0.
    unsafe {
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut local)
            .map_err(|_| "LOCAL budget unavailable")?;
        adapter
            .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut other)
            .map_err(|_| "NON_LOCAL budget unavailable")?;
    }
    Ok([
        local.Budget,
        local.CurrentUsage,
        other.Budget,
        other.CurrentUsage,
    ])
}

/// Keep Rust ownership alive exactly as long as the D3D resource itself, including
/// references held by a shader-visible descriptor after the map changes generation.
pub(super) fn keep_on_resource<T: Send + 'static>(r: &ID3D12Resource, value: T) -> Checked<()> {
    unsafe extern "system" fn release<T>(p: *mut c_void) {
        // SAFETY: one successful registration, invoked once on destruction.
        drop(unsafe { Box::from_raw(p.cast::<T>()) });
    }
    let notify = r
        .cast::<ID3DDestructionNotifier>()
        .map_err(|_| "resource lifetime notification unavailable")?;
    let p = Box::into_raw(Box::new(value));
    // SAFETY: transferred on success; reclaimed here on failure.
    if unsafe { notify.RegisterDestructionCallback(Some(release::<T>), p.cast()) }.is_err() {
        drop(unsafe { Box::from_raw(p) });
        return Err("resource lifetime notification failed");
    }
    Ok(())
}

/// Common to the submit thread and a thin worker materializing a new GPU
/// input shadow. The lock covers admission through MakeResident/creation.
pub(crate) struct Budget {
    adapter: IDXGIAdapter3,
    cap: u64,
    ledger: Mutex<Ledger>,
}
struct Ledger {
    last: u64,
    peak: u64,
    unreflected: VecDeque<(Weak<()>, u64)>,
}
impl Ledger {
    fn observed(&mut self, used: u64) -> u64 {
        let mut increase = used.saturating_sub(self.last);
        self.last = used;
        self.peak = self.peak.max(used);
        self.unreflected.retain_mut(|(live, bytes)| {
            if live.strong_count() == 0 {
                return false;
            }
            let reflected = increase.min(*bytes);
            *bytes -= reflected;
            increase -= reflected;
            *bytes != 0
        });
        self.unreflected.iter().map(|(_, n)| n).sum()
    }
}

pub(crate) struct Reservation<'a> {
    budget: &'a Budget,
    ledger: MutexGuard<'a, Ledger>,
    charge: u64,
}
impl Budget {
    pub(super) fn new(adapter: IDXGIAdapter3, cap: u64, used: u64) -> Arc<Self> {
        Arc::new(Self {
            adapter,
            cap,
            ledger: Mutex::new(Ledger {
                last: used,
                peak: used,
                unreflected: VecDeque::new(),
            }),
        })
    }
    pub(crate) fn reserve(&self, charge: u64) -> Checked<Reservation<'_>> {
        let mut l = self.ledger.lock().unwrap_or_else(|e| e.into_inner());
        let [budget, used, ..] = groups(&self.adapter)?;
        let pending = l.observed(used);
        if used.saturating_add(pending).saturating_add(charge) > limit(self.cap, budget) {
            return Err("LOCAL peak guard");
        }
        Ok(Reservation {
            budget: self,
            ledger: l,
            charge,
        })
    }
    pub(super) fn usage(&self) -> Checked<([u64; 4], u64)> {
        let mut ledger = self.ledger.lock().unwrap_or_else(|e| e.into_inner());
        let now = groups(&self.adapter)?;
        ledger.observed(now[1]);
        Ok((now, ledger.peak))
    }
}
impl Reservation<'_> {
    pub(crate) fn finish_shadow(self, r: &ID3D12Resource) -> Checked<()> {
        let mut props = D3D12_HEAP_PROPERTIES::default();
        // A two-sided shadow can put its GPU side in system memory. Account it
        // by its actual heap properties rather than the shadow's logical kind.
        unsafe { r.GetHeapProperties(Some(&mut props), None) }
            .map_err(|_| "shadow memory segment unknown")?;
        self.finish(
            r,
            props.Type == D3D12_HEAP_TYPE_DEFAULT
                || props.MemoryPoolPreference == D3D12_MEMORY_POOL_L1,
        )
    }
    pub(super) fn finish(mut self, r: &ID3D12Resource, local: bool) -> Checked<()> {
        if !local || self.charge == 0 {
            return Ok(());
        }
        let token = Arc::new(());
        self.ledger
            .unreflected
            .push_back((Arc::downgrade(&token), self.charge));
        if let Ok([_, now, ..]) = groups(&self.budget.adapter) {
            self.ledger.observed(now);
        }
        if Arc::weak_count(&token) != 0 {
            keep_on_resource(r, token)?;
        }
        Ok(())
    }
}
pub(super) struct Memory {
    pub dev: ID3D12Device5,
    pub cap: u64,
    pub start: [u64; 4],
    pub budget: Arc<Budget>,
    slabs: Vec<Weak<Slab>>,
}
impl Memory {
    pub fn new(dev: ID3D12Device5, adapter: IDXGIAdapter3) -> Checked<Self> {
        let cap = match std::env::var_os("VICTORY_DXR_VRAM_CAP_MB") {
            None => 12_800 * MIB,
            Some(v) => v
                .to_str()
                .and_then(|s| s.parse::<u64>().ok())
                .and_then(|mb| mb.checked_mul(MIB))
                .ok_or("invalid VICTORY_DXR_VRAM_CAP_MB")?,
        };
        let start = groups(&adapter)?;
        Ok(Self {
            dev,
            budget: Budget::new(adapter, cap, start[1]),
            cap,
            start,
            slabs: vec![],
        })
    }

    pub fn buffer(
        &mut self,
        bytes: u64,
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
        flags: D3D12_RESOURCE_FLAGS,
    ) -> Checked<ID3D12Resource> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        // Charge the allocation's aligned physical size, including a new 64 MiB slab,
        // not its logical slice. UMA shares its local pool with CPU-visible heaps.
        let size = unsafe { self.dev.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
        if bytes == 0 || size == u64::MAX {
            return Err("invalid allocation size");
        }
        let cpu_nonlocal = heap != D3D12_HEAP_TYPE_DEFAULT
            && unsafe { self.dev.GetCustomHeapProperties(0, D3D12_HEAP_TYPE_DEFAULT) }
                .MemoryPoolPreference
                == D3D12_MEMORY_POOL_L1;
        let charge = if cpu_nonlocal { 0 } else { size };
        let reservation = self.budget.reserve(charge)?;
        let mut r = None;
        let props = D3D12_HEAP_PROPERTIES {
            Type: heap,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
            ..Default::default()
        };
        // SAFETY: committed, resident buffer in its required initial state.
        unsafe {
            self.dev.CreateCommittedResource(
                &props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut r,
            )
        }
        .map_err(|_| "CreateCommittedResource")?;
        let r: ID3D12Resource = r.ok_or("null resource")?;
        reservation.finish(&r, true)?;
        Ok(r)
    }

    pub fn as_storage(&mut self, bytes: u64, tlas: bool) -> Checked<Storage> {
        let size = bytes
            .checked_next_multiple_of(256)
            .filter(|&n| n != 0)
            .ok_or("invalid AS size")?;
        if tlas {
            // A cached SRV owns a COM resource. Give each TLAS its own resource so an
            // old descriptor can never observe a recycled slice. It also owns its BLAS pins.
            let buf = self.buffer(
                size,
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            )?;
            return Ok(Storage {
                gpu: unsafe { buf.GetGPUVirtualAddress() },
                buffer: buf,
                bytes: size,
                _slice: None,
            });
        }
        self.slabs.retain(|s| s.strong_count() != 0);
        for s in &self.slabs {
            if let Some(slab) = s.upgrade()
                && let Some(storage) = slab.take(size)
            {
                return Ok(storage);
            }
        }
        let bytes = size.max(SLAB);
        let buf = self.buffer(
            bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        )?;
        let slab = Arc::new(Slab {
            buffer: buf,
            bytes,
            free: Mutex::new(vec![(0, bytes)]),
        });
        SLAB_BYTES.fetch_add(bytes, Ordering::Relaxed);
        self.slabs.push(Arc::downgrade(&slab));
        slab.take(size).ok_or("empty AS slab")
    }
}

struct Slab {
    buffer: ID3D12Resource,
    bytes: u64,
    free: Mutex<Vec<(u64, u64)>>,
}
impl Drop for Slab {
    fn drop(&mut self) {
        SLAB_BYTES.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}
impl Slab {
    fn take(self: &Arc<Self>, size: u64) -> Option<Storage> {
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let i = free.iter().position(|&(_, n)| n >= size)?;
        let (offset, n) = free[i];
        if n == size {
            free.remove(i);
        } else {
            free[i] = (offset + size, n - size);
        }
        Some(Storage {
            gpu: unsafe { self.buffer.GetGPUVirtualAddress() } + offset,
            buffer: self.buffer.clone(),
            bytes: size,
            _slice: Some(Slice {
                slab: self.clone(),
                offset,
                size,
            }),
        })
    }
}
struct Slice {
    slab: Arc<Slab>,
    offset: u64,
    size: u64,
}
impl Drop for Slice {
    fn drop(&mut self) {
        let mut f = self.slab.free.lock().unwrap_or_else(|e| e.into_inner());
        let mut i = f.partition_point(|&(a, _)| a < self.offset);
        f.insert(i, (self.offset, self.size));
        if i > 0 && f[i - 1].0 + f[i - 1].1 == f[i].0 {
            let (_, n) = f.remove(i);
            i -= 1;
            f[i].1 += n;
        }
        if i + 1 < f.len() && f[i].0 + f[i].1 == f[i + 1].0 {
            let (_, n) = f.remove(i + 1);
            f[i].1 += n;
        }
    }
}

pub(super) struct Storage {
    pub gpu: u64,
    pub buffer: ID3D12Resource,
    pub bytes: u64,
    _slice: Option<Slice>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dxr_peak_guard_counts_only_unobserved_live_allocations() {
        let a = Arc::new(());
        let b = Arc::new(());
        let mut l = Ledger {
            last: 1024,
            peak: 1024,
            unreflected: VecDeque::from([(Arc::downgrade(&a), 512), (Arc::downgrade(&b), 256)]),
        };
        assert_eq!(l.observed(1280), 512);
        assert_eq!(l.observed(1536), 256);
        assert_eq!(l.observed(1536), 256);
        drop(b);
        assert_eq!(l.observed(1400), 0);
        assert_eq!(l.peak, 1536);
    }
}
