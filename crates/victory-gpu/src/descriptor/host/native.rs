//! A title RTAS record may keep the same guest VA for many native generations.
use super::{HostHeap, Queued, View, native_tlas, tlas_srv};
use crate::dxr::Sidecar;
use std::{collections::BTreeMap, sync::Mutex};
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

pub(super) struct Slots {
    members: Queued,
    guests: Mutex<BTreeMap<u32, u64>>,
}

impl Slots {
    pub(super) fn forget(&self, index: u32) {
        if self.members.may(index) {
            let mut m = self.guests.lock().unwrap_or_else(|e| e.into_inner());
            m.remove(&index);
            self.members.set(index, false);
        }
    }

    pub(super) fn release(&self, first: u32, end: u32) {
        let mut m = self.guests.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|&i, _| {
            if (first..end).contains(&i) {
                self.members.set(i, false);
                false
            } else {
                true
            }
        });
    }
}

impl HostHeap {
    pub(super) fn keep_view(&self, index: u32, view: &View, resource: Option<ID3D12Resource>) {
        if resource.is_some()
            && let Some(guest) = native_tlas(view)
            && guest != 0
        {
            let slots = self.native_slots.get_or_init(|| Slots {
                members: Queued::new(self.count + self.dynamic),
                guests: Mutex::default(),
            });
            let mut m = slots.guests.lock().unwrap_or_else(|e| e.into_inner());
            self.keep_resource(index, resource);
            m.insert(index, guest);
            slots.members.set(index, true);
        } else {
            self.keep(index, resource);
        }
    }

    /// Called only under the host submission lock. A production worker can install
    /// a new guest record, but cannot advance an existing record's native generation.
    pub(crate) fn refresh_native_slots(&self, side: &Sidecar, fenced: &mut bool) {
        let Some(slots) = self.native_slots.get() else {
            return;
        };
        let slots_to_check: Vec<_> = slots
            .guests
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(&i, &k)| (i, k))
            .collect();
        for (index, guest) in slots_to_check {
            let (target, pending) = side.slot_target(guest);
            if !pending && self.kept_in(index).as_ref() == Some(&target) {
                continue;
            }
            if !*fenced {
                side.finish_slot_readers();
                *fenced = true;
            }
            let _held = self.made.lock(index);
            let m = slots.guests.lock().unwrap_or_else(|e| e.into_inner());
            if m.get(&index) != Some(&guest) {
                continue;
            }
            let (target, _) = side.slot_target(guest);
            if self.kept_in(index).as_ref() == Some(&target) {
                continue;
            }
            let host = crate::host::host().expect("active native RT device");
            let at = self.at(index);
            tlas_srv(&host.device, &target, at);
            #[cfg(feature = "diagnostics")]
            crate::replay::record::dxr_slot(at.ptr, guest);
            self.put(index);
            self.keep_resource(index, Some(target));
        }
    }
}
