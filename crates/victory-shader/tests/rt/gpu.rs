//! All GPU VAs below belong to resources kept alive until submit() fences
//! the direct queue. Only initial uploads and final/readback copies use PCIe.
use victory_shader::{
    Container, FourCC,
    emit::{
        self, HostShader,
        host::Host,
        rt_replay::{self, QueryMode},
        validator::Validator,
    },
    rt_packet::{Packet, bvh4},
};
use windows::{
    Win32::{
        Foundation::HANDLE,
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
    core::Interface,
};

pub struct Gpu {
    pub host: Host,
    device: ID3D12Device5,
    pub queue: ID3D12CommandQueue,
    alloc: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList4,
    fence: ID3D12Fence,
    heaps: [Option<ID3D12DescriptorHeap>; 2],
    serial: u64,
    uploads: Vec<ID3D12Resource>,
}

fn transition(
    r: &ID3D12Resource,
    from: D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrowed COM pointer, used synchronously to record the barrier.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: from,
                StateAfter: to,
            }),
        },
        ..Default::default()
    }
}

impl Gpu {
    pub fn new(host: Host) -> Self {
        let device: ID3D12Device5 = host.device.cast().unwrap();
        // SAFETY: all interfaces come from the same physical device.
        unsafe {
            let queue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let alloc = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap();
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let heaps = [
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
            ]
            .map(|kind| {
                Some(
                    device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: kind,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .unwrap(),
                )
            });
            Self {
                host,
                device,
                queue,
                alloc,
                list,
                fence,
                heaps,
                serial: 0,
                uploads: Vec::new(),
            }
        }
    }
    pub fn submit(&mut self) {
        // SAFETY: never reset an in-flight allocator or release staging
        // resources before completion. A null event waits synchronously.
        unsafe {
            self.list.Close().unwrap();
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
            self.serial += 1;
            self.queue.Signal(&self.fence, self.serial).unwrap();
            self.fence
                .SetEventOnCompletion(self.serial, HANDLE::default())
                .unwrap();
            self.device.GetDeviceRemovedReason().unwrap();
            assert!(self.fence.GetCompletedValue() >= self.serial);
            self.alloc.Reset().unwrap();
            self.list.Reset(&self.alloc, None).unwrap();
            self.uploads.clear();
        }
    }
    fn buffer(
        &self,
        heap: D3D12_HEAP_TYPE,
        bytes: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        assert!(bytes > 0);
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
        let mut resource = None;
        // SAFETY: device copies the descriptors, returning an owned resource.
        unsafe {
            self.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    state,
                    None,
                    &mut resource,
                )
                .unwrap();
        }
        resource.unwrap()
    }
    fn data(
        &mut self,
        words: &[u32],
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let bytes = size_of_val(words) as u64;
        let staging = self.buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let dest = self.buffer(
            D3D12_HEAP_TYPE_DEFAULT,
            bytes,
            flags,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: mapping spans exactly the allocated byte count. submit
        // owns staging through the GPU copy.
        unsafe {
            let mut dst = std::ptr::null_mut();
            staging
                .Map(0, Some(&D3D12_RANGE::default()), Some(&mut dst))
                .unwrap();
            std::ptr::copy_nonoverlapping(words.as_ptr(), dst.cast(), words.len());
            staging.Unmap(0, None);
            self.list.CopyBufferRegion(&dest, 0, &staging, 0, bytes);
            self.list
                .ResourceBarrier(&[transition(&dest, D3D12_RESOURCE_STATE_COPY_DEST, state)]);
        }
        self.uploads.push(staging);
        dest
    }
    fn input(&mut self, w: &[u32]) -> ID3D12Resource {
        self.data(
            w,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )
    }
    pub fn readback(&self, bytes: u64) -> ID3D12Resource {
        self.buffer(
            D3D12_HEAP_TYPE_READBACK,
            bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
    }
    pub fn barrier(&self) {
        // SAFETY: a null-resource UAV barrier orders all AS/UAV accesses.
        unsafe {
            self.list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            }]);
        }
    }
    pub fn timestamps(&self, n: u32) -> ID3D12QueryHeap {
        let mut heap = None;
        // SAFETY: fixed number of timestamp slots on the recording device.
        unsafe {
            self.device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: n,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .unwrap();
        }
        heap.unwrap()
    }
}

pub fn words(r: &ID3D12Resource, n: usize) -> Vec<u32> {
    // SAFETY: only used after the submitting fence; bound the mapped slice
    // by the actual allocation, then copy before unmapping.
    unsafe {
        assert!(n as u64 * 4 <= r.GetDesc().Width);
        let range = D3D12_RANGE {
            Begin: 0,
            End: n * 4,
        };
        let mut ptr = std::ptr::null_mut();
        r.Map(0, Some(&range), Some(&mut ptr)).unwrap();
        let out = std::slice::from_raw_parts(ptr.cast::<u32>(), n).to_vec();
        r.Unmap(0, Some(&D3D12_RANGE::default()));
        out
    }
}
fn va(r: &ID3D12Resource) -> u64 {
    // SAFETY: a live committed buffer.
    unsafe { r.GetGPUVirtualAddress() }
}

enum Source {
    Triangles {
        descs: Vec<D3D12_RAYTRACING_GEOMETRY_DESC>,
        _vertices: ID3D12Resource,
    },
    Instances {
        count: u32,
        buffer: Option<ID3D12Resource>,
    },
}
impl Source {
    fn inputs(&self) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        let (kind, count, data) = match self {
            Self::Triangles { descs, .. } => (
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                descs.len() as u32,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: descs.as_ptr(),
                },
            ),
            Self::Instances { count, buffer } => (
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                *count,
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: buffer.as_ref().map_or(0, va),
                },
            ),
        };
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: kind,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: data,
        }
    }
}
struct Build {
    source: Source,
    scratch: ID3D12Resource,
    result: ID3D12Resource,
}
impl Build {
    fn new(g: &Gpu, source: Source) -> Self {
        let mut sizes = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: CPU geometry descriptors in source outlive the query/build.
        unsafe {
            g.device
                .GetRaytracingAccelerationStructurePrebuildInfo(&source.inputs(), &mut sizes);
        }
        assert!(sizes.ResultDataMaxSizeInBytes > 0);
        let buffer = |bytes, state| {
            g.buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                state,
            )
        };
        Self {
            source,
            result: buffer(
                sizes.ResultDataMaxSizeInBytes,
                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
            ),
            scratch: buffer(
                sizes.ScratchDataSizeInBytes.max(256),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        }
    }
    fn record(&self, g: &Gpu) {
        let desc = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
            Inputs: self.source.inputs(),
            DestAccelerationStructureData: va(&self.result),
            ScratchAccelerationStructureData: va(&self.scratch),
            SourceAccelerationStructureData: 0,
        };
        // SAFETY: inputs, scratch, and result are retained by Replay; scratch
        // is distinct for each build and results never change resource state.
        unsafe {
            g.list.BuildRaytracingAccelerationStructure(&desc, None);
        }
    }
}

struct Pipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    bindings: Vec<(u8, bool)>,
}
impl Pipeline {
    fn new(g: &Gpu, mut shader: HostShader, expected: &[u8]) -> Self {
        let mut validator =
            Validator::load(Validator::default_path().as_deref()).expect("dxil.dll required");
        validator
            .validate(&mut shader.dxil)
            .expect("DXIL validation");
        let c = Container::parse(&shader.dxil).unwrap();
        assert_eq!(
            c.part(FourCC::DXIL).unwrap().data[..4],
            [0x66, 0, 5, 0],
            "cs_6_6"
        );
        let sfi = u64::from_le_bytes(c.part(FourCC::SFI0).unwrap().data.try_into().unwrap());
        assert_eq!(sfi & 0x10_0000 != 0, !expected.is_empty());
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let u = |at: usize| u32::from_le_bytes(psv[at..at + 4].try_into().unwrap());
        let at = 4 + u(0) as usize;
        let mut tlas = Vec::new();
        for n in 0..u(at) as usize {
            let b = at + 8 + n * u(at + 4) as usize;
            if u(b + 16) == 16 {
                assert_eq!((u(b), u(b + 4), u(b + 8)), (4, 1, u(b + 12)));
                tlas.push(u(b + 8) as u8);
            }
        }
        assert_eq!(tlas, expected);
        let mut bindings: Vec<_> = shader
            .root_srvs
            .iter()
            .map(|&s| (s, false))
            .chain(shader.root_uavs.iter().map(|&s| (s, true)))
            .collect();
        bindings.sort_unstable();
        let cost = shader.user_data_dwords().max(1) + 2 * bindings.len() as u32;
        assert!(cost <= 64);
        let (root, pso) = g.host.compute_pipeline(&shader).unwrap();
        if expected == [emit::SLOT_TLAS_0] {
            let mut oversized = shader.clone();
            oversized.root_srvs = (0..32).collect();
            assert!(
                g.host
                    .root_signature_for(&[(&oversized, D3D12_SHADER_VISIBILITY_ALL)])
                    .unwrap_err()
                    .contains("limit 64")
            );
        }
        eprintln!("PSO TLAS={tlas:?} root={cost}/64 DWORD");
        Self {
            root,
            pso,
            bindings,
        }
    }
}

pub struct Replay {
    blases: Vec<Build>,
    tlases: Vec<Build>,
    rays: ID3D12Resource,
    nodes: ID3D12Resource,
    sw: Pipeline,
    dxr: Pipeline,
    outputs: [ID3D12Resource; 2],
    n: u32,
}
impl Replay {
    pub fn new(g: &mut Gpu, p: &Packet, mode: QueryMode) -> Self {
        let packed = bvh4::pack(p).unwrap();
        let n = u32::try_from(p.rays.len()).unwrap();
        assert!(
            n.div_ceil(rt_replay::GROUP_SIZE) <= 65535,
            "packet exceeds a 1D dispatch"
        );
        let mut data = vec![n, 0, 0, 0];
        for (r, _) in &p.rays {
            data.extend(r.origin.map(f32::to_bits));
            data.push(r.t_min.to_bits());
            data.extend(r.direction.map(f32::to_bits));
            data.push(r.t_max.to_bits());
            data.extend([u32::from(r.mask), 0, 0, 0]);
        }
        let rays = g.input(&data);
        let nodes = g.input(&packed);
        let blases: Vec<_> = p
            .blases
            .iter()
            .map(|groups| {
                let vertices: Vec<_> = groups
                    .iter()
                    .flat_map(|g| g.triangles.iter().flatten().flatten().map(|x| x.to_bits()))
                    .collect();
                let vb = g.input(&vertices);
                let mut address = va(&vb);
                let descs = groups
                    .iter()
                    .map(|geom| {
                        let count = u32::try_from(geom.triangles.len() * 3).unwrap();
                        let triangles = D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                            VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                            VertexCount: count,
                            VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                                StartAddress: address,
                                StrideInBytes: 12,
                            },
                            ..Default::default()
                        };
                        address += u64::from(count) * 12;
                        D3D12_RAYTRACING_GEOMETRY_DESC {
                            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                            Flags: if geom.opaque {
                                D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE
                            } else {
                                D3D12_RAYTRACING_GEOMETRY_FLAG_NONE
                            },
                            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                                Triangles: triangles,
                            },
                        }
                    })
                    .collect();
                Build::new(
                    g,
                    Source::Triangles {
                        descs,
                        _vertices: vb,
                    },
                )
            })
            .collect();
        let tlases = p
            .tlases
            .iter()
            .map(|instances| {
                let mut data = Vec::new();
                for i in instances {
                    data.extend(i.transform.into_iter().flatten().map(f32::to_bits));
                    let address = va(&blases[i.blas as usize].result);
                    data.extend([
                        i.id | u32::from(i.mask) << 24,
                        i.contribution | ((i.front_ccw as u32) << 25),
                        address as u32,
                        (address >> 32) as u32,
                    ]);
                }
                let buffer = (!data.is_empty()).then(|| g.input(&data));
                Build::new(
                    g,
                    Source::Instances {
                        count: instances.len() as u32,
                        buffer,
                    },
                )
            })
            .collect();
        let count = p.tlases.len();
        let sw = Pipeline::new(g, rt_replay::software_shader(count).unwrap(), &[]);
        let dxr = Pipeline::new(
            g,
            rt_replay::ray_query_shader(count, mode).unwrap(),
            &[emit::SLOT_TLAS_0, emit::SLOT_TLAS_1, emit::SLOT_TLAS_2][..count],
        );
        let poison = vec![0xdead_beef; n as usize * rt_replay::HIT_WORDS];
        let outputs = std::array::from_fn(|_| {
            g.data(
                &poison,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )
        });
        g.submit();
        Self {
            blases,
            tlases,
            rays,
            nodes,
            sw,
            dxr,
            outputs,
            n,
        }
    }
    pub fn build(&self, g: &Gpu, top: bool) {
        self.record_builds(g, top);
        g.barrier();
    }
    pub fn record_builds(&self, g: &Gpu, top: bool) {
        for build in if top { &self.tlases } else { &self.blases } {
            build.record(g);
        }
    }
    pub fn trace(&self, g: &Gpu, hardware: bool) {
        self.bind(g, hardware);
        self.dispatch(g);
    }
    pub fn bind(&self, g: &Gpu, hardware: bool) {
        let pipe = if hardware { &self.dxr } else { &self.sw };
        // SAFETY: bindings are derived from the validated shader in the
        // host root signature's sorted order. Every resource stays live.
        unsafe {
            g.list.SetDescriptorHeaps(&g.heaps);
            g.list.SetPipelineState(&pipe.pso);
            g.list.SetComputeRootSignature(&pipe.root);
            g.list.SetComputeRoot32BitConstant(0, 0, 0);
            for (i, &(slot, uav)) in pipe.bindings.iter().enumerate() {
                let param = 1 + i as u32;
                if uav {
                    assert_eq!(slot, 0);
                    g.list.SetComputeRootUnorderedAccessView(
                        param,
                        va(&self.outputs[hardware as usize]),
                    );
                } else {
                    let r = match slot {
                        0 => &self.rays,
                        1 => &self.nodes,
                        249..=251 => &self.tlases[(251 - slot) as usize].result,
                        _ => panic!("unbound SRV"),
                    };
                    g.list.SetComputeRootShaderResourceView(param, va(r));
                }
            }
        }
    }
    pub fn dispatch(&self, g: &Gpu) {
        // SAFETY: bind() installed the matching root, views and pipeline.
        unsafe {
            g.list
                .Dispatch(self.n.div_ceil(rt_replay::GROUP_SIZE), 1, 1);
        }
    }
    pub fn results(&self, g: &mut Gpu) -> [Vec<u32>; 2] {
        let bytes = u64::from(self.n) * rt_replay::HIT_WORDS as u64 * 4;
        let rb = self.outputs.each_ref().map(|out| {
            let rb = g.readback(bytes);
            // SAFETY: the transitions order the completed UAV writes with
            // the copy; the fence below orders mapping with that copy.
            unsafe {
                g.list.ResourceBarrier(&[transition(
                    out,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                g.list.CopyBufferRegion(&rb, 0, out, 0, bytes);
                g.list.ResourceBarrier(&[transition(
                    out,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            rb
        });
        g.submit();
        rb.map(|r| words(&r, bytes as usize / 4))
    }
}
