// Adapted from maren-halv/VIctory-DXR @ 5b163a5 (GPL-3.0).
#define RS "RootConstants(num32BitConstants=3, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Params : register(b0) { uint count; uint first; uint cull_mask_cfg; };
ByteAddressBuffer instances : register(t0);
ByteAddressBuffer table : register(t1);
RWByteAddressBuffer output_instances : register(u0);
// missing (offset 0), reserved (offset 4), then one bit per referenced hash slot (offset 8..).
// Read only after the fence.
RWByteAddressBuffer report : register(u1);

groupshared uint g_missing;

uint hash_key(uint lo, uint hi) {
    uint h = lo ^ (hi * 0x9e3779b9u);
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u;
    return h ^ (h >> 16);
}

[RootSignature(RS)]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0u) {
        g_missing = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint i = first + tid.x;
    if (i < count) {
        uint at = i * 64u;
        uint4 last = instances.Load4(at + 48u);
        uint2 guest = last.zw;
        last.zw = 0u;
        if (any(guest)) {
            uint2 key = uint2((guest.x >> 8u) | (guest.y << 24u), guest.y >> 8u);
            uint slot = hash_key(key.x, key.y) & 65535u;
            if ((guest.x & 255u) == 0u) {
                [loop] for (uint probe = 0u; probe < 128u; ++probe) {
                    uint pos = (slot + probe) & 65535u;
                    uint4 item = table.Load4(pos * 16u);
                    if (!any(item.xy)) break;
                    if (all(item.xy == key)) {
                        last.zw = item.zw;
                        report.InterlockedOr(8u + (pos >> 5u) * 4u, 1u << (pos & 31u));
                        break;
                    }
                }
            }
            if (any(last.zw)) {
                if (cull_mask_cfg != 0u && (last.y & 0x01000000u) != 0u) {
                    uint cfg_mask = cull_mask_cfg & 0xFFu;
                    uint cur_mask = (last.x >> 24u) & 0xFFu;
                    if (cfg_mask != 0u && cur_mask != 0u) {
                        uint new_mask = ((cull_mask_cfg & 0x100u) != 0u)
                            ? (cur_mask | cfg_mask)
                            : (((cur_mask & ~cfg_mask) != 0u) ? (cur_mask & ~cfg_mask) : cur_mask);
                        if (new_mask != cur_mask) {
                            last.x = (last.x & 0x00FFFFFFu) | (new_mask << 24u);
                        }
                    }
                }
            } else {
                InterlockedAdd(g_missing, 1u);
            }
        }
        output_instances.Store4(at, instances.Load4(at));
        output_instances.Store4(at + 16u, instances.Load4(at + 16u));
        output_instances.Store4(at + 32u, instances.Load4(at + 32u));
        output_instances.Store4(at + 48u, last);
    }

    GroupMemoryBarrierWithGroupSync();
    if (gi == 0u && g_missing != 0u) {
        report.InterlockedAdd(0u, g_missing);
    }
}
