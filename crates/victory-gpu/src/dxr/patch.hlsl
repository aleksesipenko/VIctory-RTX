// Adapted from maren-halv/VIctory-DXR @ e1740b3 (GPL-3.0).
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Params : register(b0) { uint count; uint first; };
ByteAddressBuffer instances : register(t0);
ByteAddressBuffer table : register(t1);
RWByteAddressBuffer output_instances : register(u0);
// missing, reserved, then one bit per referenced hash slot. Read only after the fence.
RWByteAddressBuffer report : register(u1);
uint hash_key(uint lo, uint hi) {
    uint h = lo ^ (hi * 0x9e3779b9u);
    h ^= h >> 16; h *= 0x85ebca6bu; h ^= h >> 13; h *= 0xc2b2ae35u;
    return h ^ (h >> 16);
}
[RootSignature(RS)]
[numthreads(64,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = first + tid.x;
    if (i >= count) return;
    uint at = i * 64;
    uint4 last = instances.Load4(at + 48);
    uint2 guest = last.zw;
    last.zw = 0;
    if (any(guest)) {
        uint2 key = uint2((guest.x >> 8) | (guest.y << 24), guest.y >> 8);
        uint slot = hash_key(key.x, key.y) & 65535u;
        if ((guest.x & 255u) == 0) {
            [loop] for (uint probe = 0; probe < 128; ++probe) {
                uint pos = (slot + probe) & 65535u;
                uint4 item = table.Load4(pos * 16);
                if (!any(item.xy)) break;
                if (all(item.xy == key)) {
                    last.zw = item.zw;
                    report.InterlockedOr(8u + (pos >> 5) * 4u, 1u << (pos & 31u));
                    break;
                }
            }
        }
        if (!any(last.zw)) report.InterlockedAdd(0,1);
    }
    output_instances.Store4(at, instances.Load4(at));
    output_instances.Store4(at+16, instances.Load4(at+16));
    output_instances.Store4(at+32, instances.Load4(at+32));
    output_instances.Store4(at+48, last);
}
