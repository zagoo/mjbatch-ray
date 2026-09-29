// SPDX-License-Identifier: Apache-2.0
//
// Philox4x64-10 (Salmon et al., Random123), bit-identical to numpy.random.Philox's
// block function. numpy increments its counter *before* each block, so numpy's
// Philox(key=k, counter=c) yields block(c + 1) first; tests pin that relation.
//
// RaySimBatch draw convention ("rsb-philox-v1"): a draw is addressed by the counter
// (seq, index, stream, 0); uniform = (v0 >> 11) * 2^-53; normal = Box-Muller over
// (v0, v1) taking the cosine branch.
#pragma once

#include <cmath>
#include <cstdint>

namespace rsb {

struct Block {
  uint64_t v[4];
};

inline void MulHiLo(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo) {
  const __uint128_t p = static_cast<__uint128_t>(a) * b;
  *hi = static_cast<uint64_t>(p >> 64);
  *lo = static_cast<uint64_t>(p);
}

inline Block Philox4x64(const uint64_t ctr[4], const uint64_t key[2]) {
  uint64_t c0 = ctr[0], c1 = ctr[1], c2 = ctr[2], c3 = ctr[3];
  uint64_t k0 = key[0], k1 = key[1];
  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      k0 += 0x9E3779B97F4A7C15ULL;
      k1 += 0xBB67AE8584CAA73BULL;
    }
    uint64_t hi0, lo0, hi1, lo1;
    MulHiLo(0xD2E7470EE14C6C93ULL, c0, &hi0, &lo0);
    MulHiLo(0xCA5A826395121157ULL, c2, &hi1, &lo1);
    const uint64_t n0 = hi1 ^ c1 ^ k0;
    const uint64_t n2 = hi0 ^ c3 ^ k1;
    c0 = n0;
    c1 = lo1;
    c2 = n2;
    c3 = lo0;
  }
  return Block{{c0, c1, c2, c3}};
}

inline double U01(uint64_t x) { return static_cast<double>(x >> 11) * 0x1.0p-53; }

inline Block Draw(const uint64_t key[2], uint64_t seq, uint64_t index, uint64_t stream) {
  const uint64_t ctr[4] = {seq, index, stream, 0};
  return Philox4x64(ctr, key);
}

inline double Uniform(const uint64_t key[2], uint64_t seq, uint64_t index, uint64_t stream) {
  return U01(Draw(key, seq, index, stream).v[0]);
}

inline double Normal(const uint64_t key[2], uint64_t seq, uint64_t index, uint64_t stream) {
  const Block b = Draw(key, seq, index, stream);
  const double u1 = 1.0 - U01(b.v[0]);  // (0, 1]
  const double u2 = U01(b.v[1]);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586476925286766559 * u2);
}

// Stream identifiers shared with raysimbatch/seeds.py.
enum Stream : uint64_t {
  kStreamReset = 1,
  kStreamDomain = 2,
  kStreamObsNoise = 3,
  kStreamAction = 4,
  kStreamTask = 5,
};

}  // namespace rsb
