#include <cuda_fp16.h>
#include <mma.h>
// CUDA kernels for single-stream decoding.
//
// Decoding one token at a time means every matmul is a matrix-VECTOR product,
// which is memory-bound rather than compute-bound: the model's weights are read
// once per token and barely reused. On this hardware that is 757 GB/s (measured
// at a 135 W power limit) against 0.45 GB of f32 weights, so ~1674 tok/s is the
// ceiling regardless of how much arithmetic throughput the card has. These
// kernels are therefore written to maximise read bandwidth, not FLOPs.
//
// gemv reaches 723-770 GB/s, so it is already bandwidth-saturated.
//
// Every kernel has a scalar CPU twin in src/ops.rs. The CPU version is the
// reference; when the two disagree the GPU one is wrong.
//
// ---------------------------------------------------------------------------
// Why per-token scalars come from device memory
//
// A captured CUDA graph freezes its kernels' arguments. Passing `pos`, the
// token id, or the cache slot offset by value would bake step 0's values into
// the graph, and every replay would recompute the same token.
//
// So every value that changes per step lives in a small device buffer that
// kernels index into. One host-to-device copy of 5 ints happens before each
// replay -- one transfer per token, replacing ~170 kernel launches. Slot 0 is
// permanently zero, so call sites needing a constant offset point there and
// take the same code path rather than needing a second kernel variant.
//
//   params[0] = 0        (always)
//   params[1] = token id
//   params[2] = position
//   params[3] = pos + 1  (positions visible to attention)
//   params[4] = pos * n_kv_head * head_dim   (slot within a layer)
// ---------------------------------------------------------------------------

// No includes: NVRTC compiles without a header search path, and nothing here
// needs one. INFINITY and the intrinsics below are NVRTC builtins.
#define WARP_SIZE 32
#define NEG_INF __int_as_float(0xff800000)

// Sum a value across the warp. __shfl_down_sync is register-to-register, with
// no shared memory or barrier cost.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Sum across a whole block, via one warp-reduce then a second across warps.
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float partial[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();

    const int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    v = (threadIdx.x < n_warps) ? partial[lane] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;
}

// y = W * x, W row-major [rows, cols].
//
// One block per output row. Threads stride across the row so consecutive
// threads read consecutive addresses, which is what lets the memory controller
// coalesce each access into full cache lines -- the single most important
// property for a bandwidth-bound kernel.
extern "C" __global__ void gemv_f32(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int rows,
    const int cols,
    const int* __restrict__ params,
    const int y_base,
    const int y_idx)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* w_row = w + (size_t)row * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += w_row[i] * x[i];
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) y[y_base + params[y_idx] + row] = acc;
}

// Same, reading four floats at a time. float4 issues one 16-byte load per
// thread instead of four 4-byte loads, which reduces instruction count and
// helps the memory pipeline stay saturated. Requires cols % 4 == 0.
extern "C" __global__ void gemv_f32_vec4(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int rows,
    const int cols,
    const int* __restrict__ params,
    const int y_base,
    const int y_idx)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float4* w_row = reinterpret_cast<const float4*>(w + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        const float4 a = w_row[i];
        const float4 b = x4[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) y[y_base + params[y_idx] + row] = acc;
}

// out = x * rsqrt(mean(x^2) + eps) * weight
//
// One block for the whole vector: the reduction needs every element, and at
// d_model <= 4096 a single block covers it without a second kernel launch.
extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int n,
    const float eps)
{
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = x[i];
        acc += v * v;
    }
    acc = block_reduce_sum(acc);

    // Broadcast the scale: only thread 0 holds the reduced value.
    __shared__ float scale;
    if (threadIdx.x == 0) scale = rsqrtf(acc / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * weight[i];
    }
}

// Rotary embedding, in place, over n_heads contiguous head vectors.
//
// Uses the half-split convention: element i pairs with i + head_dim/2. The
// other common convention pairs adjacent elements and produces a model that
// runs and generates fluent nonsense.
extern "C" __global__ void rope_f32(
    float* __restrict__ v,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int n_heads,
    const int head_dim,
    const int* __restrict__ params,
    const int pos_idx,
    const int v_base,
    const int v_idx)
{
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half) return;

    const int pos = params[pos_idx];
    const int head = idx / half;
    const int i = idx % half;

    const float c = cos_table[pos * half + i];
    const float s = sin_table[pos * half + i];

    float* h = v + v_base + params[v_idx] + head * head_dim;
    const float lo = h[i];
    const float hi = h[i + half];
    h[i] = lo * c - hi * s;
    h[i + half] = lo * s + hi * c;
}

// dst += src, elementwise. The residual connection.
extern "C" __global__ void add_inplace_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// gate = silu(gate) * up, elementwise. The SwiGLU nonlinearity.
extern "C" __global__ void silu_mul_f32(
    float* __restrict__ gate,
    const float* __restrict__ up,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// Softmax over the first `n` elements of `x`, in place, one block.
extern "C" __global__ void softmax_f32(float* __restrict__ x, const int n) {
    // Max first, so exponentials cannot overflow at long context.
    float local_max = NEG_INF;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    __shared__ float smax;
    {
        float v = local_max;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        }
        __shared__ float partial[WARP_SIZE];
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp = threadIdx.x / WARP_SIZE;
        if (lane == 0) partial[warp] = v;
        __syncthreads();
        const int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        v = (threadIdx.x < n_warps) ? partial[lane] : NEG_INF;
        if (warp == 0) {
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
            }
            if (threadIdx.x == 0) smax = v;
        }
        __syncthreads();
    }

    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float e = __expf(x[i] - smax);
        x[i] = e;
        acc += e;
    }
    acc = block_reduce_sum(acc);

    __shared__ float inv_sum;
    if (threadIdx.x == 0) inv_sum = 1.0f / acc;
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] *= inv_sum;
    }
}

// One row of the embedding table into the residual stream.
extern "C" __global__ void embed_f32(
    const float* __restrict__ table,
    float* __restrict__ out,
    const int* __restrict__ params,
    const int token_idx,
    const int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) out[i] = table[(size_t)params[token_idx] * d + i];
}

// Fused single-token attention over the KV cache.
//
// One block per query head, computing scores, softmax and the weighted sum of
// values without leaving the block. Doing it as three separate kernels would
// mean three launches per head per layer -- at 12 heads x 12 layers that is 432
// launches per token, and launch overhead alone would dominate decoding a model
// this size.
//
// Shared memory is sized for the maximum context at launch, not the current
// sequence length: a captured graph fixes its shared-memory allocation, so
// sizing it to the current length would invalidate the graph as the sequence
// grows.
extern "C" __global__ void attention_decode_f32(
    const float* __restrict__ q,        // [n_head * head_dim]
    const float* __restrict__ k_cache,  // [capacity][n_kv_head * head_dim]
    const float* __restrict__ v_cache,
    float* __restrict__ out,            // [n_head * head_dim]
    const int n_head,
    const int n_kv_head,
    const int head_dim,
    const int* __restrict__ params,
    const int seq_idx,                  // positions 0..seq_len-1, current included
    const int cache_stride,             // n_kv_head * head_dim
    const int max_seq)                  // scores capacity, so partials can follow
{
    extern __shared__ float scores[];

    const int h = blockIdx.x;
    if (h >= n_head) return;

    const int seq_len = params[seq_idx];

    // Query head h reads KV head h / n_rep, matching repeat_interleave's
    // mapping of KV head j to query heads [j*n_rep, (j+1)*n_rep).
    const int n_rep = n_head / n_kv_head;
    const int kv_h = h / n_rep;
    const float* qh = q + h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    // One warp per cached position, lanes striding across the key vector.
    //
    // The obvious version gives each THREAD a position and walks that key
    // sequentially -- but then neighbouring threads read addresses
    // cache_stride floats apart (768 bytes here), so every lane issues its own
    // memory transaction and the warp touches 32 scattered lines instead of a
    // few contiguous ones. Profiling put attention at 32% of decode, the
    // largest single stage, with this loop the cause. Lane-strided reads are
    // contiguous and coalesce.
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;

    for (int j = warp_id; j < seq_len; j += n_warps) {
        const float* kh = k_cache + (size_t)j * cache_stride + kv_h * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += WARP_SIZE) partial += qh[d] * kh[d];
        const float dot = warp_reduce_sum(partial);
        if (lane == 0) scores[j] = dot * scale;
    }
    __syncthreads();

    // Softmax across the block, max-subtracted so long contexts cannot overflow.
    __shared__ float smax;
    __shared__ float ssum;
    {
        float m = NEG_INF;
        for (int j = threadIdx.x; j < seq_len; j += blockDim.x) m = fmaxf(m, scores[j]);
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
        __shared__ float warp_max[WARP_SIZE];
        if (lane == 0) warp_max[warp_id] = m;
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = NEG_INF;
            for (int i = 0; i < n_warps; ++i) best = fmaxf(best, warp_max[i]);
            smax = best;
        }
        __syncthreads();
    }

    float local = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float e = __expf(scores[j] - smax);
        scores[j] = e;
        local += e;
    }
    local = block_reduce_sum(local);
    if (threadIdx.x == 0) ssum = 1.0f / local;
    __syncthreads();

    // Weighted sum of values.
    //
    // Reads here were already coalesced -- consecutive threads take consecutive
    // d -- but with head_dim 64 and 256 threads per block, three quarters of
    // the block sat idle. Each warp now sums a slice of the positions into its
    // own partial vector in shared memory, and the partials are combined at the
    // end. The reciprocal is applied once rather than per term.
    float* partials = scores + max_seq;   // [n_warps][head_dim]
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float acc = 0.0f;
        for (int j = warp_id; j < seq_len; j += n_warps) {
            acc += scores[j] * v_cache[(size_t)j * cache_stride + kv_h * head_dim + d];
        }
        partials[warp_id * head_dim + d] = acc;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int w = 0; w < n_warps; ++w) acc += partials[w * head_dim + d];
        out[h * head_dim + d] = acc * ssum;
    }
}

// ---------------------------------------------------------------------------
// int8 weight-only quantisation
//
// Decode is bandwidth-bound, so the win here is not arithmetic: it is reading
// one byte per weight instead of four. Activations stay f32 because they are a
// negligible fraction of the bytes moved and quantising them would cost
// accuracy for nothing.
//
// Symmetric per-output-row scales: q[r][c] = round(w[r][c] / scale[r]), with
// scale[r] = max|w[r][:]| / 127. Per-row rather than per-tensor because a
// single scale across a whole matrix is set by its largest outlier, which
// crushes the resolution of every other row.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemv_i8_f32(
    const signed char* __restrict__ w,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int rows,
    const int cols,
    const int* __restrict__ params,
    const int y_base,
    const int y_idx,
    const int accumulate)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const signed char* w_row = w + (size_t)row * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += (float)w_row[i] * x[i];
    }

    acc = block_reduce_sum(acc);
    // The row scale is applied once to the reduced value, not per term.
    if (threadIdx.x == 0) {
        const int o = y_base + params[y_idx] + row;
        y[o] = accumulate ? y[o] + acc * scales[row] : acc * scales[row];
    }
}

// Same, reading four weights per load.
//
// This matters more for int8 than it did for f32. The f32 kernel already
// saturates memory bandwidth, so wider loads bought nothing; at one byte per
// weight the kernel moves 4x less data and shifts toward being limited by
// issue rate instead, where a 4-byte load beats four 1-byte loads.
// Requires cols % 4 == 0.
extern "C" __global__ void gemv_i8_f32_vec4(
    const signed char* __restrict__ w,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int rows,
    const int cols,
    const int* __restrict__ params,
    const int y_base,
    const int y_idx,
    const int accumulate)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const char4* w_row = reinterpret_cast<const char4*>(w + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        const char4 a = w_row[i];
        const float4 b = x4[i];
        acc += (float)a.x * b.x + (float)a.y * b.y
             + (float)a.z * b.z + (float)a.w * b.w;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) {
        const int o = y_base + params[y_idx] + row;
        y[o] = accumulate ? y[o] + acc * scales[row] : acc * scales[row];
    }
}

// Embedding lookup from an int8 table, applying that row's scale.
//
// tok_emb is tied to lm_head, so one quantised tensor serves both: the output
// projection reads it as a matrix with per-row scales, and this reads a single
// row and rescales it. Keeping a separate f32 copy for the lookup would add
// 154 MB for the 120M model.
extern "C" __global__ void embed_i8(
    const signed char* __restrict__ table,
    const float* __restrict__ scales,
    float* __restrict__ out,
    const int* __restrict__ params,
    const int token_idx,
    const int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = params[token_idx];
    if (i < d) out[i] = (float)table[(size_t)token * d + i] * scales[token];
}

// ---------------------------------------------------------------------------
// Warp-per-row GEMV
//
// The block-per-row kernels above give every output row a whole block, which
// suits a long row and wastes a short one. gate_proj is 2048x768: as int8 with
// char4 loads that is 192 elements against 256 threads, so a quarter of the
// block is idle, each active thread issues exactly one load, and a full
// eight-warp block reduction runs to combine them. Profiling put the MLP at
// 28.5% of decode with this the cause.
//
// Here each WARP owns a row. Lanes stride across it and reduce with __shfl
// alone -- no shared memory, no __syncthreads, and no idle warps, since one
// block covers blockDim.x/32 rows at once.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemv_i8_f32_warp(
    const signed char* __restrict__ w,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int rows,
    const int cols,
    const int* __restrict__ params,
    const int y_base,
    const int y_idx,
    const int accumulate)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    const char4* w_row = reinterpret_cast<const char4*>(w + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;

    float acc = 0.0f;
    for (int i = lane; i < cols4; i += WARP_SIZE) {
        const char4 a = w_row[i];
        const float4 b = x4[i];
        acc += (float)a.x * b.x + (float)a.y * b.y
             + (float)a.z * b.z + (float)a.w * b.w;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const int o = y_base + params[y_idx] + row;
        // accumulate folds the residual add into this write: o_proj and
        // down_proj land directly in the residual stream, removing one kernel
        // per site. The branch is uniform across the warp, so it costs nothing.
        y[o] = accumulate ? y[o] + acc * scales[row] : acc * scales[row];
    }
}

extern "C" __global__ void gemv_f32_warp(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    const int rows,
    const int cols,
    const int* __restrict__ params,
    const int y_base,
    const int y_idx,
    const int accumulate)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    const float4* w_row = reinterpret_cast<const float4*>(w + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;

    float acc = 0.0f;
    for (int i = lane; i < cols4; i += WARP_SIZE) {
        const float4 a = w_row[i];
        const float4 b = x4[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        const int o = y_base + params[y_idx] + row;
        y[o] = accumulate ? y[o] + acc : acc;
    }
}

// ---------------------------------------------------------------------------
// Fused SwiGLU: silu(gate . x) * (up . x)
//
// Unfused this is three kernels writing three `hidden`-sized buffers: gate, up,
// then the elementwise product. Fusing removes two kernel dispatches per layer
// -- 24 per token at 12 layers -- and two round trips through a 2048-element
// buffer.
//
// CUDA graphs removed the CPU cost of launching, but each kernel still pays
// GPU-side dispatch, so kernel COUNT continues to matter. One warp per output
// row, as with the standalone GEMV, since a 768-column row starves a block.
// ---------------------------------------------------------------------------

extern "C" __global__ void mlp_swiglu_i8_warp(
    const signed char* __restrict__ gate_w,
    const float* __restrict__ gate_scales,
    const signed char* __restrict__ up_w,
    const float* __restrict__ up_scales,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows,
    const int cols)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp;
    if (row >= rows) return;

    const char4* g_row = reinterpret_cast<const char4*>(gate_w + (size_t)row * cols);
    const char4* u_row = reinterpret_cast<const char4*>(up_w + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;

    float g = 0.0f;
    float u = 0.0f;
    for (int i = lane; i < cols4; i += WARP_SIZE) {
        const float4 b = x4[i];
        const char4 a = g_row[i];
        const char4 c = u_row[i];
        g += (float)a.x * b.x + (float)a.y * b.y + (float)a.z * b.z + (float)a.w * b.w;
        u += (float)c.x * b.x + (float)c.y * b.y + (float)c.z * b.z + (float)c.w * b.w;
    }

    g = warp_reduce_sum(g);
    u = warp_reduce_sum(u);

    if (lane == 0) {
        const float gs = g * gate_scales[row];
        const float us = u * up_scales[row];
        out[row] = (gs / (1.0f + __expf(-gs))) * us;
    }
}

extern "C" __global__ void mlp_swiglu_f32(
    const float* __restrict__ gate_w,
    const float* __restrict__ up_w,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int rows,
    const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float4* g_row = reinterpret_cast<const float4*>(gate_w + (size_t)row * cols);
    const float4* u_row = reinterpret_cast<const float4*>(up_w + (size_t)row * cols);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int cols4 = cols / 4;

    float g = 0.0f;
    float u = 0.0f;
    for (int i = threadIdx.x; i < cols4; i += blockDim.x) {
        const float4 b = x4[i];
        const float4 a = g_row[i];
        const float4 c = u_row[i];
        g += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        u += c.x * b.x + c.y * b.y + c.z * b.z + c.w * b.w;
    }

    g = block_reduce_sum(g);
    u = block_reduce_sum(u);

    if (threadIdx.x == 0) {
        out[row] = (g / (1.0f + __expf(-g))) * u;
    }
}

// ---------------------------------------------------------------------------
// Split-position attention (flash-decoding)
//
// The single-block-per-head kernel above is correct and coalesced, and still
// took 38.8% of decode. The reason is not the kernel body: it launches one
// block per head, so 12 blocks, on a GPU with dozens of SMs. At position 256 it
// reads ~4.7 MB of KV cache per token, which at 757 GB/s should take ~6 us; it
// took 427. Most of the machine was idle.
//
// This splits the sequence too: grid is (n_head, n_chunks), each block reduces
// one chunk of positions into a partial softmax, and a second kernel combines
// the partials. Standard online-softmax algebra -- each chunk records its local
// max m, the sum of exp(score - m), and the unnormalised weighted value sum, and
// the combine rescales each by exp(m - global_max).
//
// n_chunks is fixed at CAPACITY, not at the current sequence length: a captured
// CUDA graph freezes grid dimensions, so a length-dependent grid would make the
// graph invalid as the sequence grows. Chunks past the end write a neutral
// contribution (m = -inf, l = 0) and exit.
// ---------------------------------------------------------------------------

extern "C" __global__ void attention_partial_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ partial_o,   // [n_head][n_chunks][head_dim]
    float* __restrict__ partial_m,   // [n_head][n_chunks]
    float* __restrict__ partial_l,   // [n_head][n_chunks]
    const int n_head,
    const int n_kv_head,
    const int head_dim,
    const int* __restrict__ params,
    const int seq_idx,
    const int cache_stride,
    const int chunk_size)
{
    extern __shared__ float shared[];

    const int h = blockIdx.x;
    const int chunk = blockIdx.y;
    if (h >= n_head) return;

    const int n_chunks = gridDim.y;
    const int slot = h * n_chunks + chunk;
    const int seq_len = params[seq_idx];
    const int start = chunk * chunk_size;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;

    // Past the end of the sequence: contribute nothing. exp(-inf - M) is 0 in
    // the combine, so these vanish without a special case there.
    if (start >= seq_len) {
        if (threadIdx.x == 0) {
            partial_m[slot] = NEG_INF;
            partial_l[slot] = 0.0f;
        }
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            partial_o[(size_t)slot * head_dim + d] = 0.0f;
        }
        return;
    }

    const int end = min(start + chunk_size, seq_len);
    const int len = end - start;

    const int n_rep = n_head / n_kv_head;
    const int kv_h = h / n_rep;
    const float* qh = q + h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float* scores = shared;                    // [chunk_size]
    float* partials = shared + chunk_size;     // [n_warps][head_dim]

    // Scores: one warp per position, lanes striding the key so reads coalesce.
    for (int j = warp_id; j < len; j += n_warps) {
        const float* kh = k_cache + (size_t)(start + j) * cache_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += WARP_SIZE) dot += qh[d] * kh[d];
        dot = warp_reduce_sum(dot);
        if (lane == 0) scores[j] = dot * scale;
    }
    __syncthreads();

    // Local max over this chunk only.
    __shared__ float smax;
    {
        float m = NEG_INF;
        for (int j = threadIdx.x; j < len; j += blockDim.x) m = fmaxf(m, scores[j]);
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
        __shared__ float warp_max[WARP_SIZE];
        if (lane == 0) warp_max[warp_id] = m;
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = NEG_INF;
            for (int i = 0; i < n_warps; ++i) best = fmaxf(best, warp_max[i]);
            smax = best;
        }
        __syncthreads();
    }

    float local = 0.0f;
    for (int j = threadIdx.x; j < len; j += blockDim.x) {
        const float e = __expf(scores[j] - smax);
        scores[j] = e;
        local += e;
    }
    local = block_reduce_sum(local);

    if (threadIdx.x == 0) {
        partial_m[slot] = smax;
        partial_l[slot] = local;   // unnormalised: the combine divides once
    }

    // Unnormalised weighted value sum for this chunk.
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float acc = 0.0f;
        for (int j = warp_id; j < len; j += n_warps) {
            acc += scores[j] * v_cache[(size_t)(start + j) * cache_stride
                                       + kv_h * head_dim + d];
        }
        partials[warp_id * head_dim + d] = acc;
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int w = 0; w < n_warps; ++w) acc += partials[w * head_dim + d];
        partial_o[(size_t)slot * head_dim + d] = acc;
    }
}

// Combine per-chunk partial softmaxes into the final attention output.
//
// One block per head. Rescales each chunk by exp(m_chunk - m_global), which is
// what makes splitting the softmax exact rather than approximate.
extern "C" __global__ void attention_combine_f32(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    float* __restrict__ out,
    const int n_head,
    const int head_dim,
    const int n_chunks)
{
    const int h = blockIdx.x;
    if (h >= n_head) return;

    __shared__ float m_global;
    __shared__ float l_global;

    if (threadIdx.x == 0) {
        float m = NEG_INF;
        for (int c = 0; c < n_chunks; ++c) {
            m = fmaxf(m, partial_m[h * n_chunks + c]);
        }
        m_global = m;

        float l = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            l += partial_l[h * n_chunks + c] * __expf(partial_m[h * n_chunks + c] - m);
        }
        l_global = l;
    }
    __syncthreads();

    const float inv_l = 1.0f / l_global;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int slot = h * n_chunks + c;
            const float w = __expf(partial_m[slot] - m_global);
            acc += partial_o[(size_t)slot * head_dim + d] * w;
        }
        out[h * head_dim + d] = acc * inv_l;
    }
}

// ===========================================================================
// Batched prefill
//
// Decode and prefill are different problems. Decode has one token in flight, so
// every matmul is a matrix-VECTOR product: no reuse, bandwidth-bound, and the
// kernels above are built for it.
//
// Prefill has the whole prompt at once and no sequential dependency between its
// tokens, so the same weights serve every row. That makes it matrix-MATRIX:
// compute-bound, with arithmetic intensity that grows with prompt length.
//
// Running prefill through the decode path -- one token at a time -- costs a
// 512-token prompt ~77,000 kernel launches and measured 888 tok/s against
// llama.cpp's 92,071. That is the entire reason these kernels exist.
// ===========================================================================

#define TILE 16

// C[M,N] = A[M,K] * W[N,K]^T, with a per-output-row scale on W.
//
// W is stored row-major as [N, K], the same layout the GEMV kernels use, so no
// repacking is needed between prefill and decode.
//
// Both operands are staged through shared memory. The B tile is loaded with
// threadIdx.x indexing K -- not N -- so consecutive lanes read consecutive
// addresses within a weight row; indexing N there would stride by K per lane
// and lose coalescing entirely.
extern "C" __global__ void gemm_i8_f32(
    const signed char* __restrict__ w,
    const float* __restrict__ scales,
    const float* __restrict__ a,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K,
    const int accumulate)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;   // token
    const int col = blockIdx.x * TILE + threadIdx.x;   // output channel

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int ak = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? a[(size_t)row * K + ak] : 0.0f;

        // Transposed store: Bs[k][n], loaded coalesced along k.
        const int bn = blockIdx.x * TILE + threadIdx.y;
        const int bk = t * TILE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] =
            (bn < N && bk < K) ? (float)w[(size_t)bn * K + bk] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        const size_t o = (size_t)row * N + col;
        const float v = acc * scales[col];
        c[o] = accumulate ? c[o] + v : v;
    }
}

extern "C" __global__ void gemm_f32(
    const float* __restrict__ w,
    const float* __restrict__ a,
    float* __restrict__ c,
    const int M,
    const int N,
    const int K,
    const int accumulate)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        const int ak = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (row < M && ak < K) ? a[(size_t)row * K + ak] : 0.0f;

        const int bn = blockIdx.x * TILE + threadIdx.y;
        const int bk = t * TILE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] =
            (bn < N && bk < K) ? w[(size_t)bn * K + bk] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; ++i) {
            acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        const size_t o = (size_t)row * N + col;
        c[o] = accumulate ? c[o] + acc : acc;
    }
}

// ---------------------------------------------------------------------------
// Row-wise operations over a whole prompt
//
// One block per token row, so a T-token prompt costs one launch instead of T.
// ---------------------------------------------------------------------------

extern "C" __global__ void rmsnorm_batch_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    const int rows,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* xr = x + (size_t)row * n;
    float* outr = out + (size_t)row * n;

    float acc = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        acc += v * v;
    }
    acc = block_reduce_sum(acc);

    __shared__ float scale;
    if (threadIdx.x == 0) scale = rsqrtf(acc / (float)n + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        outr[i] = xr[i] * scale * weight[i];
    }
}

// Rotary embedding over every position of the prompt at once.
//
// `pos_offset` lets a prompt be prefilled into a cache that already holds
// earlier tokens.
extern "C" __global__ void rope_batch_f32(
    float* __restrict__ v,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int rows,
    const int n_heads,
    const int head_dim,
    const int row_stride,
    const int pos_offset)
{
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int per_row = n_heads * half;
    if (idx >= rows * per_row) return;

    const int row = idx / per_row;
    const int rem = idx % per_row;
    const int head = rem / half;
    const int i = rem % half;

    const int pos = pos_offset + row;
    const float c = cos_table[pos * half + i];
    const float sn = sin_table[pos * half + i];

    float* h = v + (size_t)row * row_stride + head * head_dim;
    const float lo = h[i];
    const float hi = h[i + half];
    h[i] = lo * c - hi * sn;
    h[i + half] = lo * sn + hi * c;
}

extern "C" __global__ void swiglu_batch_f32(
    float* __restrict__ gate,
    const float* __restrict__ up,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + __expf(-g))) * up[i];
}

extern "C" __global__ void embed_batch_f32(
    const float* __restrict__ table,
    const int* __restrict__ tokens,
    float* __restrict__ out,
    const int rows,
    const int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    out[i] = table[(size_t)tokens[i / d] * d + (i % d)];
}

extern "C" __global__ void embed_batch_i8(
    const signed char* __restrict__ table,
    const float* __restrict__ scales,
    const int* __restrict__ tokens,
    float* __restrict__ out,
    const int rows,
    const int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    const int tok = tokens[i / d];
    out[i] = (float)table[(size_t)tok * d + (i % d)] * scales[tok];
}

// Copy computed K/V rows into the layer's cache region.
//
// Prefill produces K and V as dense [T, kv_dim] matrices; the cache is indexed
// by absolute position, so this places them at the right offset.
extern "C" __global__ void cache_store_f32(
    const float* __restrict__ src,
    float* __restrict__ cache,
    const int rows,
    const int kv_dim,
    const int layer_base,
    const int pos_offset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kv_dim) return;
    const int row = i / kv_dim;
    const int col = i % kv_dim;
    cache[layer_base + (size_t)(pos_offset + row) * kv_dim + col] = src[i];
}

// Causal attention over a whole prompt.
//
// One block per (query position, head). Each block attends over positions
// 0..=query, reading K and V from the cache the projections just filled, so
// prefill and decode share one cache layout.
extern "C" __global__ void attention_prefill_f32(
    const float* __restrict__ q,        // [T, n_head * head_dim]
    const float* __restrict__ k_cache,  // layer region, [capacity][kv_dim]
    const float* __restrict__ v_cache,
    float* __restrict__ out,            // [T, n_head * head_dim]
    const int n_head,
    const int n_kv_head,
    const int head_dim,
    const int cache_stride,
    const int pos_offset)
{
    extern __shared__ float scores[];

    const int h = blockIdx.x;
    const int row = blockIdx.y;
    if (h >= n_head) return;

    const int seq_len = pos_offset + row + 1;   // causal: 0..=this position
    const int n_rep = n_head / n_kv_head;
    const int kv_h = h / n_rep;
    const float* qh = q + (size_t)row * n_head * head_dim + h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;

    for (int j = warp_id; j < seq_len; j += n_warps) {
        const float* kh = k_cache + (size_t)j * cache_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = lane; d < head_dim; d += WARP_SIZE) dot += qh[d] * kh[d];
        dot = warp_reduce_sum(dot);
        if (lane == 0) scores[j] = dot * scale;
    }
    __syncthreads();

    __shared__ float smax;
    __shared__ float ssum;
    {
        float m = NEG_INF;
        for (int j = threadIdx.x; j < seq_len; j += blockDim.x) m = fmaxf(m, scores[j]);
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
        __shared__ float warp_max[WARP_SIZE];
        if (lane == 0) warp_max[warp_id] = m;
        __syncthreads();
        if (threadIdx.x == 0) {
            float best = NEG_INF;
            for (int i = 0; i < n_warps; ++i) best = fmaxf(best, warp_max[i]);
            smax = best;
        }
        __syncthreads();
    }

    float local = 0.0f;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float e = __expf(scores[j] - smax);
        scores[j] = e;
        local += e;
    }
    local = block_reduce_sum(local);
    if (threadIdx.x == 0) ssum = 1.0f / local;
    __syncthreads();

    float* dst = out + (size_t)row * n_head * head_dim + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            acc += scores[j] * v_cache[(size_t)j * cache_stride + kv_h * head_dim + d];
        }
        dst[d] = acc * ssum;
    }
}

// ===========================================================================
// Tensor-core GEMM for prefill
//
// The tiled GEMM above sustains ~3.4 TFLOP/s: 20% of this GPU's measured FP32
// peak and 4.5% of its BF16 tensor-core peak. Prefill is compute-bound, so that
// gap is the whole remaining distance to llama.cpp, which dispatches to cuBLAS.
//
// Two design choices worth stating:
//
// Weights stay int8 in memory and are converted to half *in shared memory*.
// Keeping a half copy of the model would cost 226 MB and give back most of what
// int8 quantisation bought. int8 -> half is exact (the whole int8 range is
// representable), so this conversion loses nothing.
//
// Activations are f32 and convert to half on load, which does lose mantissa
// bits. That is a real numerical change, not a free win, and the effect on
// cross-entropy is measured rather than assumed.
//
// One block computes a 16 x 64 tile of C: four warps, each owning one 16x16
// fragment, sharing the single A tile between them.
// ===========================================================================

// ===========================================================================
// Tensor-core GEMM for prefill, in two tile sizes
//
// Both are the same algorithm with different block tiles, generated from one
// template so they cannot drift apart. The choice between them is made per
// launch in gpu.rs, because it is not a constant:
//
//   seq   16x64      64x64      (int8 prefill, tok/s, 5 interleaved trials)
//   128   20325       9950
//   256   32763      18749
//   512   39731      30033
//  1024   30779      32170
//
// The larger tile has strictly better arithmetic intensity -- it produces four
// times the output per block while only doubling the loads, ~64 FLOP per
// element loaded against ~25. That reasoning is correct and, below seq 1024,
// irrelevant: M is the sequence length, so a 64-row tile at seq=128 with
// n_embd=768 launches ceil(128/64) * ceil(768/64) = 24 blocks and leaves most
// of the GPU idle. The 16-row tile launches 96. Intensity only starts paying
// once there is enough work to fill the machine, which is why the big tile wins
// at 1024 and loses by 2x at 128.
//
// Two design choices shared by both:
//
// Weights stay int8 in memory and convert to half *in shared memory*. Keeping a
// half copy of the model would cost 226 MB and give back most of what int8
// quantisation bought. int8 -> half is exact, so that conversion loses nothing.
//
// Activations are f32 and convert to half on load, which does lose mantissa
// bits. That is a real numerical change, not a free win. Measured: it moves
// held-out cross-entropy by ~1e-5, about thirty times less than int8
// quantisation's own cost. See `gpu-eval --prefill-ctx`.
// ===========================================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TK 32
// Pad the shared leading dimension past the K step. Without it, column c of
// every row lands in the same bank group and the fragment loads serialise.
#define LD (TK + 8)

// TM x TN block tile; warps laid out WR x WC, each owning (TM/WR) x (TN/WC).
template <int TM, int TN, int WR, int WC>
__device__ __forceinline__ void gemm_wmma_impl(
    const signed char* __restrict__ w8,   // one of w8/wf is null
    const float* __restrict__ wf,
    const float* __restrict__ scales,     // [N], null when wf is used
    const float* __restrict__ a,          // [M, K]
    float* __restrict__ c,                // [M, N]
    const int M, const int N, const int K, const int accumulate)
{
    using namespace nvcuda;

    constexpr int FM = TM / (WR * WMMA_M);   // fragments per warp, row-wise
    constexpr int FN = TN / (WC * WMMA_N);

    __shared__ half As[TM * LD];
    __shared__ half Bs[TN * LD];
    __shared__ float Cs[WR * WC][WMMA_M * WMMA_N];

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int wrow = (warp / WC) * (FM * WMMA_M);   // warp origin in the tile
    const int wcol = (warp % WC) * (FN * WMMA_N);
    const int tile_m = blockIdx.y * TM;
    const int tile_n = blockIdx.x * TN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
    for (int i = 0; i < FM; ++i)
        for (int j = 0; j < FN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int kt = 0; kt < K; kt += TK) {
        // Both tiles are loaded by the whole block. Consecutive threads take
        // consecutive k, so the reads coalesce along the contiguous axis of
        // both A [M, K] and W [N, K].
        for (int i = threadIdx.x; i < TM * TK; i += blockDim.x) {
            const int m = i / TK, k = i % TK;
            const int gm = tile_m + m, gk = kt + k;
            As[m * LD + k] = (gm < M && gk < K)
                ? __float2half(a[(size_t)gm * K + gk]) : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < TN * TK; i += blockDim.x) {
            const int n = i / TK, k = i % TK;
            const int gn = tile_n + n, gk = kt + k;
            const bool live = (gn < N && gk < K);
            const size_t o = (size_t)gn * K + gk;
            Bs[n * LD + k] = !live ? __float2half(0.0f)
                : (w8 ? __short2half_rn((short)w8[o]) : __float2half(wf[o]));
        }
        __syncthreads();

        for (int kk = 0; kk < TK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> af[FM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bf[FN];
            for (int i = 0; i < FM; ++i)
                wmma::load_matrix_sync(af[i], &As[(wrow + i * WMMA_M) * LD + kk], LD);
            // col_major reads element (k, n) at n*ld + k, which is exactly how
            // W's row-major [N, K] rows sit in Bs -- no transpose needed.
            for (int j = 0; j < FN; ++j)
                wmma::load_matrix_sync(bf[j], &Bs[(wcol + j * WMMA_N) * LD + kk], LD);
            for (int i = 0; i < FM; ++i)
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Per-output-channel scale applied once, on the way out.
    for (int i = 0; i < FM; ++i) {
        for (int j = 0; j < FN; ++j) {
            wmma::store_matrix_sync(Cs[warp], acc[i][j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < WMMA_M * WMMA_N; e += WARP_SIZE) {
                const int gm = tile_m + wrow + i * WMMA_M + e / WMMA_N;
                const int gn = tile_n + wcol + j * WMMA_N + e % WMMA_N;
                if (gm < M && gn < N) {
                    const size_t o = (size_t)gm * N + gn;
                    const float v = scales ? Cs[warp][e] * scales[gn] : Cs[warp][e];
                    c[o] = accumulate ? c[o] + v : v;
                }
            }
            __syncwarp();
        }
    }
}

extern "C" __global__ void gemm_i8_wmma(
    const signed char* __restrict__ w, const float* __restrict__ scales,
    const float* __restrict__ a, float* __restrict__ c,
    const int M, const int N, const int K, const int accumulate)
{
    gemm_wmma_impl<16, 64, 1, 4>(w, nullptr, scales, a, c, M, N, K, accumulate);
}

extern "C" __global__ void gemm_i8_wmma_big(
    const signed char* __restrict__ w, const float* __restrict__ scales,
    const float* __restrict__ a, float* __restrict__ c,
    const int M, const int N, const int K, const int accumulate)
{
    gemm_wmma_impl<64, 64, 2, 2>(w, nullptr, scales, a, c, M, N, K, accumulate);
}

extern "C" __global__ void gemm_f32_wmma(
    const float* __restrict__ w, const float* __restrict__ a,
    float* __restrict__ c,
    const int M, const int N, const int K, const int accumulate)
{
    gemm_wmma_impl<16, 64, 1, 4>(nullptr, w, nullptr, a, c, M, N, K, accumulate);
}

extern "C" __global__ void gemm_f32_wmma_big(
    const float* __restrict__ w, const float* __restrict__ a,
    float* __restrict__ c,
    const int M, const int N, const int K, const int accumulate)
{
    gemm_wmma_impl<64, 64, 2, 2>(nullptr, w, nullptr, a, c, M, N, K, accumulate);
}
