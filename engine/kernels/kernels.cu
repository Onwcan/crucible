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
// Row-wise argmax
//
// The scheduler needs one token id per request, but the logits row it scanned
// to get that is 50304 floats. At batch 16 that is 3.2 MB crossing PCIe every
// step so the host can find one integer. This does the reduction on the device
// and returns 4 bytes per request instead.
//
// Tie-breaking has to match the host implementation exactly, or generated text
// diverges on the first tie. The host does:
//
//     best = 0; for i: if v[i] > v[best] { best = i }
//
// which is a strict comparison, so the LOWEST index wins a tie, and NaN never
// displaces anything -- it can only win by starting at index 0. Stated without
// reference to iteration order, that is: the winner is the lowest index i such
// that no j satisfies v[j] > v[i]. That form is order-independent, which is what
// makes it safe to evaluate as a tree.
//
// Hence the merge rule below: take the challenger if it strictly dominates,
// otherwise keep whichever index is lower. Equal values and NaN comparisons
// both fall into "neither dominates", so both resolve by index, matching the
// host in every case including all-NaN rows.
// ===========================================================================

#define ARGMAX_NO_INDEX 0x7fffffff

__device__ __forceinline__ void argmax_merge(
    float& bv, int& bi, const float ov, const int oi)
{
    // (ov > bv)            challenger strictly dominates
    // (!(bv > ov) && ...)  neither dominates -- equal, or a NaN is involved --
    //                      so the lower index wins, exactly as the host's
    //                      strict > leaves the earlier element in place.
    if (ov > bv || (!(bv > ov) && oi < bi)) {
        bv = ov;
        bi = oi;
    }
}

// One block per row. Grid is (rows), so requests never interact.
extern "C" __global__ void argmax_rows_f32(
    const float* __restrict__ x,     // [rows, cols]
    int* __restrict__ out,           // [rows]
    const int rows,
    const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* r = x + (size_t)row * cols;

    // Identity loses to any real value, and carries the largest index so it
    // also loses every tie.
    float bv = NEG_INF;
    int bi = ARGMAX_NO_INDEX;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        argmax_merge(bv, bi, r[i], i);
    }

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffff, bv, off);
        const int oi = __shfl_down_sync(0xffffffff, bi, off);
        argmax_merge(bv, bi, ov, oi);
    }

    __shared__ float warp_val[WARP_SIZE];
    __shared__ int warp_idx[WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    if (lane == 0) {
        warp_val[warp] = bv;
        warp_idx[warp] = bi;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float fv = warp_val[0];
        int fi = warp_idx[0];
        for (int w = 1; w < n_warps; ++w) {
            argmax_merge(fv, fi, warp_val[w], warp_idx[w]);
        }
        out[row] = fi;
    }
}

// ===========================================================================
// Row-wise top-k extraction
//
// Device argmax fixed the greedy path: one id per request instead of a 201 KB
// logits row. A sampled request still needed the whole row, because top-k and
// the threshold walk ran on the host -- so a batch of 16 sampled requests moved
// 3.2 MB per step to use 16 * 40 * 4 bytes of it, and measured 3960 tok/s
// against greedy's 8358. This kernel closes that gap: it returns the k best
// (value, id) pairs per row, and the host samples from those.
//
// ---------------------------------------------------------------------------
// Canonical order
// ---------------------------------------------------------------------------
//
// The host walks the candidate list in order, so the order decides which token
// a given random threshold selects. Letting thread scheduling decide it would
// make generation non-reproducible. The order is therefore defined:
//
//     higher logit first; equal logits resolved to the LOWER token id
//
// Because token ids are unique, that is a strict total order -- no two
// candidates are ever tied -- so the top-k set has exactly one valid
// arrangement and the kernel cannot disagree with the host reference by
// accident. It is expressed as a single 64-bit key, descending:
//
//     [63:32] order-preserving map of the float
//     [31:0]  complement of the token id, so lower ids sort first
//
// The float map has to reproduce the host comparator exactly, which means two
// special cases: NaN sorts below everything including -inf (the host's cmp_desc
// treats NaN as smallest, and the argmax kernel never lets NaN win), and -0.0
// must key identically to +0.0 (IEEE says they are equal, so the host resolves
// them by id). Both are done on the bit pattern rather than with float
// comparisons, so --use_fast_math cannot change the answer.
//
// ---------------------------------------------------------------------------
// Why radix select rather than a heap or a sorting network
// ---------------------------------------------------------------------------
//
// Extracting the max k times is k passes over 50304 floats -- at k=40 that is
// 8 MB per row and it dominates the step. A per-thread top-k in registers needs
// k slots per thread to be correct in the worst case (one thread can own
// several of the global top-k), which does not fit. Radix select instead finds
// the exact threshold by counting: one histogram pass per digit, narrowing the
// key from the top down, then one pass to collect everything at or above it.
//
// The common case costs 4 passes: three digits resolve the 32-bit float key
// (11 + 11 + 10 bits), and by then the k-th value is unique so the index half
// of the key is never examined. The index digits exist for exact ties, where
// "the k lowest ids among equal logits" has to be selected rather than
// whichever ids a thread happened to see first. Only the first pass touches
// DRAM; the rest hit L2.
// ===========================================================================

// Candidate capacity per row. Requests asking for more than this fall back to
// the full-logit path on the host, which stays as the reference implementation.
// 128 covers the default top-k of 40 and every value anyone has reason to use;
// the cost of the choice is the D2H block, 8 bytes per slot per row, so at
// batch 16 the whole candidate transfer is 16 KB against 3.2 MB.
#define TOPK_MAX 128
// Wide blocks because the row scan is latency bound, not compute bound: one
// block owns a whole 50304-float row, and at 256 threads that is 197 dependent
// iterations per pass.
#define TOPK_THREADS 1024
// Threads taking part in the bucket scan, independent of block width. The scan
// finishes serially in one thread, so widening the block must not lengthen it.
#define TOPK_PARTS 256
// 2^11, the widest digit used. Buckets are scanned in TOPK_PARTS chunks, so
// every digit width must divide evenly: 2048/256 = 8 and 1024/256 = 4.
#define TOPK_BUCKETS 2048

// Order-preserving map from a float to an unsigned int.
//
// Bitwise throughout: no float comparison, so flush-to-zero and fast-math
// cannot move a value across a boundary. Key 0 is reserved for NaN and no
// other input can produce it -- that would need u == 0xFFFFFFFF, itself a NaN.
//
// Masks rather than branches, matching the host `value_key` instruction for
// instruction: the sign of a logit is a coin flip, and on the host the branchy
// version measured 3.7x slower over a 50304-entry row.
__device__ __forceinline__ unsigned int topk_value_key(float v)
{
    unsigned int u = __float_as_uint(v);
    u &= ~(unsigned int)(-(int)(u == 0x80000000u));        // -0.0 keys as +0.0
    const unsigned int key = u ^ (((unsigned int)((int)u >> 31)) | 0x80000000u);
    return key & (unsigned int)(-(int)((u & 0x7fffffffu) <= 0x7f800000u));
}

__device__ __forceinline__ unsigned long long topk_key64(float v, int i)
{
    return ((unsigned long long)topk_value_key(v) << 32)
         | (unsigned long long)(0xFFFFFFFFu - (unsigned int)i);
}

// One block per row; grid is (rows), so requests never interact.
//
// row_k <= 0 skips the row entirely, which is how greedy requests avoid paying
// for this. The launch stays in the decode graph unconditionally and reads
// row_k from a device buffer, so which rows sample -- and with what k -- can
// change every step without recapturing anything.
extern "C" __global__ void topk_rows_f32(
    const float* __restrict__ x,        // [rows, cols]
    const int* __restrict__ row_k,      // [rows], <= 0 to skip
    float* __restrict__ out_vals,       // [rows, TOPK_MAX]
    int* __restrict__ out_ids,          // [rows, TOPK_MAX]
    const int rows,
    const int cols)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    int k = row_k[row];
    if (k <= 0) return;
    if (k > cols) k = cols;
    if (k > TOPK_MAX) k = TOPK_MAX;

    const float* r = x + (size_t)row * cols;
    const int tid = threadIdx.x;

    __shared__ int hist[TOPK_BUCKETS];
    __shared__ int part[TOPK_PARTS];
    __shared__ int grp[WARP_SIZE];
    __shared__ unsigned long long s_bucket;
    __shared__ int s_above;
    __shared__ int s_done;
    __shared__ float cand_v[TOPK_MAX];
    __shared__ int cand_i[TOPK_MAX];
    __shared__ int cand_n;

    // Bits of the 64-bit key still undecided, narrowed one digit at a time.
    unsigned long long prefix = 0ULL;
    // Elements whose key is strictly above the current prefix. Always < k.
    int above = 0;
    int shift = 0;

    for (int d = 0; d < 6; ++d) {
        // Digits are 11/11/10 over each 32-bit half: the value key first, then
        // the index complement, which is only reached when logits tie exactly.
        const int half = d / 3, j = d % 3;
        const int width = (j == 2) ? 10 : 11;
        shift = (1 - half) * 32 + ((j == 0) ? 21 : (j == 1) ? 10 : 0);
        const int nb = 1 << width;
        const int above_shift = shift + width;
        // d == 0 covers the whole key, and shifting a 64-bit value by 64 is
        // undefined rather than zero.
        const bool match_all = (above_shift >= 64);

        __syncthreads();
        for (int b = tid; b < nb; b += TOPK_THREADS) hist[b] = 0;
        __syncthreads();

        for (int i = tid; i < cols; i += TOPK_THREADS) {
            const unsigned long long kk = topk_key64(r[i], i);
            if (match_all || (kk >> above_shift) == prefix) {
                atomicAdd(&hist[(int)((kk >> shift) & (unsigned long long)(nb - 1))], 1);
            }
        }
        __syncthreads();

        // Finding the boundary bucket is a suffix scan, and it ends in one
        // thread walking down until the running count reaches k. That walk is
        // serial and dependent, so its length is what matters: three levels
        // bring 2048 buckets down to at most 31 + 7 + 7 steps, where a flat
        // scan over the 256 chunk totals alone measured longer than the whole
        // histogram pass it was summarising.
        const int per = nb / TOPK_PARTS;
        if (tid < TOPK_PARTS) {
            int sum = 0;
            for (int b = 0; b < per; ++b) sum += hist[tid * per + b];
            part[tid] = sum;
        }
        __syncthreads();
        if (tid < WARP_SIZE) {
            const int wide = TOPK_PARTS / WARP_SIZE;
            int sum = 0;
            for (int b = 0; b < wide; ++b) sum += part[tid * wide + b];
            grp[tid] = sum;
        }
        __syncthreads();

        if (tid == 0) {
            const int wide = TOPK_PARTS / WARP_SIZE;
            int acc = above;
            int g = WARP_SIZE - 1;
            for (; g > 0; --g) {
                if (acc + grp[g] >= k) break;
                acc += grp[g];
            }
            int chunk = (g + 1) * wide - 1;
            for (; chunk > g * wide; --chunk) {
                if (acc + part[chunk] >= k) break;
                acc += part[chunk];
            }
            int b = (chunk + 1) * per - 1;
            for (; b > chunk * per; --b) {
                if (acc + hist[b] >= k) break;
                acc += hist[b];
            }
            s_above = acc;
            s_bucket = (unsigned long long)b;
            // Done when the boundary bucket contributes exactly the shortfall.
            // At the last digit the key is fully determined and unique, so the
            // bucket holds one element and this always holds; asserting it
            // there keeps the loop terminating on a defined threshold.
            s_done = (acc + hist[b] == k || d == 5) ? 1 : 0;
        }
        __syncthreads();

        above = s_above;
        prefix = (prefix << width) | s_bucket;
        if (s_done) break;
    }

    // Everything at or above the threshold, which is now exactly k elements.
    const unsigned long long thresh = prefix << shift;
    if (tid == 0) cand_n = 0;
    __syncthreads();
    for (int i = tid; i < cols; i += TOPK_THREADS) {
        if (topk_key64(r[i], i) >= thresh) {
            const int p = atomicAdd(&cand_n, 1);
            if (p < TOPK_MAX) {
                cand_v[p] = r[i];
                cand_i[p] = i;
            }
        }
    }
    __syncthreads();
    const int n_cand = cand_n < TOPK_MAX ? cand_n : TOPK_MAX;

    // Collection order is whatever the atomics produced, so rank each candidate
    // against the others. Keys are unique, so ranks are a permutation and this
    // writes the canonical order regardless of how the block was scheduled.
    // n_cand <= TOPK_MAX <= TOPK_THREADS, so one thread per candidate.
    if (tid < n_cand) {
        const unsigned long long mine = topk_key64(cand_v[tid], cand_i[tid]);
        int rank = 0;
        for (int j = 0; j < n_cand; ++j) {
            if (topk_key64(cand_v[j], cand_i[j]) > mine) ++rank;
        }
        out_vals[(size_t)row * TOPK_MAX + rank] = cand_v[tid];
        out_ids[(size_t)row * TOPK_MAX + rank] = cand_i[tid];
    }
    // Unused slots are marked, so a host reading past k sees it immediately
    // rather than sampling stale candidates from an earlier step.
    for (int p = n_cand + tid; p < TOPK_MAX; p += TOPK_THREADS) {
        out_vals[(size_t)row * TOPK_MAX + p] = NEG_INF;
        out_ids[(size_t)row * TOPK_MAX + p] = -1;
    }
}

// ===========================================================================
// Batched GEMV for small-M decode
//
// Y[batch, rows] = X[batch, cols] @ W[rows, cols]^T, int8 weights with per-row
// scales. This exists because the tiled prefill GEMM has almost no parallelism
// at decode shapes: at M=1 it launches ceil(768/64) x ceil(1/64) = 12 blocks on
// a 60-SM GPU, and profiling showed its cost is flat from batch 1 to batch 16 --
// sixteen times the work for the same wall time, which is what an
// occupancy-starved kernel looks like. The projections are 83% of a batched
// decode step, so that is the whole problem.
//
// The mapping is one warp per OUTPUT ROW, carrying every request's accumulator
// at once.
//
// The obvious alternative -- one warp per (row, request), grid (rows, batch) --
// also restores parallelism, and re-reads the whole weight matrix once per
// request. Decode is bandwidth-bound, so at batch 16 that is 16x the traffic on
// the one tensor that dominates it. Keeping the batch inside the warp reads
// each weight row exactly once no matter the batch size, and pays instead with
// BMAX registers per lane and BMAX reads of X, which is small and L1-resident.
//
// BMAX is a template parameter rather than a runtime count because `acc[b]`
// indexed by a runtime value is not a register array -- it lands in local
// memory, which is exactly the failure the big-tile register investigation
// documented. Instantiating at 1/2/4/8/16 and rounding the batch up wastes at
// most 2x of the accumulate work while keeping the accumulators in registers.
// ===========================================================================

template <int BMAX>
__device__ __forceinline__ void gemv_batch_i8_impl(
    const signed char* __restrict__ w,   // [rows, cols]
    const float* __restrict__ scales,    // [rows]
    const float* __restrict__ x,         // [batch, cols]
    float* __restrict__ y,               // [batch, rows]
    const int rows,
    const int cols,
    const int batch,
    const int accumulate)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;

    // Vectorised exactly as the single-request warp GEMV: char4 against float4,
    // which is what made that kernel bandwidth-bound in the first place.
    const char4* w_row = reinterpret_cast<const char4*>(w + (size_t)row * cols);
    const int cols4 = cols / 4;

    float acc[BMAX];
#pragma unroll
    for (int b = 0; b < BMAX; ++b) acc[b] = 0.0f;

    for (int i = lane; i < cols4; i += WARP_SIZE) {
        const char4 a = w_row[i];
        const float ax = (float)a.x, ay = (float)a.y, az = (float)a.z, aw = (float)a.w;
#pragma unroll
        for (int b = 0; b < BMAX; ++b) {
            // Rows past the active batch still read: the buffer is allocated
            // for max_batch so this is in bounds, and the result is discarded
            // at the guarded store below. Branching here instead would
            // serialise the unrolled loop for no gain.
            const float4 v = reinterpret_cast<const float4*>(x + (size_t)b * cols)[i];
            acc[b] += ax * v.x + ay * v.y + az * v.z + aw * v.w;
        }
    }

#pragma unroll
    for (int b = 0; b < BMAX; ++b) {
        const float sum = warp_reduce_sum(acc[b]);
        if (lane == 0 && b < batch) {
            const size_t o = (size_t)b * rows + row;
            const float val = sum * scales[row];
            // accumulate folds the residual add into this write, the same
            // fusion the single-request decode path uses for o_proj and
            // down_proj.
            y[o] = accumulate ? y[o] + val : val;
        }
    }
}

#define GEMV_BATCH_ARGS                                                        \
    const signed char* __restrict__ w, const float* __restrict__ scales,       \
    const float* __restrict__ x, float* __restrict__ y,                        \
    const int rows, const int cols, const int batch, const int accumulate

extern "C" __global__ void gemv_batch_i8_b1(GEMV_BATCH_ARGS)
{ gemv_batch_i8_impl<1>(w, scales, x, y, rows, cols, batch, accumulate); }
extern "C" __global__ void gemv_batch_i8_b2(GEMV_BATCH_ARGS)
{ gemv_batch_i8_impl<2>(w, scales, x, y, rows, cols, batch, accumulate); }
extern "C" __global__ void gemv_batch_i8_b4(GEMV_BATCH_ARGS)
{ gemv_batch_i8_impl<4>(w, scales, x, y, rows, cols, batch, accumulate); }
extern "C" __global__ void gemv_batch_i8_b8(GEMV_BATCH_ARGS)
{ gemv_batch_i8_impl<8>(w, scales, x, y, rows, cols, batch, accumulate); }
extern "C" __global__ void gemv_batch_i8_b16(GEMV_BATCH_ARGS)
{ gemv_batch_i8_impl<16>(w, scales, x, y, rows, cols, batch, accumulate); }

// ===========================================================================
// Paged KV cache
//
// The contiguous cache above stores position j of a layer at a fixed linear
// offset, which forces one sequence to own one contiguous span for its whole
// possible lifetime. Paging replaces that with fixed-size physical pages and a
// per-sequence table of page ids:
//
//   pool: [n_pages][n_layer][PAGE_TOKENS][kv_dim]
//   offset(page, layer, slot) =
//       ((page * n_layer + layer) * PAGE_TOKENS + slot) * kv_dim
//
// PAGE_TOKENS is a compile-time constant so translation is a shift and a mask
// rather than a division in the attention inner loop. src/paged.rs asserts that
// its own PAGE_TOKENS matches this value; the two must not drift.
//
// These kernels read the paged representation directly. Gathering pages into a
// contiguous buffer before attention would work and would also defeat the
// purpose -- the copy would cost more than the fragmentation it avoids.
// ===========================================================================

#define PAGE_TOKENS 16
#define PAGE_SHIFT 4
#define PAGE_MASK 15

__device__ __forceinline__ size_t paged_offset(
    const int page, const int slot, const int n_layer, const int layer, const int kv_dim)
{
    return ((size_t)(page * n_layer + layer) * PAGE_TOKENS + slot) * kv_dim;
}

// Scatter a dense [rows, kv_dim] block into the paged pool.
//
// Used by prefill, where K and V are produced contiguously for the whole prompt
// and then placed. Decode does not need this: its projection writes straight
// into the pool, because the destination splits into a per-launch layer
// constant and one per-step scalar the kernel already reads from memory.
extern "C" __global__ void cache_store_paged_f32(
    const float* __restrict__ src,          // [rows][kv_dim]
    float* __restrict__ pool,
    const int* __restrict__ page_table,     // one sequence's table
    const int rows,
    const int kv_dim,
    const int n_layer,
    const int layer,
    const int pos_offset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kv_dim) return;
    const int row = i / kv_dim;
    const int col = i % kv_dim;
    const int pos = pos_offset + row;
    const int page = page_table[pos >> PAGE_SHIFT];
    const int slot = pos & PAGE_MASK;
    pool[paged_offset(page, slot, n_layer, layer, kv_dim) + col] = src[i];
}

// RoPE where every row sits at its own position.
//
// rope_batch_f32 assumes rows are consecutive positions of one sequence, which
// is true for prefill and false for a decode batch: request A may be at
// position 7 while request B is at 511. Taking positions from an array is the
// whole difference.
extern "C" __global__ void rope_rows_f32(
    float* __restrict__ v,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int rows,
    const int n_heads,
    const int head_dim,
    const int row_stride,
    const int* __restrict__ positions)
{
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int per_row = n_heads * half;
    if (idx >= rows * per_row) return;

    const int row = idx / per_row;
    const int rem = idx % per_row;
    const int head = rem / half;
    const int i = rem % half;

    const int pos = positions[row];
    const float c = cos_table[pos * half + i];
    const float sn = sin_table[pos * half + i];

    float* h = v + (size_t)row * row_stride + head * head_dim;
    const float lo = h[i];
    const float hi = h[i + half];
    h[i] = lo * c - hi * sn;
    h[i + half] = lo * sn + hi * c;
}

// Scatter one row per request into each request's own page.
//
// Each row carries its own page table and its own logical position, so a batch
// writes into pages belonging to different sequences in one launch. This is the
// decode counterpart of cache_store_paged_f32.
extern "C" __global__ void cache_store_rows_paged_f32(
    const float* __restrict__ src,          // [rows][kv_dim]
    float* __restrict__ pool,
    const int* __restrict__ page_tables,    // [rows][table_stride]
    const int* __restrict__ positions,      // [rows]
    const int rows,
    const int kv_dim,
    const int table_stride,
    const int n_layer,
    const int layer)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * kv_dim) return;
    const int row = i / kv_dim;
    const int col = i % kv_dim;
    const int pos = positions[row];
    const int page = page_tables[(size_t)row * table_stride + (pos >> PAGE_SHIFT)];
    pool[paged_offset(page, pos & PAGE_MASK, n_layer, layer, kv_dim) + col] = src[i];
}

// Batched single-token decode attention over paged KV.
//
// grid is (n_head, batch): one block per (head, request). Each request brings
// its own page table and its own sequence length, so a batch may mix a
// 7-position request with a 511-position one without padding either to the
// other. Nothing in the block reads outside its own request's pages.
//
// Batch size 1 is the single-request case, not a special path.
extern "C" __global__ void attention_decode_paged_f32(
    const float* __restrict__ q,            // [batch][n_head * head_dim]
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    float* __restrict__ out,                // [batch][n_head * head_dim]
    const int* __restrict__ page_tables,    // [batch][table_stride]
    const int* __restrict__ seq_lens,       // [batch]
    const int n_head,
    const int n_kv_head,
    const int head_dim,
    const int table_stride,
    const int n_layer,
    const int layer,
    const int kv_dim,
    const int max_seq)                      // scores capacity in shared memory
{
    extern __shared__ float scores[];

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    if (h >= n_head) return;

    const int seq_len = seq_lens[b];
    if (seq_len <= 0) return;                       // request not active

    const int* table = page_tables + (size_t)b * table_stride;
    const int n_rep = n_head / n_kv_head;
    const int kv_h = h / n_rep;
    const float* qh = q + (size_t)b * n_head * head_dim + h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    // One warp per cached position, lanes striding across the key vector --
    // the same access shape the contiguous kernel was tuned into, since slots
    // inside a page are still contiguous.
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;

    for (int j = warp_id; j < seq_len; j += n_warps) {
        const int page = table[j >> PAGE_SHIFT];
        const float* kh = k_pool
            + paged_offset(page, j & PAGE_MASK, n_layer, layer, kv_dim)
            + kv_h * head_dim;
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += WARP_SIZE) partial += qh[d] * kh[d];
        const float dot = warp_reduce_sum(partial);
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

    float* partials = scores + max_seq;   // [n_warps][head_dim]
    for (int d = lane; d < head_dim; d += WARP_SIZE) {
        float acc = 0.0f;
        for (int j = warp_id; j < seq_len; j += n_warps) {
            const int page = table[j >> PAGE_SHIFT];
            acc += scores[j] * v_pool[
                paged_offset(page, j & PAGE_MASK, n_layer, layer, kv_dim)
                + kv_h * head_dim + d];
        }
        partials[warp_id * head_dim + d] = acc;
    }
    __syncthreads();

    float* dst = out + (size_t)b * n_head * head_dim + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int w = 0; w < n_warps; ++w) acc += partials[w * head_dim + d];
        dst[d] = acc * ssum;
    }
}

// Causal prefill attention over paged KV, one block per (head, prompt row).
extern "C" __global__ void attention_prefill_paged_f32(
    const float* __restrict__ q,            // [T, n_head * head_dim]
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    float* __restrict__ out,                // [T, n_head * head_dim]
    const int* __restrict__ page_table,     // one sequence's table
    const int n_head,
    const int n_kv_head,
    const int head_dim,
    const int n_layer,
    const int layer,
    const int kv_dim,
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
        const int page = page_table[j >> PAGE_SHIFT];
        const float* kh = k_pool
            + paged_offset(page, j & PAGE_MASK, n_layer, layer, kv_dim)
            + kv_h * head_dim;
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
            const int page = page_table[j >> PAGE_SHIFT];
            acc += scores[j] * v_pool[
                paged_offset(page, j & PAGE_MASK, n_layer, layer, kv_dim)
                + kv_h * head_dim + d];
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
