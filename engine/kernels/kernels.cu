// CUDA kernels for single-stream decoding.
//
// Decoding one token at a time means every matmul is a matrix-VECTOR product,
// which is memory-bound rather than compute-bound: the model's weights are read
// once per token and barely reused. On this hardware that is 757 GB/s (measured
// at a 135 W power limit) against 0.45 GB of f32 weights, so ~1674 tok/s is the
// ceiling regardless of how much arithmetic throughput the card has. These
// kernels are therefore written to maximise read bandwidth, not FLOPs.
//
// gemv reaches 723-770 GB/s, so it is already bandwidth-saturated: the only
// remaining lever for decode speed is reading fewer bytes, i.e. quantisation.
//
// Every kernel has a scalar CPU twin in src/ops.rs. The CPU version is the
// reference; when the two disagree the GPU one is wrong.

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
    const int y_offset)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const float* w_row = w + (size_t)row * cols;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        acc += w_row[i] * x[i];
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0) y[y_offset + row] = acc;
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
    const int y_offset)
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
    if (threadIdx.x == 0) y[y_offset + row] = acc;
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
    const int pos,
    const int v_offset)
{
    const int half = head_dim / 2;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * half) return;

    const int head = idx / half;
    const int i = idx % half;

    const float c = cos_table[pos * half + i];
    const float s = sin_table[pos * half + i];

    float* h = v + v_offset + head * head_dim;
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
    const int token,
    const int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) out[i] = table[(size_t)token * d + i];
}

// Fused single-token attention over the KV cache.
//
// One block per query head, computing scores, softmax and the weighted sum of
// values without leaving the block. Doing it as three separate kernels would
// mean three launches per head per layer -- at 12 heads x 12 layers that is 432
// launches per token, and launch overhead alone would dominate decoding a model
// this size.
//
// Scores live in dynamic shared memory, so `seq_len` floats must be requested
// at launch.
extern "C" __global__ void attention_decode_f32(
    const float* __restrict__ q,        // [n_head * head_dim]
    const float* __restrict__ k_cache,  // [capacity][n_kv_head * head_dim]
    const float* __restrict__ v_cache,
    float* __restrict__ out,            // [n_head * head_dim]
    const int n_head,
    const int n_kv_head,
    const int head_dim,
    const int seq_len,                  // positions 0..seq_len-1, current included
    const int cache_stride)             // n_kv_head * head_dim
{
    extern __shared__ float scores[];

    const int h = blockIdx.x;
    if (h >= n_head) return;

    // Query head h reads KV head h / n_rep, matching repeat_interleave's
    // mapping of KV head j to query heads [j*n_rep, (j+1)*n_rep).
    const int n_rep = n_head / n_kv_head;
    const int kv_h = h / n_rep;
    const float* qh = q + h * head_dim;
    const float scale = rsqrtf((float)head_dim);

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        const float* kh = k_cache + (size_t)j * cache_stride + kv_h * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kh[d];
        scores[j] = dot * scale;
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
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp = threadIdx.x / WARP_SIZE;
        if (lane == 0) warp_max[warp] = m;
        __syncthreads();
        if (threadIdx.x == 0) {
            const int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
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

    // Weighted sum of values. One thread per output element, each walking the
    // cache; the reciprocal is applied once at the end rather than per term.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            acc += scores[j] * v_cache[(size_t)j * cache_stride + kv_h * head_dim + d];
        }
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
    const int y_offset)
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
    if (threadIdx.x == 0) y[y_offset + row] = acc * scales[row];
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
    const int y_offset)
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
    if (threadIdx.x == 0) y[y_offset + row] = acc * scales[row];
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
    const int token,
    const int d)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) out[i] = (float)table[(size_t)token * d + i] * scales[token];
}
