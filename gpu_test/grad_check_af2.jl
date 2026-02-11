#!/usr/bin/env julia
#=
  Gradient check for AlphaFold2.jl on GPU.

  Tests backward pass through:
    A. Single EvoformerIteration block
    B. Structure module (FoldIterationCore + sidechain)
    C. Confidence heads (PredictedLDDTHead)
    D. Evoformer → structure → pLDDT (end-to-end)
    E. Gradient w.r.t. continuous sequence input

  Usage:
    JULIA_CUDA_HARD_MEMORY_LIMIT=64GiB julia --project=Alphafold2.jl gpu_test/grad_check_af2.jl
=#

using Printf
using Statistics

# Load Alphafold2 (brings in CUDA, Flux, NNlib, Onion, etc.)
using Alphafold2

using Zygote
using CUDA
using Flux


# ─── Helpers ───────────────────────────────────────────────────────────────

# Recursively walk gradient NamedTuple and count array stats
function _walk_grads!(stats, g)
    if g === nothing || g isa Number
        return
    elseif g isa AbstractArray{<:Number}
        ga = g isa CuArray ? Array(g) : g
        stats[:total] += length(ga)
        stats[:nonzero] += count(!=(0f0), ga)
        stats[:nan_count] += count(isnan, ga)
    elseif g isa NamedTuple
        for v in values(g)
            _walk_grads!(stats, v)
        end
    elseif g isa Tuple
        for v in g
            _walk_grads!(stats, v)
        end
    elseif g isa AbstractVector
        for v in g
            _walk_grads!(stats, v)
        end
    end
end

function grad_summary(g; label="")
    stats = Dict{Symbol,Int}(:total => 0, :nonzero => 0, :nan_count => 0)
    _walk_grads!(stats, g)
    total, nonzero, nan_count = stats[:total], stats[:nonzero], stats[:nan_count]
    pct = total > 0 ? 100.0 * nonzero / total : 0.0
    @printf("  %s: %d/%d nonzero (%.1f%%), %d NaN\n", label, nonzero, total, pct, nan_count)
    return (total=total, nonzero=nonzero, nan_count=nan_count)
end

function test_header(name)
    println("\n" * "="^60)
    println("  TEST: $name")
    println("="^60)
end

pass_fail(ok) = ok ? "PASS" : "FAIL"

# ─── Model dimensions (AF2 monomer defaults) ──────────────────────────────
const C_M = 256    # MSA channel dim
const C_Z = 128    # pair channel dim
const C_S = 384    # single (structure module) channel dim
const L = 12       # sequence length (small for testing)
const B = 1        # batch size
const N_SEQ = 4    # MSA depth (small for testing)

results = Dict{String, Bool}()

# ═══════════════════════════════════════════════════════════════════════════
# TEST A: Single EvoformerIteration
# ═══════════════════════════════════════════════════════════════════════════
test_header("A: EvoformerIteration backward")

block = EvoformerIteration(
    C_M, C_Z;
    num_head_msa=8, msa_head_dim=32,
    num_head_pair=4, pair_head_dim=32,
    c_outer=32, c_tri_mul=128,
) |> Flux.gpu

msa_act = CUDA.randn(Float32, C_M, N_SEQ, L, B)
pair_act = CUDA.randn(Float32, C_Z, L, L, B)
msa_mask = CUDA.ones(Float32, N_SEQ, L, B)
pair_mask = CUDA.ones(Float32, L, L, B)

println("  Forward...")
msa_out, pair_out = block(msa_act, pair_act, msa_mask, pair_mask)
@printf("  msa_out: %s, pair_out: %s\n", size(msa_out), size(pair_out))

println("  Backward (loss = mean(pair_out))...")
loss_a, grads_a = Zygote.withgradient(block) do b
    _, p = b(msa_act, pair_act, msa_mask, pair_mask)
    Float32(mean(p))
end
@printf("  loss = %.6f\n", loss_a)

stats_a = grad_summary(grads_a[1]; label="block weights")
ok_a = stats_a.nonzero > 0 && stats_a.nan_count == 0
results["A: EvoformerIteration"] = ok_a
@printf("  → %s\n", pass_fail(ok_a))

CUDA.reclaim()

# ═══════════════════════════════════════════════════════════════════════════
# TEST B: Structure module (FoldIterationCore + sidechain)
# ═══════════════════════════════════════════════════════════════════════════
test_header("B: StructureModuleCore backward")

structure = StructureModuleCore(
    C_S, C_Z, 128, 12, 4, 8, 3, 8, 10.0f0, 128, 2;
) |> Flux.gpu

single_in = CUDA.randn(Float32, C_S, L, B)
pair_in = CUDA.randn(Float32, C_Z, L, L, B) .* 0.1f0
seq_mask = CUDA.ones(Float32, L, B)
aatype_cpu = rand(0:19, L)  # integer amino acid types (stays on CPU for gather)
aatype = aatype_cpu

println("  Forward...")
struct_out = structure(single_in, pair_in, seq_mask, aatype)
@printf("  act: %s, atom_pos: %s, affine: %s\n",
    size(struct_out.act), size(struct_out.atom_pos), size(struct_out.affine))

# B1: Loss on :act (neural network state — should definitely work)
println("  Backward (loss = mean(act))...")
loss_b1, grads_b1 = Zygote.withgradient(structure) do s
    out = s(single_in, pair_in, seq_mask, aatype)
    Float32(mean(out.act))
end
@printf("  loss = %.6f\n", loss_b1)
stats_b1 = grad_summary(grads_b1[1]; label="struct weights (act loss)")
ok_b1 = stats_b1.nonzero > 0 && stats_b1.nan_count == 0
results["B1: StructureModule (act loss)"] = ok_b1
@printf("  → %s\n", pass_fail(ok_b1))

# B2: Loss on :atom_pos (geometry — tests @ignore_derivatives fix)
println("  Backward (loss = mean(atom_pos))...")
loss_b2, grads_b2 = Zygote.withgradient(structure) do s
    out = s(single_in, pair_in, seq_mask, aatype)
    Float32(mean(out.atom_pos))
end
@printf("  loss = %.6f\n", loss_b2)
stats_b2 = grad_summary(grads_b2[1]; label="struct weights (atom_pos loss)")
ok_b2 = stats_b2.nonzero > 0 && stats_b2.nan_count == 0
results["B2: StructureModule (atom_pos loss)"] = ok_b2
@printf("  → %s\n", pass_fail(ok_b2))

CUDA.reclaim()

# ═══════════════════════════════════════════════════════════════════════════
# TEST C: PredictedLDDTHead
# ═══════════════════════════════════════════════════════════════════════════
test_header("C: PredictedLDDTHead backward")

lddt_head = PredictedLDDTHead(C_S) |> Flux.gpu
act_in = CUDA.randn(Float32, C_S, L, B)

println("  Forward...")
lddt_out = lddt_head(act_in)
logits = lddt_out.logits
@printf("  logits: %s\n", size(logits))

plddt_val = compute_plddt(logits)
@printf("  plddt: %s, mean=%.2f\n", size(plddt_val), mean(Array(plddt_val)))

println("  Backward (loss = mean(plddt))...")
loss_c, grads_c = Zygote.withgradient(lddt_head) do h
    out = h(act_in)
    p = compute_plddt(out.logits)
    Float32(mean(p))
end
@printf("  loss = %.4f\n", loss_c)
stats_c = grad_summary(grads_c[1]; label="lddt_head weights")
ok_c = stats_c.nonzero > 0 && stats_c.nan_count == 0
results["C: PredictedLDDTHead"] = ok_c
@printf("  → %s\n", pass_fail(ok_c))

# Also test gradient w.r.t. input
println("  Backward w.r.t. input...")
g_input = Zygote.gradient(act_in) do x
    out = lddt_head(x)
    Float32(mean(compute_plddt(out.logits)))
end[1]
gi = g_input isa CuArray ? Array(g_input) : g_input
nz_input = count(!=(0f0), gi)
nan_input = count(isnan, gi)
@printf("  input grad: %d/%d nonzero, %d NaN\n", nz_input, length(gi), nan_input)
ok_c_input = nz_input > 0 && nan_input == 0
results["C2: pLDDT input grad"] = ok_c_input
@printf("  → %s\n", pass_fail(ok_c_input))

CUDA.reclaim()

# ═══════════════════════════════════════════════════════════════════════════
# TEST D: End-to-end: evoformer block → structure → pLDDT
# ═══════════════════════════════════════════════════════════════════════════
test_header("D: Block → StructureModule → pLDDT (end-to-end)")

# Use a single evoformer block + structure module + plddt head
block_d = EvoformerIteration(
    C_M, C_Z;
    num_head_msa=8, msa_head_dim=32,
    num_head_pair=4, pair_head_dim=32,
    c_outer=32, c_tri_mul=128,
) |> Flux.gpu

# Simpler structure module (fewer layers for speed)
struct_d = StructureModuleCore(
    C_S, C_Z, 128, 12, 4, 8, 3, 4, 10.0f0, 128, 2;  # 4 layers instead of 8
) |> Flux.gpu

lddt_d = PredictedLDDTHead(C_S) |> Flux.gpu

# We need a projection from MSA first row to single representation
single_proj = LinearFirst(C_M, C_S) |> Flux.gpu

msa_d = CUDA.randn(Float32, C_M, N_SEQ, L, B)
pair_d = CUDA.randn(Float32, C_Z, L, L, B) .* 0.1f0
msa_mask_d = CUDA.ones(Float32, N_SEQ, L, B)
pair_mask_d = CUDA.ones(Float32, L, L, B)
seq_mask_d = CUDA.ones(Float32, L, B)
aatype_d = rand(0:19, L)

println("  Forward...")
msa_out_d, pair_out_d = block_d(msa_d, pair_d, msa_mask_d, pair_mask_d)
# Extract first MSA row as single representation
single_d = single_proj(msa_out_d[:, 1:1, :, :])
single_d = dropdims(single_d; dims=2)  # (C_S, L, B)
struct_out_d = struct_d(single_d, pair_out_d, seq_mask_d, aatype_d)
plddt_d = compute_plddt(lddt_d(struct_out_d.act).logits)
@printf("  mean_plddt = %.2f\n", mean(Array(plddt_d)))

println("  Backward (loss = mean(plddt))...")
loss_d, grads_d = Zygote.withgradient(block_d, struct_d, lddt_d, single_proj) do b, s, h, proj
    mo, po = b(msa_d, pair_d, msa_mask_d, pair_mask_d)
    si = proj(mo[:, 1:1, :, :])
    si = dropdims(si; dims=2)
    so = s(si, po, seq_mask_d, aatype_d)
    p = compute_plddt(h(so.act).logits)
    Float32(mean(p))
end
@printf("  loss = %.4f\n", loss_d)

stats_d_block = grad_summary(grads_d[1]; label="evoformer block")
stats_d_struct = grad_summary(grads_d[2]; label="structure module")
stats_d_lddt = grad_summary(grads_d[3]; label="lddt head")
stats_d_proj = grad_summary(grads_d[4]; label="single projection")

ok_d = (stats_d_block.nonzero > 0 && stats_d_struct.nonzero > 0 &&
        stats_d_lddt.nonzero > 0 && stats_d_proj.nonzero > 0 &&
        stats_d_block.nan_count == 0 && stats_d_struct.nan_count == 0 &&
        stats_d_lddt.nan_count == 0 && stats_d_proj.nan_count == 0)
results["D: E2E pLDDT backward"] = ok_d
@printf("  → %s\n", pass_fail(ok_d))

CUDA.reclaim()

# ═══════════════════════════════════════════════════════════════════════════
# TEST E: Gradient w.r.t. continuous sequence input
# ═══════════════════════════════════════════════════════════════════════════
test_header("E: Gradient w.r.t. continuous sequence input")

# Simulate continuous sequence features (like soft one-hot or embedding)
# target_feat in AF2 is (22, L, 1) — one-hot amino acid type + unknown
target_feat = CUDA.randn(Float32, 22, L, B)  # continuous relaxation
preprocess_1d = LinearFirst(22, C_M) |> Flux.gpu
left_single = LinearFirst(22, C_Z) |> Flux.gpu
right_single = LinearFirst(22, C_Z) |> Flux.gpu

println("  Forward with continuous input...")
msa_init = preprocess_1d(target_feat)
msa_init = reshape(msa_init, C_M, 1, L, B)  # single MSA row
pair_init = reshape(left_single(target_feat), C_Z, L, 1, B) .+ reshape(right_single(target_feat), C_Z, 1, L, B)
@printf("  msa_init: %s, pair_init: %s\n", size(msa_init), size(pair_init))

# Run through evoformer block
msa_mask_e = CUDA.ones(Float32, 1, L, B)
pair_mask_e = CUDA.ones(Float32, L, L, B)
msa_out_e, pair_out_e = block_d(msa_init, pair_init, msa_mask_e, pair_mask_e)

println("  Backward w.r.t. continuous target_feat...")
loss_e, grad_e = Zygote.withgradient(target_feat) do tf
    mi = preprocess_1d(tf)
    mi = reshape(mi, C_M, 1, L, B)
    pi = reshape(left_single(tf), C_Z, L, 1, B) .+ reshape(right_single(tf), C_Z, 1, L, B)
    _, po = block_d(mi, pi, msa_mask_e, pair_mask_e)
    Float32(mean(po))
end
@printf("  loss = %.6f\n", loss_e)

g_tf = grad_e[1]
if g_tf === nothing
    @printf("  target_feat grad: nothing!\n")
    ok_e = false
else
    ga = g_tf isa CuArray ? Array(g_tf) : g_tf
    nz = count(!=(0f0), ga)
    nan_ct = count(isnan, ga)
    @printf("  target_feat grad: %d/%d nonzero (%.1f%%), %d NaN\n",
        nz, length(ga), 100.0*nz/length(ga), nan_ct)
    ok_e = nz > 0 && nan_ct == 0
end
results["E: Continuous input grad"] = ok_e
@printf("  → %s\n", pass_fail(ok_e))

CUDA.reclaim()

# ═══════════════════════════════════════════════════════════════════════════
# TEST F: Full pipeline with continuous input → pLDDT loss
# ═══════════════════════════════════════════════════════════════════════════
test_header("F: Continuous input → block → struct → pLDDT")

println("  Backward w.r.t. continuous target_feat (full pipeline)...")
loss_f, grad_f = Zygote.withgradient(target_feat) do tf
    mi = preprocess_1d(tf)
    mi = reshape(mi, C_M, 1, L, B)
    pi = reshape(left_single(tf), C_Z, L, 1, B) .+ reshape(right_single(tf), C_Z, 1, L, B)
    mo, po = block_d(mi, pi, msa_mask_e, pair_mask_e)
    si = single_proj(mo[:, 1:1, :, :])
    si = dropdims(si; dims=2)
    so = struct_d(si, po, seq_mask_d, aatype_d)
    p = compute_plddt(lddt_d(so.act).logits)
    Float32(mean(p))
end
@printf("  loss (mean pLDDT) = %.4f\n", loss_f)

g_f = grad_f[1]
if g_f === nothing
    @printf("  target_feat grad: nothing!\n")
    ok_f = false
else
    ga = g_f isa CuArray ? Array(g_f) : g_f
    nz = count(!=(0f0), ga)
    nan_ct = count(isnan, ga)
    @printf("  target_feat grad: %d/%d nonzero (%.1f%%), %d NaN\n",
        nz, length(ga), 100.0*nz/length(ga), nan_ct)
    ok_f = nz > 0 && nan_ct == 0
end
results["F: Full pipeline input grad"] = ok_f
@printf("  → %s\n", pass_fail(ok_f))

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "="^60)
println("  SUMMARY")
println("="^60)
let n_pass = 0, n_total = 0
    for (name, ok) in sort(collect(results); by=first)
        @printf("  [%s] %s\n", pass_fail(ok), name)
        n_total += 1
        n_pass += ok ? 1 : 0
    end
    @printf("\n  %d/%d tests passed\n", n_pass, n_total)
end
println("="^60)
