#!/usr/bin/env julia
#=
  Finite-difference gradient verification for AlphaFold2.jl layers.

  Compares Zygote AD gradients against central finite differences on CPU
  with small dimensions so FD is tractable.

  Usage:
    julia --project=Alphafold2.jl gpu_test/grad_check_fd.jl
=#

using Printf
using Statistics
using LinearAlgebra

using Alphafold2
using Zygote

# ─── FD helpers ────────────────────────────────────────────────────────────

function fd_gradient(f, x::AbstractArray; eps=1e-4)
    g = similar(x)
    x_work = copy(x)
    for i in eachindex(x)
        old = x_work[i]
        x_work[i] = old + eps
        f_plus = f(x_work)
        x_work[i] = old - eps
        f_minus = f(x_work)
        x_work[i] = old
        g[i] = (f_plus - f_minus) / (2 * eps)
    end
    return g
end

function check_gradient(label, grad_ad, grad_fd; rtol=2e-2, atol=1e-5)
    if grad_ad === nothing
        @printf("  %s: AD gradient is nothing  [FAIL]\n", label)
        return false
    end

    ad = vec(Float64.(grad_ad))
    fd = vec(Float64.(grad_fd))

    max_abs_diff = maximum(abs.(ad .- fd))
    scale = max(maximum(abs.(fd)), maximum(abs.(ad)), 1.0)
    max_rel_diff = max_abs_diff / scale

    cosine = if norm(ad) > 0 && norm(fd) > 0
        dot(ad, fd) / (norm(ad) * norm(fd))
    else
        0.0
    end

    pass = max_rel_diff < rtol || max_abs_diff < atol

    @printf("  %s:\n", label)
    @printf("    max |AD-FD|: %.2e  relative: %.2e  cosine: %.6f  [%s]\n",
        max_abs_diff, max_rel_diff, cosine, pass ? "PASS" : "FAIL")
    return pass
end

pass_fail(ok) = ok ? "PASS" : "FAIL"

results = Dict{String, Bool}()

println("=" ^ 60)
println("  Finite Difference Gradient Checks (CPU)")
println("=" ^ 60)

# ─── Small dimensions for tractable FD ─────────────────────────────────────
const C_small = 16     # channel dim
const C_z = 8          # pair channel dim
const L = 4            # sequence length
const B = 1            # batch

# Random weight vectors used to break symmetry in loss functions.
# plain sum() after layernorm is invariant to input (degenerate), so we use
# weighted sums: sum(output .* loss_weights) instead.

# ═══════════════════════════════════════════════════════════════════════════
# 1. Transition layer
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 1. Transition ---")

transition = Transition(C_small, 2.0)
x_tr = randn(Float32, C_small, L, B)
tr_weights = randn(Float32, C_small, L, B)  # weighted loss (Transition contains LN)

# Grad w.r.t. input
mask_tr = ones(Float32, L, B)
g_ad = Zygote.gradient(x -> sum(transition(x, mask_tr) .* tr_weights), x_tr)[1]
g_fd = fd_gradient(x -> sum(transition(x, mask_tr) .* tr_weights), x_tr)
results["1a: Transition input"] = check_gradient("Transition d/d(input)", g_ad, g_fd)

# Grad w.r.t. weight of first linear
g_w_ad = Zygote.gradient(transition) do t
    sum(t(x_tr, mask_tr) .* tr_weights)
end[1]
# Extract first linear weight grad from NamedTuple
w1 = transition.transition1.weight
g_w_ad_arr = g_w_ad.transition1.weight
g_w_fd = fd_gradient(w1) do w
    t2 = Transition(
        transition.input_layer_norm,
        LinearFirst(w, transition.transition1.bias, transition.transition1.use_bias),
        transition.transition2,
    )
    sum(t2(x_tr, mask_tr) .* tr_weights)
end
results["1b: Transition weight"] = check_gradient("Transition d/d(weight1)", g_w_ad_arr, g_w_fd)

# ═══════════════════════════════════════════════════════════════════════════
# 2. LayerNormFirst
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 2. LayerNormFirst ---")

ln = LayerNormFirst(C_small)
x_ln = randn(Float32, C_small, L, B)
# Use weighted sum — plain sum(layernorm(x)) is ~constant w.r.t. input
# because LN normalizes to zero mean, so sum over channels = sum(b) ≈ 0.
ln_weights = randn(Float32, C_small, L, B)

g_ad_ln = Zygote.gradient(x -> sum(ln(x) .* ln_weights), x_ln)[1]
g_fd_ln = fd_gradient(x -> sum(ln(x) .* ln_weights), x_ln)
results["2a: LayerNorm input"] = check_gradient("LayerNorm d/d(input)", g_ad_ln, g_fd_ln)

g_w_ad_ln = Zygote.gradient(ln) do l
    sum(l(x_ln) .* ln_weights)
end[1]
g_w_fd_ln = fd_gradient(ln.w) do w
    l2 = LayerNormFirst(w, ln.b, ln.eps)
    sum(l2(x_ln) .* ln_weights)
end
results["2b: LayerNorm scale"] = check_gradient("LayerNorm d/d(scale)", g_w_ad_ln.w, g_w_fd_ln)

# ═══════════════════════════════════════════════════════════════════════════
# 3. LinearFirst
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 3. LinearFirst ---")

linear = LinearFirst(C_small, C_z)
x_lin = randn(Float32, C_small, L, B)

g_ad_lin = Zygote.gradient(x -> sum(linear(x)), x_lin)[1]
g_fd_lin = fd_gradient(x -> sum(linear(x)), x_lin)
results["3a: Linear input"] = check_gradient("Linear d/d(input)", g_ad_lin, g_fd_lin)

g_w_ad_lin = Zygote.gradient(linear) do l
    sum(l(x_lin))
end[1]
g_w_fd_weight = fd_gradient(linear.weight) do w
    l2 = LinearFirst(w, linear.bias, linear.use_bias)
    sum(l2(x_lin))
end
results["3b: Linear weight"] = check_gradient("Linear d/d(weight)", g_w_ad_lin.weight, g_w_fd_weight)

# ═══════════════════════════════════════════════════════════════════════════
# 4. TriangleMultiplication
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 4. TriangleMultiplication ---")

tri_mul = TriangleMultiplication(C_z, C_z; outgoing=true)
x_tri = randn(Float32, C_z, L, L, B) .* 0.1f0
pair_mask = ones(Float32, L, L, B)

g_ad_tri = Zygote.gradient(x -> sum(tri_mul(x, pair_mask)), x_tri)[1]
g_fd_tri = fd_gradient(x -> Float64(sum(tri_mul(x, pair_mask))), x_tri)
results["4: TriangleMul input"] = check_gradient("TriangleMul d/d(input)", g_ad_tri, g_fd_tri)

# ═══════════════════════════════════════════════════════════════════════════
# 5. PredictedLDDTHead + compute_plddt
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 5. PredictedLDDTHead + compute_plddt ---")

const C_S = 32  # smaller single dim for FD
lddt_head = PredictedLDDTHead(C_S; num_channels=16, num_bins=10)
x_lddt = randn(Float32, C_S, L, B) .* 0.1f0  # smaller input for FD stability

function lddt_loss(x)
    logits = lddt_head(x).logits
    p = compute_plddt(logits)
    return Float64(mean(p))
end

g_ad_lddt = Zygote.gradient(lddt_loss, x_lddt)[1]
g_fd_lddt = fd_gradient(lddt_loss, x_lddt; eps=5e-5)
results["5: pLDDT input"] = check_gradient("pLDDT d/d(input)", g_ad_lddt, g_fd_lddt)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Geometry: frames_and_literature_positions_to_atom14_pos
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 6. Geometry (atom14_pos w.r.t. rigid translation) ---")

using Alphafold2: Rigid, RotMatRotation, rigid_identity
using Alphafold2: restype_rigid_group_default_frame, restype_atom14_to_rigid_group
using Alphafold2: restype_atom14_mask, restype_atom14_rigid_group_positions
using Alphafold2: frames_and_literature_positions_to_atom14_pos

n_groups = 8
aatype_geo = rand(0:19, L, B)

# Identity rotation, random translation (Float64 for FD)
rot_geo = zeros(Float64, 3, 3, n_groups, L, B)
for g in 1:n_groups, l in 1:L, b in 1:B
    rot_geo[1,1,g,l,b] = 1.0; rot_geo[2,2,g,l,b] = 1.0; rot_geo[3,3,g,l,b] = 1.0
end
trans_geo = randn(Float64, 3, n_groups, L, B) .* 0.1

df = Float64.(restype_rigid_group_default_frame)
gi_geo = restype_atom14_to_rigid_group
am_geo = Float64.(restype_atom14_mask)
lp_geo = Float64.(restype_atom14_rigid_group_positions)

function geo_loss_trans(t_flat)
    t = reshape(t_flat, 3, n_groups, L, B)
    r = Rigid(RotMatRotation(rot_geo), t)
    pred = frames_and_literature_positions_to_atom14_pos(r, aatype_geo, df, gi_geo, am_geo, lp_geo)
    return sum(pred)
end

g_ad_geo = Zygote.gradient(geo_loss_trans, vec(trans_geo))[1]
g_fd_geo = fd_gradient(geo_loss_trans, vec(trans_geo); eps=1e-6)
results["6: Geometry trans"] = check_gradient("Geometry d/d(trans)", g_ad_geo, g_fd_geo; rtol=1e-2)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Geometry E2E: torsion_angles → atom14_pos w.r.t. angles
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 7. Geometry E2E (torsion angles → atom14_pos) ---")

using Alphafold2: torsion_angles_to_frames

bb_rot = zeros(Float64, 3, 3, L, B)
for l in 1:L, b in 1:B
    bb_rot[1,1,l,b] = 1.0; bb_rot[2,2,l,b] = 1.0; bb_rot[3,3,l,b] = 1.0
end
bb_trans = randn(Float64, 3, L, B)
bb_rigid = Rigid(RotMatRotation(bb_rot), bb_trans)

alpha_geo = randn(Float64, 2, 7, L, B) .* 0.3
# Normalize
for j in 1:7, l in 1:L, b in 1:B
    n = sqrt(alpha_geo[1,j,l,b]^2 + alpha_geo[2,j,l,b]^2)
    if n > 0
        alpha_geo[1,j,l,b] /= n; alpha_geo[2,j,l,b] /= n
    else
        alpha_geo[1,j,l,b] = 1.0; alpha_geo[2,j,l,b] = 0.0
    end
end

function geo_e2e_loss(a_flat)
    a = reshape(a_flat, 2, 7, L, B)
    frames = torsion_angles_to_frames(bb_rigid, a, aatype_geo, df)
    pred = frames_and_literature_positions_to_atom14_pos(frames, aatype_geo, df, gi_geo, am_geo, lp_geo)
    return sum(pred)
end

g_ad_e2e = Zygote.gradient(geo_e2e_loss, vec(alpha_geo))[1]
g_fd_e2e = fd_gradient(geo_e2e_loss, vec(alpha_geo); eps=1e-6)
results["7: Geometry angles"] = check_gradient("Geometry d/d(alpha)", g_ad_e2e, g_fd_e2e; rtol=1e-2)

# ═══════════════════════════════════════════════════════════════════════════
# 8. Small EvoformerIteration (input grad)
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 8. Small EvoformerIteration ---")

# Very small dims for tractable FD
evo_c_m = 8
evo_c_z = 8
evo_L = 3
evo_N = 2

evo_block = EvoformerIteration(
    evo_c_m, evo_c_z;
    num_head_msa=2, msa_head_dim=4,
    num_head_pair=2, pair_head_dim=4,
    c_outer=4, c_tri_mul=8,
)

pair_in_evo = randn(Float32, evo_c_z, evo_L, evo_L, B) .* 0.1f0
msa_in_evo = randn(Float32, evo_c_m, evo_N, evo_L, B) .* 0.1f0
msa_mask_evo = ones(Float32, evo_N, evo_L, B)
pair_mask_evo = ones(Float32, evo_L, evo_L, B)
evo_weights = randn(Float32, evo_c_z, evo_L, evo_L, B)  # weighted loss

function evo_loss(p_in)
    _, po = evo_block(msa_in_evo, p_in, msa_mask_evo, pair_mask_evo)
    return Float64(sum(po .* evo_weights))
end

g_ad_evo = Zygote.gradient(evo_loss, pair_in_evo)[1]
g_fd_evo = fd_gradient(evo_loss, pair_in_evo; eps=1e-4)
results["8: Evoformer pair input"] = check_gradient("Evoformer d/d(pair_in)", g_ad_evo, g_fd_evo)

# ═══════════════════════════════════════════════════════════════════════════
# 9. FoldIterationCore (input grad)
# ═══════════════════════════════════════════════════════════════════════════
println("\n--- 9. FoldIterationCore ---")

fic_c_s = 16
fic_c_z = 8
fic = FoldIterationCore(fic_c_s, fic_c_z, 8, 4, 2, 2, 2)

act_fic = randn(Float32, fic_c_s, L, B) .* 0.1f0
z_fic = randn(Float32, fic_c_z, L, L, B) .* 0.1f0
mask_fic = ones(Float32, L, B)
# Weighted loss to break layernorm invariance (sum after LN with unit weights = 0)
fic_weights = randn(Float32, fic_c_s, L, B)

function fic_loss(a)
    act_out, _, _ = fic(a, z_fic, mask_fic)
    return Float64(sum(act_out .* fic_weights))
end

g_ad_fic = Zygote.gradient(fic_loss, act_fic)[1]
g_fd_fic = fd_gradient(fic_loss, act_fic; eps=1e-4)
results["9: FoldIterCore input"] = check_gradient("FoldIterCore d/d(act)", g_ad_fic, g_fd_fic)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 60)
println("  SUMMARY")
println("=" ^ 60)
let n_pass = 0, n_total = 0
    for (name, ok) in sort(collect(results); by=first)
        @printf("  [%s] %s\n", pass_fail(ok), name)
        n_total += 1
        n_pass += ok ? 1 : 0
    end
    @printf("\n  %d/%d tests passed\n", n_pass, n_total)
end
println("=" ^ 60)
