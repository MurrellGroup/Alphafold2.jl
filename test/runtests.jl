using Test

include(joinpath(@__DIR__, "..", "src", "Alphafold2.jl"))
using .Alphafold2

const _HAS_ZYGOTE = let
    try
        @eval using Zygote
        true
    catch
        false
    end
end

@testset "Layout helpers" begin
    x = randn(Float32, 9, 9, 5)
    y = af2_to_first_2d(x)
    @test size(y) == (5, 9, 9, 1)

    z = dropdims(first_to_af2_2d(y); dims=1)
    @test z ≈ x atol=0 rtol=0
end

@testset "TriangleMultiplication shapes" begin
    cz, ch, L, B = 13, 7, 11, 2
    x = randn(Float32, cz, L, L, B)
    m = rand(Float32, L, L, B) .> 0.2f0
    mask = Float32.(m)

    out_out = TriangleMultiplication(cz, ch; outgoing=true)(x, mask)
    out_in = TriangleMultiplication(cz, ch; outgoing=false)(x, mask)

    @test size(out_out) == size(x)
    @test size(out_in) == size(x)
end

@testset "TriangleAttention shapes" begin
    cz, h, hd, L, B = 20, 4, 5, 9, 2
    x = randn(Float32, cz, L, L, B)
    mask = Float32.(rand(Float32, L, L, B) .> 0.2f0)

    out_row = TriangleAttention(cz, h, hd; orientation=:per_row)(x, mask)
    out_col = TriangleAttention(cz, h, hd; orientation=:per_column)(x, mask)

    @test size(out_row) == size(x)
    @test size(out_col) == size(x)
end

@testset "Transition shapes" begin
    c, n, b = 31, 17, 5
    x = randn(Float32, c, n, b)
    mask = Float32.(rand(Float32, n, b) .> 0.2f0)
    tr = Transition(c, 2.0)
    out = tr(x, mask)
    @test size(out) == size(x)
end

@testset "OuterProductMean shapes" begin
    c_m, c_outer, c_z = 19, 11, 23
    n_seq, n_res, b = 7, 13, 2
    act = randn(Float32, c_m, n_seq, n_res, b)
    mask = Float32.(rand(Float32, n_seq, n_res, b) .> 0.2f0)
    opm = OuterProductMean(c_m, c_outer, c_z)
    out = opm(act, mask)
    @test size(out) == (c_z, n_res, n_res, b)
end

@testset "MSA attention shapes" begin
    c_m, c_z = 17, 19
    n_seq, n_res, b = 6, 11, 2
    h, hd = 4, 5

    msa_act = randn(Float32, c_m, n_seq, n_res, b)
    msa_mask = Float32.(rand(Float32, n_seq, n_res, b) .> 0.2f0)
    pair_act = randn(Float32, c_z, n_res, n_res, b)

    row_att = MSARowAttentionWithPairBias(c_m, c_z, h, hd)
    col_att = MSAColumnAttention(c_m, h, hd)

    out_row = row_att(msa_act, msa_mask, pair_act)
    out_col = col_att(msa_act, msa_mask)

    @test size(out_row) == size(msa_act)
    @test size(out_col) == size(msa_act)
end

@testset "InvariantPointAttention shapes" begin
    c_s, c_z = 31, 17
    c_hidden, h, qk_pts, v_pts = 7, 5, 3, 4
    n, b = 19, 2

    s = randn(Float32, c_s, n, b)
    z = randn(Float32, c_z, n, n, b)
    mask = Float32.(rand(Float32, n, b) .> 0.2f0)
    r = rigid_identity((n, b), s; fmt=:quat)

    ipa = InvariantPointAttention(c_s, c_z, c_hidden, h, qk_pts, v_pts)
    out = ipa(s, z, r, mask)

    @test size(out) == size(s)
end

@testset "EvoformerIteration shapes" begin
    c_m, c_z = 17, 19
    n_seq, n_res, b = 7, 13, 2

    msa = randn(Float32, c_m, n_seq, n_res, b)
    pair = randn(Float32, c_z, n_res, n_res, b)
    msa_mask = Float32.(rand(Float32, n_seq, n_res, b) .> 0.2f0)
    pair_mask = Float32.(rand(Float32, n_res, n_res, b) .> 0.2f0)

    evo = EvoformerIteration(
        c_m,
        c_z;
        num_head_msa=4,
        msa_head_dim=5,
        num_head_pair=4,
        pair_head_dim=5,
        c_outer=11,
        c_tri_mul=13,
        outer_first=true,
    )

    msa_out, pair_out = evo(msa, pair, msa_mask, pair_mask)
    @test size(msa_out) == size(msa)
    @test size(pair_out) == size(pair)
end

@testset "FoldIterationCore shapes" begin
    c_s, c_z = 31, 17
    c_hidden, h, qk_pts, v_pts = 7, 5, 3, 4
    n, b = 13, 2
    ntrans = 3

    act = randn(Float32, c_s, n, b)
    z = randn(Float32, c_z, n, n, b)
    mask = Float32.(rand(Float32, n, b) .> 0.2f0)
    rigid = rigid_identity((n, b), act; fmt=:quat)

    fold = FoldIterationCore(c_s, c_z, c_hidden, h, qk_pts, v_pts, ntrans)
    out_act, out_rigid, out_affine_update = fold(act, z, mask, rigid)

    @test size(out_act) == size(act)
    @test size(out_affine_update) == (6, n, b)
    @test size(Alphafold2.to_tensor_7(out_rigid)) == (7, n, b)
end

@testset "GenerateAffinesCore shapes" begin
    c_s, c_z = 31, 17
    c_hidden, h, qk_pts, v_pts = 7, 5, 3, 4
    n, b = 11, 2
    ntrans = 3
    nlayers = 4

    single = randn(Float32, c_s, n, b)
    pair = randn(Float32, c_z, n, n, b)
    seq_mask = Float32.(rand(Float32, n, b) .> 0.2f0)

    m = GenerateAffinesCore(c_s, c_z, c_hidden, h, qk_pts, v_pts, ntrans, nlayers)
    out_act, out_affine = m(single, pair, seq_mask)

    @test size(out_act) == size(single)
    @test size(out_affine) == (nlayers, 7, n, b)
end

@testset "MultiRigidSidechain shapes" begin
    c_in, c_hidden = 31, 17
    n, b = 13, 2
    nres = 2

    act = randn(Float32, c_in, n, b)
    initial_act = randn(Float32, c_in, n, b)
    rigids = rigid_identity((n, b), act; fmt=:quat)
    aatype = rand(0:19, n, b)

    sc = MultiRigidSidechain(c_in, c_hidden, nres)
    out = sc(rigids, [act, initial_act], aatype)

    @test size(out[:angles_sin_cos]) == (2, 7, n, b)
    @test size(out[:unnormalized_angles_sin_cos]) == (2, 7, n, b)
    @test size(out[:atom_pos]) == (3, 14, n, b)
    @test size(Alphafold2.to_tensor_4x4(out[:frames])) == (4, 4, 8, n, b)
end

@testset "StructureModuleCore shapes" begin
    c_s, c_z = 31, 17
    c_hidden, h, qk_pts, v_pts = 7, 5, 3, 4
    n, b = 11, 2
    ntrans = 3
    nlayers = 4
    position_scale = 10f0
    side_c = 13
    side_nres = 2

    single = randn(Float32, c_s, n, b)
    pair = randn(Float32, c_z, n, n, b)
    seq_mask = Float32.(rand(Float32, n, b) .> 0.2f0)
    aatype = rand(0:19, n, b)

    m = StructureModuleCore(
        c_s,
        c_z,
        c_hidden,
        h,
        qk_pts,
        v_pts,
        ntrans,
        nlayers,
        position_scale,
        side_c,
        side_nres,
    )

    out = m(single, pair, seq_mask, aatype)

    @test size(out[:act]) == size(single)
    @test size(out[:affine]) == (nlayers, 7, n, b)
    @test size(out[:angles_sin_cos]) == (nlayers, 2, 7, n, b)
    @test size(out[:unnormalized_angles_sin_cos]) == (nlayers, 2, 7, n, b)
    @test size(out[:atom_pos]) == (nlayers, 3, 14, n, b)
    @test size(out[:frames]) == (nlayers, 4, 4, 8, n, b)
end

@testset "Training Utils features" begin
    aatype = reshape(Int32[0, 1, 20, 5], 4, 1)

    hard = build_hard_sequence_features(aatype)
    @test size(hard[:target_feat]) == (22, 4, 1)
    @test size(hard[:msa_feat]) == (49, 1, 4, 1)

    seq_logits = randn(Float32, 21, 4, 1)
    soft = build_soft_sequence_features(seq_logits)
    @test size(soft[:seq_probs]) == (21, 4, 1)
    @test size(soft[:target_feat]) == (22, 4, 1)
    @test size(soft[:msa_feat]) == (49, 1, 4, 1)

    seq_mask, msa_mask, residue_index = build_basic_masks(aatype; n_msa_seq=1)
    @test size(seq_mask) == (4, 1)
    @test size(msa_mask) == (1, 4, 1)
    @test size(residue_index) == (4, 1)
    @test residue_index[:, 1] == collect(0:3)

    lddt_logits = randn(Float32, 50, 4, 1)
    @test isfinite(mean_plddt_loss(lddt_logits))
end

if _HAS_ZYGOTE
    @testset "Zygote full-path smoke" begin
        function _relpos_one_hot_test(residue_index::AbstractMatrix{Int}, max_relative_feature::Int)
            L, B = size(residue_index)
            offset = reshape(residue_index, L, 1, B) .- reshape(residue_index, 1, L, B)
            idx = clamp.(offset .+ max_relative_feature, 0, 2 * max_relative_feature)
            oh = Alphafold2.one_hot_last(idx, 2 * max_relative_feature + 1)
            return Float32.(permutedims(oh, (4, 1, 2, 3)))
        end

        c_m, c_z, c_s = 8, 8, 8
        n, b = 5, 1
        h, hd = 2, 4

        aatype = reshape(rand(0:19, n), n, b)
        seq_mask, msa_mask, residue_index = build_basic_masks(aatype; n_msa_seq=1)
        pair_mask = reshape(seq_mask, n, 1, b) .* reshape(seq_mask, 1, n, b)

        model = (
            preprocess_1d=LinearFirst(22, c_m),
            preprocess_msa=LinearFirst(49, c_m),
            left_single=LinearFirst(22, c_z),
            right_single=LinearFirst(22, c_z),
            prev_pos_linear=LinearFirst(15, c_z),
            pair_relpos=LinearFirst(2 * 2 + 1, c_z),
            block=EvoformerIteration(
                c_m,
                c_z;
                num_head_msa=h,
                msa_head_dim=hd,
                num_head_pair=h,
                pair_head_dim=hd,
                c_outer=6,
                c_tri_mul=6,
                outer_first=false,
            ),
            single_activations=LinearFirst(c_m, c_s),
            structure=StructureModuleCore(c_s, c_z, 6, h, 2, 3, 2, 2, 10f0, 8, 2),
            lddt=PredictedLDDTHead(c_s),
        )

        function loss(model, seq_logits)
            seq_feats = build_soft_sequence_features(seq_logits)
            target_feat = seq_feats[:target_feat]
            msa_feat = seq_feats[:msa_feat]

            msa_act = model.preprocess_msa(msa_feat) .+ reshape(model.preprocess_1d(target_feat), c_m, 1, n, b)
            left = model.left_single(target_feat)
            right = model.right_single(target_feat)
            pair_act = reshape(left, c_z, n, 1, b) .+ reshape(right, c_z, 1, n, b)
            pair_act = pair_act .+ model.prev_pos_linear(zeros(Float32, 15, n, n, b))
            pair_act = pair_act .+ model.pair_relpos(_relpos_one_hot_test(residue_index, 2))

            msa_act, pair_act = model.block(msa_act, pair_act, msa_mask, pair_mask)
            single = model.single_activations(view(msa_act, :, 1, :, :))
            struct_out = model.structure(single, pair_act, seq_mask, aatype)

            lddt_logits = model.lddt(struct_out[:act])[:logits]
            return mean_plddt_loss(lddt_logits)
        end

        seq_logits = randn(Float32, 21, n, b)
        g_model, g_seq = Zygote.gradient(loss, model, seq_logits)

        @test sum(abs, g_seq) > 0f0
        @test sum(abs, g_model.preprocess_1d.weight) > 0f0
        @test sum(abs, g_model.block.msa_transition.transition1.weight) > 0f0
        @test sum(abs, g_model.structure.fold_iteration_core.ipa.linear_q.weight) > 0f0
        @test sum(abs, g_model.lddt.logits.weight) > 0f0
    end
else
    @testset "Zygote full-path smoke" begin
        @test_skip false
    end
end

include(joinpath(@__DIR__, "pure_julia_regression_pdb.jl"))
