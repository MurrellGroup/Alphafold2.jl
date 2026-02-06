using Test

include(joinpath(@__DIR__, "..", "src", "Alphafold2.jl"))
using .Alphafold2

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
