# Shared pure-Julia regression case definitions used by fixture generation and tests.

function default_regression_params(repo_root::AbstractString)
    root_parent = normpath(joinpath(repo_root, ".."))
    return (
        monomer=get(ENV, "AF2_MONOMER_PARAMS", joinpath(root_parent, "af2_weights_official", "params_npz", "params_model_1_ptm.npz")),
        multimer=get(ENV, "AF2_MULTIMER_PARAMS", joinpath(root_parent, "af2_weights_official", "params_npz", "params_model_1_multimer_v3.npz")),
    )
end

function pure_julia_regression_cases(repo_root::AbstractString)
    msa_dir = joinpath(repo_root, "test", "regression", "msa")
    template_dir = joinpath(repo_root, "test", "regression", "templates")

    glucagon_seq = "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"
    gcn4_seq = "MKQLEDKVEELLSKNYHLENEVARLKKLV"

    return [
        (
            name="monomer_seq_only",
            model=:monomer,
            params_kind=:monomer,
            sequence_arg="ACDEFGHIK",
            num_recycle=1,
            msa_files=String[],
            template_pdbs=String[],
            template_chains=String[],
            expected_chain_ids=['A'],
            description="Monomer from single sequence only",
        ),
        (
            name="monomer_msa_only",
            model=:monomer,
            params_kind=:monomer,
            sequence_arg="ACDEFGHIK",
            num_recycle=1,
            msa_files=[joinpath(msa_dir, "monomer_short.a3m")],
            template_pdbs=String[],
            template_chains=String[],
            expected_chain_ids=['A'],
            description="Monomer from sequence + user MSA",
        ),
        (
            name="monomer_template_only",
            model=:monomer,
            params_kind=:monomer,
            sequence_arg=glucagon_seq,
            num_recycle=1,
            msa_files=String[],
            template_pdbs=[joinpath(template_dir, "glucagon_template.pdb")],
            template_chains=["A"],
            expected_chain_ids=['A'],
            description="Monomer from sequence + template only",
        ),
        (
            name="monomer_template_msa",
            model=:monomer,
            params_kind=:monomer,
            sequence_arg=glucagon_seq,
            num_recycle=1,
            msa_files=[joinpath(msa_dir, "monomer_glucagon.a3m")],
            template_pdbs=[joinpath(template_dir, "glucagon_template.pdb")],
            template_chains=["A"],
            expected_chain_ids=['A'],
            description="Monomer from sequence + MSA + template",
        ),
        (
            name="multimer_seq_only",
            model=:multimer,
            params_kind=:multimer,
            sequence_arg=string(gcn4_seq, ",", gcn4_seq),
            num_recycle=5,
            msa_files=String[],
            template_pdbs=String[],
            template_chains=String[],
            expected_chain_ids=['A', 'B'],
            description="Two-chain homodimer multimer from sequences only (GCN4)",
        ),
        (
            name="multimer_msa_only",
            model=:multimer,
            params_kind=:multimer,
            sequence_arg=string(gcn4_seq, ",", gcn4_seq),
            num_recycle=5,
            msa_files=[
                joinpath(msa_dir, "multimer_chainA_gcn4.a3m"),
                joinpath(msa_dir, "multimer_chainB_gcn4.a3m"),
            ],
            template_pdbs=String[],
            template_chains=String[],
            expected_chain_ids=['A', 'B'],
            description="Two-chain homodimer multimer from sequences + per-chain MSAs (GCN4)",
        ),
        (
            name="multimer_template_msa",
            model=:multimer,
            params_kind=:multimer,
            sequence_arg=string(gcn4_seq, ",", gcn4_seq),
            num_recycle=5,
            msa_files=[
                joinpath(msa_dir, "multimer_chainA_gcn4.a3m"),
                joinpath(msa_dir, "multimer_chainB_gcn4.a3m"),
            ],
            template_pdbs=[
                joinpath(template_dir, "gcn4_dimer_template.pdb"),
                joinpath(template_dir, "gcn4_dimer_template.pdb"),
            ],
            template_chains=["A", "B"],
            expected_chain_ids=['A', 'B'],
            description="Two-chain homodimer multimer from sequences + per-chain MSAs + per-chain templates (GCN4)",
        ),
    ]
end
