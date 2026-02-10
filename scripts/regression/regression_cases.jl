# Shared pure-Julia regression case definitions used by fixture generation and tests.
#
# 9 cases: 4 monomer + 5 multimer, covering sequence-only, MSA-only, template-only,
# and combined inputs. All multimer cases use the GCN4 homodimer (2 x 30 residues).
#
# Expected quality (CPU, deterministic):
#   Monomer cases (9-29 residues):
#     - monomer_seq_only:       pLDDT ~43, ≤1 clash (short sequence, low confidence)
#     - monomer_msa_only:       pLDDT ~43, 0 clashes
#     - monomer_template_only:  pLDDT ~58, ≤3 clashes (template-guided, still short)
#     - monomer_template_msa:   pLDDT ~58, ≤3 clashes
#   Multimer cases (60 residues, 2 chains):
#     - multimer_seq_only:      pLDDT ~95, 0 clashes
#     - multimer_msa_only:      pLDDT ~95, 0 clashes
#     - multimer_template_msa:  pLDDT ~95, 0 clashes
#     - multimer_template_msa_multi:   pLDDT ~95, 0 clashes
#     - multimer_template_msa_uneven:  pLDDT ~95, 0 clashes
#
# Clash detection: non-bonded atoms (residue sequence gap > 1) with distance < 2.0 Å.
# Multimer cases should ALWAYS have 0 clashes. Small monomer clashes are acceptable
# due to short sequence lengths (the model has very little signal to work with).
#
# GPU notes: TF32 (FAST_MATH) introduces small numerical differences vs CPU. PDBs
# won't be byte-identical, but pLDDT and clash counts should match CPU expectations.
# Use compare_pdb_coordinates() to verify max_abs coordinate diff < 0.2 Å.

function default_regression_params(repo_root::AbstractString)
    return (
        monomer=get(ENV, "AF2_MONOMER_PARAMS", "alphafold2_model_1_ptm_dm_2022-12-06.safetensors"),
        multimer=get(ENV, "AF2_MULTIMER_PARAMS", "alphafold2_model_1_multimer_v3_dm_2022-12-06.safetensors"),
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
            num_recycle=3,
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
            num_recycle=3,
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
            num_recycle=3,
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
            num_recycle=3,
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
        (
            name="multimer_template_msa_multi",
            model=:multimer,
            params_kind=:multimer,
            sequence_arg=string(gcn4_seq, ",", gcn4_seq),
            num_recycle=5,
            msa_files=[
                joinpath(msa_dir, "multimer_chainA_gcn4.a3m"),
                joinpath(msa_dir, "multimer_chainB_gcn4.a3m"),
            ],
            template_pdbs=[
                string(joinpath(template_dir, "gcn4_dimer_template.pdb"), "+", joinpath(template_dir, "gcn4_dimer_template.pdb")),
                string(joinpath(template_dir, "gcn4_dimer_template.pdb"), "+", joinpath(template_dir, "gcn4_dimer_template.pdb")),
            ],
            template_chains=["A+B", "B+A"],
            expected_chain_ids=['A', 'B'],
            description="Two-chain homodimer multimer with two template rows per chain (GCN4)",
        ),
        (
            name="multimer_template_msa_uneven",
            model=:multimer,
            params_kind=:multimer,
            sequence_arg=string(gcn4_seq, ",", gcn4_seq),
            num_recycle=5,
            msa_files=[
                joinpath(msa_dir, "multimer_chainA_gcn4.a3m"),
                joinpath(msa_dir, "multimer_chainB_gcn4.a3m"),
            ],
            template_pdbs=[
                string(joinpath(template_dir, "gcn4_dimer_template.pdb"), "+", joinpath(template_dir, "gcn4_dimer_template.pdb")),
                joinpath(template_dir, "gcn4_dimer_template.pdb"),
            ],
            template_chains=["A+B", "B"],
            expected_chain_ids=['A', 'B'],
            description="Two-chain homodimer multimer with uneven template rows across chains (GCN4)",
        ),
    ]
end
