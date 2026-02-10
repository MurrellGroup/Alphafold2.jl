# Monomer feature builder — returns Dict{String,Any} in-process (no NPZ round-trip).

"""
    build_monomer_features(query_seq; num_recycle=1, msa_file="",
                           template_pdb_arg="", template_chain_arg="A") -> Dict{String,Any}

Build AF2 monomer input features from a single amino-acid sequence. Returns a Dict
suitable for passing directly to `_infer()`.

Environment variables respected:
- `AF2_MONOMER_MAX_MSA_CLUSTERS` (default 512)
- `AF2_MONOMER_MAX_TEMPLATES` (default 4)
- `AF2_MONOMER_REDUCE_MSA_BY_TEMPLATES` (default true)
- `AF2_MONOMER_MAX_EXTRA_MSA` (default 5120)
"""
function build_monomer_features(
    query_seq_raw::AbstractString;
    num_recycle::Integer=1,
    msa_file::AbstractString="",
    template_pdb_arg::AbstractString="",
    template_chain_arg::AbstractString="A",
)::Dict{String,Any}
    query_seq = uppercase(strip(query_seq_raw))
    num_recycle = Int(num_recycle)
    msa_file = strip(msa_file)
    template_pdb_arg = strip(template_pdb_arg)
    template_chain_arg = strip(template_chain_arg)

    isempty(query_seq) && error("Sequence must be non-empty.")

    # --- Templates -----------------------------------------------------------
    has_templates = !isempty(template_pdb_arg)
    template_parsed = Tuple{String,Vector{Int32},Array{Float32,4},Array{Float32,3}}[]
    if has_templates
        pdb_paths = [strip(x) for x in split(template_pdb_arg, ",") if !isempty(strip(x))]
        chain_args = [strip(x) for x in split(template_chain_arg, ",") if !isempty(strip(x))]
        if length(chain_args) == 1 && length(pdb_paths) > 1
            chain_args = fill(chain_args[1], length(pdb_paths))
        end
        length(chain_args) == length(pdb_paths) || error("template_chain count must match template_pdb count (or be a single chain to broadcast)")
        for c in chain_args
            length(c) == 1 || error("Each template_chain entry must be a single character; got: $(c)")
        end
        for i in eachindex(pdb_paths)
            push!(template_parsed, _fp_parse_template_chain(pdb_paths[i], chain_args[i][1]; use_x=false))
        end
    end

    # --- Sequence features ---------------------------------------------------
    aatype = _fp_aatype_from_sequence(query_seq; use_x=false)
    Lq = length(aatype)
    T = length(template_parsed)

    template_aatype_aligned = zeros(Int32, T, Lq)
    template_pos_aligned = zeros(Float32, T, Lq, 37, 3)
    template_mask_aligned = zeros(Float32, T, Lq, 37)
    if has_templates
        for t in 1:T
            template_seq, template_aatype, template_pos, template_mask = template_parsed[t]
            aa_t, pos_t, mask_t, _ = _fp_align_template_to_query(
                query_seq,
                template_seq,
                template_aatype,
                template_pos,
                template_mask,
            )
            template_aatype_aligned[t, :] .= aa_t
            template_pos_aligned[t, :, :, :] .= pos_t
            template_mask_aligned[t, :, :] .= mask_t
        end
    end

    # --- MSA -----------------------------------------------------------------
    seq_mask = ones(Float32, length(aatype))
    query_msa_row = _fp_hhblits_ids_from_sequence(query_seq)
    msa, deletion_matrix = if isempty(msa_file)
        reshape(copy(query_msa_row), 1, :), zeros(Float32, 1, length(aatype))
    else
        _fp_load_msa_file(msa_file, length(aatype), query_msa_row; track_taxa=false)
    end

    _fp_correct_msa_restypes!(msa)
    msa_mask_raw = ones(Float32, size(msa, 1), size(msa, 2))
    residue_index = Int32.(collect(0:(length(aatype) - 1)))

    # --- MSA clustering ------------------------------------------------------
    max_msa_clusters = tryparse(Int, get(ENV, "AF2_MONOMER_MAX_MSA_CLUSTERS", "512"))
    max_msa_clusters === nothing && error("Invalid AF2_MONOMER_MAX_MSA_CLUSTERS")
    max_templates = tryparse(Int, get(ENV, "AF2_MONOMER_MAX_TEMPLATES", "4"))
    max_templates === nothing && error("Invalid AF2_MONOMER_MAX_TEMPLATES")
    reduce_msa_by_templates = _fp_bool_env("AF2_MONOMER_REDUCE_MSA_BY_TEMPLATES", true)
    max_extra_msa = tryparse(Int, get(ENV, "AF2_MONOMER_MAX_EXTRA_MSA", "5120"))
    max_extra_msa === nothing && error("Invalid AF2_MONOMER_MAX_EXTRA_MSA")

    msa_cluster_rows = max_msa_clusters - (reduce_msa_by_templates ? max_templates : 0)
    msa_cluster_rows = max(msa_cluster_rows, 1)
    L = size(msa, 2)
    Sraw = size(msa, 1)
    Ssel = min(Sraw, msa_cluster_rows)

    msa_model = zeros(Int32, msa_cluster_rows, L)
    deletion_matrix_model = zeros(Float32, msa_cluster_rows, L)
    msa_mask_model = zeros(Float32, msa_cluster_rows, L)
    msa_model[1:Ssel, :] .= msa[1:Ssel, :]
    deletion_matrix_model[1:Ssel, :] .= deletion_matrix[1:Ssel, :]
    msa_mask_model[1:Ssel, :] .= Float32.(msa[1:Ssel, :] .!= Int32(21))

    extra_msa = zeros(Int32, max_extra_msa, L)
    extra_deletion_matrix = zeros(Float32, max_extra_msa, L)
    extra_msa_mask = zeros(Float32, max_extra_msa, L)
    nextra_raw = max(0, Sraw - Ssel)
    nextra_keep = min(nextra_raw, max_extra_msa)
    if nextra_keep > 0
        src = (Ssel + 1):(Ssel + nextra_keep)
        extra_msa[1:nextra_keep, :] .= msa[src, :]
        extra_deletion_matrix[1:nextra_keep, :] .= deletion_matrix[src, :]
        extra_msa_mask[1:nextra_keep, :] .= Float32.(msa[src, :] .!= Int32(21))
    end

    target_feat = _fp_build_target_feat(aatype)
    msa_feat = _fp_build_msa_feat(msa_model, deletion_matrix_model; msa_mask=msa_mask_model)
    extra_has_deletion = Float32.(extra_deletion_matrix .> 0f0)
    extra_deletion_value = _fp_deletion_value_transform(extra_deletion_matrix)

    # --- Assemble payload ----------------------------------------------------
    payload = Dict{String,Any}(
        "aatype" => reshape(aatype, :, 1),
        "seq_mask" => seq_mask,
        "residue_index" => residue_index,
        "msa" => msa,
        "deletion_matrix" => deletion_matrix,
        "msa_mask" => msa_mask_raw,
        "msa_mask_model" => msa_mask_model,
        "target_feat" => target_feat,
        "msa_feat" => msa_feat,
        "extra_msa" => extra_msa,
        "extra_deletion_matrix" => extra_deletion_matrix,
        "extra_msa_mask" => extra_msa_mask,
        "extra_has_deletion" => extra_has_deletion,
        "extra_deletion_value" => extra_deletion_value,
        "num_recycle" => Int32(num_recycle),
    )

    if has_templates
        Tp = max_templates
        useT = min(T, Tp)
        tmpl_aatype = zeros(Int32, Tp, Lq)
        tmpl_pos = zeros(Float32, Tp, Lq, 37, 3)
        tmpl_mask = zeros(Float32, Tp, Lq, 37)
        tmpl_present = zeros(Float32, Tp)
        tmpl_sum_probs = zeros(Float32, Tp)
        if useT > 0
            tmpl_aatype[1:useT, :] .= template_aatype_aligned[1:useT, :]
            tmpl_pos[1:useT, :, :, :] .= template_pos_aligned[1:useT, :, :, :]
            tmpl_mask[1:useT, :, :] .= template_mask_aligned[1:useT, :, :]
            tmpl_present[1:useT] .= 1f0
            tmpl_sum_probs[1:useT] .= 1f0
        end
        payload["template_aatype"] = tmpl_aatype
        payload["template_all_atom_positions"] = tmpl_pos
        payload["template_all_atom_masks"] = tmpl_mask
        payload["template_mask"] = tmpl_present
        payload["template_sum_probs"] = tmpl_sum_probs
    end

    return payload
end
