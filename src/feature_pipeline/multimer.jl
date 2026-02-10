# Multimer feature builder — returns Dict{String,Any} in-process (no NPZ round-trip).
# Contains multimer-only helpers prefixed `_fp_` plus the public `build_multimer_features()`.
# Random (MersenneTwister, shuffle!) is imported via feature_pipeline.jl.

function _fp_row_similarity(row::AbstractVector{Int32}, query_row::AbstractVector{Int32})
    length(row) == length(query_row) || error("Similarity row length mismatch.")
    matches = 0
    for i in eachindex(row)
        matches += row[i] == query_row[i] ? 1 : 0
    end
    return Float32(matches) / Float32(length(row))
end

function _fp_normalize_pairing_mode(raw::AbstractString)
    s = lowercase(strip(replace(raw, r"\s+" => " ")))
    if s in ("", "block diagonal", "block_diagonal", "blockdiag", "none", "unpaired")
        return :block_diagonal
    elseif s in ("taxon labels matched", "taxon_labels_matched", "taxon", "species")
        return :taxon_labels_matched
    elseif s in ("pair by row index", "pair_by_row_index", "row index", "row_index", "row")
        return :pair_by_row_index
    elseif s in ("random pairing", "random_pairing", "random")
        return :random_pairing
    else
        error("Unsupported pairing mode: $(raw). Supported: block diagonal, taxon labels matched, pair by row index, random pairing")
    end
end

function _fp_split_csv(arg::AbstractString)
    return [uppercase(strip(x)) for x in split(arg, ",") if !isempty(strip(x))]
end

function _fp_parse_template_groups(
    template_pdb_arg::AbstractString,
    template_chain_arg::AbstractString,
    n_chains::Int,
)
    pdb_entries = isempty(strip(template_pdb_arg)) ? String[] : [strip(x) for x in split(template_pdb_arg, ",")]
    chain_entries = isempty(strip(template_chain_arg)) ? String[] : [strip(x) for x in split(template_chain_arg, ",")]

    if isempty(pdb_entries)
        return [String[] for _ in 1:n_chains], [Char[] for _ in 1:n_chains]
    end

    if length(pdb_entries) == 1 && n_chains > 1
        pdb_entries = fill(pdb_entries[1], n_chains)
    end
    length(pdb_entries) == n_chains || error("template_pdbs count ($(length(pdb_entries))) must match sequence count ($(n_chains))")

    if isempty(chain_entries)
        chain_entries = fill("A", n_chains)
    elseif length(chain_entries) == 1 && n_chains > 1
        chain_entries = fill(chain_entries[1], n_chains)
    end
    length(chain_entries) == n_chains || error("template_chains count ($(length(chain_entries))) must match sequence count ($(n_chains))")

    template_pdbs = Vector{Vector{String}}(undef, n_chains)
    template_chains = Vector{Vector{Char}}(undef, n_chains)

    for ci in 1:n_chains
        pdb_group = [strip(x) for x in split(pdb_entries[ci], "+") if !isempty(strip(x))]
        chain_group = [strip(x) for x in split(chain_entries[ci], "+") if !isempty(strip(x))]

        if isempty(pdb_group)
            template_pdbs[ci] = String[]
            template_chains[ci] = Char[]
            continue
        end

        if isempty(chain_group)
            chain_group = ["A"]
        end
        if length(chain_group) == 1 && length(pdb_group) > 1
            chain_group = fill(chain_group[1], length(pdb_group))
        end
        length(chain_group) == length(pdb_group) || error(
            "template_chains entry $(ci) must match template_pdbs entry count; got $(length(chain_group)) chains for $(length(pdb_group)) templates",
        )

        chain_chars = Vector{Char}(undef, length(chain_group))
        for ti in eachindex(chain_group)
            c = chain_group[ti]
            length(c) == 1 || error("Each template chain must be a single character; got: $(c)")
            chain_chars[ti] = c[1]
        end

        template_pdbs[ci] = pdb_group
        template_chains[ci] = chain_chars
    end

    return template_pdbs, template_chains
end

function _fp_entity_and_sym_ids(seqs::Vector{String})
    seq_to_entity = Dict{String,Int32}()
    next_entity = Int32(1)
    entity_by_chain = Vector{Int32}(undef, length(seqs))
    for i in eachindex(seqs)
        s = seqs[i]
        if !haskey(seq_to_entity, s)
            seq_to_entity[s] = next_entity
            next_entity += 1
        end
        entity_by_chain[i] = seq_to_entity[s]
    end

    entity_counts = Dict{Int32,Int32}()
    sym_by_chain = Vector{Int32}(undef, length(seqs))
    for i in eachindex(seqs)
        e = entity_by_chain[i]
        n = get(entity_counts, e, Int32(0)) + Int32(1)
        entity_counts[e] = n
        sym_by_chain[i] = n
    end

    return entity_by_chain, sym_by_chain
end

function _fp_build_multimer_msa_rows(
    chain_msa::Vector{Array{Int32,2}},
    chain_del::Vector{Array{Float32,2}},
    chain_taxa::Vector{Vector{String}},
    chain_lens::Vector{Int},
    starts::Vector{Int},
    total_len::Int,
    pairing_mode::Symbol;
    random_seed::Int=0,
)
    n_chains = length(chain_msa)
    rows = Vector{Vector{Int32}}()
    dels = Vector{Vector{Float32}}()
    cluster_bias = Float32[]

    if pairing_mode == :block_diagonal
        for ci in 1:n_chains
            st = starts[ci]
            en = st + chain_lens[ci] - 1
            for r in 1:size(chain_msa[ci], 1)
                row = fill(Int32(21), total_len)
                del = zeros(Float32, total_len)
                row[st:en] .= vec(chain_msa[ci][r, :])
                del[st:en] .= vec(chain_del[ci][r, :])
                push!(rows, row)
                push!(dels, del)
                push!(cluster_bias, r == 1 ? 1f0 : 0f0)
            end
        end
        return rows, dels, cluster_bias
    end

    # Paired modes: all-chain query row first.
    query_row = Int32[]
    query_del = Float32[]
    for ci in 1:n_chains
        append!(query_row, vec(chain_msa[ci][1, :]))
        append!(query_del, zeros(Float32, chain_lens[ci]))
    end
    push!(rows, query_row)
    push!(dels, query_del)
    push!(cluster_bias, 1f0)

    used = [Set{Int}() for _ in 1:n_chains]
    unpaired_orders = [collect(2:size(chain_msa[ci], 1)) for ci in 1:n_chains]

    if pairing_mode == :pair_by_row_index || pairing_mode == :random_pairing
        nonquery_orders = [collect(2:size(chain_msa[ci], 1)) for ci in 1:n_chains]
        if pairing_mode == :random_pairing
            rng = MersenneTwister(random_seed)
            for ci in 1:n_chains
                shuffle!(rng, nonquery_orders[ci])
            end
        end
        unpaired_orders = nonquery_orders
        k = minimum(length(order) for order in nonquery_orders)
        for t in 1:k
            paired_row = Int32[]
            paired_del = Float32[]
            for ci in 1:n_chains
                r = nonquery_orders[ci][t]
                push!(used[ci], r)
                append!(paired_row, vec(chain_msa[ci][r, :]))
                append!(paired_del, vec(chain_del[ci][r, :]))
            end
            push!(rows, paired_row)
            push!(dels, paired_del)
            push!(cluster_bias, 0f0)
        end
    elseif pairing_mode == :taxon_labels_matched
        species_dicts = Vector{Dict{String,Vector{Int}}}(undef, n_chains)
        for ci in 1:n_chains
            d = Dict{String,Vector{Int}}()
            query = vec(chain_msa[ci][1, :])
            for r in 2:size(chain_msa[ci], 1)
                label = strip(chain_taxa[ci][r])
                isempty(label) && continue
                push!(get!(d, label, Int[]), r)
            end
            for label in keys(d)
                sort!(d[label]; by=r -> (-_fp_row_similarity(view(chain_msa[ci], r, :), query), r))
            end
            species_dicts[ci] = d
        end

        species = sort(collect(union((Set(keys(d)) for d in species_dicts)...)))
        for label in species
            present = [haskey(species_dicts[ci], label) && !isempty(species_dicts[ci][label]) for ci in 1:n_chains]
            present_count = count(identity, present)
            present_count <= 1 && continue
            if any(present[ci] && length(species_dicts[ci][label]) > 600 for ci in 1:n_chains)
                continue
            end
            k = minimum(length(species_dicts[ci][label]) for ci in 1:n_chains if present[ci])
            for t in 1:k
                row = fill(Int32(21), total_len)
                del = zeros(Float32, total_len)
                for ci in 1:n_chains
                    present[ci] || continue
                    r = species_dicts[ci][label][t]
                    push!(used[ci], r)
                    st = starts[ci]
                    en = st + chain_lens[ci] - 1
                    row[st:en] .= vec(chain_msa[ci][r, :])
                    del[st:en] .= vec(chain_del[ci][r, :])
                end
                push!(rows, row)
                push!(dels, del)
                push!(cluster_bias, 0f0)
            end
        end
    else
        error("Unsupported pairing mode: $(pairing_mode)")
    end

    # Append leftover unpaired rows block-diagonal.
    for ci in 1:n_chains
        st = starts[ci]
        en = st + chain_lens[ci] - 1
        paired_seq_set = Set{String}()
        if pairing_mode == :taxon_labels_matched
            for r in used[ci]
                push!(paired_seq_set, join(vec(chain_msa[ci][r, :]), ","))
            end
        end

        row_order = pairing_mode == :random_pairing ? unpaired_orders[ci] : collect(2:size(chain_msa[ci], 1))

        for r in row_order
            (r in used[ci]) && continue
            if pairing_mode == :taxon_labels_matched
                seq_key = join(vec(chain_msa[ci][r, :]), ",")
                seq_key in paired_seq_set && continue
            end
            row = fill(Int32(21), total_len)
            del = zeros(Float32, total_len)
            row[st:en] .= vec(chain_msa[ci][r, :])
            del[st:en] .= vec(chain_del[ci][r, :])
            push!(rows, row)
            push!(dels, del)
            push!(cluster_bias, 0f0)
        end
    end

    return rows, dels, cluster_bias
end

"""
    build_multimer_features(seqs; num_recycle=1, msa_files=String[],
                            template_pdb_arg="", template_chain_arg="",
                            pairing_mode_raw="block diagonal",
                            pairing_seed=0) -> Dict{String,Any}

Build AF2 multimer input features from a vector of chain sequences. Returns a Dict
suitable for passing directly to `_infer()`.

Environment variables respected:
- `AF2_MULTIMER_PAIRING_MODE` (default "block diagonal")
- `AF2_MULTIMER_MIN_MSA_ROWS` (default 129)
- `AF2_MULTIMER_MAX_TEMPLATES` (default 4)
"""
function build_multimer_features(
    seqs_raw::Vector{<:AbstractString};
    num_recycle::Integer=1,
    msa_files::Vector{<:AbstractString}=String[],
    template_pdb_arg::AbstractString="",
    template_chain_arg::AbstractString="",
    pairing_mode_raw::AbstractString=get(ENV, "AF2_MULTIMER_PAIRING_MODE", "block diagonal"),
    pairing_seed::Integer=0,
)::Dict{String,Any}
    seqs = [uppercase(strip(s)) for s in seqs_raw]
    num_recycle = Int(num_recycle)
    template_pdb_arg = strip(template_pdb_arg)
    template_chain_arg = strip(template_chain_arg)
    pairing_mode = _fp_normalize_pairing_mode(pairing_mode_raw)
    pairing_seed = Int(pairing_seed)

    isempty(seqs) && error("No sequences provided.")
    length(seqs) >= 2 || error("Expected multimer input with at least 2 chains.")
    if !isempty(msa_files)
        length(msa_files) == length(seqs) || error("msa_files count ($(length(msa_files))) must match sequence count ($(length(seqs)))")
    end

    chain_lens = [length(s) for s in seqs]
    starts = cumsum(vcat(1, chain_lens[1:end-1]))
    total_len = sum(chain_lens)

    template_pdb_groups, template_chain_groups = _fp_parse_template_groups(template_pdb_arg, template_chain_arg, length(seqs))

    chain_aatype = [_fp_aatype_from_sequence(s; use_x=true) for s in seqs]
    aatype = vcat(chain_aatype...)
    seq_mask = ones(Float32, total_len)

    residue_index = zeros(Int32, total_len)
    asym_id = zeros(Int32, total_len)
    entity_by_chain, sym_by_chain = _fp_entity_and_sym_ids(seqs)
    entity_id = zeros(Int32, total_len)
    sym_id = zeros(Int32, total_len)

    for ci in eachindex(seqs)
        Lc = chain_lens[ci]
        st = starts[ci]
        en = st + Lc - 1
        residue_index[st:en] .= Int32.(0:(Lc - 1))
        asym_id[st:en] .= Int32(ci)
        entity_id[st:en] .= entity_by_chain[ci]
        sym_id[st:en] .= sym_by_chain[ci]
    end

    chain_query_rows = [_fp_hhblits_ids_from_sequence(s) for s in seqs]
    chain_msa = Vector{Array{Int32,2}}(undef, length(seqs))
    chain_del = Vector{Array{Float32,2}}(undef, length(seqs))
    chain_taxa = Vector{Vector{String}}(undef, length(seqs))
    dedup_before_pair = !(pairing_mode in (:pair_by_row_index, :random_pairing))

    for ci in eachindex(seqs)
        if isempty(msa_files) || isempty(strip(msa_files[ci]))
            q = chain_query_rows[ci]
            chain_msa[ci] = reshape(copy(q), 1, :)
            chain_del[ci] = zeros(Float32, 1, length(q))
            chain_taxa[ci] = [""]
        else
            chain_msa[ci], chain_del[ci], chain_taxa[ci] = _fp_load_msa_file(
                msa_files[ci],
                chain_lens[ci],
                chain_query_rows[ci];
                deduplicate=dedup_before_pair,
                track_taxa=true,
            )
        end
    end

    rows, dels, cluster_bias_real = _fp_build_multimer_msa_rows(
        chain_msa,
        chain_del,
        chain_taxa,
        chain_lens,
        starts,
        total_len,
        pairing_mode;
        random_seed=pairing_seed,
    )

    S = length(rows)
    msa = zeros(Int32, S, total_len)
    deletion_matrix = zeros(Float32, S, total_len)
    for s in 1:S
        msa[s, :] .= rows[s]
        deletion_matrix[s, :] .= dels[s]
    end

    _fp_correct_msa_restypes!(msa)

    min_msa_rows = tryparse(Int, get(ENV, "AF2_MULTIMER_MIN_MSA_ROWS", "129"))
    min_msa_rows === nothing && error("Invalid AF2_MULTIMER_MIN_MSA_ROWS")
    n_real_rows = size(msa, 1)
    if size(msa, 1) < min_msa_rows
        pad = min_msa_rows - size(msa, 1)
        msa = vcat(msa, zeros(Int32, pad, total_len))
        deletion_matrix = vcat(deletion_matrix, zeros(Float32, pad, total_len))
    end

    msa_mask = ones(Float32, size(msa, 1), total_len)
    if size(msa, 1) > n_real_rows
        msa_mask[(n_real_rows + 1):end, :] .= 0f0
    end
    extra_msa = copy(msa)
    extra_deletion_matrix = copy(deletion_matrix)
    extra_msa_mask = copy(msa_mask)
    cluster_bias_mask = zeros(Float32, size(msa, 1))
    length(cluster_bias_real) == n_real_rows || error("cluster bias length mismatch")
    cluster_bias_mask[1:n_real_rows] .= cluster_bias_real

    max_templates = tryparse(Int, get(ENV, "AF2_MULTIMER_MAX_TEMPLATES", "4"))
    max_templates === nothing && error("Invalid AF2_MULTIMER_MAX_TEMPLATES")
    max_templates >= 0 || error("AF2_MULTIMER_MAX_TEMPLATES must be non-negative")

    template_aatype = zeros(Int32, 0, total_len)
    template_all_atom_positions = zeros(Float32, 0, total_len, 37, 3)
    template_all_atom_masks = zeros(Float32, 0, total_len, 37)
    has_templates = any(!isempty(g) for g in template_pdb_groups)
    if has_templates
        Tmpl = max_templates
        template_aatype = zeros(Int32, Tmpl, total_len)
        template_all_atom_positions = zeros(Float32, Tmpl, total_len, 37, 3)
        template_all_atom_masks = zeros(Float32, Tmpl, total_len, 37)

        for ci in eachindex(seqs)
            chain_seq = seqs[ci]
            st = starts[ci]
            en = st + chain_lens[ci] - 1
            n_chain_templates = min(length(template_pdb_groups[ci]), Tmpl)
            for ti in 1:n_chain_templates
                t_seq, t_aa, t_pos, t_mask = _fp_parse_template_chain(
                    template_pdb_groups[ci][ti],
                    template_chain_groups[ci][ti];
                    use_x=true,
                )
                aa_aligned, pos_aligned, mask_aligned, _ = _fp_align_template_to_query(
                    chain_seq,
                    t_seq,
                    t_aa,
                    t_pos,
                    t_mask,
                )
                template_aatype[ti, st:en] .= aa_aligned
                template_all_atom_positions[ti, st:en, :, :] .= pos_aligned
                template_all_atom_masks[ti, st:en, :] .= mask_aligned
            end
        end
    end

    out_dict = Dict{String,Any}(
        "aatype" => reshape(aatype, :, 1),
        "seq_mask" => seq_mask,
        "residue_index" => residue_index,
        "asym_id" => asym_id,
        "entity_id" => entity_id,
        "sym_id" => sym_id,
        "msa" => msa,
        "deletion_matrix" => deletion_matrix,
        "msa_mask" => msa_mask,
        "cluster_bias_mask" => cluster_bias_mask,
        "extra_msa" => extra_msa,
        "extra_deletion_matrix" => extra_deletion_matrix,
        "extra_msa_mask" => extra_msa_mask,
        "num_recycle" => Int32(num_recycle),
    )
    if has_templates
        out_dict["template_aatype"] = template_aatype
        out_dict["template_all_atom_positions"] = template_all_atom_positions
        out_dict["template_all_atom_masks"] = template_all_atom_masks
    end

    return out_dict
end
