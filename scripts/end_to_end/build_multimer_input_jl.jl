using NPZ
using Random

if !isdefined(Main, :Alphafold2)
    include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
end
using .Alphafold2

const _HHBLITS_AA_TO_ID = Dict{Char,Int32}(
    'A' => 0, 'B' => 2, 'C' => 1, 'D' => 2, 'E' => 3, 'F' => 4, 'G' => 5, 'H' => 6,
    'I' => 7, 'J' => 20, 'K' => 8, 'L' => 9, 'M' => 10, 'N' => 11, 'O' => 20, 'P' => 12,
    'Q' => 13, 'R' => 14, 'S' => 15, 'T' => 16, 'U' => 1, 'V' => 17, 'W' => 18, 'X' => 20,
    'Y' => 19, 'Z' => 3, '-' => 21,
)

const _MAP_HHBLITS_TO_AF2 = Int32[
    0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20, 21,
]

function _aatype_from_sequence(seq::AbstractString)
    out = Vector{Int32}(undef, length(seq))
    i = 1
    for ch in seq
        aa = string(uppercase(ch))
        out[i] = Int32(get(Alphafold2.restype_order_with_x, aa, 20))
        i += 1
    end
    return out
end

function _hhblits_ids_from_sequence(seq::AbstractString)
    out = Vector{Int32}(undef, length(seq))
    i = 1
    for ch in seq
        out[i] = get(_HHBLITS_AA_TO_ID, uppercase(ch), Int32(20))
        i += 1
    end
    return out
end

function _parse_fasta_entries(path::AbstractString)
    entries = Tuple{String,String}[]
    cur = IOBuffer()
    current_desc = ""
    saw_header = false
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            if startswith(line, '>')
                if saw_header
                    push!(entries, (current_desc, String(take!(cur))))
                end
                current_desc = strip(line[2:end])
                saw_header = true
                truncate(cur, 0)
                seekstart(cur)
                continue
            end
            saw_header || error("Invalid FASTA/A3M: sequence line before header in $(path)")
            print(cur, line)
        end
    end
    if saw_header
        push!(entries, (current_desc, String(take!(cur))))
    end
    return entries
end

function _extract_taxon_label(desc::AbstractString)
    m = match(r"OX=(\d+)", desc)
    m !== nothing && return "OX:" * m.captures[1]
    m = match(r"(?i)TaxID=(\d+)", desc)
    m !== nothing && return "TAXID:" * m.captures[1]
    m = match(r"(?i)taxon[:=]([A-Za-z0-9_.-]+)", desc)
    m !== nothing && return "TAXON:" * lowercase(m.captures[1])
    m = match(r"(?i)OS=([^=]+?)(?:\s[A-Z]{1,3}=|$)", desc)
    m !== nothing && return "OS:" * lowercase(strip(m.captures[1]))
    return ""
end

function _parse_fasta_sequences(path::AbstractString)
    return [seq for (_, seq) in _parse_fasta_entries(path)]
end

function _a3m_row_to_aligned_and_deletions(seq::AbstractString)
    aligned = IOBuffer()
    del = Int32[]
    del_count = 0
    for ch in seq
        if islowercase(ch)
            del_count += 1
        else
            print(aligned, uppercase(ch))
            push!(del, Int32(del_count))
            del_count = 0
        end
    end
    return String(take!(aligned)), del
end

function _msa_ids_from_aligned_seq(seq::AbstractString)
    out = Vector{Int32}(undef, length(seq))
    i = 1
    for ch in seq
        out[i] = get(_HHBLITS_AA_TO_ID, uppercase(ch), Int32(20))
        i += 1
    end
    return out
end

function _load_msa_file(
    msa_path::AbstractString,
    L::Int,
    query_row::AbstractVector{Int32};
    deduplicate::Bool=true,
)
    entries = _parse_fasta_entries(msa_path)
    isempty(entries) && error("No sequences found in MSA file: $(msa_path)")

    rows = Vector{Vector{Int32}}()
    dels = Vector{Vector{Int32}}()
    taxa = String[]
    seen = Set{String}()
    for (desc, seq) in entries
        aligned, del = _a3m_row_to_aligned_and_deletions(seq)
        length(aligned) == L || error("MSA aligned row length $(length(aligned)) != query length $(L)")
        if deduplicate && (aligned in seen)
            continue
        end
        deduplicate && push!(seen, aligned)
        push!(rows, _msa_ids_from_aligned_seq(aligned))
        push!(dels, del)
        push!(taxa, _extract_taxon_label(desc))
    end

    isempty(rows) && error("No usable rows after parsing MSA file: $(msa_path)")
    if rows[1] != query_row
        rows = vcat([collect(query_row)], rows)
        dels = vcat([zeros(Int32, L)], dels)
        taxa = vcat([""], taxa)
    else
        taxa[1] = ""
    end

    S = length(rows)
    msa = zeros(Int32, S, L)
    deletion_matrix = zeros(Float32, S, L)
    for s in 1:S
        msa[s, :] .= rows[s]
        deletion_matrix[s, :] .= Float32.(dels[s])
    end
    return msa, deletion_matrix, taxa
end

function _row_similarity(row::AbstractVector{Int32}, query_row::AbstractVector{Int32})
    length(row) == length(query_row) || error("Similarity row length mismatch.")
    matches = 0
    for i in eachindex(row)
        matches += row[i] == query_row[i] ? 1 : 0
    end
    return Float32(matches) / Float32(length(row))
end

function _normalize_pairing_mode(raw::AbstractString)
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

function _parse_template_chain(pdb_path::AbstractString, chain_id::Char)
    residues = Vector{Dict{String,Any}}()
    residue_index = Dict{Tuple{String,Char},Int}()

    open(pdb_path, "r") do io
        for line in eachline(io)
            startswith(line, "ATOM") || continue
            length(line) >= 54 || continue
            line[22] == chain_id || continue

            resseq = strip(line[23:26])
            icode = line[27]
            key = (resseq, icode)
            if !haskey(residue_index, key)
                residue_index[key] = length(residues) + 1
                push!(residues, Dict{String,Any}(
                    "resname" => strip(line[18:20]),
                    "atoms" => Dict{String,NTuple{3,Float32}}(),
                ))
            end

            atom_name = strip(line[13:16])
            haskey(Alphafold2.atom_order, atom_name) || continue

            x = parse(Float32, strip(line[31:38]))
            y = parse(Float32, strip(line[39:46]))
            z = parse(Float32, strip(line[47:54]))
            residues[residue_index[key]]["atoms"][atom_name] = (x, y, z)
        end
    end

    isempty(residues) && error("No residues parsed from $(pdb_path) chain $(chain_id)")

    seq = join([get(Alphafold2.restype_3to1, r["resname"], "X") for r in residues], "")
    L = length(residues)
    template_aatype = _aatype_from_sequence(seq)
    template_pos = zeros(Float32, 1, L, 37, 3)
    template_mask = zeros(Float32, 1, L, 37)
    for i in 1:L
        atoms = residues[i]["atoms"]
        for (name, xyz) in atoms
            idx = Alphafold2.atom_order[name] + 1
            template_pos[1, i, idx, 1] = xyz[1]
            template_pos[1, i, idx, 2] = xyz[2]
            template_pos[1, i, idx, 3] = xyz[3]
            template_mask[1, i, idx] = 1f0
        end
    end
    return seq, template_aatype, template_pos, template_mask
end

function _global_align_query_to_template(query_seq::AbstractString, template_seq::AbstractString)
    Lq = length(query_seq)
    Lt = length(template_seq)
    match_score = 1
    mismatch_score = -1
    gap_penalty = -1

    score = zeros(Int, Lq + 1, Lt + 1)
    trace = zeros(UInt8, Lq + 1, Lt + 1) # 1=diag, 2=up, 3=left

    for i in 1:Lq
        score[i + 1, 1] = score[i, 1] + gap_penalty
        trace[i + 1, 1] = 2
    end
    for j in 1:Lt
        score[1, j + 1] = score[1, j] + gap_penalty
        trace[1, j + 1] = 3
    end

    for i in 1:Lq
        qi = query_seq[i]
        for j in 1:Lt
            tj = template_seq[j]
            diag = score[i, j] + (uppercase(qi) == uppercase(tj) ? match_score : mismatch_score)
            up = score[i, j + 1] + gap_penalty
            left = score[i + 1, j] + gap_penalty
            best = diag
            dir = UInt8(1)
            if up > best
                best = up
                dir = UInt8(2)
            end
            if left > best
                best = left
                dir = UInt8(3)
            end
            score[i + 1, j + 1] = best
            trace[i + 1, j + 1] = dir
        end
    end

    mapping = zeros(Int, Lq) # query index -> template index (0 if gap)
    i = Lq
    j = Lt
    while i > 0 || j > 0
        dir = trace[i + 1, j + 1]
        if dir == 1
            mapping[i] = j
            i -= 1
            j -= 1
        elseif dir == 2
            mapping[i] = 0
            i -= 1
        else
            j -= 1
        end
    end
    return mapping
end

function _align_template_to_query(
    query_seq::AbstractString,
    template_seq::AbstractString,
    template_aatype::AbstractVector{Int32},
    template_pos::AbstractArray,
    template_mask::AbstractArray,
)
    Lq = length(query_seq)
    Lt = length(template_seq)

    aatype_aligned = fill(Int32(20), Lq)
    pos_aligned = zeros(Float32, Lq, 37, 3)
    mask_aligned = zeros(Float32, Lq, 37)

    if Lq == Lt
        aatype_aligned .= template_aatype
        pos_aligned .= template_pos[1, :, :, :]
        mask_aligned .= template_mask[1, :, :]
        return aatype_aligned, pos_aligned, mask_aligned, Lq
    end

    mapping = _global_align_query_to_template(query_seq, template_seq)
    mapped = 0
    for qi in 1:Lq
        tj = mapping[qi]
        if tj > 0
            aatype_aligned[qi] = template_aatype[tj]
            pos_aligned[qi, :, :] .= template_pos[1, tj, :, :]
            mask_aligned[qi, :] .= template_mask[1, tj, :]
            mapped += 1
        end
    end
    return aatype_aligned, pos_aligned, mask_aligned, mapped
end

function _split_csv(arg::AbstractString)
    return [uppercase(strip(x)) for x in split(arg, ",") if !isempty(strip(x))]
end

function _parse_template_groups(
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

function _entity_and_sym_ids(seqs::Vector{String})
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

function _build_multimer_msa_rows(
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
                sort!(d[label]; by=r -> (-_row_similarity(view(chain_msa[ci], r, :), query), r))
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

function build_multimer_input(
    seqs_raw::Vector{String},
    out_path::AbstractString;
    num_recycle::Integer=1,
    msa_files::Vector{String}=String[],
    template_pdb_arg::AbstractString="",
    template_chain_arg::AbstractString="",
    pairing_mode_raw::AbstractString=get(ENV, "AF2_MULTIMER_PAIRING_MODE", "block diagonal"),
    pairing_seed::Integer=0,
)
    seqs = [uppercase(strip(s)) for s in seqs_raw]
    out_path = String(out_path)
    num_recycle = Int(num_recycle)
    template_pdb_arg = strip(template_pdb_arg)
    template_chain_arg = strip(template_chain_arg)
    pairing_mode = _normalize_pairing_mode(pairing_mode_raw)
    pairing_seed = Int(pairing_seed)

    isempty(seqs) && error("No sequences provided.")
    length(seqs) >= 2 || error("Expected multimer input with at least 2 chains.")
    if !isempty(msa_files)
        length(msa_files) == length(seqs) || error("msa_files count ($(length(msa_files))) must match sequence count ($(length(seqs)))")
    end

    chain_lens = [length(s) for s in seqs]
    starts = cumsum(vcat(1, chain_lens[1:end-1]))
    total_len = sum(chain_lens)

    template_pdb_groups, template_chain_groups = _parse_template_groups(template_pdb_arg, template_chain_arg, length(seqs))

    chain_aatype = [_aatype_from_sequence(s) for s in seqs]
    aatype = vcat(chain_aatype...)
    seq_mask = ones(Float32, total_len)

    residue_index = zeros(Int32, total_len)
    asym_id = zeros(Int32, total_len)
    entity_by_chain, sym_by_chain = _entity_and_sym_ids(seqs)
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

    chain_query_rows = [_hhblits_ids_from_sequence(s) for s in seqs]
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
            chain_msa[ci], chain_del[ci], chain_taxa[ci] = _load_msa_file(
                msa_files[ci],
                chain_lens[ci],
                chain_query_rows[ci];
                deduplicate=dedup_before_pair,
            )
        end
    end

    rows, dels, cluster_bias_real = _build_multimer_msa_rows(
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

    # Convert HHblits MSA IDs into AF2 residue index order (matching Python
    # `feature_processing._correct_msa_restypes`).
    for s in 1:size(msa, 1), i in 1:size(msa, 2)
        tok = Int(msa[s, i])
        (0 <= tok <= 21) || error("MSA token out of range [0,21]: $(tok)")
        msa[s, i] = _MAP_HHBLITS_TO_AF2[tok + 1]
    end

    # Keep MSA row budget small enough for CPU-friendly parity/regression while
    # preserving AF2 multimer's sampled-msa regime.
    min_msa_rows = tryparse(Int, get(ENV, "AF2_MULTIMER_MIN_MSA_ROWS", "129"))
    min_msa_rows === nothing && error("Invalid AF2_MULTIMER_MIN_MSA_ROWS")
    n_real_rows = size(msa, 1)
    if size(msa, 1) < min_msa_rows
        pad = min_msa_rows - size(msa, 1)
        # Match AF2 multimer `pad_msa`: padded token value is 0.
        msa = vcat(msa, zeros(Int32, pad, total_len))
        deletion_matrix = vcat(deletion_matrix, zeros(Float32, pad, total_len))
    end

    # Match AF2 multimer `_make_msa_mask`: ones for real rows, zeros for
    # padded rows; residue-level seq padding is absent for native builders.
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
        # Match AF2 multimer merged-template contract by building per-template
        # rows across chains (up to max_templates), with zero-padded rows.
        T = max_templates
        template_aatype = zeros(Int32, T, total_len)
        template_all_atom_positions = zeros(Float32, T, total_len, 37, 3)
        template_all_atom_masks = zeros(Float32, T, total_len, 37)

        for ci in eachindex(seqs)
            chain_seq = seqs[ci]
            st = starts[ci]
            en = st + chain_lens[ci] - 1
            n_chain_templates = min(length(template_pdb_groups[ci]), T)
            for ti in 1:n_chain_templates
                t_seq, t_aa, t_pos, t_mask = _parse_template_chain(
                    template_pdb_groups[ci][ti],
                    template_chain_groups[ci][ti],
                )
                aa_aligned, pos_aligned, mask_aligned, mapped = _align_template_to_query(
                    chain_seq,
                    t_seq,
                    t_aa,
                    t_pos,
                    t_mask,
                )
                template_aatype[ti, st:en] .= aa_aligned
                template_all_atom_positions[ti, st:en, :, :] .= pos_aligned
                template_all_atom_masks[ti, st:en, :] .= mask_aligned
                println(
                    "Template chain ", ci, " row ", ti, ": mapped residues ", mapped, "/", chain_lens[ci],
                    " (query len ", chain_lens[ci], ", template len ", length(t_aa), ")",
                )
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

    mkpath(dirname(abspath(out_path)))
    NPZ.npzwrite(out_path, out_dict)

    println("Saved multimer input NPZ to ", out_path)
    println("  chains: ", length(seqs))
    println("  total residues: ", total_len)
    println("  msa rows: ", size(msa, 1))
    println("  num_recycle: ", num_recycle)
    println("  pairing_mode: ", string(pairing_mode))
end

function main()
    if length(ARGS) < 2
        error("Usage: julia build_multimer_input_jl.jl <sequences_csv> <out.npz> [num_recycle] [msa_files_csv] [template_pdbs_csv] [template_chains_csv] [pairing_mode] [pairing_seed]")
    end

    seqs = _split_csv(ARGS[1])
    out_path = ARGS[2]
    num_recycle = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    msa_files = if length(ARGS) >= 4
        [strip(x) for x in split(ARGS[4], ",")]
    else
        String[]
    end
    template_pdb_arg = length(ARGS) >= 5 ? ARGS[5] : ""
    template_chain_arg = length(ARGS) >= 6 ? ARGS[6] : ""
    pairing_mode = length(ARGS) >= 7 ? ARGS[7] : get(ENV, "AF2_MULTIMER_PAIRING_MODE", "block diagonal")
    pairing_seed_raw = length(ARGS) >= 8 ? ARGS[8] : get(ENV, "AF2_MULTIMER_PAIRING_SEED", "0")
    pairing_seed = tryparse(Int, strip(pairing_seed_raw))
    pairing_seed === nothing && error("Invalid pairing seed: $(pairing_seed_raw)")

    return build_multimer_input(
        seqs,
        out_path;
        num_recycle=num_recycle,
        msa_files=msa_files,
        template_pdb_arg=template_pdb_arg,
        template_chain_arg=template_chain_arg,
        pairing_mode_raw=pairing_mode,
        pairing_seed=pairing_seed,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
