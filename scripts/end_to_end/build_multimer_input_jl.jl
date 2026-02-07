using NPZ

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
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

function _parse_fasta_sequences(path::AbstractString)
    seqs = String[]
    cur = IOBuffer()
    open(path, "r") do io
        for raw in eachline(io)
            line = strip(raw)
            isempty(line) && continue
            if startswith(line, '>')
                if position(cur) > 0
                    push!(seqs, String(take!(cur)))
                end
                continue
            end
            print(cur, line)
        end
    end
    if position(cur) > 0
        push!(seqs, String(take!(cur)))
    end
    return seqs
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

function _load_msa_file(msa_path::AbstractString, L::Int, query_row::AbstractVector{Int32})
    seqs = _parse_fasta_sequences(msa_path)
    isempty(seqs) && error("No sequences found in MSA file: $(msa_path)")

    rows = Vector{Vector{Int32}}()
    dels = Vector{Vector{Int32}}()
    seen = Set{String}()
    for s in seqs
        aligned, del = _a3m_row_to_aligned_and_deletions(s)
        length(aligned) == L || error("MSA aligned row length $(length(aligned)) != query length $(L)")
        if aligned in seen
            continue
        end
        push!(seen, aligned)
        push!(rows, _msa_ids_from_aligned_seq(aligned))
        push!(dels, del)
    end

    if rows[1] != query_row
        rows = vcat([collect(query_row)], rows)
        dels = vcat([zeros(Int32, L)], dels)
    end

    S = length(rows)
    msa = zeros(Int32, S, L)
    deletion_matrix = zeros(Float32, S, L)
    for s in 1:S
        msa[s, :] .= rows[s]
        deletion_matrix[s, :] .= Float32.(dels[s])
    end
    return msa, deletion_matrix
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

function main()
    if length(ARGS) < 2
        error("Usage: julia build_multimer_input_jl.jl <sequences_csv> <out.npz> [num_recycle] [msa_files_csv] [template_pdbs_csv] [template_chains_csv]")
    end

    seqs = _split_csv(ARGS[1])
    out_path = ARGS[2]
    num_recycle = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    msa_files = if length(ARGS) >= 4
        parts = [strip(x) for x in split(ARGS[4], ",")]
        length(parts) == length(seqs) || error("msa_files count ($(length(parts))) must match sequence count ($(length(seqs)))")
        parts
    else
        String[]
    end
    template_pdbs = if length(ARGS) >= 5
        [strip(x) for x in split(ARGS[5], ",") if !isempty(strip(x))]
    else
        String[]
    end
    template_chains = if length(ARGS) >= 6
        [strip(x) for x in split(ARGS[6], ",") if !isempty(strip(x))]
    else
        String[]
    end

    isempty(seqs) && error("No sequences provided.")
    length(seqs) >= 2 || error("Expected multimer input with at least 2 chains.")

    chain_lens = [length(s) for s in seqs]
    starts = cumsum(vcat(1, chain_lens[1:end-1]))
    total_len = sum(chain_lens)

    if !isempty(template_pdbs)
        if length(template_pdbs) == 1 && length(seqs) > 1
            template_pdbs = fill(template_pdbs[1], length(seqs))
        end
        if isempty(template_chains)
            template_chains = fill("A", length(seqs))
        elseif length(template_chains) == 1 && length(seqs) > 1
            template_chains = fill(template_chains[1], length(seqs))
        end
        length(template_pdbs) == length(seqs) || error("template_pdbs count ($(length(template_pdbs))) must match sequence count ($(length(seqs)))")
        length(template_chains) == length(seqs) || error("template_chains count ($(length(template_chains))) must match sequence count ($(length(seqs)))")
        for c in template_chains
            length(c) == 1 || error("Each template chain must be a single character; got: $(c)")
        end
    end

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

    for ci in eachindex(seqs)
        if isempty(msa_files)
            q = chain_query_rows[ci]
            chain_msa[ci] = reshape(copy(q), 1, :)
            chain_del[ci] = zeros(Float32, 1, length(q))
        else
            chain_msa[ci], chain_del[ci] = _load_msa_file(msa_files[ci], chain_lens[ci], chain_query_rows[ci])
        end
    end

    rows = Vector{Vector{Int32}}()
    dels = Vector{Vector{Float32}}()
    min_rows = minimum(size(m, 1) for m in chain_msa)

    # Homomer-like case (all chains share one entity id): match AF2 merge behavior
    # by row-wise dense concatenation across chains.
    if length(unique(entity_by_chain)) == 1
        for r in 1:min_rows
            dense_row = Int32[]
            dense_del = Float32[]
            for ci in eachindex(seqs)
                append!(dense_row, vec(chain_msa[ci][r, :]))
                append!(dense_del, vec(chain_del[ci][r, :]))
            end
            push!(rows, dense_row)
            push!(dels, dense_del)
        end
    else
        # Full concatenated query row first.
        push!(rows, vcat(chain_query_rows...))
        push!(dels, zeros(Float32, total_len))

        # Add row-index paired rows if all chains have extra rows.
        if min_rows > 1
            for r in 2:min_rows
                paired_row = Int32[]
                paired_del = Float32[]
                for ci in eachindex(seqs)
                    append!(paired_row, vec(chain_msa[ci][r, :]))
                    append!(paired_del, vec(chain_del[ci][r, :]))
                end
                push!(rows, paired_row)
                push!(dels, paired_del)
            end
        end

        # Add unpaired per-chain rows with gaps outside each chain segment.
        for ci in eachindex(seqs)
            Lc = chain_lens[ci]
            st = starts[ci]
            en = st + Lc - 1
            for r in 2:size(chain_msa[ci], 1)
                row = fill(Int32(21), total_len)
                del = zeros(Float32, total_len)
                row[st:en] .= vec(chain_msa[ci][r, :])
                del[st:en] .= vec(chain_del[ci][r, :])
                push!(rows, row)
                push!(dels, del)
            end
        end
    end

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
    cluster_bias_mask[1] = 1f0

    template_aatype = zeros(Int32, 0, total_len)
    template_all_atom_positions = zeros(Float32, 0, total_len, 37, 3)
    template_all_atom_masks = zeros(Float32, 0, total_len, 37)
    has_templates = !isempty(template_pdbs)
    if has_templates
        # Match AF2 multimer merged-template contract: stack has up to 4 rows,
        # with row 1 carrying merged chain templates and remaining rows zero.
        T = 4
        template_aatype = zeros(Int32, T, total_len)
        template_all_atom_positions = zeros(Float32, T, total_len, 37, 3)
        template_all_atom_masks = zeros(Float32, T, total_len, 37)
        merged_template_aatype = fill(Int32(20), total_len)
        merged_template_positions = zeros(Float32, total_len, 37, 3)
        merged_template_masks = zeros(Float32, total_len, 37)

        for ci in eachindex(seqs)
            chain_seq = seqs[ci]
            t_seq, t_aa, t_pos, t_mask = _parse_template_chain(template_pdbs[ci], template_chains[ci][1])
            aa_aligned, pos_aligned, mask_aligned, mapped = _align_template_to_query(
                chain_seq,
                t_seq,
                t_aa,
                t_pos,
                t_mask,
            )
            st = starts[ci]
            en = st + chain_lens[ci] - 1
            merged_template_aatype[st:en] .= aa_aligned
            merged_template_positions[st:en, :, :] .= pos_aligned
            merged_template_masks[st:en, :] .= mask_aligned
            println(
                "Template chain ", ci, ": mapped residues ", mapped, "/", chain_lens[ci],
                " (query len ", chain_lens[ci], ", template len ", length(t_aa), ")",
            )
        end

        template_aatype[1, :] .= merged_template_aatype
        template_all_atom_positions[1, :, :, :] .= merged_template_positions
        template_all_atom_masks[1, :, :] .= merged_template_masks
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
end

main()
