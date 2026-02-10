# Shared feature-pipeline helpers used by both monomer and multimer builders.
# All functions are prefixed `_fp_` to avoid collisions with names in other files.

function _fp_aatype_from_sequence(seq::AbstractString; use_x::Bool=false)
    order = use_x ? restype_order_with_x : restype_order
    out = Vector{Int32}(undef, length(seq))
    i = 1
    for ch in seq
        aa = string(uppercase(ch))
        out[i] = Int32(get(order, aa, 20))
        i += 1
    end
    return out
end

function _fp_hhblits_ids_from_sequence(seq::AbstractString)
    out = Vector{Int32}(undef, length(seq))
    i = 1
    for ch in seq
        out[i] = get(HHBLITS_AA_TO_ID, uppercase(ch), Int32(20))
        i += 1
    end
    return out
end

function _fp_parse_fasta_entries(path::AbstractString)
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

function _fp_parse_fasta_sequences(path::AbstractString)
    return [seq for (_, seq) in _fp_parse_fasta_entries(path)]
end

function _fp_a3m_row_to_aligned_and_deletions(seq::AbstractString)
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

function _fp_msa_ids_from_aligned_seq(seq::AbstractString)
    out = Vector{Int32}(undef, length(seq))
    i = 1
    for ch in seq
        out[i] = get(HHBLITS_AA_TO_ID, uppercase(ch), Int32(20))
        i += 1
    end
    return out
end

"""
    _fp_load_msa_file(path, L, query_row; deduplicate=true, track_taxa=false)

Load an A3M/FASTA MSA file, returning `(msa, deletion_matrix)` when `track_taxa=false`,
or `(msa, deletion_matrix, taxa)` when `track_taxa=true`.
"""
function _fp_load_msa_file(
    msa_path::AbstractString,
    L::Int,
    query_row::AbstractVector{Int32};
    deduplicate::Bool=true,
    track_taxa::Bool=false,
)
    entries = _fp_parse_fasta_entries(msa_path)
    isempty(entries) && error("No sequences found in MSA file: $(msa_path)")

    rows = Vector{Vector{Int32}}()
    dels = Vector{Vector{Int32}}()
    taxa = track_taxa ? String[] : nothing
    seen = deduplicate ? Set{String}() : nothing

    for (desc, seq) in entries
        aligned, del = _fp_a3m_row_to_aligned_and_deletions(seq)
        length(aligned) == L || error("MSA aligned row length $(length(aligned)) != query length $(L)")
        if deduplicate && (aligned in seen)
            continue
        end
        deduplicate && push!(seen, aligned)
        push!(rows, _fp_msa_ids_from_aligned_seq(aligned))
        push!(dels, del)
        track_taxa && push!(taxa, _fp_extract_taxon_label(desc))
    end

    isempty(rows) && error("No usable rows after parsing MSA file: $(msa_path)")

    if rows[1] != query_row
        rows = vcat([collect(query_row)], rows)
        dels = vcat([zeros(Int32, L)], dels)
        if track_taxa
            taxa = vcat([""], taxa)
        end
    else
        track_taxa && (taxa[1] = "")
    end

    S = length(rows)
    msa = zeros(Int32, S, L)
    deletion_matrix = zeros(Float32, S, L)
    for s in 1:S
        msa[s, :] .= rows[s]
        deletion_matrix[s, :] .= Float32.(dels[s])
    end

    if track_taxa
        return msa, deletion_matrix, taxa
    else
        return msa, deletion_matrix
    end
end

function _fp_parse_template_chain(pdb_path::AbstractString, chain_id::Char; use_x::Bool=false)
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
            haskey(atom_order, atom_name) || continue

            x = parse(Float32, strip(line[31:38]))
            y = parse(Float32, strip(line[39:46]))
            z = parse(Float32, strip(line[47:54]))
            residues[residue_index[key]]["atoms"][atom_name] = (x, y, z)
        end
    end

    isempty(residues) && error("No residues parsed from $(pdb_path) chain $(chain_id)")

    seq = join([get(restype_3to1, r["resname"], "X") for r in residues], "")
    L = length(residues)
    template_aatype = _fp_aatype_from_sequence(seq; use_x=use_x)

    template_all_atom_positions = zeros(Float32, 1, L, 37, 3)
    template_all_atom_masks = zeros(Float32, 1, L, 37)
    for i in 1:L
        atoms = residues[i]["atoms"]
        for (name, xyz) in atoms
            idx = atom_order[name] + 1
            template_all_atom_positions[1, i, idx, 1] = xyz[1]
            template_all_atom_positions[1, i, idx, 2] = xyz[2]
            template_all_atom_positions[1, i, idx, 3] = xyz[3]
            template_all_atom_masks[1, i, idx] = 1f0
        end
    end

    return seq, template_aatype, template_all_atom_positions, template_all_atom_masks
end

function _fp_global_align_query_to_template(query_seq::AbstractString, template_seq::AbstractString)
    Lq = length(query_seq)
    Lt = length(template_seq)
    match_score = 1
    mismatch_score = -1
    gap_penalty = -1

    score = zeros(Int, Lq + 1, Lt + 1)
    trace = zeros(UInt8, Lq + 1, Lt + 1)

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

    mapping = zeros(Int, Lq)
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

function _fp_align_template_to_query(
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

    mapping = _fp_global_align_query_to_template(query_seq, template_seq)
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

function _fp_deletion_value_transform(x::AbstractArray)
    return atan.(Float32.(x) ./ 3f0) .* (2f0 / Float32(pi))
end

function _fp_build_target_feat(aatype::AbstractVector{Int32})
    L = length(aatype)
    target_feat = zeros(Float32, L, 22)
    for i in 1:L
        idx = clamp(Int(aatype[i]), 0, 20)
        target_feat[i, 1] = 0f0
        target_feat[i, idx + 2] = 1f0
    end
    return target_feat
end

function _fp_build_msa_feat(
    msa::AbstractMatrix{Int32},
    deletion_matrix::AbstractMatrix{Float32};
    msa_mask::Union{Nothing,AbstractMatrix{Float32}}=nothing,
)
    S, L = size(msa)
    size(deletion_matrix, 1) == S || error("deletion matrix row mismatch")
    size(deletion_matrix, 2) == L || error("deletion matrix col mismatch")

    feat = zeros(Float32, S, L, 49)
    del_val = _fp_deletion_value_transform(deletion_matrix)
    for s in 1:S, i in 1:L
        if msa_mask !== nothing && msa_mask[s, i] <= 0f0
            continue
        end
        tok = clamp(Int(msa[s, i]), 0, 22)
        feat[s, i, tok + 1] = 1f0
        has_del = deletion_matrix[s, i] > 0f0 ? 1f0 : 0f0
        feat[s, i, 24] = has_del
        feat[s, i, 25] = del_val[s, i]
        feat[s, i, 25 + tok + 1] = 1f0
        feat[s, i, 49] = del_val[s, i]
    end
    return feat
end

function _fp_bool_env(name::AbstractString, default::Bool)
    s = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    return s in ("1", "true", "yes", "y", "on")
end

function _fp_correct_msa_restypes!(msa::AbstractMatrix{Int32})
    for s in 1:size(msa, 1), i in 1:size(msa, 2)
        tok = Int(msa[s, i])
        (0 <= tok <= 21) || error("MSA token out of range [0,21]: $(tok)")
        msa[s, i] = MAP_HHBLITS_TO_AF2[tok + 1]
    end
    return msa
end

# _fp_extract_taxon_label is needed by _fp_load_msa_file when track_taxa=true,
# and is also used directly by multimer code. Defined here so the load function
# can reference it.
function _fp_extract_taxon_label(desc::AbstractString)
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
