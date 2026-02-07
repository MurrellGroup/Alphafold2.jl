using NPZ

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

function _aatype_from_sequence(seq::AbstractString)
    out = Vector{Int32}(undef, lastindex(seq))
    i = 1
    for ch in seq
        aa = string(uppercase(ch))
        out[i] = Int32(get(Alphafold2.restype_order, aa, 20))
        i += 1
    end
    return out
end

const _HHBLITS_AA_TO_ID = Dict{Char,Int32}(
    'A' => 0, 'B' => 2, 'C' => 1, 'D' => 2, 'E' => 3, 'F' => 4, 'G' => 5, 'H' => 6,
    'I' => 7, 'J' => 20, 'K' => 8, 'L' => 9, 'M' => 10, 'N' => 11, 'O' => 20, 'P' => 12,
    'Q' => 13, 'R' => 14, 'S' => 15, 'T' => 16, 'U' => 1, 'V' => 17, 'W' => 18, 'X' => 20,
    'Y' => 19, 'Z' => 3, '-' => 21,
)

const _MAP_HHBLITS_TO_AF2 = Int32[
    0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20, 21,
]

function _hhblits_ids_from_sequence(seq::AbstractString)
    out = Vector{Int32}(undef, lastindex(seq))
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
    out = Vector{Int32}(undef, lastindex(seq))
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

    template_all_atom_positions = zeros(Float32, 1, L, 37, 3)
    template_all_atom_masks = zeros(Float32, 1, L, 37)
    for i in 1:L
        atoms = residues[i]["atoms"]
        for (name, xyz) in atoms
            idx = Alphafold2.atom_order[name] + 1
            template_all_atom_positions[1, i, idx, 1] = xyz[1]
            template_all_atom_positions[1, i, idx, 2] = xyz[2]
            template_all_atom_positions[1, i, idx, 3] = xyz[3]
            template_all_atom_masks[1, i, idx] = 1f0
        end
    end

    return seq, template_aatype, template_all_atom_positions, template_all_atom_masks
end

function _global_align_query_to_template(query_seq::AbstractString, template_seq::AbstractString)
    Lq = length(query_seq)
    Lt = length(template_seq)
    match_score = 1
    mismatch_score = -1
    gap_penalty = -1

    score = zeros(Int, Lq + 1, Lt + 1)
    trace = zeros(UInt8, Lq + 1, Lt + 1) # 1=diag, 2=up(query char vs gap), 3=left(gap vs template char)

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

    aatype_aligned = zeros(Int32, Lq)
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
        else
            aatype_aligned[qi] = Int32(20) # unknown
        end
    end
    return aatype_aligned, pos_aligned, mask_aligned, mapped
end

function main()
    if length(ARGS) < 2
        error("Usage: julia build_monomer_input_jl.jl <sequence> <out.npz> [num_recycle] [msa_file] [template_pdb_or_csv] [template_chain_or_csv]")
    end

    query_seq = uppercase(strip(ARGS[1]))
    out_path = ARGS[2]
    num_recycle = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    msa_file = length(ARGS) >= 4 ? strip(ARGS[4]) : ""
    template_pdb_arg = length(ARGS) >= 5 ? strip(ARGS[5]) : ""
    template_chain_arg = length(ARGS) >= 6 ? strip(ARGS[6]) : "A"

    isempty(query_seq) && error("Sequence must be non-empty.")

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
            push!(template_parsed, _parse_template_chain(pdb_paths[i], chain_args[i][1]))
        end
    end

    aatype = _aatype_from_sequence(query_seq)
    Lq = length(aatype)
    T = length(template_parsed)

    template_aatype_aligned = zeros(Int32, T, Lq)
    template_pos_aligned = zeros(Float32, T, Lq, 37, 3)
    template_mask_aligned = zeros(Float32, T, Lq, 37)
    if has_templates
        for t in 1:T
            template_seq, template_aatype, template_pos, template_mask = template_parsed[t]
            aa_t, pos_t, mask_t, mapped = _align_template_to_query(
                query_seq,
                template_seq,
                template_aatype,
                template_pos,
                template_mask,
            )
            template_aatype_aligned[t, :] .= aa_t
            template_pos_aligned[t, :, :, :] .= pos_t
            template_mask_aligned[t, :, :] .= mask_t
            println(
                "Template ", t, ": mapped residues ", mapped, "/", Lq,
                " (raw template length ", length(template_aatype), ")",
            )
        end
    end

    seq_mask = ones(Float32, length(aatype))
    query_msa_row = _hhblits_ids_from_sequence(query_seq)
    msa, deletion_matrix = if isempty(msa_file)
        reshape(copy(query_msa_row), 1, :), zeros(Float32, 1, length(aatype))
    else
        _load_msa_file(msa_file, length(aatype), query_msa_row)
    end
    # Match AF2 model-entry convention after `correct_msa_restypes`.
    for s in 1:size(msa, 1), i in 1:size(msa, 2)
        tok = Int(msa[s, i])
        (0 <= tok <= 21) || error("MSA token out of range [0,21]: $(tok)")
        msa[s, i] = _MAP_HHBLITS_TO_AF2[tok + 1]
    end
    msa_mask = ones(Float32, size(msa, 1), size(msa, 2))
    residue_index = Int32.(collect(0:(length(aatype) - 1)))

    mkpath(dirname(abspath(out_path)))
    payload = Dict{String,Any}(
        "aatype" => reshape(aatype, :, 1),
        "seq_mask" => seq_mask,
        "residue_index" => residue_index,
        "msa" => msa,
        "deletion_matrix" => deletion_matrix,
        "msa_mask" => msa_mask,
        "extra_msa" => msa,
        "extra_deletion_matrix" => deletion_matrix,
        "extra_msa_mask" => msa_mask,
        "num_recycle" => Int32(num_recycle),
    )

    if has_templates
        payload["template_aatype"] = template_aatype_aligned
        payload["template_all_atom_positions"] = template_pos_aligned
        payload["template_all_atom_masks"] = template_mask_aligned
    end
    NPZ.npzwrite(out_path, payload)

    println("Saved monomer input NPZ to ", out_path)
    println("  num_templates: ", T)
    println("  query length: ", Lq)
    println("  msa rows: ", size(msa, 1))
    println("  num_recycle: ", num_recycle)
end

main()
