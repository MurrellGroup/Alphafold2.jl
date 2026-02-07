using NPZ

include(joinpath(@__DIR__, "..", "..", "src", "Alphafold2.jl"))
using .Alphafold2

const _HHBLITS_AA_TO_ID = Dict{Char,Int32}(
    'A' => 0, 'B' => 2, 'C' => 1, 'D' => 2, 'E' => 3, 'F' => 4, 'G' => 5, 'H' => 6,
    'I' => 7, 'J' => 20, 'K' => 8, 'L' => 9, 'M' => 10, 'N' => 11, 'O' => 20, 'P' => 12,
    'Q' => 13, 'R' => 14, 'S' => 15, 'T' => 16, 'U' => 1, 'V' => 17, 'W' => 18, 'X' => 20,
    'Y' => 19, 'Z' => 3, '-' => 21,
)

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
    for s in seqs
        aligned, del = _a3m_row_to_aligned_and_deletions(s)
        length(aligned) == L || error("MSA aligned row length $(length(aligned)) != query length $(L)")
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
        error("Usage: julia build_multimer_input_jl.jl <sequences_csv> <out.npz> [num_recycle] [msa_files_csv]")
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

    isempty(seqs) && error("No sequences provided.")
    length(seqs) >= 2 || error("Expected multimer input with at least 2 chains.")

    chain_lens = [length(s) for s in seqs]
    starts = cumsum(vcat(1, chain_lens[1:end-1]))
    total_len = sum(chain_lens)

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

    # Full concatenated query row first.
    push!(rows, vcat(chain_query_rows...))
    push!(dels, zeros(Float32, total_len))

    # Optionally add naive row-index paired rows if all chains have extra rows.
    min_rows = minimum(size(m, 1) for m in chain_msa)
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

    S = length(rows)
    msa = zeros(Int32, S, total_len)
    deletion_matrix = zeros(Float32, S, total_len)
    for s in 1:S
        msa[s, :] .= rows[s]
        deletion_matrix[s, :] .= dels[s]
    end

    msa_mask = Float32.(msa .!= Int32(21))
    extra_msa = copy(msa)
    extra_deletion_matrix = copy(deletion_matrix)
    extra_msa_mask = copy(msa_mask)

    mkpath(dirname(abspath(out_path)))
    NPZ.npzwrite(out_path, Dict(
        "aatype" => reshape(aatype, :, 1),
        "seq_mask" => seq_mask,
        "residue_index" => residue_index,
        "asym_id" => asym_id,
        "entity_id" => entity_id,
        "sym_id" => sym_id,
        "msa" => msa,
        "deletion_matrix" => deletion_matrix,
        "msa_mask" => msa_mask,
        "extra_msa" => extra_msa,
        "extra_deletion_matrix" => extra_deletion_matrix,
        "extra_msa_mask" => extra_msa_mask,
        "num_recycle" => Int32(num_recycle),
    ))

    println("Saved multimer input NPZ to ", out_path)
    println("  chains: ", length(seqs))
    println("  total residues: ", total_len)
    println("  msa rows: ", S)
    println("  num_recycle: ", num_recycle)
end

main()
