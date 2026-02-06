# Layout conversion helpers for AF2 parity checks.
# AF2 python layout is feature-last with optional leading batch.
# Julia layout in this repo is feature-first, batch-last.

# [N, N, C] -> [C, N, N, 1]
function af2_to_first_2d(x::AbstractArray)
    y = permutedims(x, (3, 1, 2))
    return reshape(y, size(y, 1), size(y, 2), size(y, 3), 1)
end

# [N, N, C, B] (python-style with C last) -> [C, N, N, B]
function af2_to_first_2d_batched(x::AbstractArray)
    return permutedims(x, (3, 1, 2, 4))
end

# [C, N, N, B] -> [B, N, N, C]
function first_to_af2_2d(x::AbstractArray)
    return permutedims(x, (4, 2, 3, 1))
end

# [N, C] -> [C, N, 1]
function af2_to_first_3d(x::AbstractArray)
    y = permutedims(x, (2, 1))
    return reshape(y, size(y, 1), size(y, 2), 1)
end

# [C, N, B] -> [B, N, C]
function first_to_af2_3d(x::AbstractArray)
    return permutedims(x, (3, 2, 1))
end
