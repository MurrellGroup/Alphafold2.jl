# Feature pipeline — in-process feature builders for monomer and multimer inputs.
# Replaces the subprocess + NPZ round-trip previously used by fold() and regression helpers.

using Random: MersenneTwister, shuffle!

const HHBLITS_AA_TO_ID = Dict{Char,Int32}(
    'A' => 0, 'B' => 2, 'C' => 1, 'D' => 2, 'E' => 3, 'F' => 4, 'G' => 5, 'H' => 6,
    'I' => 7, 'J' => 20, 'K' => 8, 'L' => 9, 'M' => 10, 'N' => 11, 'O' => 20, 'P' => 12,
    'Q' => 13, 'R' => 14, 'S' => 15, 'T' => 16, 'U' => 1, 'V' => 17, 'W' => 18, 'X' => 20,
    'Y' => 19, 'Z' => 3, '-' => 21,
)

const MAP_HHBLITS_TO_AF2 = Int32[
    0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20, 21,
]

include("feature_pipeline/common.jl")
include("feature_pipeline/monomer.jl")
include("feature_pipeline/multimer.jl")
