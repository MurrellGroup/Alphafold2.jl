module AF2SafeTensors

using JSON
using Mmap

const DTYPE_MAP = Dict(
    "F16" => Float16,
    "F32" => Float32,
    "I64" => Int64,
    "I32" => Int32,
    "I16" => Int16,
    "I8" => Int8,
    "U8" => UInt8,
)

struct HeaderEntry
    dtype::String
    shape::Vector{Int}
    data_offsets::Tuple{Int,Int}
end

struct Reader
    path::String
    data::Vector{UInt8}
    header::Dict{String,HeaderEntry}
    base::Int
end

function _read_header(path::String)
    open(path, "r") do io
        header_len = read(io, UInt64)
        header_json = String(read(io, header_len))
        raw = JSON.parse(header_json)
        header = Dict{String,HeaderEntry}()
        for (k, v) in raw
            k == "__metadata__" && continue
            dtype = v["dtype"]
            shape = Vector{Int}(v["shape"])
            offsets = v["data_offsets"]
            header[k] = HeaderEntry(dtype, shape, (offsets[1], offsets[2]))
        end
        base = 8 + Int(header_len)
        return header, base
    end
end

function Reader(path::String)
    header, base = _read_header(path)
    data = Mmap.mmap(path)
    return Reader(path, data, header, base)
end

function read_tensor(reader::Reader, name::String)
    haskey(reader.header, name) || error("Missing tensor in safetensors file: $(name)")
    entry = reader.header[name]
    T = get(DTYPE_MAP, entry.dtype, nothing)
    T === nothing && error("Unsupported safetensors dtype $(entry.dtype) for tensor $(name)")
    start = reader.base + entry.data_offsets[1] + 1
    stop = reader.base + entry.data_offsets[2]
    bytes = @view reader.data[start:stop]
    arr = reinterpret(T, bytes)
    shape = entry.shape
    if isempty(shape)
        return arr[1]
    end
    if length(shape) == 1
        return reshape(arr, shape[1])
    end

    # safetensors are row-major; convert to Julia column-major axis order.
    rev_shape = reverse(shape)
    src = reshape(arr, rev_shape...)
    perm = reverse(1:length(shape))
    return PermutedDimsArray(src, perm)
end

end
