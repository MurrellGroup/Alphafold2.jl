# AGENTS

## Core Rule
No fallbacks. Code that does something unexpected should error. This rule should always be followed.

## Unimplemented Features Ledger Rule
Before any commit, read `/Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/README.md` section `Features not yet implemented` and update it if (and only if) you have implemented and validated an item.

If you discover any unimplemented or partially implemented feature, add it to the README `Features not yet implemented` list immediately with concrete scope and validation status.

## Implementation Placement Rule
For unresolved features, implement behavior to match official Python first. Prefer putting preprocessing-style logic in the monomer-style preprocessing location for codebase sanity when semantics remain equivalent. If official behavior is fundamentally forward-time (for example multimer MSA sampling/masking tied to recycle/ensemble execution), keep it in forward and match official semantics there.
