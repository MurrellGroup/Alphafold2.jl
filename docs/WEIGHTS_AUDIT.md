# AF2 Weights Audit and Safetensors Conversion

Generated artifacts (local machine):
- NPZ source directory: `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_npz`
- Safetensors output directory: `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/params_safetensors`
- JSON manifest: `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/AF2_WEIGHTS_AUDIT.json`
- Markdown manifest: `/Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official/AF2_WEIGHTS_AUDIT.md`

Conversion command used:

```bash
python3.11 /Users/benmurrell/JuliaM3/AF2JuliaPort/Alphafold2.jl/scripts/weights/audit_and_convert_af2_weights.py \
  --weights-root /Users/benmurrell/JuliaM3/AF2JuliaPort/af2_weights_official \
  --skip-download \
  --skip-extract
```

Summary:
- total checkpoints audited: `15`
- supported by `Alphafold2.jl`: `15 / 15`
- support levels:
  - `SUPPORTED_PARTIAL_CONFIDENCE`: 5 (`params_model_{1..5}.npz`)
  - `SUPPORTED_FULL_MONOMER`: 5 (`params_model_{1..5}_ptm.npz`)
  - `SUPPORTED_FULL_MULTIMER`: 5 (`params_model_{1..5}_multimer_v3.npz`)

| source_npz | variant | support_level | safetensors filename |
|---|---|---|---|
| `params_model_1.npz` | `monomer` | `SUPPORTED_PARTIAL_CONFIDENCE` | `alphafold2_model_1_dm_2022-12-06.safetensors` |
| `params_model_1_ptm.npz` | `monomer_ptm` | `SUPPORTED_FULL_MONOMER` | `alphafold2_model_1_ptm_dm_2022-12-06.safetensors` |
| `params_model_1_multimer_v3.npz` | `multimer` | `SUPPORTED_FULL_MULTIMER` | `alphafold2_model_1_multimer_v3_dm_2022-12-06.safetensors` |
| `params_model_2.npz` | `monomer` | `SUPPORTED_PARTIAL_CONFIDENCE` | `alphafold2_model_2_dm_2022-12-06.safetensors` |
| `params_model_2_ptm.npz` | `monomer_ptm` | `SUPPORTED_FULL_MONOMER` | `alphafold2_model_2_ptm_dm_2022-12-06.safetensors` |
| `params_model_2_multimer_v3.npz` | `multimer` | `SUPPORTED_FULL_MULTIMER` | `alphafold2_model_2_multimer_v3_dm_2022-12-06.safetensors` |
| `params_model_3.npz` | `monomer` | `SUPPORTED_PARTIAL_CONFIDENCE` | `alphafold2_model_3_dm_2022-12-06.safetensors` |
| `params_model_3_ptm.npz` | `monomer_ptm` | `SUPPORTED_FULL_MONOMER` | `alphafold2_model_3_ptm_dm_2022-12-06.safetensors` |
| `params_model_3_multimer_v3.npz` | `multimer` | `SUPPORTED_FULL_MULTIMER` | `alphafold2_model_3_multimer_v3_dm_2022-12-06.safetensors` |
| `params_model_4.npz` | `monomer` | `SUPPORTED_PARTIAL_CONFIDENCE` | `alphafold2_model_4_dm_2022-12-06.safetensors` |
| `params_model_4_ptm.npz` | `monomer_ptm` | `SUPPORTED_FULL_MONOMER` | `alphafold2_model_4_ptm_dm_2022-12-06.safetensors` |
| `params_model_4_multimer_v3.npz` | `multimer` | `SUPPORTED_FULL_MULTIMER` | `alphafold2_model_4_multimer_v3_dm_2022-12-06.safetensors` |
| `params_model_5.npz` | `monomer` | `SUPPORTED_PARTIAL_CONFIDENCE` | `alphafold2_model_5_dm_2022-12-06.safetensors` |
| `params_model_5_ptm.npz` | `monomer_ptm` | `SUPPORTED_FULL_MONOMER` | `alphafold2_model_5_ptm_dm_2022-12-06.safetensors` |
| `params_model_5_multimer_v3.npz` | `multimer` | `SUPPORTED_FULL_MULTIMER` | `alphafold2_model_5_multimer_v3_dm_2022-12-06.safetensors` |

