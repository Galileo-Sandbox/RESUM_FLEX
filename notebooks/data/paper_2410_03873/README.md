# Aggregate data provenance for arXiv:2410.03873

These two CSV files are the portable, read-only inputs used by
`notebooks/paper_2410_03873_reproduction.ipynb`. They were copied from commit
`3a7dc839bccda187b0c704f19d804c0dde7c2c8a` of the original RESuM repository:

| Bundled file | Original path |
|---|---|
| `cnp_v1.6_output.csv` | `examples/legend-neutron-moderator/out/cnp/cnp_v1.6_output.csv` |
| `hf_validation_data_v1.2.csv` | `examples/legend-neutron-moderator/in/mfgp/hf_validation_data_v1.2.csv` |

The bundled files have a normalized final newline. Their SHA-256 checksums are:

```text
1903f03a82c3442389cf588a01227a909e679196b2ba94eb715d14bd6fe1c2a3  cnp_v1.6_output.csv
307723b5b40b42acd68152f4a30f59b120a7b8d0b18d57602c4916eecc6151bf  hf_validation_data_v1.2.csv
```

The files contain aggregate trial-level values, not the event-level simulation
corpus. Their presence makes the notebook independent of a local checkout of
the original codebase, but it does not make CNP retraining or exact historical
model reconstruction possible.
