# Flashinfer + Openevolve DataLog

Feb 11: Initialization.

* Justification: Flashinfer MoE baseline release, solidifcation of baseline intent.

Agreed architecture split:

1. A5000 machine (Ampere, sm_86) → Ollama LLM serving only
2. RTX 4090 laptop (Ada, sm_89)  → OpenEvolve + kernel benchmarks (FP8 support)

* Justification: Ada Lovelace architecture sm_89 FP8 native support, Ampere sm_86 incompatible.
