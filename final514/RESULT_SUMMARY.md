# Ringkasan Hasil Final514 ErrorLogLoss

## Identitas run

- Tanggal: 30 Juli 2026.
- Modal profile: `berlianasrlta`.
- App ID: `ap-MFJZa8nbIW2v2h4vDt5U4s`.
- GPU: NVIDIA A100-SXM4-40GB.
- Model: `google/siglip2-so400m-patch16-naflex`.
- Revision: `cc24074f717b612951c2dead130904ab9b65a81e`.
- Seed: 2026.
- Waktu run: `2026-07-30T01:51:08Z` sampai `2026-07-30T03:09:47Z`.

## Recipe

- Feature manifest: 259 perubahan.
- Final manifest: 514 perubahan.
- Head: Balanced Logistic Regression.
- Kandidat `C`: `0.1`, `0.2`, `0.3`, `0.5`, `0.75`, `1.0`.
- Ranking: error naik, log-loss naik, Macro-F1 turun, lalu `C` naik.
- `C` terpilih: `0.2`.
- Tidak ada data-scenario ablation, calibration, gate, atau ensemble.

## Hasil

- Validation error: 42 / 5.308.
- Validation Macro-F1: `0.9935154625965161`.
- Evaluasi lokal pasca-run: 0 / 1.458 error.
- Margin inference minimum: `0.026033` pada ID261.

Outcome lokal diperiksa setelah recipe dipilih dari validation. Outcome tersebut bukan input bagi ranking `C`.

## Artefak

Artefak ringan berada di `results/`. Checkpoint dan embedding penuh berada di Modal Volume:

`/last_hope_final_naflex_audited514_error_logloss_balancedlr_a100_ajeng`

Seluruh 514 gambar audit tersedia dalam 20 halaman pada `swap_contact_sheets/`.
