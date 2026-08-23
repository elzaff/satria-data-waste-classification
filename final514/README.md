# SigLIP2 NaFlex Audited514 ErrorLogLoss BalancedLR A100 Ajeng

Pipeline standalone clean retrain dari dataset mentah. Encoder memakai feature manifest259; BalancedLR memakai final manifest514.

## Pemilihan head

Satu-satunya ablasi ialah kandidat `C`: `0.1`, `0.2`, `0.3`, `0.5`, `0.75`, dan `1.0`.

Urutan ranking validation:

1. error paling sedikit;
2. log-loss terendah;
3. macro-F1 tertinggi;
4. `C` terkecil.

`selected_c` selalu diambil dari baris peringkat pertama `validation_head_ablation.tsv`. Tidak ada `C` recipe, override, kalibrasi, gate, ensemble, atau skenario manifest.

## Konfigurasi

- Model: `google/siglip2-so400m-patch16-naflex`.
- Revision: `cc24074f717b612951c2dead130904ab9b65a81e`.
- NaFlex: maksimum 256 patch.
- Split: 5-fold `StratifiedGroupKFold`, seed 2026; Fold 0 untuk validation.
- GPU: A100 40 GB, BF16.
- Head: Balanced Logistic Regression.
- Training: sama dengan Ajeng456 dan recipe Audited514 sebelumnya.

## Run

```powershell
modal profile activate berlianasrlta
modal run --detach "SATRIA DATA - EKSPERIMEN/evidence/18_naflex_audit514_error_logloss/modal_pipeline.py"
```

Clean ulang root khusus recipe ini:

```powershell
modal run --detach "SATRIA DATA - EKSPERIMEN/evidence/18_naflex_audit514_error_logloss/modal_pipeline.py" --force
```

Artefak besar tersimpan pada volume `bdc2026-model-cache` di:

`/last_hope_final_naflex_audited514_error_logloss_balancedlr_a100_ajeng`

Artefak ringan otomatis masuk folder `results`.

## Hasil clean run 30 Juli 2026

- `C=0.2`, dipilih otomatis oleh validation.
- Validation: 42 error, Macro-F1 `0.9935154626`.
- Evaluasi lokal pasca-run: 0 error dari 1.458 gambar.
- Single model; tanpa calibration, gate, atau ensemble.
- Seluruh 514 perubahan dapat diperiksa pada `swap_contact_sheets/`.
