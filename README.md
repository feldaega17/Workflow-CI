# Workflow CI & MLProject

## Cara Menjalankan Lokal
1. `cd MLProject`
2. `pip install -r requirements.txt`
3. `mlflow run . --env-manager=local`

## GitHub Actions
File `.github/workflows/retrain.yml` akan berjalan otomatis saat push ke main. Artifact `mlruns` akan diupload.

## Checklist
- [x] Basic
- [x] Skilled
- [x] Advance
