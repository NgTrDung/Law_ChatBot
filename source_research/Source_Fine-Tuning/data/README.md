---
language:
- vi
dataset_info:
  features:
  - name: text
    dtype: string
  - name: meta
    struct:
    - name: issuing_agency
      dtype: string
    - name: promulgation_date
      dtype: string
    - name: sign_number
      dtype: string
    - name: signer
      dtype: string
    - name: type
      dtype: string
  - name: content
    dtype: string
  - name: citation
    dtype: string
  splits:
  - name: dieu
    num_bytes: 2492050352
    num_examples: 909509
  - name: khoan
    num_bytes: 2149637870
    num_examples: 1168531
  - name: diem
    num_bytes: 1269688327
    num_examples: 849900
  download_size: 1766417120
  dataset_size: 5911376549
configs:
- config_name: default
  data_files:
  - split: dieu
    path: data/dieu-*
  - split: khoan
    path: data/khoan-*
  - split: diem
    path: data/diem-*
---
