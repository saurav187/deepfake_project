#!/usr/bin/env bash
# Start the second training tar once the train and dev tars are verified.
cd "$(dirname "$0")"
until [[ -f asvspoof5/flac_T_aa.tar.ok && -f asvspoof5/flac_D_aa.tar.ok ]]; do sleep 60; done
exec /home/saurav/deepfake_project/venv/bin/python pdownload.py \
  "https://zenodo.org/api/records/14498691/files/flac_T_ab.tar/content" asvspoof5/flac_T_ab.tar \
  --size 7489587200 --md5 d05be3f4be7a343fdbdd0ed29fdff2e1 --connections 8 >> pdownload_T_ab.log 2>&1
