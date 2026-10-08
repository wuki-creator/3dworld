#!/usr/bin/env bash
SEG="/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/1857 Obese Climp-63 Segmentation"
OUT="/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/unpacked"
mkdir -p "$OUT"
cd "$SEG" || exit 1
for f in *.zip; do
  echo "UNZIP_START $f $(date +%H:%M:%S)"
  if unzip -o -q "$f" -d "$OUT"; then
    echo "UNZIP_OK $f $(date +%H:%M:%S)"
  else
    echo "UNZIP_BAD $f $(date +%H:%M:%S)"
  fi
done
echo "ALL_UNZIPPED $(date +%H:%M:%S)"
for d in "$OUT"/*/; do
  echo "COUNT $(basename "$d") $(find "$d" -type f | wc -l) files $(du -sh "$d" | cut -f1)"
done
echo "STATS_DONE"
