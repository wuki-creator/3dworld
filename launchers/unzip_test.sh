#!/usr/bin/env bash
cd "/local_nvme_data/zizhuo/virtual_life/empiar-10791/1857-obese-climp63/1857 Obese Climp-63 Segmentation" || exit 1
for f in *.zip; do
  if unzip -t "$f" > /dev/null 2>&1; then
    echo "ZIP_OK $f"
  else
    echo "ZIP_BAD $f"
  fi
done
echo ALL_TESTED
