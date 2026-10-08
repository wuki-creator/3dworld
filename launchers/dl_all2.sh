cd /nfs_beijing_os/zizhuo_vcc/pertseq/norman
URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_filtered_matrix.mtx.gz"
SIZE=1130430844
N=16
CHUNK=$((SIZE/N+1))

dl_seg() {
  i=$1
  s=$((i*CHUNK)); e=$(( (i+1)*CHUNK - 1 ))
  if [ $e -ge $SIZE ]; then e=$((SIZE-1)); fi
  want=$((e-s+1))
  for t in $(seq 1 60); do
    cur=0
    if [ -f seg_$i ]; then
      if [ "$(head -c2 seg_$i | xxd -p)" != "1f8b" ]; then
        rm -f seg_$i
      else
        cur=$(stat -c%s seg_$i)
      fi
    fi
    if [ $cur -eq $want ]; then echo "seg_$i OK"; return 0; fi
    curl -sfSL -C - --connect-timeout 15 -r $s-$e -o seg_$i "$URL" || true
    sleep 5
  done
  echo "seg_$i FAILED"
  return 1
}

export -f dl_seg
export URL SIZE CHUNK
seq 0 $((N-1)) | xargs -P 4 -I{} bash -c 'dl_seg {}'
total=0
for i in $(seq 0 $((N-1))); do
  s=$(stat -c%s seg_$i 2>/dev/null || echo 0)
  total=$((total+s))
done
echo "TOTAL=$total EXPECT=$SIZE"
if [ $total -eq $SIZE ]; then
  cat $(for i in $(seq 0 $((N-1))); do echo seg_$i; done) > GSE133344_filtered_matrix.mtx.gz
  gzip -t GSE133344_filtered_matrix.mtx.gz && echo NORMAN_DL_OK || echo NORMAN_DL_CORRUPT
fi
