cd /nfs_beijing_os/zizhuo_vcc/pertseq/norman
URL=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE133nnn/GSE133344/suppl/GSE133344_filtered_matrix.mtx.gz
SIZE=1130430844
N=16
CHUNK=$((SIZE/N+1))
for i in $(seq 0 $((N-1))); do
  s=$((i*CHUNK)); e=$(( (i+1)*CHUNK - 1 ))
  if [ $e -ge $SIZE ]; then e=$((SIZE-1)); fi
  ( for t in 1 2 3 4 5 6 7 8 9 10 11 12; do
      curl -sSL -C - --connect-timeout 15 --max-time 900 -r $s-$e -o seg_$i "$URL" && break
      sleep 4
    done ) &
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo seg_$i; done) > GSE133344_filtered_matrix.mtx.gz
gzip -t GSE133344_filtered_matrix.mtx.gz && echo NORMAN_DL_OK || echo NORMAN_DL_CORRUPT
ls -la GSE133344_filtered_matrix.mtx.gz
