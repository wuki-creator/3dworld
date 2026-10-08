p = "/nfs_beijing_os/zizhuo_vcc/work/ensemble_predict_h1.py"
s = open(p, encoding="utf-8").read()
CRLF = chr(13) + chr(10)

a_old = 'p.add_argument("--cells-per-target", type=int, default=400)'
a_new = a_old + chr(10) + 'p.add_argument("--per-target-counts", default="")'
assert s.count(a_old) == 1, "anchor a %d" % s.count(a_old)
s = s.replace(a_old, a_new)

b_old = "a = p.parse_args()"
inject = chr(10).join([
    "",
    "import csv as _csv",
    "count_map = {}",
    "if a.per_target_counts:",
    "    with open(a.per_target_counts, newline='') as _fh:",
    "        for _row in _csv.reader(_fh):",
    "            if len(_row) >= 2 and _row[1].strip().isdigit():",
    "                count_map[_row[0].strip()] = int(_row[1])",
    "    print('per-target counts loaded: %d' % len(count_map), flush=True)",
    "",
])
assert s.count(b_old) == 1, "anchor b %d" % s.count(b_old)
s = s.replace(b_old, b_old + inject)

c_old = "    for position, target in enumerate(targets):" + chr(10) + "        rng = np.random.default_rng(a.seed + 10000 * context_index + position)"
c_new = "    for position, target in enumerate(targets):" + chr(10) + "        n_cells = count_map.get(target, a.cells_per_target)" + chr(10) + "        rng = np.random.default_rng(a.seed + 10000 * context_index + position)"
assert s.count(c_old) == 1, "anchor c %d" % s.count(c_old)
s = s.replace(c_old, c_new)

d_old = "decoded = V.decode_tight(gene_mean, cal_pos, a.cells_per_target, rng,"
d_new = "decoded = V.decode_tight(gene_mean, cal_pos, n_cells, rng,"
assert s.count(d_old) == 1, "anchor d %d" % s.count(d_old)
s = s.replace(d_old, d_new)

e1_old = "obs_targets.extend([target] * a.cells_per_target)"
e1_new = "obs_targets.extend([target] * n_cells)"
assert s.count(e1_old) == 1, "anchor e1 %d" % s.count(e1_old)
s = s.replace(e1_old, e1_new)

e2_old = "obs_contexts.extend([context] * a.cells_per_target)"
e2_new = "obs_contexts.extend([context] * n_cells)"
assert s.count(e2_old) == 1, "anchor e2 %d" % s.count(e2_old)
s = s.replace(e2_old, e2_new)

open(p, "w", encoding="utf-8", newline="").write(s)
print("patched ok")
