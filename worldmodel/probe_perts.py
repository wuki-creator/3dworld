# -*- coding: utf-8 -*-
"""探查 xiehon / aissa / dat17 的 perturbation 标签格式（为 05 模型确认命名）"""
import sys
sys.path.insert(0, r"C:\Users\Zizhuo WU\Documents\Kimi\Workspaces\vcc")
from ssh_server import open_channel, run_remote

cli, ch = open_channel(verbose=False)
cmd = (
    "/nfs_beijing/zizhuo/vcc/venv311/bin/python - <<'PY'\n"
    "import scanpy as sc\n"
    "for name in ['xiehon','aissa','dat17']:\n"
    "    try:\n"
    "        ad = sc.read_h5ad(f'/nfs_beijing/zizhuo/vcc/results/core/{name}_processed.h5ad')\n"
    "        p = ad.obs['perturbation'].astype(str)\n"
    "        print(f'== {name}: {ad.n_obs} cells, {p.nunique()} perts')\n"
    "        print(sorted(p.unique())[:15])\n"
    "        print('counts head:', p.value_counts().head(8).to_dict())\n"
    "    except Exception as e:\n"
    "        print(name, 'ERR', str(e)[:100])\n"
    "PY\n"
    "echo __RC__$?"
)
rc, out = run_remote(ch, cmd, timeout=240, verbose=True)
print(out)
print("EXIT:", rc)
ch.close(); cli.close()
