import torch, numpy as np, json, os

srcs = {
    "seed113": "/home/zizhuo/maglab_deploy/checkpoints/magworld_h1_v13_full_seed113.pt",
    "seed227": "/home/zizhuo/maglab_deploy/checkpoints/magworld_h1_v13_full_seed227.pt",
}
outdir = "/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat"
os.makedirs(outdir, exist_ok=True)


def jdefault(o):
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


for tag, path in srcs.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    print(tag, "top-level keys:", list(ck.keys()), flush=True)
    arrays = {}
    meta = {}
    for k, v in ck.items():
        if isinstance(v, torch.Tensor):
            arrays[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
            for name, t in v.items():
                arrays[k + "::" + name] = t.detach().cpu().numpy()
        else:
            meta[k] = v
    np.savez(os.path.join(outdir, tag + ".npz"), **arrays)
    with open(os.path.join(outdir, tag + "_meta.json"), "w") as f:
        json.dump(meta, f, default=jdefault, indent=1)
    print(tag, "meta keys:", sorted(meta.keys()), "| array count:", len(arrays), flush=True)
print("DUMP_DONE", flush=True)
