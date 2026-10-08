import torch, numpy as np, json, os

outdir = "/nfs_beijing_os/zizhuo_vcc/ckpts/v13_compat"
for tag in ["seed113", "seed227"]:
    z = np.load(os.path.join(outdir, tag + ".npz"))
    with open(os.path.join(outdir, tag + "_meta.json")) as f:
        meta = json.load(f)
    ck = dict(meta)
    model_state = {}
    for name in z.files:
        arr = z[name]
        if "::" in name:
            top, sub = name.split("::", 1)
            model_state[sub] = torch.from_numpy(arr)
        else:
            ck[name] = torch.from_numpy(arr)
    if model_state:
        ck["model_state"] = model_state
    dst = os.path.join(outdir, "magworld_h1_v13_full_%s_np1.pt" % tag)
    torch.save(ck, dst)
    print("saved", dst, "| keys:", list(ck.keys()), flush=True)
print("REBUILD_DONE", flush=True)
