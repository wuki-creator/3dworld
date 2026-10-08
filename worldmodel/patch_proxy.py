
import io
p = "/nfs_beijing_os/zizhuo_vcc/work/proxy_score.py"
src = open(p, encoding="utf-8").read()
old = '("model_world_h1_v20", "WorldModelH1V20"),'
new = '("model_world_h1_v24", "WorldModelH1V24"),
        ' + old
assert src.count(old) == 1, "anchor count %d" % src.count(old)
assert "model_world_h1_v24" not in src
open(p, "w", encoding="utf-8").write(src.replace(old, new))
print("patched ok")
