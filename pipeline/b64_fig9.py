from PIL import Image
import base64, io
im = Image.open("/nfs_beijing/zizhuo/vcc/results/magworld/figures_v2/Fig9_worldmodel.png")
im.thumbnail((1600, 1600))
b = io.BytesIO()
im.save(b, "PNG")
open("/tmp/f9.b64", "w").write(base64.b64encode(b.getvalue()).decode())
print("B64_DONE")
