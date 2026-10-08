import glob
import os
import gears

print("GEARS_OK", gears.__file__)
pkg = os.path.dirname(gears.__file__)
print("pkg data:", glob.glob(os.path.join(pkg, "data", "*")))
try:
    import numpy
    print("numpy", numpy.__version__)
    import scanpy
    print("scanpy", scanpy.__version__)
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("IMPORT_WARN", repr(e))
