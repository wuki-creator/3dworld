import glob
import inspect
import os

import gears
from gears import PertData, GEARS

pkg = os.path.dirname(gears.__file__)
src = open(os.path.join(pkg, "pertdata.py"), encoding="utf-8").read()
for i, line in enumerate(src.splitlines()):
    if "go_essential" in line or "data/" in line or "default_pert_graph" in line:
        print("pertdata L%d: %s" % (i, line.strip()))
src2 = open(os.path.join(pkg, "gears.py"), encoding="utf-8").read()
for i, line in enumerate(src2.splitlines()):
    if "go_essential" in line or "pert_graph" in line:
        print("gears L%d: %s" % (i, line.strip()))
print("--- signatures ---")
print("PertData.__init__", inspect.signature(PertData.__init__))
print("PertData.load", inspect.signature(PertData.load))
print("PertData.prepare_split", inspect.signature(PertData.prepare_split))
print("GEARS.__init__", inspect.signature(GEARS.__init__))
print("GEARS.train", inspect.signature(GEARS.train))
print("GEARS methods:", [m for m in dir(GEARS) if not m.startswith("_")])
