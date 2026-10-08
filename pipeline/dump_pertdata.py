import os

import gears

pkg = os.path.dirname(gears.__file__)
src = open(os.path.join(pkg, "pertdata.py"), encoding="utf-8").read()
lines = src.splitlines()
for a, b in [(100, 150), (150, 260)]:
    print(f"===== L{a}-{b} =====")
    for i in range(a, min(b, len(lines))):
        print("L%d: %s" % (i, lines[i]))
