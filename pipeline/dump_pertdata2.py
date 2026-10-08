import os

import gears

pkg = os.path.dirname(gears.__file__)
src = open(os.path.join(pkg, "pertdata.py"), encoding="utf-8").read()
lines = src.splitlines()
for a, b in [(30, 100), (260, 430)]:
    print(f"===== L{a}-{b} =====")
    for i in range(a, min(b, len(lines))):
        print("L%d: %s" % (i, lines[i]))
