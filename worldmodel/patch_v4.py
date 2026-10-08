p = "/home/zizhuo/maglab_deploy/src/train_magworld_h1_v4.py"
src = open(p).read()
shim = '''
# py3.8 compatibility shim (added by zizhuo vcc pipeline)
if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, required=False, help=None, **kwargs):
            super().__init__(option_strings, dest, nargs=0, default=default, required=required, help=help)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))
    argparse.BooleanOptionalAction = _BooleanOptionalAction
'''
if "_BooleanOptionalAction" in src:
    print("already patched")
elif "BooleanOptionalAction" in src:
    lines = src.splitlines(True)
    done = False
    for i, l in enumerate(lines):
        if l.startswith("import argparse") or l.startswith("import "):
            lines.insert(i + 1, shim)
            done = True
            break
    if not done:
        lines.insert(0, "import argparse\n" + shim)
    open(p, "w").write("".join(lines))
    print("patched OK")
else:
    print("no BooleanOptionalAction use found")
