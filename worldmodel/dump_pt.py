import sys, types
import numpy
# shim: numpy2 pickles reference numpy._core.*; map to numpy.core.*
import numpy.core.multiarray, numpy.core.numeric, numpy.core.numerictypes
core = numpy.core
sys.modules['numpy._core'] = core
for name in ['multiarray', 'numeric', 'numerictypes', 'fromnumeric', 'records', 'function_base']:
    try:
        mod = __import__('numpy.core.' + name, fromlist=['x'])
        sys.modules['numpy._core.' + name] = mod
    except Exception:
        pass

import torch, json
pt = torch.load('/home/zizhuo/maglab_deploy/checkpoints/magworld_h1_v13_full_seed113.pt',
                map_location='cpu', weights_only=False)
print('KEYS:', list(pt.keys()) if isinstance(pt, dict) else type(pt))
if isinstance(pt, dict):
    for k, v in pt.items():
        if k == 'history':
            h = v
            print('HISTORY type:', type(h), 'len:', len(h) if hasattr(h, '__len__') else '?')
            if isinstance(h, list) and h and isinstance(h[0], dict):
                print('entry keys:', list(h[0].keys()))
                for i, e in enumerate(h):
                    if i < 3 or i % 10 == 0 or i >= len(h) - 3:
                        val = e.get('validation') or {}
                        met = val.get('metrics') or {}
                        print(i, 'epoch=', e.get('epoch'), 'train_loss=', e.get('train_loss'),
                              'val=', {kk: met.get(kk) for kk in list(met.keys())[:10]})
            elif isinstance(h, dict):
                print('history dict keys:', list(h.keys())[:30])
                for kk in list(h.keys())[:10]:
                    print(kk, str(h[kk])[:150])
        elif k == 'training_args':
            print('TRAINING_ARGS:')
            print(json.dumps(v, indent=1, default=str))
        elif k == 'state_dict':
            print('state_dict tensors:', len(v))
        else:
            print(k, '=', type(v), str(v)[:300])
