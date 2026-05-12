import sys
sys.path.insert(0, 'src')
from train_models import create_resnet50, create_efficientnet_b0

for fn in (create_resnet50, create_efficientnet_b0):
    try:
        m = fn(16)
        print(f"{fn.__name__}: OK")
    except Exception as e:
        print(f"{fn.__name__}: ERROR -> {type(e).__name__}: {e}")
