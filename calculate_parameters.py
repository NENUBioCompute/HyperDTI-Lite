import torch
import geoopt
from torch import nn
from  model import  HyperTriplet
from hyperparameter import  hyperparameter


hp = hyperparameter()
# ========== 统计可学习参数函数 ==========
def count_learnable_params(model, verbose=True):
    total_params = 0
    param_details = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num = param.numel()
        total_params += num
        param_details.append((name, num, tuple(param.shape), type(param).__name__))

    if verbose:
        print("\n========== Learnable Parameters Summary ==========")
        for name, num, shape, ptype in param_details:
            print(f"{name:35s} | {str(shape):20s} | {num:10,d} | {ptype}")
        print("--------------------------------------------------")
        print(f"Total learnable parameters: {total_params:,} "
              f"({total_params / 1e6:.4f} M)")
        print("==================================================\n")

    return total_params


# ========== 主流程 ==========
if __name__ == "__main__":
    model = HyperTriplet(hp)
    total_params = count_learnable_params(model)



# ================================================================
