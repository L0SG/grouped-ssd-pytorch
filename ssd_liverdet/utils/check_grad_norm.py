
def check_grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        if p is None:
            print("?")
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm