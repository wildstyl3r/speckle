import torch


def param_count(model: torch.nn.Module):
    # tch VarStore flags every variable trainable at creation; requires_grad_(false)
    # only freezes autograd, so Rust scheme.txt always reports trainable == total.
    total = sum(t.numel() for t in model.state_dict().values())
    trainable = total
    return total, trainable


def write_summary(model: torch.nn.Module, wr) -> None:
    grad_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    sorted_vars = sorted(model.state_dict().items())
    wr.write("-" * 85 + "\n")
    for name, tensor in sorted_vars:
        shape = str(list(tensor.shape))
        flag = "(trainable)" if grad_flags.get(name, False) else ""
        wr.write(f"{name:<60} {shape} {flag}\n".rstrip() + "\n")
    wr.write("-" * 85 + "\n")
