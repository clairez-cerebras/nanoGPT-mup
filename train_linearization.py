import os
import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

def get_layer(model, layer_of_interest_name):
    """
    Get a layer/module from the model by its name.
    """
    return dict(model.named_modules())[layer_of_interest_name]


def get_param(model, param_of_interest_name):
    """
    Get a parameter from the model by its name.
    """
    return dict(model.named_parameters())[param_of_interest_name]


def get_activation_and_grad_wrt_param(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_of_interest_name: str,
    param_of_interest: torch.Tensor,
    return_vector=False
):
    """
    Get the activation from a specific layer and the gradient of that activation w.r.t. a specific parameter.
    """

    activation: Optional[torch.Tensor] = None    
    def forward_hook(module, in_t, out_t):
        nonlocal activation
        activation = out_t

    layer_of_interest = get_layer(model, layer_of_interest_name)
    hook_handle = layer_of_interest.register_forward_hook(forward_hook)

    # Forward pass
    model.zero_grad(set_to_none=True)
    out = model(inputs)  # Could be logits or next layer's hidden states, doesn't matter
    hook_handle.remove()

    # Now `activation` should hold h^l
    if activation is None:
        raise RuntimeError(f"No activation captured from layer {layer_of_interest_name}.")

    # For the simplest approach: we do d( sum(h^l) ) / d param
    # That is effectively a shape match if we pass grad_outputs = ones_like(activation).
    grad_outputs = torch.ones_like(activation[0,0], requires_grad=True)

    # Get derivative of the [0,0] entry (first batch, first nueron).
    grad_tuple = torch.autograd.grad(
        outputs=activation[0,0,0],    # or out if we want ∂out/∂W, but here we do activation
        inputs=param_of_interest,
        # grad_outputs=grad_outputs,
        retain_graph=False,  # if we want to do more grads afterwards
        create_graph=False,  # if we don't need higher order derivatives
        allow_unused=True
    )
    grad = grad_tuple[0]

    # Return both the captured h^l and the gradient
    return activation[0,0,0] if not return_vector else activation[0,0,:], grad


def get_first_order_taylor(delta_w, grad):
    """
    Compute the first-order Taylor expansion given a parameter perturbation and the gradient.
    """
    return torch.sum(delta_w[0,:] * grad[0:])



# ========== TESTING ========== #

class SimpleLayer(nn.Module):
    """
    A simple layer that applies a linear transformation after a ReLU activation.
    """
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        phi_x = torch.relu(x)  # This is your φ(x)
        return self.linear(phi_x)
    

def test_derivative_wrt_first_row(in_dim, out_dim):
    """
    Test that the derivative of the output w.r.t. the first row of W matches the expected value.
    """

    model = SimpleLayer(in_dim=in_dim, out_dim=out_dim)

    x = torch.randn(1, in_dim, requires_grad=True)
    phi_x = torch.relu(x)

    W = get_param(model, "linear.weight")
    
    act, grad = get_activation_and_grad_wrt_param(
        model=model,
        inputs=x,
        layer_of_interest_name="linear",
        param_of_interest=W,
        return_vector=True
    )

    # The derivative of (W φ(x))_0 w.r.t. W is a matrix whose first row is φ(x), others are 0
    expected_grad = torch.zeros_like(W)
    expected_grad[0, :] = phi_x

    assert torch.allclose(grad, expected_grad, atol=1e-6), f"Gradient mismatch:\nGot: {grad}\nExpected: {expected_grad}"

    print("Test passed!")


if __name__ == "__main__":

    torch.manual_seed(0)
    in_dim = 4
    out_dim = 3

    test_derivative_wrt_first_row(in_dim, out_dim)