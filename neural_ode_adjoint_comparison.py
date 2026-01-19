import jax
import jax.numpy as jnp
from jax import grad, vjp, jit
import numpy as np

"""
NEURAL ODE ADJOINT METHOD - HIGHLIGHTING ADJOINT
------------------------------------------------------
This script demonstrates the "Reverse-mode differentiation of ODEs"
(The Adjoint Method) from scratch using JAX.

We will:
1. Define a custom ODE solver (RK4).
2. Define a Neural Network that acts as the dynamics f(z, t, theta).
3. Implement the Adjoint Method (The backward augmented pass).
4. Verify that the Adjoint gradients match standard backprop gradients.
"""

def rk4_step(func, t, y, dt, params):
    """
    Performs a single Runge-Kutta 4 step.

    Crucial Update: uses jax.tree_util.tree_map to handle both simple array states
    (Forward pass) and complex tuple states (Backward Augmented pass).
    """
    dt2 = 0.5 * dt
    dt6 = dt / 6.0

    k1 = func(y, t, params)

    # k2 = func(y + 0.5 * dt * k1, ...)
    y2 = jax.tree_util.tree_map(lambda s, k: s + dt2 * k, y, k1)
    k2 = func(y2, t + dt2, params)

    # k3 = func(y + 0.5 * dt * k2, ...)
    y3 = jax.tree_util.tree_map(lambda s, k: s + dt2 * k, y, k2)
    k3 = func(y3, t + dt2, params)

    # k4 = func(y + dt * k3, ...)
    y4 = jax.tree_util.tree_map(lambda s, k: s + dt * k, y, k3)
    k4 = func(y4, t + dt, params)

    # y_next = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    y_next = jax.tree_util.tree_map(
        lambda s, k1_, k2_, k3_, k4_: s + dt6 * (k1_ + 2*k2_ + 2*k3_ + k4_),
        y, k1, k2, k3, k4
    )
    return y_next

def odeint(func, y0, t_span, params, num_steps=100):
    """
    Solves ODE from t_span[0] to t_span[1].

    Args:
        func: The dynamics function f(y, t, params)
        y0: Initial state (Can be Array or Tuple)
        t_span: Tuple (t0, t1)
        params: Parameters for f
    """
    t0, t1 = t_span
    dt = (t1 - t0) / num_steps

    # Use jax.lax.scan for efficiency (it compiles the loop)
    def body_fun(carry, _):
        curr_y, curr_t = carry
        next_y = rk4_step(func, curr_t, curr_y, dt, params)
        return (next_y, curr_t + dt), None

    (y_final, _), _ = jax.lax.scan(body_fun, (y0, t0), jnp.arange(num_steps))
    return y_final

def neural_dynamics(state, t, params):
    """
    f(z, t, theta) modeled as a simple MLP.
    State dim: 2
    Params: (Weights, Biases)
    """
    W, b = params
    # Simple Tanh activation dynamics: dz/dt = Tanh(Wz + b)
    return jnp.tanh(jnp.dot(W, state) + b)

# Initialize random parameters
key = jax.random.PRNGKey(0)
D = 2 # Dimension of state
W = jax.random.normal(key, (D, D))
b = jax.random.normal(key, (D,))
true_params = (W, b)

def augmented_dynamics(aug_state, t, params):
    """
    The dynamics of the AUGMENTED system [z, a, dL/dtheta].
    aug_state: A tuple (z, a, grad_accum_W, grad_accum_b)
    """
    z, a, _, _ = aug_state

    # --- Vector-Jacobian Products (VJP) ---
    # We calculate VJP of f w.r.t (z, W, b).
    primals = (z, params[0], params[1])

    # 1. Run forward f(z)
    # 2. Get 'vjp_fun' which corresponds to J^T * v
    f_val, vjp_fun = jax.vjp(
        lambda z_, W_, b_: neural_dynamics(z_, t, (W_, b_)),
        *primals
    )

    # Compute the VJPs (-a^T * J)
    # vjp_fun returns gradients matching the shape of 'primals'
    vjp_z, vjp_W, vjp_b = vjp_fun(a)

    # The Dynamics Equations:
    dz_dt = f_val           # dz/dt = f(z)
    da_dt = -vjp_z          # da/dt = -a^T * df/dz
    dgrad_W_dt = -vjp_W     # d(grad)/dt = -a^T * df/dtheta
    dgrad_b_dt = -vjp_b

    return (dz_dt, da_dt, dgrad_W_dt, dgrad_b_dt)

def adjoint_gradient(params, y0, t_span, target):
    """
    Computes gradients using the Adjoint Method (O(1) Memory).
    """
    t0, t1 = t_span

    # 1. Forward Pass: Solve ODE from t0 to t1 to get z(t1)
    z_final = odeint(neural_dynamics, y0, (t0, t1), params)

    # 2. Compute Loss Gradients at the boundary (t1)
    diff = z_final - target
    loss = 0.5 * jnp.sum(diff**2)
    a_final = diff  # dL/dz(t1)

    # 3. Setup Backward Solve (t1 -> t0)
    # Initial condition for backward pass: [ z(t1), a(t1), 0, 0 ]
    grad_W_init = jnp.zeros_like(params[0])
    grad_b_init = jnp.zeros_like(params[1])
    aug_state_init = (z_final, a_final, grad_W_init, grad_b_init)

    # Solve augmented ODE backwards
    final_aug_state = odeint(
        augmented_dynamics,
        aug_state_init,
        (t1, t0),
        params
    )

    # Extract results
    z0_reconstructed, a0, grad_W, grad_b = final_aug_state

    return loss, (grad_W, grad_b)
    
def standard_loss(params, y0, t_span, target):
    """
    Computes loss by backpropagating through the solver internals (O(N) Memory).
    """
    z_final = odeint(neural_dynamics, y0, t_span, params)
    return 0.5 * jnp.sum((z_final - target)**2)




if __name__ == "__main__":
    print("--- Neural ODE Adjoint Method Demo ---")

    try:
        devices = jax.devices()
        print(f"JAX Devices found: {devices}")
        platform = devices[0].platform.upper()
        print(f"Active Platform: {platform}")

        if platform == "CPU":
            print("WARNING: Running on CPU. For GPU support, install 'jax[cuda]'.")
        elif platform == "GPU":
            print("SUCCESS: Running on GPU! All JIT-compiled functions will use the GPU.")
    except Exception as e:
        print(f"Device check failed: {e}")

    y0 = jnp.array([1.0, 0.0])  # Initial state
    target = jnp.array([0.0, 1.0]) # Target state at t1
    t_span = (0.0, 1.0)

    # A. Compute with Standard JAX Autodiff
    print("\n1. Computing Gradients via Standard Backprop (Autodiff)...")
    std_grad_fn = jit(jax.value_and_grad(standard_loss))
    std_loss, std_grads = std_grad_fn(true_params, y0, t_span, target)
    print(f"   Loss: {std_loss:.6f}")

    # B. Compute with Adjoint Method
    print("\n2. Computing Gradients via Adjoint Method (ODE Solve)...")
    adj_grad_fn = jit(adjoint_gradient)
    adj_loss, adj_grads = adj_grad_fn(true_params, y0, t_span, target)
    print(f"   Loss: {adj_loss:.6f}")

    # C. Comparison
    print("\n3. Verification:")

    # Check Weight Gradients
    diff_W = jnp.linalg.norm(std_grads[0] - adj_grads[0])
    print(f"   Difference in Weight Gradients: {diff_W:.8f}")

    # Check Bias Gradients
    diff_b = jnp.linalg.norm(std_grads[1] - adj_grads[1])
    print(f"   Difference in Bias Gradients:   {diff_b:.8f}")

    if diff_W < 1e-4 and diff_b < 1e-4:
        print("\nSUCCESS: Adjoint gradients match Autodiff gradients!")
    else:
        print("\nWARNING: Discrepancy detected (likely due to solver discretization errors).")
        print("Note: Exact matches require the solver to be reversible or small step sizes.")

