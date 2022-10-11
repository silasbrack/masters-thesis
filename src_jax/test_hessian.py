import jax
from jax import numpy as jnp


def nested_vmap(fun, n: int):
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


def gvp(inner_fun, outer_fun, p_in, t_in):
    """
    Calculates the GGN-approximated Hessian HVP
    ggn_hessian(outer_fun(inner_fun(p_in))) * t_in

    p_in: pytree_0
    t_in: pytree_1
    inner_fun: pytree_1 -> pytree_2
    outer_fun: pytree_1 -> pytree_2
    """
    p_out, f_l = jax.linearize(inner_fun, p_in)  # (pytree_1), (pytree_0 -> pytree_1)
    f_lt_tuple = jax.linear_transpose(f_l, p_in)  # pytree_1 -> pytree_0
    f_lt = lambda x: f_lt_tuple(x)[0]  # primals tuple only contain one primal
    Jt = f_l(t_in)  # pytree_1
    d_outer, HJt = jax.jvp(jax.jacrev(outer_fun, argnums=0), (p_out,), (Jt,))
    # pytree_2(pytree_1), pytree_2(pytree_1) with prepended shape leaves
    shapes = jax.eval_shape(outer_fun, p_out)  # pytree_2
    Gt = jax.tree_map(lambda s, h: nested_vmap(f_lt, len(s.shape))(h), shapes, HJt)  # h: pytree_1
    return p_out, d_outer, Gt


def f(x):
    return jax.tree_map(lambda x: x + 0.5, x)


def g(x):
    return jnp.sum(x**2)


def hvp(fun, p, t):
    return jax.jvp(jax.jacrev(fun, argnums=0), p, t)[1]


key = jax.random.PRNGKey(0)

input_size = (5,)
print(f"{input_size=}")

x = jnp.ones(input_size)
epsilon = jax.random.normal(key, input_size)


hessian_prod = hvp(lambda x: g(f(x)), (x,), (epsilon,))  # Exact HVP
p_out, d_outer, Gt = gvp(f, g, p_in=x, t_in=epsilon)  # GGN HVP
hessian_fn = jax.jit(jax.hessian(lambda x: g(f(x))))
hessian = hessian_fn(x)  # Exact Hessian
print(f"{hessian=}")
assert jnp.allclose(hessian @ epsilon, hessian_prod)
assert jnp.allclose(hessian_prod, Gt)
assert jnp.allclose(hessian @ epsilon, Gt)

opt_fn = lambda e: hvp(lambda x: g(f(x)), (x,), (e,))
samples_cg, _ = jax.scipy.sparse.linalg.cg(opt_fn, epsilon)
samples_inv = jax.scipy.linalg.inv(hessian) @ epsilon
print(f"{samples_cg=}")
assert jnp.allclose(samples_cg, samples_inv)
