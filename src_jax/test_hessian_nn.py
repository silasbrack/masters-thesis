from functools import partial
import time
import equinox as eqx
import jax
from equinox import nn
from jax import nn as jnn
from jax import numpy as jnp
import numpy as np

from src_jax.model.conv import ConvNet
from src_jax.test_hessian_toy import test_cg_hvp, test_gvp, test_cg_gvp


key = jax.random.PRNGKey(0)
model = ConvNet(key)
model_call_kwargs = {"key": key}
batch_size = 256
posterior_samples = 1
image_size = (1, 32, 32)
input_size = image_size
input_size = (batch_size,) + image_size


def random_split_like_tree(rng_key, target=None, treedef=None):
    # https://github.com/google/jax/discussions/9508
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    # https://github.com/google/jax/discussions/9508
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_util.tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype) if eqx.is_array(l) else None,
        # lambda l, k: jax.random.normal(k, (posterior_samples,) + l.shape, l.dtype) if eqx.is_array(l) else None,
        target,
        keys_tree,
    )


num_params = sum(x.size if hasattr(x, "size") else 0 for x in jax.tree_util.tree_leaves(model))
print(f"{num_params=}")


x = jnp.ones(input_size)
epsilon = tree_random_normal_like(key, model)

params, static = eqx.partition(model, eqx.is_array)


def model_fn(params):
    model = eqx.combine(params, static)
    model = partial(model, **model_call_kwargs)
    return model(x)


def model_fn_vmap(params):
    model = eqx.combine(params, static)
    model = partial(model, **model_call_kwargs)
    model = jax.vmap(model)
    return model(x)


def loss_fn(y):
    return 0.5 * (y.T @ y)


@jax.vmap
def loss_fn_vmap(y):
    return 0.5 * (y.T @ y)


def model_loss_fn(params):
    model = eqx.combine(params, static)
    model = partial(model, **model_call_kwargs)
    return loss_fn(model(x))


def model_loss_fn_vmap(params):
    model = eqx.combine(params, static)
    model = partial(model, **model_call_kwargs)
    loss_model_fn = jax.vmap(lambda x: loss_fn(model(x)))
    return loss_model_fn(x)


def hvp(f, p, v):
    return jax.jvp(jax.jacrev(f), (p,), (v,))[1]


def nested_vmap(fun, n: int):
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


def gvp(inner_fun, outer_fun, p_in, t_in):
    """
    Calculates the GGN-approximated Hessian HVP
    ggn_hessian(outer_fun(inner_fun(p_in))) * t_in

    Credit to https://github.com/YouJiacheng from https://github.com/google/jax/discussions/9980.

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


def cg_hvp(f, p, V):
    opt_fn = lambda v: hvp(f, p, v)
    return jax.scipy.sparse.linalg.cg(opt_fn, V)[0]


def vmap_hvp(f, p, V):
    return jax.vmap(lambda v: hvp(f, p, v))(V)  # VMAP over V


def vmap_cg_hvp(f, p, V):
    opt_fn = lambda v: vmap_hvp(f, p, v)
    return jax.scipy.sparse.linalg.cg(opt_fn, V)[0]


if x.ndim == 3:
    print("no vmap")

    # t0 = time.perf_counter()
    # hvp_ = hvp(model_loss_fn, params, epsilon)
    # wall_prod_hvp = time.perf_counter() - t0
    # print(f"{wall_prod_hvp=}")

    # t0 = time.perf_counter()
    # gvp_ = test_gvp(loss_fn, model_fn, params, epsilon)
    # wall_prod_gvp = time.perf_counter() - t0
    # print(f"{wall_prod_gvp=}")

    # t0 = time.perf_counter()
    # h = jax.hessian(lambda p: loss_fn(model_fn(p)))(params)
    # wall_prod_exact = time.perf_counter() - t0
    # print(f"{wall_prod_exact=}")

    step_size = 1e-3
    t0 = time.perf_counter()
    gradient = jax.grad(lambda p: loss_fn(model_fn(p)))(params)
    print("grad:", time.perf_counter() - t0)
    t0 = time.perf_counter()
    natural_gradient = hvp(model_loss_fn, params, gradient)
    print("ng:", time.perf_counter() - t0)
    t0 = time.perf_counter()
    updates = jax.tree_map(lambda ng: -ng * step_size, natural_gradient)
    new_params = eqx.apply_updates(params, updates)
    print("update:", time.perf_counter() - t0)

    # t0 = time.perf_counter()
    # hvp = test_cg_hvp(loss_fn, model_fn, params, epsilon)
    # wall_prod_hvp = time.perf_counter() - t0
    # print(f"{wall_prod_hvp=}")

    # t0 = time.perf_counter()
    # gvp = test_cg_gvp(loss_fn, model_fn, params, epsilon)
    # wall_prod_gvp = time.perf_counter() - t0
    # print(f"{wall_prod_gvp=}")

elif x.ndim == 4 and epsilon.fc1.weight.ndim == 2:
    print("vmap over batch")

    t0 = time.perf_counter()
    hvp_ = hvp(model_loss_fn_vmap, params, epsilon)
    wall_prod_hvp = time.perf_counter() - t0
    print(f"{wall_prod_hvp=}")

    t0 = time.perf_counter()
    gvp_ = test_gvp(loss_fn_vmap, model_fn_vmap, params, epsilon)
    wall_prod_gvp = time.perf_counter() - t0
    print(f"{wall_prod_gvp=}")

    @jax.jit
    @jax.vmap
    def cg_hvp_vmap(x):
        def model_loss_fn(params):
            model = eqx.combine(params, static)
            model = partial(model, **model_call_kwargs)
            return loss_fn(model(x))

        opt_fn = lambda v: hvp(model_loss_fn, params, v)
        return jax.scipy.sparse.linalg.cg(opt_fn, epsilon)[0]

    t0 = time.perf_counter()
    hvp_ = cg_hvp_vmap(x)
    wall_prod_hvp = time.perf_counter() - t0
    print(f"{wall_prod_hvp=}")

    @jax.jit
    @jax.vmap
    def cg_gvp_vmap(x):
        def model_fn(params):
            model = eqx.combine(params, static)
            model = partial(model, **model_call_kwargs)
            return model(x)

        opt_fn = lambda v: gvp(model_fn, loss_fn, params, v)[2]
        return jax.scipy.sparse.linalg.cg(opt_fn, epsilon)[0]

    t0 = time.perf_counter()
    gvp_ = cg_gvp_vmap(x)
    wall_prod_gvp = time.perf_counter() - t0
    print(f"{wall_prod_gvp=}")

elif x.ndim == 4 and epsilon.fc1.weight.ndim == 3:
    print("vmap over batch and epsilon")

    @jax.jit
    @jax.vmap
    def cg_hvp_vmap(x):
        def model_loss_fn(params):
            model = eqx.combine(params, static)
            model = partial(model, **model_call_kwargs)
            return loss_fn(model(x))

        opt_fn = lambda v: hvp(model_loss_fn, params, v)
        cg_fn = jax.vmap(lambda e: jax.scipy.sparse.linalg.cg(opt_fn, e)[0])
        return cg_fn(epsilon)

    t0 = time.perf_counter()
    hvp_ = cg_hvp_vmap(x)
    wall_prod_hvp = time.perf_counter() - t0
    print(f"{wall_prod_hvp=}")

    @jax.jit
    @jax.vmap
    def cg_gvp_vmap(x):
        def model_fn(params):
            model = eqx.combine(params, static)
            model = partial(model, **model_call_kwargs)
            return model(x)

        opt_fn = lambda v: gvp(model_fn, loss_fn, params, v)[2]
        cg_fn = jax.vmap(lambda e: jax.scipy.sparse.linalg.cg(opt_fn, e)[0])
        return cg_fn(epsilon)

    t0 = time.perf_counter()
    gvp_ = cg_gvp_vmap(x)
    wall_prod_gvp = time.perf_counter() - t0
    print(f"{wall_prod_gvp=}")
