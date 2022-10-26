import os
import shutil
import subprocess
import time
from functools import partial
from typing import Dict

import jax
import pandas as pd
from jax import numpy as jnp


def nested_vmap(fun, n: int):
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


def gvp(inner_fun, outer_fun, p_in, t_in):
    """Calculates the GGN-approximated Hessian HVP.

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


def f(x):
    return jax.tree_map(lambda x: x + 0.5, x)


def g(x):
    return jax.tree_map(lambda x: jnp.sum(x**2), x)


def hvp(fun, p, t):
    return jax.jvp(jax.jacrev(fun, argnums=0), p, t)[1]


@partial(jax.jit, static_argnums=(0, 1))
def test_exact_hessian_prod(g, f, x, epsilon):
    hessian = jax.hessian(lambda x: g(f(x)))(x)
    hessian = jnp.squeeze(hessian)
    return hessian.T @ epsilon


@partial(jax.jit, static_argnums=(0, 1))
def test_hvp(g, f, x, epsilon):
    return hvp(lambda x: g(f(x)), (x,), (epsilon,))


@partial(jax.jit, static_argnums=(0, 1))
def test_gvp(g, f, x, epsilon):
    return gvp(f, g, p_in=x, t_in=epsilon)[2]


@partial(jax.jit, static_argnums=(0, 1))
def test_exact_inv(g, f, x, epsilon):
    hessian = jax.hessian(lambda x: g(f(x)))(x)
    return jax.scipy.linalg.inv(hessian).T @ epsilon


@partial(jax.jit, static_argnums=(0, 1))
def test_cg_hvp(g, f, x, epsilon):
    opt_fn = lambda t: hvp(lambda x: g(f(x)), (x,), (t,))
    return jax.scipy.sparse.linalg.cg(opt_fn, epsilon)[0]


@partial(jax.jit, static_argnums=(0, 1))
def test_cg_gvp(g, f, x, epsilon):
    opt_fn = lambda t: gvp(f, g, p_in=x, t_in=t)[2]
    return jax.scipy.sparse.linalg.cg(opt_fn, epsilon)[0]


def get_gpu_memory_map():
    """Get the current gpu usage.

    Return:
        A dictionary in which the keys are device ids as integers and
        values are memory usage as integers in MB.
    Raises:
        FileNotFoundError:
            If nvidia-smi installation not found
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        raise FileNotFoundError("nvidia-smi: command not found")
    result = subprocess.run(
        [nvidia_smi_path, "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8",
        # capture_output=True,          # valid for python version >=3.7
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
    )

    # Convert lines into a dictionary
    gpu_memory = [float(x) for x in result.stdout.strip().split(os.linesep)]
    gpu_memory_map = {f"gpu_id: {gpu_id}/memory.used (MB)": memory for gpu_id, memory in enumerate(gpu_memory)}
    return gpu_memory_map, gpu_memory


def main(n_params: int):
    results = {"n_params": n_params}

    print("Input size is", n_params)
    key = jax.random.PRNGKey(0)

    input_size = (n_params,)

    x = jnp.ones(input_size)
    epsilon = jax.random.normal(key, input_size)

    print("Hessian-vector product")
    t0 = time.perf_counter()
    hessian_prod_hvp = test_hvp(g, f, x, epsilon)
    hessian_prod_hvp.block_until_ready()
    wall_prod_hvp = time.perf_counter() - t0
    results["wall_hvp"] = wall_prod_hvp
    print(f"{wall_prod_hvp:.2f} seconds for Hessian-vector product via HVP")

    t0 = time.perf_counter()
    hessian_prod_gvp = test_gvp(g, f, x, epsilon)
    hessian_prod_gvp.block_until_ready()
    wall_prod_gvp = time.perf_counter() - t0
    results["wall_gvp"] = wall_prod_gvp
    print(f"{wall_prod_gvp:.2f} seconds for Hessian-vector product via GVP")

    assert jnp.allclose(hessian_prod_hvp, hessian_prod_gvp)

    if input_size[0] <= 10000:
        t0 = time.perf_counter()
        hessian_prod_exact = test_exact_hessian_prod(g, f, x, epsilon)
        hessian_prod_exact.block_until_ready()
        wall_prod_exact = time.perf_counter() - t0
        results["wall_manual"] = wall_prod_exact
        print(f"{wall_prod_exact:.2f} seconds for exact Hessian-vector product")

        assert jnp.allclose(hessian_prod_exact, hessian_prod_hvp)
        assert jnp.allclose(hessian_prod_exact, hessian_prod_gvp)

    print("Inverse Hessian-vector product")

    t0 = time.perf_counter()
    samples_cg = test_cg_hvp(g, f, x, epsilon)
    samples_cg.block_until_ready()
    wall_prod_hvp = time.perf_counter() - t0
    results["wall_cg_hvp"] = wall_prod_hvp
    print(f"{wall_prod_hvp:.2f} seconds for Hessian-vector product via HVP + CG")

    t0 = time.perf_counter()
    samples_cg_ggn = test_cg_gvp(g, f, x, epsilon)
    samples_cg_ggn.block_until_ready()
    wall_prod_gvp = time.perf_counter() - t0
    results["wall_cg_gvp"] = wall_prod_gvp
    print(f"{wall_prod_gvp:.2f} seconds for Hessian-vector product via GVP + CG")

    assert jnp.allclose(samples_cg, samples_cg_ggn)

    if input_size[0] <= 10000:
        t0 = time.perf_counter()
        samples_inv = test_exact_inv(g, f, x, epsilon)
        samples_inv.block_until_ready()
        wall_prod_exact = time.perf_counter() - t0
        results["wall_manual_inverse"] = wall_prod_exact
        print(f"{wall_prod_exact:.2f} seconds for exact Hessian-vector inverse product")

        assert jnp.allclose(samples_cg, samples_inv)
        assert jnp.allclose(samples_cg_ggn, samples_inv)

    # jax.profiler.save_device_memory_profile("memory.prof")
    return results


if __name__ == "__main__":
    # results = [main(int(n_params)) for n_params in [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]]
    # df = pd.DataFrame(results)
    # df.to_csv("results/hessian_profile.csv", mode="a", header=False)
    # print(df)
    main(10000)
    # main(100000000)
