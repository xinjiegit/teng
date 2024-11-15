import argparse
import copy
from argparse import ArgumentParser
import time
from datetime import datetime
import os
import shutil
import logging
import random
import json
import pickle
from functools import partial

import jax
from jax import config

config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)

import jax.random as jrnd
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

import optax

import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from src.utils import boolargparse, jsonargparse, ArgsKwargsParseAction  # types of argument used for argparser
from src.model import CreateSimpleRealNVP, SimpleARRQSFlow, SimplePDENet, SimplePDENet2, SimplePDENet0, SimplePDENet3, \
    DirichletPDENet, PDENet, SimplePDENet4, CircularDirichletPDENet, CircularDirichletPDENet2, DirichletPDENet2
from src.sampler import SimpleFlowSampler, ContinuousUniformSampler, PeriodicQuadratureSampler, OpenQuadratureSampler, \
    CircularQuadratureSampler
from src.var_state import SimpleVarStateReal
from src.operator import FokkerPlanckOperator, HeatOperatorNoLog
from src.lstsq_solver import cgls_solve, cgls_solve2, lsqr_solve
from src.utils import StackedRandomNaturalPolicyGradLS, RandomNaturalPolicyGradLS, simple_q_policy_grad, \
    natural_q_policy_grad_ls, natural_q_policy_grad_minsr, natural_q_policy_grad_tdvp, compute_observables, \
    compute_observables_and_fidelities

now = datetime.now()


def get_config():
    parser = ArgumentParser()

    ### general configs ###
    parser.add_argument("--nb_dims", type=int, default=2)
    parser.add_argument("--nb_steps", type=int, default=30000)
    parser.add_argument("--save_dir", type=str, nargs='?', default=None)  # can be emtpy
    # do this by setting enviroment variable from outside
    # parser.add_argument("--use_double", type=boolargparse, default=True,
    #                     help='whether to use double precision, accepts true/false, yes/no, t/f, y/n, 1/0')
    parser.add_argument("--load_config_from", type=str, nargs='?', default=None,
                        help='if specified, will load the config from the provided json file and overwrite current config, save_dir and load_config_from will not be overwritten')  # can be empty

    ### model configs ###
    # parser.add_argument("--model", type=str, default='SimplePDENet',
    #                     choices=['CreateSimpleRealNVP', 'SimpleARRQSFlow', 'SimplePDENet'])
    # parser.add_argument("--load_model_state_from", type=str, nargs='?', default='results/PDyNG/test_03-26-2024-03-46-34_1024_7_40_no_resnet_tanh_large_lr2_dirichlet_load_cosine_decay_circ_init/model_state.pickle')
    # parser.add_argument("--load_model_state_from", type=str, nargs='?', default='results/PDyNG/test_03-26-2024-06-08-44_1024_7_40_no_resnet_tanh_large_lr2_dirichlet_load_cosine_decay_circ_init_correct/model_state.pickle')
    # parser.add_argument("--load_model_state_from", type=str, nargs='?', default='results/PDyNG/test_03-25-2024-21-46-07_1024_7_40_no_resnet_tanh_large_lr2_dirichlet_tdvp_ac_init/model_state.pickle')
    # test_03-26-2024-06-08-44_1024_7_40_no_resnet_tanh_large_lr2_dirichlet_load_cosine_decay_circ_init_correct
    parser.add_argument("--load_model_state_from", type=str, nargs='?', default=None)
    parser.add_argument("--model_seed", type=int, default=1234)

    ### sampler configs ###
    # parser.add_argument("--sampler", type=str, default='PeriodicQuadratureSampler',
    #                     choices=['SimpleFlowSampler', 'ContinuousUniformSampler', 'PeriodicQuadratureSampler'])
    parser.add_argument("--load_sampler_state_from", type=str, nargs='?', default=None)
    # sampler configs for nonexact sampler
    parser.add_argument("--sampler_seed", type=int, default=4321)
    parser.add_argument("--nb_samples", type=int, default=65536,
                        help='number of samples')

    ### optimier configs ###
    parser.add_argument("--load_optimizer_state_from", type=str, nargs='?', default=None)
    parser.add_argument("--optimizer", type=str, default="cgls", nargs='?',
                        help="choose one of optax optimizer or one of 'cgls', 'minsr', 'tdvp', 'mclanchlan'")
    # parser.add_argument("--optimizer_args", action=ArgsKwargsParseAction, nargs='*', default=([], {}),
    #                     help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    # parser.add_argument("--optimizer_args", action=ArgsKwargsParseAction, nargs='*', default=([], {'tol': 1e-8, 'atol': 1e-8, 'maxiter': 100}), #{'rcond': 1e-6}),
    #                     help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    parser.add_argument("--optimizer_args", action=ArgsKwargsParseAction, nargs='*',
                        default=([], {'maxiter': 100, 'use_jax_scan': True}),  # {'rcond': 1e-6}),
                        help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    # parser.add_argument("--optimizer_args", action=ArgsKwargsParseAction, nargs='*', default=([], {'rcond': 1e-14}),
    #                     help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    # parser.add_argument("--scheduler", type=str, default="exponential_decay")
    # parser.add_argument("--scheduler_args", action=ArgsKwargsParseAction, nargs='*', default=([], {'init_value': 0.1, 'transition_steps': 50000, 'decay_rate':0.6}),
    #                     help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    # parser.add_argument("--scheduler", type=str, default="cosine_decay_schedule")
    # parser.add_argument("--scheduler_args", action=ArgsKwargsParseAction, nargs='*', default=([], {'init_value': 0.09, 'decay_steps': 10000}),
    #                     help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    parser.add_argument("--scheduler", type=str, default="warmup_cosine_decay_schedule")
    parser.add_argument("--scheduler_args", action=ArgsKwargsParseAction, nargs='*', default=(
    [], {'init_value': 0.06, 'peak_value': 0.12, 'warmup_steps': 50, 'decay_steps': 30000}),
                        help="given input of the form 1, 2, 3, a=4, b=5, the parser parsers it to (args kwargs)")
    parser.add_argument("--weight_decay", type=float, default=0.00000)

    args = parser.parse_args()
    if args.save_dir is None or args.save_dir.lower() == 'none':
        args.save_dir = f'./results/PDyNG/test_{now.strftime("%m-%d-%Y-%H-%M-%S")}_1024_7_40_no_resnet_tanh_large_lr2_dirichlet_load_cosine_decay_ac_init_correct_try_again/'
    else:
        args.save_dir = "./results/" + args.save_dir

    os.makedirs(args.save_dir, exist_ok=True)
    backup_dir = os.path.join(args.save_dir, 'code_backup')
    os.makedirs(backup_dir, exist_ok=True)
    if args.load_config_from is not None and args.save_dir.lower() != 'none':
        with open(args.save_dir + '/config_overwritten.json', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        with open(args.load_config_from, 'r') as f:
            new_arg_dict = json.load(f)
        new_args = argparse.Namespace(**new_arg_dict)
        new_args.save_dir = args.save_dir
        new_args.load_config_from = args.load_config_from
        args = new_args
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    for filename in os.listdir('./'):
        if '.sh' in filename or \
                '.swb' in filename or \
                '.py' in filename:
            if filename == '.pylint.d':
                continue
            if '.swp' in filename:
                continue
            if '__pycache__' in filename:
                continue
            shutil.copy(filename, backup_dir)
        shutil.copytree('./src', os.path.join(backup_dir, 'src'), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('*__pycache__*'))

    return args


def write_to_file(file, *items, flush=False):
    if type(items[0]) == list or type(items[0]) == tuple:
        items = items[0]
    for item in items:
        file.write('%s ' % item)
    file.write('\n')
    if flush:
        file.flush()


# def forward_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old, return_loss_value=True):
#     """
#     compute the forward KL divergence reward
#     where forward KL divergence is defined as KL(p_target(x) || p_new(x)) with p_target = p_old + delta_p_old
#     this function may have an unknown scaling of the reward depending on delta_p_old_over_p_old
#     """
#     # p_target(x) / p_new(x) = p_old(x) / p_new(x) * (1 + delta_p_old(x) / p_old(x))
#     reward = -jnp.exp(log_p_old - log_p_new) * (1 + delta_p_old_over_p_old) # negative because we want to maximize the reward
#     if not return_loss_value:
#         return reward
#     else:
#         # p_target(x) / p_new(x) * log(p_target(x) / p_new(x)) = -reward(x) * (log_p_old(x) - log_p_new(x) + log(1 + delta_p_old(x) / p_old(x))
#         log_p_new_over_p_target = log_p_old - log_p_new + jnp.log1p(delta_p_old_over_p_old)
#         log_Z_target = jsp.special.logsumexp(log_p_new_over_p_target) - jnp.log(log_p_new_over_p_target.size)
#         loss_value = -reward * log_p_new_over_p_target#  / jnp.exp(log_Z_target)
#         return reward * 2, loss_value # multiply by 2 due to log_p = 2 * log_psi # MAYBE NOT CORRECT
#
# def reverse_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old, return_loss_value=True):
#     """
#     compute the reverse KL divergence reward
#     where reverse KL divergence is defined as KL(p_new(x) || p_target(x)) with p_target = p_old + delta_p_old
#     """
#     # log (p_new(x) / p_target(x)) = log p_new(x) - log p_old(x) - log(1 + delta_p_old(x) / p_old(x)) + 1
#     # the +1 in the end is optional because we only need reward - reward_mean later
#     reward = -(log_p_new - log_p_old - jnp.log1p(delta_p_old_over_p_old) + 1) # negative because we want to maximize the reward
#     if not return_loss_value:
#         return reward
#     else:
#         # log(p_new(x) / p_target(x)) = -reward(x) - 1
#         log_p_new_over_p_target = -reward - 1
#         log_Z_target = jsp.special.logsumexp(log_p_new_over_p_target) - jnp.log(log_p_new_over_p_target.size)
#         loss_value = log_p_new_over_p_target + log_Z_target # MAYBE NOT CORRECT and MAYBE NOT NECESSARY
#         return reward * 2, loss_value # multiply by 2 due to log_p = 2 * log_psi # MAYBE NOT CORRECT
#
# def symmetrized_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old, return_loss_value=True):
#     """
#     compute the symmetrized KL divergence reward
#     where symmetrized KL divergence is defined as (KL(p_target(x) || p_new(x)) + KL(p_new(x) || p_target(x)))/2 with p_target = p_old + delta_p_old
#     """
#     if not return_loss_value:
#         return (forward_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old) + reverse_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old)) / 2
#     else:
#         forward_reward, forward_loss_value = forward_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old, return_loss_value=True)
#         reverse_reward, reverse_loss_value = reverse_kl_reward(log_p_new, log_p_old, delta_p_old_over_p_old, return_loss_value=True)
#         return (forward_reward + reverse_reward) / 2, (forward_loss_value + reverse_loss_value) / 2

# def evaluate_and_plot(config, var_state, step):
#     xs = jnp.linspace(-10, 10, 1000)
#     ys = var_state.log_psi(xs.reshape(1, -1, 1, 1))
#     ys = jnp.exp(ys.squeeze() * 2)/50
#     plt.plot(xs, ys)
#     plt.savefig(os.path.join(config.save_dir, f"test_sampling_{step}.png"), dpi=200)

# def evaluate_and_plot(config, var_state, step):
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#
#     # Make data.
#     X = np.linspace(-jnp.pi*2, jnp.pi*2, 50)
#     Y = np.linspace(-jnp.pi*2, jnp.pi*2, 50)
#     X, Y = np.meshgrid(X, Y)
#     # Z = var_state.log_psi(np.stack([X, Y], -1).reshape(1, -1, 1, 2))
#     Z = var_state.log_psi(np.stack([X, Y], -1).reshape(1, -1, 2))
#     Z = jnp.exp(Z.squeeze() * 2).reshape(50, 50)
#
#     # Plot the surface.
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#
#     # Customize the z axis.
#     # ax.set_zlim(0, 0.16)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     # A StrMethodFormatter is used automatically
#     ax.zaxis.set_major_formatter('{x:.02f}')
#
#     # Add a color bar which maps values to colors.
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     plt.savefig(os.path.join(config.save_dir, f"test_sampling_{step}.png"), dpi=200)

def evaluate_and_plot(config, var_state, step):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.linspace(-jnp.pi * 2, jnp.pi * 2, 50)
    Y = np.linspace(-jnp.pi * 2, jnp.pi * 2, 50)
    X, Y = np.meshgrid(X, Y)
    # Z = var_state.evaluate(np.stack([X, Y], -1).reshape(1, -1, 1, 2))
    Z = var_state.evaluate(np.stack([X, Y], -1).reshape(1, -1, 2))
    Z = Z.reshape(50, 50)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0, 0.16)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(os.path.join(config.save_dir, f"test_sampling_{step}.png"), dpi=200)


def make_plot(func, filename):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.linspace(-jnp.pi, jnp.pi, 50)
    Y = np.linspace(-jnp.pi, jnp.pi, 50)
    X, Y = np.meshgrid(X, Y)
    # Z = var_state.evaluate(np.stack([X, Y], -1).reshape(1, -1, 1, 2))
    Z = func(np.stack([X, Y], -1).reshape(1, -1, 2))
    Z = Z.reshape(50, 50)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    # ax.set_zlim(0, 0.16)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.savefig(os.path.join(config.save_dir, f"{filename}.png"), dpi=200)


def natural_policy_grad_ls(var_state, samples, sqrt_weights, rewards, *, ls_solver=None, joint_compile=True,
                           real_output=False):
    """
    rewards: does not need to subtract mean. will be handled inside
    ls_solver: should be a callable of the signiture ls_solver(A: callable, A^T: callable, b: Array) -> (x, info_tuple)
               if it requires extra kwargs, must be handled beforehand with partial(ls_solver, **kwargs)
    """
    if joint_compile:
        if real_output:
            return natural_policy_grad_ls_real_joint_jit(var_state.get_state(), var_state.pure_funcs, samples,
                                                         sqrt_weights, rewards, ls_solver)
    assert False, "not implemented"


@partial(jax.jit, static_argnums=(1, 5))
def natural_policy_grad_ls_real_joint_jit(state, var_state_pure, samples, sqrt_weights, rewards, ls_solver):
    jvp_raw, vjp_raw, value = var_state_pure.jvp_vjp_func(state, samples)

    def jvp(tangents):
        pushforwards = jvp_raw(tangents)
        pushforwards = pushforwards * sqrt_weights

        return pushforwards

    def vjp(cotangents):
        cotangents = cotangents * sqrt_weights
        pullbacks = vjp_raw(cotangents)
        return pullbacks

    rewards = rewards * sqrt_weights

    return ls_solver(jvp, vjp, rewards)


def square_loss(u, v):
    reward = -(u - v)
    loss = reward ** 2 / 2
    return reward, loss


@partial(jax.pmap, in_axes=(None, None, None, 0), out_axes=(0, 0, 0, 0), static_broadcasted_argnums=(1, 2))
@partial(jax.vmap, in_axes=(None, None, None, 0), out_axes=(0, 0, 0, 0))
def fitting_loss_pure(state, var_state_pure, u_target_func, sample):
    u_func = lambda x: var_state_pure.evaluate(state, x[None, :]).squeeze(0)
    du_func = jax.jacrev(u_func)
    ddu_func = jax.jacfwd(du_func)
    du_target_func = jax.jacrev(u_target_func)
    ddu_target_func = jax.jacfwd(du_target_func)
    u = u_func(sample)
    du = du_func(sample)
    ddu = ddu_func(sample)
    u_target = u_target_func(sample)
    du_target = du_target_func(sample)
    ddu_target = ddu_target_func(sample)
    reward = jnp.concatenate([-(u - u_target)[None] * 1, (du - du_target) / 30], -1)
    # reward = jnp.concatenate([-(u - u_target)[None]*1, (jnp.diag(ddu) - jnp.diag(ddu_target)) / 30], -1)
    # reward = -(u - u_target)
    loss1 = (u - u_target) ** 2 / 2
    loss2 = (du - du_target) @ (du - du_target) / 2
    loss3 = (jnp.diag(ddu) - jnp.diag(ddu_target)) @ (jnp.diag(ddu) - jnp.diag(ddu_target)) / 2
    # reward = -(u - u_target) + (jnp.trace(ddu) - jnp.trace(ddu_target))
    # loss = (u - u_target)**2 / 2 + (du @ du - du_target @ du_target) / 2
    return reward, loss1, loss2, loss3


@partial(jax.pmap, in_axes=(None, 0), out_axes=(0, 0, 0), static_broadcasted_argnums=(0,))
@partial(jax.vmap, in_axes=(None, 0), out_axes=(0, 0, 0))
def fitting_denominator(u_target_func, sample):
    du_target_func = jax.jacrev(u_target_func)
    ddu_target_func = jax.jacfwd(du_target_func)
    u_target = u_target_func(sample)
    du_target = du_target_func(sample)
    ddu_target = ddu_target_func(sample)
    # reward = jnp.concatenate([-(u - u_target)[None]*1, (jnp.diag(ddu) - jnp.diag(ddu_target)) / 10], -1)
    # reward = -(u - u_target)
    loss1 = (u_target) ** 2 / 2
    loss2 = (du_target) @ (du_target) / 2
    loss3 = (jnp.diag(ddu_target)) @ (jnp.diag(ddu_target)) / 2
    # reward = -(u - u_target) + (jnp.trace(ddu) - jnp.trace(ddu_target))
    # loss = (u - u_target)**2 / 2 + (du @ du - du_target @ du_target) / 2
    return loss1, loss2, loss3


def fitting_loss(var_state, u_target_func, samples):
    return fitting_loss_pure(var_state.get_state(), var_state.pure_funcs, u_target_func, samples)


# def initialize(config, var_state, pde_operator, optimizer, policy_grad):

# def u_target_func(x):
#     return (jnp.sin(x[..., 0] * 3) + jnp.sin(x[..., 1] * 5))/10

# def u_target_func(x):
#     return  (jnp.exp(3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) + jnp.exp(-3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) -
#              jnp.exp(3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) - jnp.exp(-3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1]))) / 100

# def u_target_func(x):
#     return  (jnp.exp(3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) + jnp.exp(-3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) -
#              jnp.exp(3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) - jnp.exp(-3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) -
#              2 * jnp.sin(x[..., 0]/2)**2 * (jnp.exp(jnp.sin(x[..., 1])) - jnp.exp(-jnp.sin(x[..., 1])))) / 100

# def u_target_func(x):
#     return  (jnp.exp(3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) + jnp.exp(-3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) -
#              jnp.exp(3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) - jnp.exp(-3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) -
#              2 * (1 - jnp.cos(x[..., 0]/2)) * (jnp.exp(jnp.sin(x[..., 1])) - jnp.exp(-jnp.sin(x[..., 1])))) / 100
#
# def u_target_func(xy):
#     xx = xy[..., 0]
#     yy = xy[..., 1]
#     # please help to put these into an array:
#     amps = jnp.array([
#         [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
#         [0.000, -0.006, -0.013, -0.008, -0.000, 0.014, 0.016, -0.002, -0.009],
#         [0.000, 0.010, -0.007, -0.007, 0.005, 0.001, 0.009, -0.008, -0.014],
#         [0.000, 0.017, -0.009, 0.006, 0.005, 0.020, -0.006, 0.001, -0.015],
#         [0.000, 0.023, 0.005, -0.003, 0.007, 0.004, 0.018, 0.001, 0.002],
#         [0.000, -0.018, -0.011, 0.015, -0.007, 0.016, 0.003, 0.008, 0.013],
#         [0.000, 0.012, -0.032, 0.009, 0.004, 0.006, -0.003, -0.020, -0.003],
#         [0.000, -0.010, 0.010, 0.003, -0.005, -0.030, 0.012, 0.021, 0.004],
#         [0.000, -0.012, -0.008, -0.005, 0.017, 0.001, 0.016, 0.023, -0.017]
#     ])
#     inds = jnp.arange(9)
#     inds_x, inds_y = jnp.meshgrid(inds, inds)
#     return jnp.sum(amps * jnp.sin(inds_x * xx[..., None, None]) * jnp.sin(inds_y * yy[..., None, None]), axis=(-1, -2))

# def u_target_func(xyz):
#     xx = xyz[..., 0]
#     yy = xyz[..., 1]
#     zz = xyz[..., 2]
#     # please help to put these into an array:
#     sine_amps = jnp.array([[[-0.075, -0.056],
#                         [ 0.074, -0.007]],

#                        [[-0.027, -0.008],
#                         [ 0.032,  0.039]]])
#     cos_amps = jnp.array([[[ 0.043, -0.044,  0.006],
#                             [-0.021,  0.047, -0.038],
#                             [ 0.074,  0.043,  0.008]],

#                            [[-0.05 , -0.028, -0.018],
#                             [ 0.029,  0.047, -0.021],
#                             [-0.065, -0.021, -0.041]],

#                            [[ 0.021, -0.042,  0.014],
#                             [-0.047,  0.034, -0.02 ],
#                             [ 0.022,  0.024,  0.019]]])
#     inds = jnp.arange(1, 3)
#     inds_x, inds_y, inds_z = jnp.meshgrid(inds, inds, inds)
#     out = jnp.sum(sine_amps * jnp.sin(inds_x * xx[..., None, None, None]) * jnp.sin(inds_y * yy[..., None, None, None]) * jnp.sin(inds_z * zz[..., None, None, None]), axis=(-1, -2, -3))
#     inds = jnp.arange(0, 3)
#     inds_x, inds_y, inds_z = jnp.meshgrid(inds, inds, inds)
#     out = out + jnp.sum(cos_amps * jnp.cos(inds_x * xx[..., None, None, None]) * jnp.cos(inds_y * yy[..., None, None, None]) * jnp.cos(inds_z * zz[..., None, None, None]), axis=(-1, -2, -3))
#     return out

# def u_target_func(xy):
#     xx = xy[..., 0]
#     yy = xy[..., 1]
#     return jnp.exp((jnp.cos(jnp.pi * xx - 2) + jnp.sin(yy - 1)) * 2) / 50


bessel_zeros = [sp.special.jn_zeros(i, 5) for i in range(5)]

rs = np.linspace(0, 20, 1000000)
jns = [sp.special.jn(n, rs) for n in range(5)]


# jax_jns = [[jnp.interp(rs, sp.special.jn(0, rs))]]

def diskharmonic(r, theta, m, n):
    # return sp.special.jn(m, bessel_zeros[m][n-1] * r) * np.cos(m * theta)
    return jnp.interp(bessel_zeros[m][n - 1] * r, rs, jns[m]) * jnp.cos(m * theta)
    # return sp.special.jn(m, bessel_zeros[m][n-1] * r) * jnp.cos(m * theta)


def u_target_func(xy):
    xx = xy[..., 0]
    yy = xy[..., 1]
    r = jnp.sqrt(xx ** 2 + yy ** 2)
    theta = jnp.arctan2(yy, xx)
    return (diskharmonic(r, theta, 0, 1) - \
            diskharmonic(r, theta, 0, 2) / 4 + \
            diskharmonic(r, theta, 0, 3) / 16 - \
            diskharmonic(r, theta, 0, 4) / 64 + \
            diskharmonic(r, theta, 1, 1) - \
            diskharmonic(r, theta, 1, 2) / 2 + \
            diskharmonic(r, theta, 1, 3) / 4 - \
            diskharmonic(r, theta, 1, 4) / 8 + \
            diskharmonic(r, theta, 2, 1) + \
            diskharmonic(r, theta, 3, 1) + \
            diskharmonic(r, theta, 4, 1)) / 4


# def u_target_func(x):
#     return  (jnp.exp(3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) + jnp.exp(-3 * jnp.sin(x[..., 0]) + jnp.sin(x[..., 1])) -
#              jnp.exp(3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) - jnp.exp(-3 * jnp.sin(x[..., 0]) - jnp.sin(x[..., 1])) -
#              (2 * jnp.exp(3 * jnp.sin(x[..., 0])) + 2 * jnp.exp(-3 * jnp.sin(x[..., 0]))) * jnp.tanh(jnp.cos(x[..., 1] / 2) * 20) * x[..., 1] / 20 * 2 / jnp.pi) / 100


# def u_target_func(x):
#     return (jnp.sin(x[..., 0] * 5))/10

# def u_target_func(x):
#     x = x[..., 0]
#     return jnp.tanh(2 * jnp.sin(x)) / 3 - jnp.exp(-23.5 * (x - jnp.pi / 2) ** 2) + jnp.exp(-27 * (x - 4.2) ** 2) + jnp.exp(-38 * (x - 5.4) ** 2)

def train(config, var_state_new, optimizer, policy_grad):
    with open(os.path.join(config.save_dir, f'steps.txt'), 'w') as fsteps:
        # copy new state to old state
        # intialize optimizer at each step
        opt_state = optimizer.init(var_state_new.get_parameters(flatten=True))
        for step in range(config.nb_steps):
            # get samples from var_state_new
            samples, (boundaries, sqrt_weights_b), sqrt_weights = var_state_new.sample()
            samples = jnp.concatenate([samples, boundaries], 1)
            sqrt_weights = jnp.concatenate([sqrt_weights, sqrt_weights_b], 1)
            # print(var_state_new(samples).max())
            # u_new = var_state_new.evaluate(samples)
            # u_target = u_target_func(samples)
            # reward, loss_value = square_loss(u_new, u_target)
            reward, loss_value1, loss_value2, loss_value3 = fitting_loss(var_state_new, u_target_func, samples)
            # update, info = policy_grad(var_state_new, samples, sqrt_weights, reward)
            update, info = policy_grad(samples, sqrt_weights, reward.reshape(1, -1), resample_params=True)
            info = tuple(
                each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
            update, opt_state = optimizer.update(update, opt_state, var_state_new.get_parameters(flatten=True))
            var_state_new.update_parameters(update)
            loss1 = (loss_value1 * sqrt_weights ** 2).sum().item()
            loss2 = (loss_value2 * sqrt_weights ** 2).sum().item()
            loss3 = (loss_value3 * sqrt_weights ** 2).sum().item()
            logging.info(f'{step=}, {loss1=}, {loss2=}, {loss3=}, {info=}')
            write_to_file(fsteps, step, loss1, loss2, loss3, *info, flush=True)

            # log_psi_new = var_state_new.log_psi(samples)
            # log_p_new = log_psi_new * 2
            # # compute the local operator O p_old(x)/p_old(x)
            # log_psi_old = var_state_old.log_psi(samples)
            # log_p_old = log_psi_old * 2
            # p_old_dot_over_p_old = pde_operator(var_state_old, samples, log_psi_old, compile=True) # p_old_dot / p_old
            # delta_p_old_over_p_old = p_old_dot_over_p_old * dt # delta_p_old / p_old
            # psi_new = jnp.exp(log_psi_new)
            # p_target = jnp.exp(log_p_old) * (1 + delta_p_old_over_p_old)
            # psi_target = jnp.sqrt(p_target)
            # reward, loss_value = square_loss(psi_new, psi_target) # TEST ONLY
            # update, info = policy_grad(var_state_new, samples, sqrt_weights, reward)
            # info = tuple(
            #     each_info.item() if isinstance(each_info, jnp.ndarray) else each_info for each_info in info)
            # update, opt_state = optimizer.update(update, opt_state, var_state_new.get_parameters(flatten=True))
            # var_state_new.update_parameters(update)
            # loss = (loss_value * sqrt_weights**2).sum().item()
            # logging.info(f'{step=}, {T=}, {iter=}, {loss=}, {info=}')
            # write_to_file(fsteps, T, step, iter, loss, *info, flush=True)
            if step % 100 == 0:
                var_state_new.save_state(os.path.join(config.save_dir, 'model_state.pickle'))
                plt.figure(dpi=200)
                xs = jnp.linspace(-1, 1, 1000)
                ys = jnp.linspace(-1, 1, 1000)
                X, Y = jnp.meshgrid(xs, ys)
                in_disk = X ** 2 + Y ** 2 <= 1
                Z = u_target_func(jnp.stack([X, Y], -1).reshape(1, -1, 2)).reshape(1000, 1000)
                if isinstance(var_state_new.net, CircularDirichletPDENet2):
                    Z = Z * in_disk
                print(Z.min(), Z.max())
                plt.imshow(Z, extent=(-1, 1, -1, 1), origin='lower', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
                plt.colorbar()
                plt.savefig(os.path.join(config.save_dir, f"u_target_{step}.png"))
                plt.close()
                plt.figure(dpi=200)
                xs = jnp.linspace(-1, 1, 1000)
                ys = jnp.linspace(-1, 1, 1000)
                X, Y = jnp.meshgrid(xs, ys)
                in_disk = X ** 2 + Y ** 2 <= 1
                Z2 = var_state_new.evaluate(jnp.stack([X, Y], -1).reshape(1, -1, 2)).reshape(1000, 1000)
                if isinstance(var_state_new.net, CircularDirichletPDENet2):
                    Z2 = Z2 * in_disk
                print(Z2.min(), Z2.max())
                plt.imshow(Z2, extent=(-1, 1, -1, 1), origin='lower', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
                plt.colorbar()
                plt.savefig(os.path.join(config.save_dir, f"var_state_{step}.png"))
                plt.close()
                # plot difference
                plt.figure(dpi=200)
                plt.imshow(Z - Z2, extent=(-1, 1, -1, 1), origin='lower')
                plt.colorbar()
                plt.savefig(os.path.join(config.save_dir, f"diff_{step}.png"))
                plt.close()
        # evaluate_and_plot(config, var_state_new, step)
    return config, var_state_new, optimizer, opt_state, policy_grad


def main():
    # get config
    config = get_config()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    file_handler = logging.FileHandler(os.path.join(config.save_dir, "log.txt"), mode="w")
    logger.addHandler(file_handler)

    # define the neural network model
    # assert config.model == 'CreateSimpleRealNVP'
    # net = CreateSimpleRealNVP(nb_rows=1, nb_columns=config.nb_dims,
    #                          hidden_dim=2, nb_layers=1, kernel_size=(1, config.nb_dims), nb_flow_layers=0,
    #                          scale_and_shift_after=True) #TEST ONLY
    # assert config.model == 'SimpleARRQSFlow'
    # net = SimpleARRQSFlow(nb_sites=config.nb_dims, nb_intervals=1, min_bin_size=1e-8, min_derivative=1e-8,
    #                       hidden_dim=8, nb_layers=4, embedding_dim=2, bias_init_final=nn.initializers.uniform(scale=0.),
    #                       use_embedding=False) #TEST ONLY
    # assert config.model == 'SimplePDENet'
    # net = SimplePDENet4(width=40, depth=7, periods=(2, 2 * jnp.pi))
    # net = DirichletPDENet2(width=40, depth=7, minvals=(0, 0), maxvals=(jnp.pi, jnp.pi))
    net = PDENet(width=40, depth=7)
    # net = CircularDirichletPDENet2(width=40, depth=7, radius=1.)

    # net = SimplePDENet3(width=40, depth=7, period=jnp.pi*2)

    # define the sampler
    # assert config.sampler == 'SimpleFlowSampler'
    # sampler = SimpleFlowSampler(system_shape=(1, config.nb_dims), nb_samples=config.nb_samples, rand_seed=config.sampler_seed)
    # sampler = SimpleFlowSampler(system_shape=(config.nb_dims,), nb_samples=config.nb_samples,
    #                             rand_seed=config.sampler_seed)
    # assert config.sampler == 'PeriodicQuadratureSampler'
    # sampler = PeriodicQuadratureSampler(nb_sites=config.nb_dims,
    #                                    nb_samples=config.nb_samples,
    #                                    minvals=(0., 0.), maxvals=(2, jnp.pi*2), rand_seed=config.sampler_seed)
    # sampler = OpenQuadratureSampler(nb_sites=config.nb_dims,
    #                                    nb_samples=config.nb_samples,
    #                                    minvals=0, maxvals=jnp.pi, rand_seed=config.sampler_seed)

    sampler = CircularQuadratureSampler(nb_sites=config.nb_dims,
                                        nb_samples=config.nb_samples,
                                        radius=1., rand_seed=config.sampler_seed)

    # load sampler state is needed
    if config.load_sampler_state_from is not None and config.load_sampler_state_from.lower() != 'none':
        sampler.load_state(config.load_sampler_state_from)

    # define the var_state (in this case it is vps)
    # we need to define two copies of the var_state
    # the net can be shared because it is just a pure function, which will not cause any issue
    # var_state_new = Simplevar_stateReal(net=net, system_shape=(1, config.nb_dims), sampler=sampler, init_seed=config.model_seed)
    # var_state_old = Simplevar_stateReal(net=net, system_shape=(1, config.nb_dims), sampler=sampler, init_seed=config.model_seed)

    var_state_new = SimpleVarStateReal(net=net, system_shape=(config.nb_dims,), sampler=sampler,
                                       init_seed=config.model_seed)
    print(var_state_new.count_parameters())

    # load model state if needed
    if config.load_model_state_from is not None and config.load_model_state_from.lower() != 'none':
        # both can load from the same location because the var_state_new will be updated in the first step of training
        var_state_new.load_state(config.load_model_state_from, allow_missing=True)

    # define the optimizer
    lr_schedule = vars(optax)[config.scheduler](*config.scheduler_args[0], **config.scheduler_args[1])
    if config.optimizer is None or config.optimizer.lower() in ['none', 'minsr', 'mclanchlan', 'tdvp', 'dirac', 'cgls']:
        optimizer = optax.chain(
            optax.add_decayed_weights(weight_decay=-config.weight_decay, mask=None),
            # need a negative sign since our update will be the negative gradient
            optax.scale_by_schedule(lr_schedule),
        )
    else:
        optimizer = optax.chain(
            vars(optax)[f'scale_by_{config.optimizer}'](*config.optimizer_args[0], **config.optimizer_args[1]),
            optax.add_decayed_weights(weight_decay=-config.weight_decay, mask=None),
            # need a negative sign since our update will be the negative gradient
            optax.scale_by_schedule(lr_schedule),
        )
    # if config.load_optimizer_state_from is not None and config.load_optimizer_state_from.lower() != 'none':
    #     with open(config.load_optimizer_state_form, "rb") as f:
    #         opt_state = pickle.load(f)

    # define policy grad function
    if config.optimizer.lower() == 'cgls':
        # must define outside of the loop otherwise jit will not work
        # cgls_solver = partial(cgls_solve2, **config.optimizer_args[1])  # we will ignore args and only use kwargs
        cgls_solver = partial(lsqr_solve, **config.optimizer_args[1])  # we will ignore args and only use kwargs
        # policy_grad = partial(natural_policy_grad_ls, ls_solver=cgls_solver, joint_compile=True,
        #                       real_output=True)
        policy_grad = StackedRandomNaturalPolicyGradLS(var_state_new, cgls_solver, nb_params_to_take=None)
    # elif config.optimizer.lower() == 'minsr':
    #     policy_grad = partial(natural_q_policy_grad_minsr, joint_compile=True, real_output=True
    #                           **config.optimizer_args[1])  # we will ignore args and only use kwargs
    # elif config.optimizer.lower() == 'tdvp':
    #     assert False, 'tdvp does not work here'
    # elif config.optimizer.lower() == 'mclanchlan':
    #     policy_grad = partial(natural_q_policy_grad_tdvp, version='mclanchlan', joint_compile=True,
    #                           **config.optimizer_args[1])  # we will ignore args and only use kwargs
    else:
        policy_grad = partial(simple_q_policy_grad, joint_compile=True,
                              **config.optimizer_args[1])  # we will ignore args and only use kwargs
    # else:
    #     assert False, f'optimizer {config.optimizer=} not supported'
    # evaluate_and_plot(config, var_state_new, 0)
    config, var_state_new, optimizer, opt_state, policy_grad = train(config, var_state_new, optimizer, policy_grad)
    var_state_new.save_state(os.path.join(config.save_dir, 'model_state.pickle'))
    return


if __name__ == '__main__':
    main()