#!/usr/bin/env python3

import time

import netket as nk
import netket.experimental as nkx
import optax
import flax
from jax import numpy as jnp
from jax.nn.initializers import truncated_normal, zeros
# from netket import experimental as nkx
from netket.jax import dtype_real
from netket.nn import log_cosh
from netket.optimizer import identity_preconditioner
from functools import partial

from args import args
from ham import ColoredJ1J2, Hubbard, tVModel, J1J2OneD
from utils import init_out_dir, tree_size_real_nonzero
import sym_sinekan
import mlp

def get_ham():
    L = args.L
    L2 = args.L2

    if args.boundary == "peri":
        pbc = True
    elif args.boundary == "open":
        pbc = False
    else:
        raise ValueError(f"Unknown boundary: {args.boundary}")

    if args.ham.endswith("tri"):
        assert args.ham_dim == 2
        graph = ColoredJ1J2((L, L2), pbc, back_diag=False)
    elif args.ham.endswith("kag"):
        assert args.ham_dim == 2
        # Nikita's kagome_sqrt18: extent = [3, 2]
        graph = nk.graph.Lattice(
            basis_vectors=[[1, 0], [-0.5, 0.75**0.5]],
            extent=[L, L2],
            pbc=pbc,
            site_offsets=[[0.5, 0], [0.25, 0.75**0.5 / 2], [0.75, 0.75**0.5 / 2]],
        )
    elif args.ham == "j1j2":
        assert args.ham_dim == 2
        graph = ColoredJ1J2((L, L2), pbc)
    else:
        if args.ham_dim == 2:
            extent = [L, L2]
        else:
            extent = [L] * args.ham_dim
        graph = nk.graph.Grid(extent=extent, pbc=pbc)

    if args.ham == "hubb":
        assert not args.zero_mag
        hilbert = nkx.hilbert.SpinOrbitalFermions(
            n_orbitals=graph.n_nodes, s=1 / 2, n_fermions=(args.Nf,) * 2
        )
    elif args.ham == "tv":
        assert not args.zero_mag
        hilbert = nkx.hilbert.SpinOrbitalFermions(
            n_orbitals=graph.n_nodes, n_fermions=args.Nf
        )
    else:
        assert not args.Nf
        if args.zero_mag:
            hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)
        else:
            hilbert = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)

    J = 1
    sign = args.sign == "mars"

    if args.ham == "ising":
        assert args.sign == "none"
        assert not args.J2
        assert not args.U
        assert not args.V
        H = nk.operator.IsingJax(hilbert=hilbert, graph=graph, J=-J, h=args.h)
    elif args.ham.startswith("heis"):
        assert not args.J2
        assert not args.U
        assert not args.V
        assert not args.h
        if args.ham.endswith("tri"):
            H = nk.operator.Heisenberg(
                hilbert=hilbert, graph=graph, J=[J, J], sign_rule=[sign, False]
            )
        else:
            H = nk.operator.Heisenberg(
                hilbert=hilbert, graph=graph, J=J, sign_rule=sign
            )
    elif args.ham == "j1j2":
        assert not args.U
        assert not args.V
        assert not args.h
        H = nk.operator.Heisenberg(
            hilbert=hilbert, graph=graph, J=[J, args.J2], sign_rule=[sign, False], dtype=args.dtype,
        )
    elif args.ham == "hubb":
        assert args.sign == "none"
        assert not args.J2
        assert not args.V
        assert not args.h
        H = Hubbard(hilbert=hilbert, graph=graph, U=args.U)
    elif args.ham == "tv":
        assert args.sign == "none"
        assert not args.J2
        assert not args.U
        assert not args.h
        H = tVModel(hilbert=hilbert, graph=graph, V=args.V)
    elif args.ham == "j1j2_1d":
        H, graph, hilbert = J1J2OneD(L, args.J2, pbc, use_marshall=args.sign=="mars", 
                                     total_sz=0 if args.zero_mag else None)
    else:
        raise ValueError(f"Unknown ham: {args.ham}")
        
    return graph, hilbert, H


def get_net(graph, hilbert):
    N = hilbert.size
    if args.net == "jas":
        assert args.layers == 1
        assert args.features == 1
        return nk.models.Jastrow(
            param_dtype=args.dtype, kernel_init=truncated_normal(stddev=1 / N)
        )
    elif args.net == "rbm":
        assert args.layers == 1
        alpha = args.features
        if jnp.issubdtype(args.dtype, jnp.floating):
            kernel_init = truncated_normal(stddev=1 / (alpha**0.5 * N))
        else:
            kernel_init = truncated_normal(stddev=1 / (alpha**0.25 * N**0.75))

        return nk.models.RBM(
            alpha=alpha,
            param_dtype=args.dtype,
            activation=log_cosh,
            kernel_init=kernel_init,
            hidden_bias_init=zeros,
            visible_bias_init=zeros,
        )
    elif args.net == "gcnn":
        return nk.models.GCNN(
            symmetries=graph,
            layers=args.layers,
            features=args.features,
            param_dtype=args.dtype,
        )
    elif args.net == "rnn_lstm":
        return nkx.models.FastLSTMNet(
            hilbert=hilbert,
            layers=args.layers,
            features=args.features,
            graph=graph,
            param_dtype=args.dtype,
        )
    elif args.net == "mlp":
        return mlp.MLP(
            layers_hidden=args.layers_hidden,
        )
    elif args.net == "symmlp":
        return mlp.SymmetricMLP(
            layers_hidden=args.layers_hidden,
        )
    elif args.net == "sinekan":
        return sym_sinekan.SineKAN(
            layers_hidden=args.layers_hidden,
            grid_size=args.grid_size,
        )
    elif args.net == "sym_sinekan":
        return sym_sinekan.SymmetricSineKAN1D(
            layers_hidden=args.layers_hidden,
            grid_size=args.grid_size,
        )
    else:
        raise ValueError(f"Unknown net: {args.net}")


def get_sampler(graph, hilbert):
    if args.ham in ["hubb", "tv"] or args.zero_mag:
        if args.net.startswith("rnn"):
            raise NotImplementedError
        elif args.net.endswith("sinekan"):
            return nk.sampler.MetropolisLocal(
                hilbert, n_chains=args.batch_size, dtype=dtype_real(args.dtype)
            )
        else:
            if args.ham == "hubb":
                graph = nk.graph.disjoint_union(graph, graph)
            return nk.sampler.MetropolisExchange(
                hilbert,
                graph=graph,
                n_chains=args.batch_size,
                dtype=dtype_real(args.dtype),
            )
    else:
        if args.net.startswith("rnn"):
            return nk.sampler.ARDirectSampler(hilbert, dtype=dtype_real(args.dtype))
        else:
            return nk.sampler.MetropolisLocal(
                hilbert, n_chains=args.batch_size, dtype=dtype_real(args.dtype)
            )


def get_vstate(sampler, model):
    return nk.vqs.MCState(
        sampler,
        model,
        n_samples=args.batch_size,
        n_discard_per_chain=0,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

def custom_learning_rate_schedule(
    step,
    drop_step=15000,
    lr_threshold=5e-4,
    lr_decaytime=1000,
):
    # Phase 1: fixed learning rate up to 'initial_steps'
    phase1 = step < drop_step

    # Phase 2: decayed learning rate after 'initial_steps'
    steps_after_initial = jnp.maximum(step - drop_step, 0)
    lr_phase2 = lr_threshold / (1.0 + steps_after_initial / lr_decaytime)

    # Combine all phases
    lr = jnp.where(phase1, lr_threshold, lr_phase2)
    return lr


def get_optimizer():
    if args.optimizer == "custom":
        # Define the schedule using a partial function with warmup
        schedule = partial(
            custom_learning_rate_schedule,
            drop_step=args.drop_step,
            lr_threshold=args.lr,
            lr_decaytime=args.decay_time,
        )
        # Create the Adam optimizer with the custom schedule as the learning rate
        optimizer = optax.adam(learning_rate=schedule, eps=1e-5)
    elif args.optimizer == "adam" or args.optimizer == "sr":
        optimizer = optax.adam(learning_rate=args.lr, eps=1e-5)
    elif args.optimizer == "sgd":
        optimizer = optax.sgd(learning_rate=args.lr)
    elif args.optimizer == "sr":
        # Define the schedule using a partial function with warmup
        schedule = partial(
            custom_learning_rate_schedule,
            drop_step=args.drop_step,
            lr_threshold=-args.lr,
            lr_decaytime=args.decay_time,
        )
        optimizer = optax.chain(optax.scale_by_schedule(schedule))

    # Optionally add gradient clipping
    if args.grad_clip:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optimizer
        )

    # Optionally handle splitting real and imaginary parts
    if args.split_real:
        optimizer = optax.chain(
            optimizer,
            optax.contrib.split_real_and_imaginary()
        )
    else:
        preconditioner = identity_preconditioner

    return optimizer, preconditioner

def get_vmc(H, vstate, optimizer, preconditioner):
    if args.optimizer != "sr":
        vmc = nk.VMC(
            H,
            variational_state=vstate,
            optimizer=optimizer,
            preconditioner=preconditioner,
        )
    else:
        diag_shift = optax.linear_schedule(
            args.diag_shift, 0, args.max_step
        )
        vmc = nkx.driver.VMC_SRt(
            H,
            variational_state=vstate,
            optimizer=optimizer,
            diag_shift=diag_shift,
        )

    logger = nk.logging.JsonLog(
        args.log_filename,
        "w",
        save_params_every=max(args.max_step // 100, 1),
        write_every=max(args.max_step // 100, 1),
    )

    return vmc, logger


def main():
    init_out_dir()
    print(args.log_filename)

    graph, hilbert, H = get_ham()
    model = get_net(graph, hilbert)
    
    sampler = get_sampler(graph, hilbert)
    vstate = get_vstate(sampler, model)

    print("n_params ", tree_size_real_nonzero(vstate.parameters))

    optimizer, preconditioner = get_optimizer()
    vmc, logger = get_vmc(H, vstate, optimizer, preconditioner)

    print("start_time (full training)", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_time = time.time()

    # Full training
    vmc.run(n_iter=args.max_step, out=logger, show_progress=args.show_progress)
    used_time = time.time() - start_time
    print("used_time (full training)", used_time)

    # Save the final variational state
    with open(args.full_out_dir + "final_vstate.mpack", "wb") as file:
        file.write(flax.serialization.to_bytes(vstate))

    vstate.n_samples = args.estim_size
    energy = vstate.expect(H).mean
    print("energy", energy)

if __name__ == "__main__":
    main()
