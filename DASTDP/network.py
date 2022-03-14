from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.spatial.distance import euclidean
from torch.nn.modules.utils import _pair
from typing import Dict, Iterable, Optional, Type
from DASTDP import *

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection

class myNet(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """

    def __init__(
        self,
        n_inpt: int,
        n_output: int,
        n_neurons: int = 100,
        exc: float = 22.5,
        inh: float = 17.5,
        dt: float = 1.0,
        baseDA: float = 1.0,
        DAdecay: float = 0.03,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: float = 0.0,
        wmax: float = 1.0,
        weight_decay_s=2e-3,
        weight_decay_l=1e-4,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        observation_period = 20,
        inpt_shape: Optional[Iterable[int]] = None,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_ouput = n_output
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
        self.baseDA = baseDA
        self.DA = baseDA #concentration of dopamine
        self.DAdecay = DAdecay
        self.observation_period = observation_period
        self.weight_decay_s = weight_decay_s
        self.weight_decay_l = weight_decay_l

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = AdaptiveLIFNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-70.0,
            thresh=-52.0,
            refrac=4,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        # inh_layer = LIFNodes(
        #     n=self.n_neurons,
        #     traces=False,
        #     rest=-60.0,
        #     reset=-45.0,
        #     thresh=-40.0,
        #     tc_decay=10.0,
        #     refrac=2,
        #     tc_trace=20.0,
        # )

        out_layer = AdaptiveLIFNodes(
            n=self.n_ouput,
            traces=True,
            rest=-65.0,
            reset=-70.0,
            thresh=-52.0,
            refrac=4,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        # Connections
        ws = 0.15 * torch.rand(self.n_inpt, self.n_neurons) + 0.1
        wl = 0.05 * torch.rand(self.n_inpt, self.n_neurons)

        input_exc_conn = DAConnection(
            source=input_layer,
            target=exc_layer,
            ws=ws,
            wl=wl,
            weight_decay_s = weight_decay_s,
            weight_decay_l = weight_decay_l,
            update_rule=DASTDP,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        # w = self.exc * torch.diag(torch.ones(self.n_neurons))
        # exc_inh_conn = Connection(
        #     source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        # )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_exc_conn = Connection(
            source=exc_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        w_out_s = 0.15 * torch.rand(self.n_neurons, self.n_ouput)
        w_out_l = 0.15 * torch.rand(self.n_neurons, self.n_ouput)+0.1

        exc_out_conn = DAConnection(
            source=exc_layer,
            target=out_layer,
            ws=w_out_s,
            wl=w_out_l,
            weight_decay_s = weight_decay_s,
            weight_decay_l = weight_decay_l,
            update_rule=DASTDP,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )

        w_out_recurrent = -self.inh * (
                torch.ones(self.n_ouput, self.n_ouput)
                - torch.diag(torch.ones(self.n_ouput))
        )
        recurrent_connection = Connection(
            source=out_layer,
            target=out_layer,
            w=w_out_recurrent,
            wmin=-self.inh,
            wmax=0,
        )


        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        #self.add_layer(inh_layer, name="Ai")
        self.add_layer(out_layer, name="Y")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        #self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(recurrent_exc_conn, source="Ae", target="Ae")
        self.add_connection(exc_out_conn, source="Ae", target="Y")
        self.add_connection(recurrent_connection, source="Y", target="Y")

    def run(
            self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, label=None, **kwargs
    ) -> None:
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        #print(inputs["X"][230, 0, 0])


        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)
                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())
            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))
                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(x=torch.zeros(self.layers[l].s.shape))

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0
            #if self.layers['Y'].s.sum():
                #print(self.layers['Y'].s.sum())

            #Apply dopamine
            if t and not t % self.observation_period:
                out_spikes = (
                    self.monitors["output"]
                        .get("s")
                        .view(self.observation_period, self.n_ouput, 1)
                )
                self.sum_spikes = (
                    out_spikes.sum(0).sum(1)
                )
                if self.sum_spikes.max() != self.sum_spikes.min():
                    pred_label = torch.argmax(self.sum_spikes)
                    target_spikes = self.sum_spikes[label]
                    pred_spikes = self.sum_spikes[pred_label]
                    self.DA += (label==pred_label)*0.5

            # Run synapse updates.
            for c in self.connections:
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, DA=self.DA-self.baseDA, **kwargs
                )

            # # Get input to all layers.
            # current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

            self.DA -= (self.DA-self.baseDA)*self.DAdecay

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()