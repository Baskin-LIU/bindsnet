from abc import ABC, abstractmethod
from torch.nn import Module, Parameter
from typing import Optional, Sequence, Union
from bindsnet.network.nodes import CSRMNodes, Nodes

import numpy as np
import torch

from bindsnet.network.topology import (
    AbstractConnection,
    LocalConnection,
)

from bindsnet.learning.learning import LearningRule

class DASTDP(LearningRule):
    # language=rst
    """
    Dopamine-modulated STDP.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        weight_decay_s: float = 0.0,
        weight_decay_l: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        """
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, (float, int)):
            nu = [nu, nu]

        self.nu = torch.zeros(2, dtype=torch.float)
        self.nu[0] = nu[0]
        self.nu[1] = nu[1]

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            if self.source.batch_size == 1:
                self.reduction = torch.squeeze
            else:
                self.reduction = torch.sum
        else:
            self.reduction = reduction

        # Weight decay.
        self.weight_decay = weight_decay
        self.weight_decay_s = 1.0 - weight_decay_s if weight_decay_s else 1.0
        self.weight_decay_l = 1.0 - weight_decay_l if weight_decay_l else 1.0

        if isinstance(connection, (DAConnection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        da = kwargs["DA"]
        batch_size = self.source.batch_size
        #####################
        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                # batch_size, *self.source.shape, device=self.source.s.device
                batch_size,
                self.source.n,
                device=self.source.s.device,
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                # batch_size, *self.target.shape, device=self.target.s.device
                batch_size,
                self.target.n,
                device=self.target.s.device,
            )
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()

        # Parse keyword arguments.
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        reward_update = self.nu[0] * self.eligibility.squeeze()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            self.p_plus.unsqueeze(2), target_s.unsqueeze(1)
        ) + torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))
        #############  prepost STDP##############################
        nonDA_update = 0.
        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            nonDA_update -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1]:
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            nonDA_update += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        self.connection.ws += nonDA_update
        self.connection.ws += reward_update * da/(da+1)
        self.connection.wl += reward_update * da/(da+1) * da

        self.connection.ws *= self.weight_decay_s
        self.connection.wl *= self.weight_decay_l

        self.connection.w.copy_(self.connection.ws + self.connection.wl)

            # Bound weights.
        if ((self.connection.wmin != -np.inf).any()
                or (self.connection.wmax != np.inf).any()) :
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)



    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """

        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-1, kernel_height, stride)
                .reshape(
                    batch_size,
                    height_out,
                    in_channels * kernel_height,
                )
                .repeat(
                    1,
                    out_channels,
                    1,
                )
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(out_channels * height_out).to(
                self.connection.w.device
            )

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(
                batch_size,
                height_out,
                in_channels * kernel_height,
            )
            .repeat(
                1,
                out_channels,
                1,
            )
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )

        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()


class DAConnection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        weight_decay_s: float = 0.0,
        weight_decay_l: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        Module.__init__(self)

        assert isinstance(source, Nodes), "Source is not a Nodes object"
        assert isinstance(target, Nodes), "Target is not a Nodes object"

        self.source = source
        self.target = target

        # self.nu = nu
        self.weight_decay = weight_decay
        self.reduction = reduction

        self.update_rule = kwargs.get("update_rule", DASTDP)

        # Float32 necessary for comparisons with +/-inf

        self.norm = kwargs.get("norm", None)
        self.decay = kwargs.get("decay", None)

        if self.update_rule is None:
            self.update_rule = DASTDP



        ######################weights##################################

        self.wmin = Parameter(
            torch.as_tensor(kwargs.get("wmin", -np.inf), dtype=torch.float32),
            requires_grad=False,
        )
        #print(self.wmin)
        self.wmax = Parameter(
            torch.as_tensor(kwargs.get("wmax", np.inf), dtype=torch.float32),
            requires_grad=False,
        )


        ws = kwargs.get("ws", None)
        wl = kwargs.get("wl", None)
        if ws is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                ws = torch.clamp(torch.rand(source.n, target.n), self.wmin/2, self.wmax/2)
            else:
                ws = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)/2
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                ws = torch.clamp(torch.as_tensor(ws), self.wmin/2, self.wmax/2)

        if wl is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                wl = torch.clamp(torch.rand(source.n, target.n), self.wmin / 2, self.wmax / 2)
            else:
                wl = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin) / 2
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                wl = torch.clamp(torch.as_tensor(wl), self.wmin / 2, self.wmax / 2)

        self.ws = Parameter(ws, requires_grad=False)
        self.wl = Parameter(wl, requires_grad=False)

        self.w = self.ws + self.wl
        self.w = Parameter(self.w, requires_grad=False)
        #for abstract class

        #############################bias##############################
        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

        self.update_rule = self.update_rule(
            connection=self,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            weight_decay_S=weight_decay_s,
            weight_decay_L=weight_decay_l,
            **kwargs,
        )

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        #print(self.w)
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ self.w
        else:
            post = s.view(s.size(0), -1).float() @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """ """

        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
