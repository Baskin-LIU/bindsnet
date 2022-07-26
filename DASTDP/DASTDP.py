from abc import ABC, abstractmethod
from torch.nn import Module, Parameter
from typing import Optional, Sequence, Union, Iterable
from bindsnet.network.nodes import CSRMNodes, Nodes

import numpy as np
import torch

from bindsnet.network.topology import (
    AbstractConnection,
    LocalConnection,
    Connection,
)

from bindsnet.learning.learning import LearningRule
from bindsnet.network.nodes import DiehlAndCookNodes

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
        baseDA: float = 0.5,
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
        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))
        self.baseDA = baseDA

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
        self.weight_decay = 1.0 - weight_decay if weight_decay else 1.0
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
        boosted = da > self.baseDA
        #####################
        batch_size = self.source.batch_size


        update = torch.zeros_like(self.connection.w)
        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            update -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1]:
            target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            update += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        update *= (self.connection.wmax - self.connection.w)/self.connection.wmax
        #transfer = np.exp(da - 6.5)
        transfer = boosted * da * 1e-3

        self.connection.wl += self.connection.ws * transfer
        self.connection.ws += update - self.connection.ws * transfer

        self.connection.ws *= self.weight_decay_s
        self.connection.wl *= self.weight_decay_l

        self.connection.w.copy_(self.connection.ws + self.connection.wl)

        # print(self.connection.w.mean())
        # print(self.connection.ws.mean())
        # print(self.connection.wl.mean())
        # print('#####################')

    def _connection_update_v1(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        da = kwargs["DA"]
        boosted = da > self.baseDA
        #####################
        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros((self.source.n), device=self.source.s.device)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros((self.target.n), device=self.target.s.device)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.outer(self.p_plus, target_s) + torch.outer(
            source_s, self.p_minus
        )


        reward_update = da * self.nu[0] * self.connection.dt * self.eligibility_trace


        reward_update *= (self.connection.wmax - self.connection.w)/self.connection.wmax
        #print(reward_update.mean())

        self.connection.ws += 50 * reward_update * 1./(da+1.)
        self.connection.wl += 10 * boosted * reward_update * da/(da+1.)

        self.connection.ws *= self.weight_decay_s
        self.connection.wl *= self.weight_decay_l

        self.connection.w.copy_(self.connection.ws + self.connection.wl)


class iSTDP(LearningRule):
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
        p0: float = 0.01,
        baseDA: float = 0.5,
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
        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))
        self.baseDA = baseDA

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, (float, int)):
            nu = [nu, nu]

        self.nu = torch.zeros(3, dtype=torch.float)
        self.nu[0] = nu[0]
        self.nu[1] = nu[1]
        self.nu[2] = nu[2]
        self.p0 = p0

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            if self.source.batch_size == 1:
                self.reduction = torch.squeeze
            else:
                self.reduction = torch.sum
        else:
            self.reduction = reduction

        # Weight decay.
        self.weight_decay = 1.0 - weight_decay if weight_decay else 1.0
        self.weight_decay_s = 1.0 - weight_decay_s if weight_decay_s else 1.0
        self.weight_decay_l = 1.0 - weight_decay_l if weight_decay_l else 1.0

        if isinstance(connection, (DAConnection, Connection)):
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
        boosted = da > self.baseDA
        #####################
        batch_size = self.source.batch_size

        update = torch.zeros_like(self.connection.w)
        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = (self.target.x.view(batch_size, -1).unsqueeze(1)) * self.nu[0]
            update += self.reduction(torch.bmm(source_s, target_x), dim=0)# * (self.wmin - self.connection.w)

            update += self.reduction(source_s.repeat(1, 1, target_x.shape[0]), dim=0) * self.nu[2] * self.connection.w

            del source_s, target_x
        # Post-synaptic update.
        if self.nu[1]:
            target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            update += self.reduction(torch.bmm(source_x, target_s), dim=0)
            #update -= self.reduction(source_x.repeat(1,1,10), dim=0) * self.nu[1] * self.connection.w * da
            del source_x, target_s

        self.connection.w -= update

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
        # print(self.w.mean())
        # print(self.ws.mean())
        # print(self.wl.mean())
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
            weight_decay_s=weight_decay_s,
            weight_decay_l=weight_decay_l,
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
            w_abs_sum = self.wl.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.wl *= 400 / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """


class AdaptiveLIF(DiehlAndCookNodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        one_spike: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            thresh=thresh,
            rest=rest,
            reset=reset,
            refrac=refrac,
            tc_decay=tc_decay,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            lbound=lbound,
            one_spike=one_spike,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            #self.theta += torch.log(self.theta+1.01) * self.theta_plus * self.s.float().sum(0)
            self.theta += self.theta_plus * self.s.float().sum(0)

        if self.one_spike:
            if self.s.any():
                _any = self.s.view(self.batch_size, -1).any(1)
                ind = torch.multinomial(
                    self.s.float().view(self.batch_size, -1)[_any], 1
                )
                _any = _any.nonzero()
                self.s.zero_()
                self.s.view(self.batch_size, -1)[_any, ind] = 1

        # voltage clipping to lowerbound
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        if self.traces:
            # Decay and set spike traces.
            self.x *= self.trace_decay

            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s.bool(), self.trace_scale)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += x.float()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        self.s.zero_()

        if self.traces:
            self.x.zero_()  # Spike traces.

        if self.sum_input:
            self.summed.zero_()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.


def Advantage(spikes, label, num_labels):

    entropy = torch.exp(spikes[label])/torch.exp(spikes).sum()
    advantage = entropy-1/num_labels

    return advantage



