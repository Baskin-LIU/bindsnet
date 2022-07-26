import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from network import *

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=350)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=300)
parser.add_argument("--load", type=str, default=None)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--teach", dest="teach", action="store_true")

parser.set_defaults(plot=False, gpu=True, teach=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
test = not train
plot = args.plot
gpu = args.gpu
wmax = 10

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()


n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
n_classes = 10
# Build network.
network = myNet(
    n_inpt=784,
    n_output=n_classes,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
    wmax=wmax,
)
if args.load:
    network = torch.load(open(args.load, "rb"))
    w = network.Ai_to_Ae.w
    print(network.Y.theta)
    #print(w.min(),w.max(),w.mean())
    #print(w[0])

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
class NormalizeIntensity():
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        scale = self.mean / tensor.mean()
        return tensor.mul_(scale)

train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=train,
    transform=transforms.Compose(
        [transforms.ToTensor(),
         #NormalizeIntensity(0.12),
         transforms.Lambda(lambda x: x * intensity)] # modify intensity to increase Hz
    ),
)

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.

assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
out_voltage_monitor = Monitor(
    network.layers["Y"], ["v"], time=int(time / dt), device=device
)
# inh_voltage_monitor = Monitor(
#     network.layers["Ai"], ["v"], time=int(time / dt), device=device
# )
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(out_voltage_monitor, name="out_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

output_spike = Monitor(
        network.layers['Y'], state_vars=["s"], time=None, device=device
    )
network.add_monitor(output_spike, name="output")

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
out_weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
print("\nBegin training.\n")
network.train(mode=train)
start = t()
for epoch in range(n_epochs):
    labels = []
    pred_labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )
            labels = torch.stack(labels, dim=0).squeeze(1)
            pred_labels = torch.stack(pred_labels, dim=0)
            #print(labels)
            #print(pred_labels)
            acc = (labels == pred_labels).sum()/len(labels)
            print("predict acc is", acc)

            labels = []
            pred_labels = []

        if step and not step % (2*update_interval):
            network.save("./ckp/%d.ckp" % step)

        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1, label=batch["label"][0].to(device), teach=args.teach)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        out_voltages = out_voltage_monitor.get("v")
        #inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        out_spikes = (
            spikes['Y'].get("s")
                .view(time, n_classes, 1)
        )
        sstep = args.teach*400
        sum_spikes = (
            out_spikes[sstep:].sum(0).sum(1)
        )

        pred_label = torch.argmax(sum_spikes).cpu()
        if not sum_spikes.sum() or sum_spikes.sum() == 10*sum_spikes.max():
            pred_label = torch.tensor(-1)
        pred_labels.append(pred_label)
        #print(batch["label"], pred_label, sum_spikes)
        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )

            exc_out_weights = network.connections[("Ae", "Y")]
            #print(exc_out_weights.w.shape)
            #print('############')


            #print(network.DA)
            #print(exc_out_weights.ws.mean())
            square_out_weights = get_square_weights(
                exc_out_weights.w.view(n_neurons, n_classes), 5, n_sqrt
            )
            square_out_weights_l = get_square_weights(
                exc_out_weights.wl.view(n_neurons, n_classes), 5, n_sqrt
            )
            square_out_weights[30:50,] = square_out_weights_l[:20,]
            spikes_ = {layer: spikes[layer].get("s") for layer in ['X', 'Ae', 'Y']}
            voltages = {"Y": out_voltages}
            #print(out_voltages[:, :10])
            #voltages = {"Ae": exc_voltages, "Y": out_voltages}
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            #weights_im = plot_weights(square_weights, im=weights_im)
            out_weights_im = plot_weights(square_out_weights, im=out_weights_im, wmax=wmax)
            #perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line")


            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )

            #print(exc_out_weights.w.mean(), exc_out_weights.ws.max(), exc_out_weights.wl.min())
            #print(spikes_['Ae'].sum(), input_exc_weights.mean())
            #print(network.connections[("Ai", "Y")].w.min(), exc_out_weights.w.mean())
            plt.pause(1e-6)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")


# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))


print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
