# Simulating the effect of PC synchrony on downstream neurons

## Overview
The code in this repository seeks to understand the effect of non-zero
covariance (synchronous firing) in a population of Purkinje cells on 
a downstream neuron in the deep cerebellar nucleus. Current recording
methodolgies allow us to record for pairs (or sometimes triplets) of
Purkinje cells simultaneously, but estimates suggest that approximately
40-50 Purkinje cells synapse on a single cerebellar nucleus neuron 
(c.f., Person and Raman, 2012). As we cannot record from the entire
presynaptic Purkinje cell population simultaneously, we use simulation
to estimate the combined effect of synchronous spiking across this
population of 40-50 Purkinje cells.

To answer this question, the code in the repository constructs two
populations of simulated Purkinje cells. These Purkinje cells are
modeled as Poisson distributed point-process neurons. In the first
"independent" population, we assume that all simulated Purkinje cells
have zero covariance. In the second population, we bootstrap a
covariance matrix by choosing random covariance values between pairs
of Purkinje cells from the distribution that we actually measured.
By keeping the mean firing rates of the two populations the same,
we can ask what the effect of changes in the temporal spiking patterns
might have on the downstream nuclei.

## Data requirements
Raw data necessary for these simulations are stored in the Open Science
Framework repository located [here](https://osf.io/wjg32/). In particular, 
you will need an HDF5 file called `pairwise_covariance.h5`. This file
contains the measured pair-wise covariance values from our Purkinje
cell population. This HDF5 file should be placed in the top-level 
directory of this package after checkout.

## Software requirements
The simulations were performed and tested in Julia v1.8, but should
be compatible with any version of Julia greater than v1.0. Following
installation of Julia, several additional Julia packages will need to
be installed to read from the HDF5 file and perform plotting.

All commands necessary to run the simulation, including the installation
of additional packages, can be found in the Jupyter notebook 
(`pc_synchrony_simulation.ipynb`). Note that you do not need to install
Jupyter to run these commands. Rather, these commands can be copy-and-pasted
directly into a Julia terminal in the order seen in the notebook.

## Estimating the fraction of fully synchronous Purkinje cells
To aid in comparison with past results, namely those from Person and Raman (2012),
our goal was to describe how many Purkinje cells would need to have identical
spiketrains to produce the average distribution of spikes in the independent
and non-zero covariance populations. If all Purkinje cellss are independent, 
then one spike in one Purkinje cell would be accompanied, on average, by 
$(N-1)\bar{M}\Delta{}t$ additional synchronous spikes from the point of view of 
a downstream neuron. Here, $N$ is the number of neurons in the Purkinje cell 
population (e.g., N=40), $\bar{M}$ is the mean firing rate of the 
Purkinje cells in spikes/s, and $\Delta{}t$ is the temporal resolution (1 ms). 
If all PCs are fully synchronous, then the firing of one PC completely 
predicts the firing of the rest of the population, resulting in $(N-1)$ 
spikes arriving at the downstream neuron simultaneously. We solved this model for 
all values of synchrony. Let $x$ be the fraction of the population that is fully 
synchronous, ranging from 0 (fully independent) to 1 (completely synchronous). 
Then, the firing of a single PC in the population predicts the arrival of 
additional spikes at the downstream neuron according to the following equation:
```math
 x \left[ (Nx-1)+(N-Nx) \bar{M}\Delta{}t\right] + (1-x)(N-1)\bar{M}\Delta{}t
```
