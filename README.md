# Reveal Cancer Invasion Strategy with Reinforcement Learning
We demonstrate how cancer cells learn to reproduce and invade using reinforcement learning. The idea originated from a proposal to Cell Fate Symposium 2018 (see proposal [here](https://www.dropbox.com/s/er54z58cbda2bkn/cell%20fate%20full%20proposal.pdf?dl=0)).

At the beginning of each episode, one cancer cell at the center of a fixed square domain with even-distributed nutrients is initialized. We imagine that an agent is playing a game in which it takes cues from the microenvironment of a cancer cell and instructs the cell whether to reproduce or migrate. The cost of nutrient to reproduce is higher than moving around. The nutrient is limited and diffuse within the domain with a no-flux boundary condition. When the nutrient is below certain threshold the cells start dying. The goal is to maximize the number of offsprings that successfully exit the domain. For this goal, we design rewarding as follows:

* reproduction results in a positive reward and
* death leads to a negative reward of the same amount.
* extra reward is given when a cancer cell is successfully exit the domain.

Here we employed Policy Gradient algorithm to learn possible strategy of cancer invasion. Policy function takes input of ambient nutrient level and stochasticaly maps to the action that can be taken by the cancer cell: moving to one of the nearby sites or reproduce. Here policy function is approximated as a neutral net with one hidden layer. Run the "cell" below to see how the cancer cells improve its reward with training with more and more episode. The bottom graph is the running average reward over the episodes. The two grid plots show details of several typical episodes. Note salient behavior changes after 100 episode the cancer cells compare to the 1st episode.

This is a simple sketch of the idea. In the future, we hope to reveal interesting morphological patterns of cancer invasion by adding more hidden layers and more biologically meaningfully features to the current model.

A demo is avaible at [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/juhang62/cancerRL/master). After it is loaded, simply open the jupyter notebook named demo.ipynb and run the cell. 
