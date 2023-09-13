# Decision-Pretrained Transformer

This repository contains an implemention of the Decision-Pretrained Transformer (DPT) from the paper [Supervised Pretraining Can Learn In-Context Reinforcement Learning](https://arxiv.org/abs/2306.14892).
DPT is a transformer pretrained via supervised learning that can be deployed in new reinforcement learning (RL) tasks and solve them in-context. The method is intended to work and be studied in Meta-RL-like settings.

This repo supports pretraining and evaluating in the following settings:
- Bandits
- Dark Room (2D sparse-reward navigation)
- A variant of [Miniworld](https://github.com/Farama-Foundation/Miniworld)

### Abstract
> Large transformer models trained on diverse datasets have shown a remarkable ability to learn in-context, achieving high few-shot performance on tasks they were not explicitly trained to solve. In this paper, we study the in-context learning capabilities of transformers in decision-making problems, i.e., reinforcement learning (RL) for bandits and Markov decision processes. To do so, we introduce and study Decision-Pretrained Transformer (DPT), a supervised pretraining method where the transformer predicts an optimal action given a query state and an in-context dataset of interactions, across a diverse set of tasks. This procedure, while simple, produces a model with several surprising capabilities. We find that the pretrained transformer can be used to solve a range of RL problems in-context, exhibiting both exploration online and conservatism offline, despite not being explicitly trained to do so. The model also generalizes beyond the pretraining distribution to new tasks and automatically adapts its decision-making strategies to unknown structure. Theoretically, we show DPT can be viewed as an efficient implementation of Bayesian posterior sampling, a provably sample-efficient RL algorithm. We further leverage this connection to provide guarantees on the regret of the in-context algorithm yielded by DPT, and prove that it can learn faster than algorithms used to generate the pretraining data. These results suggest a promising yet simple path towards instilling strong in-context decision-making abilities in transformers.

## Instructions for Setting Up the Environment


To create a new conda environment, open your terminal and run the following command:

```bash
conda create --name dpt python=3.9.15
```

Install PyTorch by following the [official instructions here](https://pytorch.org/get-started/locally/) appropriately for your system. The recommended versions for the related packages are as follows with CUDA 11.7:

```bash
torch==1.13.0
torchvision==0.14.0
```
For example, you might run:

```bash
conda install pytorch=1.13.0 torchvision=0.14.0 cudatoolkit=11.7 -c pytorch -c nvidia
```

The remaining requirements are fairly standard and are listed in the `requirements.txt`. These can be installed by running

```bash
pip install -r requirements.txt
```

If you want to run optional Miniworld experiments, follow these steps to install the Miniworld environment:

```bash
git clone https://github.com/jon--lee/Miniworld.git
cd Miniworld
git checkout modified
pip install -e .
```

## Running Experiments

Each experiment has three phases: (1) pretraining data collection (2) pretraining (3) evaluation of the in-context algorithm. See the paper for details. There are files `run_bandit.sh`, `run_darkroom.sh`, and `run_miniworld.sh` that show example usage to run these. Training in all settings can take several hours, so it may be prudent to start with smaller problems (e.g. fewer arms, reduced time horizon, etc.). The aboves scripts for bandits and darkroom will generate about 4gb of data total. Miniworld will be substantially larger, so please ensure that you have sufficient disk space.

It is recommended to run batches of data collection in parallel for Miniworld because it requires generating images, which is slower. 

```
@article{lee2023supervised,
  title={Supervised Pretraining Can Learn In-Context Reinforcement Learning},
  author={Lee, Jonathan N and Xie, Annie and Pacchiano, Aldo and Chandak, Yash and Finn, Chelsea and Nachum, Ofir and Brunskill, Emma},
  journal={arXiv preprint arXiv:2306.14892},
  year={2023}
}
```