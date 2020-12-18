# DQN_Navigation
This repository contains the code for the first project of the Deep RL nano degree from Udacity.

## Goal 
The objective of this project is to train a RL agent to solve the Unity Banana environment.

![The environment](images/banana.gif)

The goal is for the agent to collect yellow bananas while avoiding blue ones.
The agent can take any of 4 actions: move forward, backward, turn right or left.
The space is composed of the agent's velocity and ray-based perceptions of objects around forward direction.
A reward of +1 is received for any yellow banana collected, and one of -1 for any blue banana.

The environment is considered a to be solved if a mean reward of +13 is obtained over 100 episodes.

## Gettting Started
First, follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to create a virtual environment.

Then, the unity environment should be downloaded.
Download the version that matches your system and unzip the archive.
Remember the path to the extracted folder.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

The notebook `Navigation.ipynb` contains code for the training and rendering of trained agent.
