# Basic
Credit: 5ECTS

Professor: KOUNTOURIS Marios

Exam: Yes, no documents allowed

Project: optional

Lab: 
+ 1 for neural network
+ 1 for GAN and VAR
+ 1 for federated learning

# Write-up

## Lecture1
Basicly, how to combine machine learning with communication systems

## Lecture 2 & 3
Key concepts from supervised learning and neural networks, like:
- Error
  - Generalization error: $R(h)=\mathbb{E}_{(x,y)\sim{p_{xy}}}[l(y,h(x))]$
  - Empirical error: $\^R(h)=\frac{1}{N}\sum_{n=1}^Nl(y_i,h(x_i))$
  - Bayes error: $R^*=R(h^*)=inf_{h\in H}R(h)$

- Empirical Risk Minimization (ERM)
  - To approaximate the generalization error by the training error
  - Statistical problem
    - How well does the minimizer of the empirical isk perform?
  - Optimization problem
    - How to minimize the empirical risk?

- Structural Risk Minimization
  - Regularization-based algorithm: $h=argmin_{h\in H}\^R(h)+\lambda\phi(h)$

- Model selection
  - Excess Risk = estimation error + approaximation error
  - $\^R(h) - R^*=[\^R(h)-R(h^*)]+[R(h^*)-R^*], R(h^*)=min_{h\in H}R(h)$
  - Approaximation: deterministic and measures how much is lost from restricting the set of predictors
  - Estimation: training error is an estimate of generalization error
  - Bias-variace trade-off: a large class results in a small approaximation error and a large estimation error

- Known True Distribution
  - No need for data
  - Inference problem
- Unknown True Distribution
  - Need for data
  - Learning problem

- Gradient
  - Critical point: all partial derivatives are zero
  - Gradient: vector of all partial derivatives

- Statistical estimation
  - Maximum Liklihood Estimation (MLE)
- Bayesian learning
  - Maximum A Posteriori estimation (MAP)

- Neyman-Pearson Theory
  - For a fixed $P_{FA}$, the liklihood ratio test maximizes $P_D$ with the decision rule $$L(x)=\frac{p(x;\mathcal{H}_1)}{p(x;\mathcal{H}_0)}>\gamma,$$ where $$\int_{x;L(x)>\gamma}p(x;\mathcal{H_o})dx=P_{FA}$$

- Error Type
  - False alarm $P_{FA}$
  - Missed detection $P_{M}$
  - Probability of detection $P_{D}$

- Receiver Operating Characteristics (ROC)
  - ROC curve: the higher, the better
- Area Under ROC curve
  - threshold independent

- Metrics
  - Precision
  - Recall
  - Accuracy
  - F_score
  - True Negative Rate

- Loss Funtions
  - Binary Cross Entropy Loss
  - Cross Entropy Loss
  - Mean Squared Error Loss

## Lecture 4 & 5
Key concepts from Variational Autoencoders (VAE).
- Autoencoders
  - Goal
    - find a general way to map them to another set of data points $\mathcal{z} = {z^{(1)}, z^{(2)}, ... ,z^{(m)}}$ , where $z$’s have lower dimensionality than $x$’s & $z$’s can faithfully reconstruct $x$’s.
  - How it works
    - (1) Encode the input into a compressed and meaningful representation (latent space representation) 
    - (2) Decode it back such that the reconstructed input (lossy) is similar as possible to the original one.
  - Types
    - Regularized autoencoders
    - Variational autoencoders
  - How to interpret Autoencoder Training
    - Expectation approximated through Monte Carlo Sampling
    - Minimization through gradient descent
    - Learns effectively to maximize $I(s;\mathcal{y})=H(s)-H(s|\mathcal{y})$
    - Transmitter learns a higher order constellation
    - Receiver learns the maximum likelihood detector for a sufficiently rich $g_{\theta_R}(\mathcal{y})$

- Variational Autoencoders
  -  An AE whose training is regularized to avoid overfitting and ensure that the latent
space has good properties that enable generative process.
  - How it works
    - Variational: use probabilistic latent encoding
    - Generative process: latent space should be regular enough
  - Kullback-Leibler Divergence
    - Measures how well a probability distribution £ approximates a distribution ¢ (the “truth”)
  - Variational Inference - ELBO
    - Maximize ELBO to find the parameters that give as tight a bound as possible on the marginal probability of $x$

## Lecture 6
About GANs.
- Implicit Generative Model
  - Sampling  the code vector $z$ from a simple, fixed distribution
  - Computing a differentiable function $G$ to map $z$ to $x$ space

- GAN 
  - Baisc Idea
    - Generator
      - creates realistic-looking samples (*fake data*) from the input
      - attempts to generate samples that are likely under the true distribution
      - the random noise is interpreted as a latent code
      - training
        - differentiable modules
      - tries to maximize the log-probability of its samples being classified as *fake*
    - Discriminator
      - figures out if a sample came from the generative network or training set
      - training
        - fix discriminator weights
        - sample from generator
        - backprop error to update discriminator weights 
      - maximize the log-liklihood for the binary classification problem 
    - Theoretical perspectives
      - Zero-sum game
      - Minimax theorem
        - $max_{x\in X}min_{y\in Y}V(x,y)=min_{y\in Y}max_{x\in X}V(x,y)$
        - For every two-player zero-sum game, player1's strategy guarentees a best payoff of $V^*$ regardless of player2'strategy, similarly, player2's strategy guarentess a best payoff of $-V^*$
        - $min_{\theta_g}max_{\theta_d}\mathbb{E}_{x\sim p_{data}}logdD_{\theta_d}(x)+\mathbb{E}_{z\sim p(z)}logD_{\theta_d}(G_{\theta_g}(z))$

## Lecture 7
- Information
  - allows for quantification of the amount of informational content
- Precision of information
  - Fisher
    - amount of information
  - Shannon
    - Statistical uncertainty
- Measuring Information
  - **bit** (binary information unit)
  - one bit of information corresponds to $P_X(x_i)=0.5$
- Properties of information defination
  - **lower probalibity** yields higher information
  - **surprising** holds more information than **valuable** does
  - The information in independent events are **additive**
  - When events are **not independent**, they can be **multiplied**
- Entropy
  - a property of the underlying distribution that measures the amount of randomness or surprise in the random variable
  - $H(X)\leq logN$, with equality iff X is uniformly distributed
  - $H(X)\geq 0$, with equality iff X is deterministic
  - tells the average amount of information required to solve the uncertainty
- Relative Entropy
  - A measure of distance between probability distributions
  - $H(X) \leq H_q(X)$
- Joint Entropy
  - $H(X_1, X_2, ..., X_n) = \sum H(X_i)$ if all events are indepent
- Conditional Entropy
  - $H(X|Y)=\sum p(y)H(X|Y=y)$
  - $H(X|Y)\leq H(X)$ iff X and Y are independent
- Mutual Information
  - quantifies the **dependence** of two random variables X and Y
  - captures **non-linear dependencies**
  - tells how helpful a variable helps **reducing uncertainty**
- $f$-divergence
  - defination
    - let $f:\mathbb{R}_+\in\mathbb(R)$ be a convex function that $f(1)=0$
  - **Blackwell Theorem**
    - If a procedure A has smaller f-divergence than a procedure B (for some fixed),  then there exist some set of prior probabilities such that **procedure A has a smaller probability of error than procedure B**.
  - basic properties
    - non-negativity
    - monotonicity

## Lecture 8
- Main Issues in Large-Scale ML (massive datasets): 
  - Storage
  - Computation
  - Communication
- Parallelism
  - Use the transistors to add more parallel units to the chip (*Increases throughput, but not speed*)
  - Can never actually achieve a linear or super-linear speedup as the amount of parallel workers increases (*Amdahl’s Law*)
  - Distributed Setting
    - Many workers communicate over a network
    - Usually no shared memory abstraction
    - Latency much higher than all other types of parallelism
  - Feasible solutions
    - Scaling up: Getting the largest machine possible, with maxed out RAM
    - Scaling out: Getting a bunch of machines, and linking them together
- Basic patterns of distributed communication
  - Broadcast: Machine A sends data to many machines
  - Reduce: Compute some reduction of data on multiple machines and materialize result on B
  - All-reduce: Compute some reduction of data on multiple machines and materialize result on all those machines
  - Push: Machine A sends some data to machine B
  - Pull: Machine A requests some data from machine B
  - Wait: Pause until another machine says to continue
  - Barrier: Wait for all workers to reach some point in their code
- Distributed ML Models
  - Centralized (No data privacy): All data is in the cloud/edge + Inference and decision making in the cloud/edge
  - Distributed (Privacy-preserving): Only part of the data is in the cloud/edge + ML in the cloud + on-user-device ML
  - Decentralized (sharing models instead of data): Data fully distributed + Collaborative intelligence
- Parameter Server Model
  - A single machine (PS) has the explicit responsibility of storing/maintaining the current value of the parameters.
  - Each worker computes gradients on minibatches of the data, PS updates its parameters by using gradients that are computed by the workers, and pushed to the PS
  - Periodically, PS broadcasts its updated parameters to all other worker machines, so that they can use the updated parameters to compute gradients, workers pull an update copy of the weights from the PS
- Pushing AI/ML to the Edge
  - Communication-Efficient First-order Methods
    - Round Minimization
    - Accelerating mini-batch SGD
    - Variance Reduction
    - Bandwidth Minimization
    - reduce the size of local updates from each device
        - Gradient reuse (LAG)
        - Observation: gradients of some devices vary slowly between two consecutive communication rounds
        - How LAG works
        - gradient quantization
        - gradient sparsification
  - Communication-Efficient Second-order Methods
    - Maintain a global approximated inverse Hessian matrix in the central node
    - Solve a second-order approximation problem locally at each device
  - Over-the-Air Computation
    - Interference can be harnessed for computing functional values instead of being canceled 
  - Function Computation
    - Nomographic functions

## Lecture 8
- Straggler Mitigation
  - How to deal with stragglers? Coding Theory to the Rescue.
- Federated Learning
  - Key features
    - Privacy-Preserving Collaborative ML
    - Instead of making predictions in the cloud, distribute the model, make predictions on device
    - Features
    - On-device datasets: end users keep raw data locally
    - On-device training: end-user devices perform training on a shared model
    - Federated learning: a shared global model is trained via federated computation
    - Federated computation: server collects trained weights from end users and update the shared model 
    - Basic Federated Learning Operation
    - At devices: perform local training, send the trained weights to aggregator
    - At aggregator: average the model, redistribute to the devices
  - Federated Computation
    - Iteratively perform a local training algorithm (e.g., multiple steps of SGD) based on the dataset at each device
    - Aggregate the local updated models
  - Challenges
    - Massively Distributed
    - Limited Communication
    - Unbalanced Data
    - Highly Non-IID Data
    - Unreliable Compute Nodes
    - Dynamic Data Availability
