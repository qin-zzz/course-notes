# Basic
Credit: 2.5ECTS

Professor: MICHIARDI Pietro

Exam: Yes, no documents allowed

Lab: No, but in AML yes

# Write-up
## Lecture 1 & 2
- Deep Nets approaximate non-linear functions
  - Non-linearities distinguish Deep Nets from Linear Model
  - There is a connection between infinite basis functions (Deep Nets with infinite width), Kernel methods and Gaussian Process

- Cost function
  - Since deep nets are non-linear:
    - loss functions become non-convex
    - we use variants of gradient descent algorithm
    - there are weaker guarantees for convergence
    - initialization is important

  - A probabilistic interpretation
    - likelihood, probability of seeing the data, given the input and the model parameters
    - likelihood is unnormalized, so it's not really probability
    - A principled approach to overcome th typical overfitting of ML is to use Bayesian Inference
> Why likelihood is unnormalized?

> How does it work using bayesian inference?

- Infomation and entropy
  - Entropy, measures of uncertainty of a random variable
  - Killback-Liebler Divergence, measures the divergence between two distributions
  - Cross-Entropy, discrepancy between two distributions

$$ H(p(x),q(x)) = H(X) + KL[p(x)||q(x)] $$

> How does the value of entropy change with p in Bernoulli?

> Proof that $W_{ml} = argminH(p(x),q(x|W))$, where $X$ is the data and $q(x|W)$ is the model.

- Output Units
  - The choice of output units is tightly coupled with cost functions
    - Cross-entropy is often used
    - The choice of how to present the output determines the form of cross-entropy

- Linear Units for Gaussian Output
  - Produce the mean of a conditional Gaussian distribution
  - Equivalent to minimizing the MSE
> Linear units do not saturate, hence work well with gradient based optimization. WHY?

- Sigmoids Units for binary classification
  - Squash the output in [0,1]
  - Defina a Bernoulli distribution over $y$ conditioned on $x$

$$\sigma(z) = \frac{1}{1+e^{-z}}$$
- Softmax Units for multi-class classification
  - Squash the output in [0,1]
  - Define a categorical distribution over $y$ conditioned on $x$

$$softmax(z_i) = \frac{exp(z_i)}{\sum_{j}{exp(z_j)}}$$

- Hidden Units
  - Differentiable computations
    - Using SGD to optimize non-convex functions doesn't gurantee a global minimum
    - Numerical precision limitations
    - Settle to piece-wise differentiable units
> WHY?

- RELU
    - Most hidden units apply an element-wise non-linear function to affine transformation of th inpputs
    - When activation is zero, weights cannot be learned
$$g(z) = max {0,z}$$

-  Regularization
  - Goal
    - prevent overfitting
  - Methods
    - Extra constraints on the model
    - Extra terms in the objective function
    - Ensemble methods
    - Add noise

- Vector Norms
  - Norm
    - Any measure of the size of a vector $||x||=(\sum_i|x_i|^p)^{1/p}$
  - Squared L2 
    - weight decay
    - aka. Ridge regression, or Tikhonov regression
    - $\nabla_W\~{\mathcal{L}}(W;X,y)=\nabla_X\mathcal{L}(W;X,y)+\alpha W$
  - L1
    - $\nabla_W\~{\mathcal{L}}(W;X,y)=\nabla_X\mathcal{L}(W;X,y)+\alpha sign(W)$

> Why sign(W)?

L1 may cause parameters to be zero for large $\alpha$, while L2 does not.

- Noise Injection
  - Where?
    - Input samples
    - Weights of the deep nets
    - Labels

- Early Stopping
  - Can be optimized
  - Doesn't affect cost function
  - Theoretical understanding involved

- Dropout
  - AlexNet
  - Voting
  - Dropout probability to multiply weights at test time
  - Computational efficient
  - Avoids co-adaption
  - $f^{(l)}(h)=g(W^{(l)T}h\odot m^{(l)})$

- DropConnect
  - Only omits connections 

- Sharp & Flat minimizers 

- The structure of loss function
  - Depth has a dramatic impact
  - Deep nets without skip connection are highly non-convex
  - Shortcut connection

- Stochastic Optimization
  - Challenges related to loss surfaces
    - Differentiability
    - Computational cost
    - Model identifiability
    - Loss surface non-convex
    - Saddle points
    - Cliffs and exploding gradients

- Robbins-Monro Conditions
  - Sufficient conditions to guarentee convergence of SGD
$\sum^{\infty}\lambda_t=\infty,\sum^{\infty}\lambda_t^2<\infty$

- SGD with momentum
$$g_t=\frac{1}{m}\nabla\sum\mathcal{J}(W_T,x_i,y_i)$$
$$v_{t+1}=\alpha v_t+\lambda g_t$$
$$W_{t+1}=W_t-v_{t+1}$$
  - Goal: accelerating learning
  - How
    - hyperparameters $\alpha$ determines how quickly the contributions of previous gradients exponentially decay
        - In SGD, step size was $||g||$ multiplied by $\lambda$
        - Now, step size depends on how large nd how aligned a sequence of gradients are

- Adaptive Algorithms
  - Adagrad
    $$r_{t+1}=r_t+g_t\odot g_t$$
    $$W_{t+1}=W_t-\frac{\lambda}{\delta+\sqrt{r}}\odot g_t$$
    - RMSprop
    - better in non-convex settings
    $$r_{t+1}=\rho r_t+(1-\rho)g_t\odot g_t$$
    $$W_{t+1}=W_t-\frac{\lambda}{\delta+\sqrt{r}}\odot g_t$$
  - Adam
  - Adagrad + momentum
    $$s_{t+1}=\rho _1 s_t +(1-\rho _1)g_t$$
    $$r_{t+1}=\rho _2 r_t +(1-\rho _2)g_t\odot g_t$$
    $$\hat{s}=\frac{s_t +1}{1-\rho _1^t}, \hat{r}=\frac{r_t +1}{1-\rho _2^t}$$
    $$W_{t+1}=W_t-\frac{\lambda \hat{s}}{\delta+\sqrt{r}}$$

- Initialization
  - What happens with the scale of parameters
    - too small, signal shrinks
    - too large, signal explodes

- Normalization
  - Why batch normalization
    - the loss landscape more smooth
        - gradients more predictive
        - larger range of learning rate
        - faster convergence
  - Batch Norm
    - mean and std are the same for all training samples
  - Layer Norm
    - mean and std are the same for all features dimensions

## Lecture 3
- Why ConvNet?
  - From feature engineering to feature learning
  - Fundamental ideas:
    - Sparse Connectivity
    - Parameter Sharing
      - kernel parameters are the same for all layer inputs
      - time complexities don't change
      - space complexity greatly improves
    - Equivalent Representation

- Basic Approach
  - $L$: Input 
  - $W$: Convolution kernel of size $k \times k$ 
  - $O$: discrete cross-correaltion operation $O_{ij}=(L\times W)_{i,j}=\sum_m \sum_n L_{i+m,j+n}W_{m,n}$

> Many libraries label cross-correlation as **convolution**;
> $O$ is often called **feature map**.

- Notions
  - Strides
    - Skip some positions to reduce the computational cost
    - Coarse featur extraction
    - Gauge output size
  - Padding
    - Control the kernel and output size
  - Channels
    - Input may have more than one dimensions
  - Sizing convolutional filters
    - $K$: number of kernels
    - $h_k,w_k$: kernel height and kernel width
    - $s$: stride
    - $p$: padding
    - $h_O=\frac{h_L-h_K+2p}{s}+1$
    - $w_O=\frac{w_L-w_K+2p}{s}+1$
    - $d_O=k$
  - Convolutional layer
    - Convolutional filter
    - Non-linear activations
    - Max-pooling filter
  - Pooling layer
    - Reduce the size of feature maps
    - Use summary statistics of nearby outputs
    - Max pooling
      - Provides approaximate invariance to translation
    - Average pooling
      - Reduce the number of parameters in the next layer
    - Layer size
      - $h_O=(h_L-h_K)/s+1$
      - $w_O=(w_L-w_K)/s+1$
      - $d_O=d_L$

> Why needn't differentiate max pooling? Undifferentiable.

- Structure
  - AlexNet
    - Data Augumentation to reduce overfitting
    - RELU Non-linearity
    - Training on multiple GPUs
    - Local Normalization
    - Overlap Pooling
  - VGG-16
    - Small cov filters
    - No local normalization
    - Multi-GPU
    - Artistic learning rate decay
    - Smart initilization
  - ResNet
    - Residual learning block

- Pre-trained Models
  - Transfer learning
    - Use pre-trained weights, remove last layers
  - Fine tuning
    - Add new layers on top of the backbone (not to modifybackbone so much)
    - Fine tune the whole architecture end-to-end
    - Use small datasets but with richer labels

> [slide 76] How to understand it?

- Localization
  - Predict the coordinates of a bounding box (x,y,w,h)
  - metric: IoU (interection over union)
  - multi-task learning

- Object Detection
  - Techniques
    - Object proposal
    - Object classification
  - Approaches
    - Single-stage
      - Region-based are accurate but cost a lot
      - Cons
        - Trade accuracy for a real-time speed
        - Tend to have issues in detecting objects that are too close or too small
      - SSD
        - Single shot detector using vcg16
        - Custom filters to make predictions
      - YOLO
        - Predict bounding boxies and class probabilities
        - Using a single network in a single valuation
        - Simple model allows for real-time tasks 
    - Two-stages
      - Sliding window
      - Selective search
      - RCNN
        - Region proposal methods
        - Regions warped into fix-sized images
        - Classification and refine boundary box
      - Fast-RCNN
        - Use a feature extractor
        - Use an **external region proposal method**
        - Wrap the patch using ROI
        - FC for classification and localization
      - Faster-RCNN
        - Replaces region proposal by a deep net(RPN)

> The differences between single stage an muti stage
> How single stage works?

- Segmentation
  - Categories
    - Semantic(class-aware)
    - Instance(instance-aware)
  - Deconvolution Network
    - Skip connections
    - Sharper masks
    - Better object detection

## Lecture 4
- Sequence Learning
  - Assume data points to be related
  - Be able to deal with sequential distortions
  - Make use of context information
  - Usage
    - Time-series predictions
    - Sequence labelling
  - Methodologies
    - Auto-regressive models
    - Feedfoward neural nets
    - Linear dynamic systems
      - Generative models
    - Hidden markov models
      - Have a discrete one-of-N hidden states

- Sequence Labelling
  - Given a training set A and a testing set B, both drawn independently from a distribution $\mathcal{D}_{\mathcal{X}\times\mathcal{T}}$
  - The goal is to train an algorithm $\mathcal{h}:\mathcal{X}\rightarrow\mathcal{T}$ to label B in a way that it minimizes some task-specific error measures

- Parameter sharing
  - Achieve computational efficiency and generalization
  - Apply the model to examples of different forms
  - Different parameters for each value of the time index (**doen't generalize to sequence length not seen during training**) 

- Unfolding computational graphs
  - The unfolding operations result in parameter sharing
  - Classical recurrent form of a dynamic system $s^{(t)}=f(s^{(t-1)};\theta)$

- From dynamic system to hidden state
  - Dynamical system driven by $s^{(t)}=f(s^{(t-1)},x^{(t)};\theta)$
  - Hidden state as a summary of the past $h^{(t)}=f(h^{(t-1)},x^{(t)};\theta)$, $h^{(t)}$ is a lossy summary of task-relevant aspects of the past

- RNN
  - Hidden state $h^{(t)}=tanh[Ux^{(t)}+Wh^{(t)}+b]$
  - Output $o^{(t)}=Vh^{(t)}+c$
  - $\hat{y}^{(t)}=softmax(o^{(t)})$ 
  - teacher forcing 

> [slide25] How to understand it? What do the loss functions look like?

- Challenges
  - Expensive gradient computation
  - Space and time complexity (**unparallized**)

- Details of backprop
> How does it work?

- Issues with gradient and time-dependencies
  - $h^{(t)}=(W^t)^Th^{(0)}=Q^T\Lambda ^tQh^{(0)}$
  - Many gradients (eigen-value) > 1
    - Exploding gradients
    - Weights update too much
    - **Gradient clipping**
      - if $||g|| > v$, then $g\leftarrow \frac{g_v}{||g||}$
  - Many gradients < 1
    - Vanishing gradients
    - Weight update too little
    - Solution
      - Regularizer + gradient clipping 
      - Use ReLU instead of sigmoid and tahn
      - Initialize hidden state weights to identity matrix and biases to zero
      - Use more complex current cells with additional mechanisms to control information flow

> Do they also happen with CNN and DNN?
> Do we have a mathamatical explaination for vanishing gradients?
> Why sigmoid and tanh have such problem?

- Memory-based architectures
  - LSTM 
    - Cell state $C_t$ runs through the LSTM cell and determines what to remember and what to forget (**linear operations**)
    - Gate networks determine how much of each cell vector component should go through (**point-wise multiplication**)
      - Forget gate 
        - decides which information to **suppress** from Cell State
        - $f_{t}=\sigma(W_f[h_{t-1},x_t]+b_f)$
      - Input gate
        - decide which information to store in Cell State
        - Part1 decides which elements to update
          - $i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$
        - Part2 decides which values to store 
          - $\tilde{C}_t=tanh(W_c[h_{t-1},x_t]+b_c)$
    - Cell state update
      - Forget some elements + add new element values
      - $C_t=f_t\otimes C_{t-1}+i_t \otimes \tilde{C}_t$
    - Producing cell output
      - $o_t=\sigma(W_o[h_{t-1},x_t]+b_o)$ decides how to scale output elements
      - $h_t=o_t\otimes tanh(C_t)$ filters Cell State and squashes it

> part1(sigmoid) map the result to [0,1], decide whether or not; part2(tanh)rotate the result

> Why do we use such design? What is the inspiration? 

- LSTM variants
  - Peehole connections
    - Propagate Cell States through all gates
  - Coupling gates
    - Joint decision about what to forget and what to add
  - GRU
    - Combines forget and addition gate into a single update gate

- Problems with RNN, LSTM
  - RNN
    - long-term dependencies + vanishing gradients + computational cost
  - LSTM
    - long-term dependencies + computational cost
  - Not hardware-friendly, difficult to parallelize

## Lecture 5
- Temporal Convolutional Networks
  - Application domains
    - Auto-regressive prediction
    - Unsupported domains
      - Machine translation
      - Sequence-to-sequence 
      - the entire input sequence

> [slide60]what is out of the box? what are the workarounds?

- Causal Convolutions
  - Layer defination
    - Output same as input
    - No leakage from the future 
  - Problems
    - Keeping long-lasting memory
    - Very deep networks
    - Very fat filters
  - Dilated Convolutions
    - Enable exponentially large receptive field
  - Residual Connections
    - Contains a branch leading out to a series of transformation $\mathcal{F}$
    - Transformed output are added to the input $x$ of the block $o=Activation(x+\mathcal{F}(x))$
    - Allows layers to learn modifications to the identity mapping
  - Pros
    - Parallelism
    - Flexible receptive field size
    - Stable gradients
    - Variable input length
  - Cons
    - Data storage during evaluation (raw sequence size, while RNN only need hidden state)
    - Problems with transfer learning

> [slide64] What is identity mapping?

> [slide65] Why TCN don't suffer from exploding / vanishing gradients?

- Transformer
  - The attention machanism
    - The last decoder can decide to go back and look at a particular part of the input sentence
    - It attends to some hidden states of the input sentence
    - The decoder should learn what to decode
  - self attention
    - every output is th weighted sum over the inputs
    - Scaled dot product
      - normalized by the square root of the input dimenssions $$w_{ij}^{'}=\frac{x_i^Tx_j}{\sqrt{k}}$$
      - the input and output of self attention have similar variance
    - Query, Key, Value
      - Value determines the output
      - Query corresponds the current output
      - Key determines the weight
      - $$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{k}})V$$
    - Multi-head
      - Project the input sequence to several low-dimensioanl sequences
      - Apply a self-attention to each of those
      - Concatenate outputs
    - Masked self attention
      - Apply a mask matrix to the self attention weights
    - Positional information
      - Positional embedding
      - Positional encoding
      - Relative position
    - **Not causal**

> [slide86] why $\sqrt{k}$?

> [slide89] what is mask?

> [slide90] How to understand?

- Sequence to sequence layer
  - handle sequences of different length with same parameters
  - best of both worlds: parallel computation and long dependencies