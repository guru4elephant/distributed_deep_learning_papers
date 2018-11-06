## A Hitchhiker's Guide On Distributed Training of Deep Neural Network
Distributed Training on GPU for large dataset is critical in modern deep learning application. A benchmark dataset like Imagenet can take unto a week with single GPU card whereas it is observed that the training time can be brought down to several minutes with distributed training on multiple GPUS on gpu clusters. The survey explore some synchronous and asynchronous variants of distributed Stochastic Gradient Descent. To obtain higher throughput and lower latency on training tasks, mixed precision training, large batch training and gradient compression techniques are investigated.


### Training Algorithms
#### Synchronous SGD
1. The Algorithm
  - Nodes in the network compute gradients on their local batch of data after which each node sends their gradients to a master server. The master accumulates these gradients by averaging them to form the new global set of gradients for the weight update step.
  - convergence garantisse
  - Recent trends have gravitated towards scaling Synchronous SGD, more specifically, training networks with large batch sizes has led to promising results
    - large mini-batch algorithms
    - modulating the learning rate proportional to the batch size
    - Linear learning rate scaling, batch size = 8096
      - [Accurate, large minibatch SGD: training imagenet in 1 hour 2017.](https://arxiv.org/pdf/1706.02677.pdf)
      - [DONT DECAY THE LEARNING RATE, INCREASE THE BATCH SIZE 2017](https://arxiv.org/pdf/1711.00489.pdf)
      - [Cyclical learning rates for training neural networks, WACV 2017](https://arxiv.org/pdf/1506.01186.pdf)
    - LARS batch size = 32k
      - [Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling 2017](https://people.eecs.berkeley.edu/~youyang/publications/batch32k.pdf)
      - The authors of LARS proposed using different learning rates for different layers of the neural network since it was observed that the ratio between the norm of the weights to the norm of the gradients is different for different layers. For example, in the AlexNet model, the ratio for the first conv layer is 5.76 while the ratio for the last fully connected layer is 1345
      - Presently, LARS is the state of the art in training with large batch sizes.
    - mixed precision training, batch size = 64k
      - [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes 2018.](https://arxiv.org/pdf/1807.11205.pdf)
    - pytorch 18mins training 
      - [Fast and accurate neural nets using modern best practices](https://www.fast.ai/2018/10/02/fastai-ai/)
4. Problem
  - Stragglers
    - Slow network conditions, failed network requests, machine crashes or even byzantine errors are all possible failures that are common in a distributed network
  - Synchronization Barrier
    - wait for the slowest node to finish current mini-batch training
  - Single Point of Failure
  - Fault Tolerance
#### Asynchronous SGD
1. stability and accuracy not guaranteed
2. Compare Async and Sync SGD
  - [How to scale distributed deep learning](https://arxiv.org/pdf/1611.04581.pdf)
3. Research Problem
  - Stale Gradients
    - [Asynchronous distributed ADMM for consensus optimization 2014](http://proceedings.mlr.press/v32/zhange14.pdf)
    - [Accelerating stochastic gradient descent using predictive variance reduction NIPS 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf)
    - [Staleness-aware async-sgd for distributed deep learning IJCAI 2016](https://www.ijcai.org/Proceedings/16/Papers/335.pdf)
    - [Asynchronous stochastic gradient descent with delay compensation 2016](https://arxiv.org/pdf/1609.08326.pdf) using Taylor expansion of the gradient function and approximation of Hessian matrix to theoretically prove convergence for convex and non-convex optimization problems. Experiments on image recognition tasks show a good balance between speed and accuracy
    - allows workers to maintain their own local weights and coordinates work using an elastic force linking a center variable with the computed weights
  - [Elastic Averaging SGD](https://arxiv.org/pdf/1412.6651.pdf)
  - Gossip SGD: [How to scale distributed deep learning](https://arxiv.org/pdf/1611.04581.pdf)
#### Communication Strategies
* All reduce commonly used
  - Ring All Reduce
  - Recursive Halfing/Doubling and Binary Blocks algorithm
  - In distributed training, the computation vs communication has to be kept optimal for efficient horizontal scaling
  - Training remains optimal if the communication step is efficient and synchronized with the computation of various machines i.e computation should finish at approximately the same time across the cluster.
  - Network condition not good
    gradient compression
    mixed precision training
  - cyclic learning rates
* Tensor Fusion
  - The benefits of performing this fusion is the reduction of the overhead of the startup time of each machine and overall reduction of the frequency if network traffic
  - However, using tensor fusion for small tensors can lead to the ring All Reduce becoming inefficient and slow, [Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes 2018](https://arxiv.org/pdf/1807.11205.pdf) proposes a hierarchical All Reduce that uses a multi layered master slave setup that is observed to give lower latencies.
  - It's wide spread use in production systems lik[Horovod](https://arxiv.org/pdf/1802.05799.pdf) and [Tencent's Framework](https://arxiv.org/pdf/1807.11205.pdf) make it an important staple in modern distributed training frameworks.
#### Training Tricks
 * Mixed precision training
  - [Mixed precision training 2017](https://arxiv.org/pdf/1710.03740.pdf)
  - loss drop off and inferior arithmetic precision problem exists in mixed precision training
* GRADIENT AND PARAMETER COMPRESSION
  - Gradient Quantization [1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs, in Interspeech 2014](https://www.microsoft.com/en-us/research/publication/1-bit-stochastic-gradient-descent-and-application-to-data-parallel-distributed-training-of-speech-dnns/)
  - Sparsification [Deep gradient compression: Reducing the communication bandwidth for distributed training 2017](https://arxiv.org/pdf/1712.01887.pdf)
  - It is currently the state of the art in gradient compression.
#### Optimal training algorithms and communication strategies for different settings
#### Future work
* [FEDERATED LEARNING: STRATEGIES FOR IMPROVING COMMUNICATION EFFICIENCY Jakub Konecny 2016](https://arxiv.org/pdf/1610.05492.pdf)
