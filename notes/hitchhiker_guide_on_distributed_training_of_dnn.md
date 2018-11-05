## A Hitchhiker's Guide On Distributed Training of Deep Neural Network
Distributed Training on GPU for large dataset is critical in modern deep learning application. A benchmark dataset like Imagenet can take unto a week with single GPU card whereas it is observed that the training time can be brought down to several minutes with distributed training on multiple GPUS on gpu clusters. The survey explore some synchronous and asynchronous variants of distributed Stochastic Gradient Descent. To obtain higher throughput and lower latency on training tasks, mixed precision training, large batch training and gradient compression techniques are investigated.


### Training Algorithms
1. Synchronous SGD
* The Algorithm
  Nodes in the network compute gradients on their local batch of data after which each node sends their gradients to a master server. The master accumulates these gradients by averaging them to form the new global set of gradients for the weight update step.
* convergence garantisse
* Recent trends have gravitated towards scaling Synchronous SGD, more specifically, training networks with large batch sizes has led to promising results
- large mini-batch algorithms
- modulating the learning rate proportional to the batch size
- Linear learning rate scaling£ºbatch size = 8096
  P. Goyal, P. Dollar, R. Girshick, P. Noordhuis, L. Wesolowski, A. Ky- ? rola, A. Tulloch, Y. Jia, and K. He, ¡°Accurate, large minibatch SGD: training imagenet in 1 hour,¡± arXiv preprint arXiv:1706.02677, 2017.
- DON¡¯T DECAY THE LEARNING RATE, INCREASE THE BATCH SIZE Samuel L. Smith? , Pieter-Jan Kindermans? , Chris Ying & Quoc V. Le Google Brain
- L. N. Smith, ¡°Cyclical learning rates for training neural networks,¡± in Applications of Computer Vision (WACV), 2017 IEEE Winter Conference on. IEEE, 2017, pp. 464?472.
* LARS£ºbatch size = 32k
- B. Ginsburg, I. Gitman, and Y. You, ¡°Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling,¡± 2018.
- The authors of [13] proposed using different learning rates for different layers of the neural network since it was observed that the ratio between the norm of the weights to the norm of the gradients is different for different layers. For example, in the AlexNet model, the ratio for the first conv layer is 5.76 while the ratio for the last fully connected layer is 1345
- Presently, LARS is the state of the art in training with large batch sizes.
* mixed precision training£ºbatch size = 64k
- X. Jia, S. Song, W. He, Y. Wang, H. Rong, F. Zhou, L. Xie, Z. Guo, Y. Yang, L. Yu et al., ¡°Highly Scalable Deep Learning Training System with Mixed-Precision: Training ImageNet in Four Minutes,¡± arXiv preprint arXiv:1807.11205, 2018.
* Problem
- Stragglers
  Slow network conditions, failed network requests, machine crashes or even byzantine errors are all possible failures that are common in a distributed network
- Synchronization Barrier
- Single Point of Failure
- Fault Tolerance
2. Asynchronous SGD
* stability and accuracy not guaranteed
* Compare Async and Sync SGD
- P. H. Jin, Q. Yuan, F. Iandola, and K. Keutzer, ¡°How to scale distributed deep learning?¡± arXiv preprint arXiv:1611.04581, 2016.
* Research Problem
** Stale Gradients
* R. Zhang and J. Kwok, ¡°Asynchronous distributed ADMM for consensus optimization,¡± in International Conference on Machine Learning, 2014, pp. 1701?1709.
* R. Johnson and T. Zhang, ¡°Accelerating stochastic gradient descent using predictive variance reduction,¡± in Advances in neural information processing systems, 2013, pp. 315?323.
* W. Zhang, S. Gupta, X. Lian, and J. Liu, ¡°Staleness-aware async-sgd for distributed deep learning,¡± arXiv preprint arXiv:1511.05950, 2015
* S. Zheng, Q. Meng, T. Wang, W. Chen, N. Yu, Z.-M. Ma, and T.-Y. Liu, ¡°Asynchronous stochastic gradient descent with delay compensation,¡± arXiv preprint arXiv:1609.08326, 2016.
- using Taylor expansion of the gradient function and approximation of Hessian matrix to theoretically prove convergence for convex and non-convex optimization problems. Experiments on image recognition tasks show a good balance between speed and accuracy
- allows workers to maintain their own local weights and coordinates work using an elastic force linking a center variable with the computed weights
** Elastic Averaging SGD
- P. H. Jin, Q. Yuan, F. Iandola, and K. Keutzer, ¡°How to scale distributed deep learning?¡± arXiv preprint arXiv:1611.04581, 2016
* Gossip SGD
* Communication Strategies
* All reduce commonly used
* Ring All Reduce
* Recursive Halfing/Doubling and Binary Blocks algorithm
* In distributed training, the computation vs communication has to be kept optimal for efficient horizontal scaling
* Training remains optimal if the communication step is efficient and synchronized with the computation of various machines i.e computation should finish at approximately the same time across the cluster.
* Network condition not good
* gradient compression
* mixed precision training
* cyclic learning rates
* Tensor Fusion
* The benefits of performing this fusion is the reduction of the overhead of the startup time of each machine and overall reduction of the frequency if network traffic
* However, using tensor fusion for small tensors can lead to the ring All Reduce becoming inefficient and slow, [14] proposes a hierarchical All Reduce that uses a multi layered master slave setup that is observed to give lower latencies.
* It¡¯s wide spread use in production systems like Horovord [38] and Tencent¡¯s Framework [14] make it an important staple in modern distributed training frameworks.
* Training Tricks
* Mixed precision training
* P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. Garcia, B. Ginsburg, M. Houston, O. Kuchaev, G. Venkatesh et al., ¡°Mixed precision training,¡± arXiv preprint arXiv:1710.03740, 2017.
* loss drop off and inferior arithmetic precision problem exists in mixed precision training
* GRADIENT AND PARAMETER COMPRESSION
* Gradient Quantization
* F. Seide, H. Fu, J. Droppo, G. Li, and D. Yu, ¡°1-Bit Stochastic Gradient Descent and Application to Data-Parallel Distributed Training of Speech DNNs,¡± in Interspeech 2014, September 2014.
* Sparsification
* Y. Lin, S. Han, H. Mao, Y. Wang, and W. J. Dally, ¡°Deep gradient compression: Reducing the communication bandwidth for distributed training,¡± arXiv preprint arXiv:1712.01887, 2017.
* It is currently the state of the art in gradient compression.
* Optimal training algorithms and communication strategies for different settings
* Future work
* FEDERATED LEARNING: STRATEGIES FOR IMPROVING COMMUNICATION EFFICIENCY Jakub Konecn¡¦ y? ? , H. Brendan McMahan, Felix X. Yu, Ananda Theertha Suresh & Dave Bacon Google
