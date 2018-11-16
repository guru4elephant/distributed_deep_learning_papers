## DEEP GRADIENT COMPRESSION: REDUCING THE COMMUNICATION BANDWIDTH FOR DISTRIBUTED TRAINING
Large Scale Distributed Training requires significant communication bandwidth for gradient exchange. When we train models on serveral hundred nodes, it will be even time consuming to do communication through network. Things will be even worse if it is trained in mobile devices where low throughput and high latency exist. This paper proposes deep gradient compression based on the fact that 99% gradients are redundant. Momentum correction, local gradient clipping, momentum factor masking, and warm-up training are applied in deep gradient compression. Application scenarios: speech recognition, image classification, language modeling. gradients are compressed from 270x to 600x. Insensitive to low bandwidth. Improve accuracy actually.

## Related Work
### Gradient Quantization
- 1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns
  Frank Seide, Hao Fu, Jasha Droppo, Gang Li, and Dong Yu
  Summary: reduce gradients transfer data size and achieved 10xpeedup in traditional speech applications

- Qsgd: Randomized quantization for communication-optimal stochastic gradient descent
  Dan Alistarh, Jerry Li, Ryota Tomioka, and Milan Vojnovic
  Summary: balance the trade-off between accuracy and gradient precision

- Terngrad: Ternary gradients to reduce communication in distributed deep learning
  Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li
  Summary: uses 3-level gradients

- Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients
  Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, and Yuheng Zou
  Summary: uses 1-bit weights with 2-bit gradients

### Gradient Sparsification
- Scalable distributed dnn training using commodity gpu cloud computing. 2015
  Nikko Strom
  only send gradients larger than a predefined constant threshold

- Communication quantization for dataparallel training of deep neural networks
  Nikoli Dryden, Sam Ade Jacobs, Tim Moon, and Brian Van Essen
  chose a fixed proportion of positive and negative gradient updates separately

- Sparse communication for distributed gradient descent
  Alham Fikri Aji and Kenneth Heafield
  sparsify the gradients by a single threshold based on the absolute value, and requires layer normalization

- Adacomp: Adaptive residual gradient compression for data-parallel distributed training
  Chia-Yu Chen, Jungwook Choi, Daniel Brand, Ankur Agrawal, Wei Zhang, and Kailash Gopalakrishnan.
  automatically tunes the compression rate depending on local gradient activity

## DGC
### Gradient Sparsification
only send important gradient, only gradients larger than a threshold will be transmitted. small gradients are accumulated until they are large enough and are transmitted.
local gradient accumulation is equivalent to increase the batch size over time.

### Improved the local gradient accumulation
Momentum Correction: Momentum SGD 

Local Gradient Clipping: approximate local gradient threshold and do clipping

### Overcomming The Staleness Effect
Momentum Factor Masking: masking the momentum matrix with gradient mask

Warm-up Training: less aggressive learning rate to slow down the changing speed of the neural network at the start of training. Less aggressive learning rate to slow down the changing speed of the neural network at the start of training, less aggressive gradient sparsity, to reduce the number of extreme gradients being delayed.  Instead of linearly ramping up the learning
rate during the first several epochs, we exponentially increase the gradient sparsity from a relatively
small value to the final value, in order to help the training adapt to the gradients of larger sparsity.

## Implementation



