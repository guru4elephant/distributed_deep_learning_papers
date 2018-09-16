## An In-depth Performance Characterization of CPU- and GPU-based DNN Training on Modern Architectures

#### The authors try to answer a few questions about distributed training of deep learning models on CPU and GPU hardware.
- What are the computation and communication characteristics of popular deep learning model training?
- How various datasets and networks are handled differently in DL frameworks that executor on CPUs and GPUs?
- Can we devise any possible strategies to evaluate the performance of DL frameworks on different compute architectures
in a standard manner?
- What are the performance trends that can be observed when only a single GPU or a single CPU/processor is used?
- To what degree can scale-out of DNN training help if multiple nodes for both CPU-based and GPU-based DNN training are utilized?

#### some key conclusions
- Computations of convolution account for most of time (up to 83% time) in DNN training
- GPU performs the best in current hardwares
- Recent CPU-based optimizations like MKL-DNN and OpenMP-based thread parallelism leads to excellent speed-ups over under-optimized designs (up to 3.2X improvement for AlexNet training).