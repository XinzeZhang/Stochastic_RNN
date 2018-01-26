# [Echo state network](http://www.scholarpedia.org/article/Echo_state_network)

_Author: Herbert Jaeger, Jacobs University Bremen, Bremen, Germany_

对于前馈神经网络，一种便捷的训练方式是将其input->hidden和hidden->hidden之间的连接权重随机且固定，然后仅训练最后一层，即仅训练hidden->output层，而最后的hidden层与output层之间实际为线性模型。通过这种方式，我们得到了一个具有非线性能力同时训练又非常快速的随机前馈神经网络。

在随机前馈神经网络的启发下，同样的思想可以应用在循环神经网络Recurrent Neural Network（RNN）上
Echo state network（ESN）在循环神经网络的基础上建立了一种随机的机制。