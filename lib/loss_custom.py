from torch.autograd import Variable
import torch
from torch.nn.modules.module import Module
from torch.nn.modules.container import Sequential
from torch.nn.modules.activation import LogSoftmax
from torch.nn import functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)

class NLLLoss(_WeightedLoss):
    r"""The negative log likelihood loss. It is useful to train a classification
    problem with n classes
    If provided, the optional argument `weights` should be a 1D Tensor assigning
    weight to each of the classes.
    This is particularly useful when you have an unbalanced training set.
    The input given through a forward call is expected to contain
    log-probabilities of each class: input has to be a 2D Tensor of size
    `(minibatch, n)`
    Obtaining log-probabilities in a neural network is easily achieved by
    adding a  `LogSoftmax`  layer in the last layer of your network.
    You may use `CrossEntropyLoss`  instead, if you prefer not to add an extra
    layer.
    The target that this loss expects is a class index
    `(0 to N-1, where N = number of classes)`
    The loss can be described as::
        loss(x, class) = -x[class]
    or in the case of the weights argument it is specified as follows::
        loss(x, class) = -weights[class] * x[class]
    or in the case of ignore_index::
        loss(x, class) = class != ignoreIndex ? -weights[class] * x[class] : 0
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, has to be a Tensor of size "nclasses"
        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to False, the losses are instead summed for
           each minibatch. Default: True
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average
            is True, the loss is averaged over non-ignored targets.
    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`
        - Target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`
    Examples::
        >>> m = nn.LogSoftmax()
        >>> loss = nn.NLLLoss()
        >>> # input is of size nBatch x nClasses = 3 x 5
        >>> input = autograd.Variable(torch.randn(3, 5), requires_grad=True)
        >>> # each element in target has to have 0 <= value < nclasses
        >>> target = autograd.Variable(torch.LongTensor([1, 0, 4]))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(NLLLoss, self).__init__(weight, size_average)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        _assert_no_grad(target)
        return F.nll_loss(input, target, self.weight, self.size_average,
                          self.ignore_index)


class NLLLoss2d(NLLLoss):
    r"""This is negative log likehood loss, but for image inputs. It computes
    NLL loss per-pixel.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a 1D Tensor having as many elements,
            as there are classes.
        size_average: By default, the losses are averaged over observations
            for each minibatch. However, if the field size_average is set to
            False, the losses are instead summed for each minibatch.
            Default: True
    Shape:
        - Input: :math:`(N, C, H, W)` where `C = number of classes`
        - Target: :math:`(N, H, W)` where each value is `0 <= targets[i] <= C-1`
    Examples::
        >>> m = nn.Conv2d(16, 32, (3, 3)).float()
        >>> loss = nn.NLLLoss2d()
        >>> # input is of size nBatch x nClasses x height x width
        >>> input = autograd.Variable(torch.randn(3, 16, 10, 10))
        >>> # each element in target has to have 0 <= value < nclasses
        >>> target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
        >>> output = loss(m(input), target)
        >>> output.backward()
    """
    pass