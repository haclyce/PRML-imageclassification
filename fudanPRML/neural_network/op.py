import torch
from .activation import softmax
device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, inputs):
        raise NotImplementedError

# 实现交叉熵损失函数
class MultiCrossEntropyLoss(Op):
    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None

        self.model = model

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        输入：
            - predicts:预测值,shape=[N, C],N为样本数量
            - labels:真实标签,shape=[N, 1]
        输出：
            - 损失值:shape=[1]
        """
        self.predicts = predicts
        self.labels = labels
        self.num = self.predicts.shape[0]

        loss = 0
        loss=torch.gather(softmax(self.predicts),dim=1,index=labels.long().reshape(-1,1))
        loss=torch.sum(-torch.log2(loss))
        loss=loss/self.num
        
        return loss

    def backward(self):
        loss_grad_predicts = None

              
        first = -torch.nn.functional.one_hot(self.labels.long(), num_classes=self.predicts.shape[1])
        predicts_exp = torch.exp(self.predicts)
        second = predicts_exp / torch.sum(predicts_exp, dim=1, keepdim=True)
        loss_grad_predicts = first + second
        # 梯度反向传播
        self.model.backward(loss_grad_predicts)

#实现logistic 算子
class Logistic(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        outputs = 1.0 / (1.0 + torch.exp(-inputs))
        self.outputs = outputs
        return outputs

    def backward(self, grads):
        outputs_grad_inputs = None

        outputs_grad_inputs=self.outputs*(1.0-self.outputs)
        outputs_grad_inputs=outputs_grad_inputs*grads

        return outputs_grad_inputs

#实现线性层的算子
class Linear(Op):
    def __init__(self, input_size, output_size, name):
        self.params = {}
        self.params['W'] = torch.normal(mean=0.,std=1.,size=(input_size, output_size)).to(device).to(torch.float64)
        self.params['b'] = torch.zeros(size=(1, output_size),dtype=torch.float64).to(device)

        self.inputs = None
        self.grads = {}

        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        outputs = torch.matmul(self.inputs, self.params['W']) + self.params['b']
        return outputs

    def backward(self, grads):
        """
        输入：
            - grads:损失函数对当前层输出的导数
        输出：
            - 损失函数对当前层输入的导数
        """
        loss_grad_inputs = None

        self.grads['W']=torch.matmul(self.inputs.T, grads)
        self.grads['b']=torch.sum(grads, dim=0)
        loss_grad_inputs=torch.matmul(grads, self.params['W'].T)
            
        return loss_grad_inputs

#实现整个网络
class Model_MLP_L2(Op):
    def __init__(self, input_size, hidden_size, output_size):
        # 线性层
        self.fc1 = Linear(input_size, hidden_size, name="fc1")
        # Logistic激活函数层
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_size, output_size, name="fc2")
        self.act_fn2 = Noactivation()

        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2]

    def __call__(self, X):
        return self.forward(X)

    # 前向计算
    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        return a2
        
    # 反向计算
    def backward(self, loss_grad_a2):
  
        loss_grad_z2=self.act_fn2.backward(loss_grad_a2)
        loss_grad_a1=self.fc2.backward(loss_grad_z2)
        loss_grad_z1=self.act_fn1.backward(loss_grad_a1)
        loss_grad_input=self.fc1.backward(loss_grad_z1)
        
#没激活
class Noactivation(Op):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None
    def forward(self, inputs):
        outputs =inputs
        self.outputs = outputs
        return outputs
    def backward(self, grads):
        outputs_grad_inputs=grads
        return outputs_grad_inputs
