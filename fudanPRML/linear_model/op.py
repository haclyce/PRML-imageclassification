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

class model_SR(Op):
    def __init__(self, input_dim, output_dim):
        super(model_SR, self).__init__()
        self.params = {}
        # 将线性层的权重参数全部初始化为0
        self.params['W'] = torch.zeros([input_dim, output_dim],dtype=torch.float64).to(device)
        #self.params['W'] = torch.normal(mean=0, std=0.01,size=(input_dim, output_dim)).to(device).to(torch.float64)
        # 将线性层的偏置参数初始化为0
        self.params['b'] = torch.zeros([output_dim],dtype=torch.float64).to(device)
        self.outputs = None
        self.output_dim = output_dim
        self.X = None
        self.grads = {}
        
    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        输入：
            - inputs: shape=[N,D], N是样本数量,D是特征维度
        输出：
            - outputs:预测值,shape=[N,C],C是类别数
        """
        # 线性计算
        self.X = inputs.to(device)
        score = torch.matmul(self.X, self.params['W']) + self.params['b']
        #print("XW+b:",score)
        # Softmax 函数
        self.outputs = softmax(score)
        return self.outputs
    
    def backward(self, labels):

        N =labels.shape[0]
        labels = torch.nn.functional.one_hot(labels.long(), self.output_dim).to(torch.float64).to(device)
        self.grads['W'] = -torch.matmul(self.X.T,(labels-self.outputs))/N
        self.grads['b'] = -torch.matmul(torch.ones([N],dtype=torch.float64).to(device),(labels-self.outputs))/N


class MultiCrossEntropyLoss(Op):
    def __init__(self):
        self.predicts = None
        self.labels = None
        self.num = None

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
        loss=torch.gather(predicts,dim=1,index=labels.long().reshape(-1,1))
        loss=torch.sum(-torch.log2(loss))
        loss=loss/self.num
        return loss 
