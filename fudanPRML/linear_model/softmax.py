from .op import *
from .optimizer import SimpleBatchGD
from .metric import accuracy
from .runner import RunnerV2 
from .tool import plot
class SoftmaxClassifier(object):

    def __init__(self,input_size,output_size):
        self.input_dim=input_size
        self.output_dim=output_size
    def fit(self,X,y):
        pass

    def predict(self,X_train,y_train,X_dev,y_dev,lr,epochs):
        # 特征维度
        input_dim = self.input_dim
        # 类别数
        output_dim = self.output_dim
        # 实例化模型
        model = model_SR(input_dim=input_dim, output_dim=output_dim)
        # 指定优化器
        optimizer = SimpleBatchGD(init_lr=lr, model=model)
        # 指定损失函数
        loss_fn = MultiCrossEntropyLoss()
        # 指定评价方式
        metric = accuracy
        # 实例化RunnerV2类
        runner = RunnerV2(model, optimizer, metric, loss_fn)

        # 模型训练
        runner.train([X_train, y_train], [X_dev,y_dev], num_epochs=epochs, log_eopchs=50, eval_epochs=1, save_path="best_model.pdparams")
        #记录模型训练的结果
        self.runner=runner
    
    def plt(self):    
        #图像为
        plot(self.runner,fig_name='linear-acc2.pdf')

    def evaluate(self,dataset):
        return self.runner.evaluate(dataset)