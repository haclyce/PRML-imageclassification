from .op import *
from .optimizer import BatchGD
from .metric import accuracy
from .runner import RunnerV2_1
from .tool import plot
class MLPClassifier(object):
    def __init__(self,input_size,hide_size,output_size):
        self.input_size=input_size
        self.hidden_size=hide_size
        self.output_size=output_size
    def fit(self,X,y):
        pass

    def predict(self,X_train,y_train,X_dev,y_dev,lr,epoch_num):
    
        # 定义网络
        model = Model_MLP_L2(input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.output_size)

        # 损失函数
        loss_fn = MultiCrossEntropyLoss(model)

        # 优化器
        optimizer = BatchGD(lr, model)

        # 评价方法
        metric = accuracy

        # 实例化RunnerV2_1类，并传入训练配置
        runner = RunnerV2_1(model, optimizer, metric, loss_fn)

        runner.train([X_train, y_train], [X_dev, y_dev], num_epochs=epoch_num, log_epochs=100)
        self.runner=runner
    
    def plt(self):
        plot(self.runner, 'fw-acc.pdf')
    
    def evaluate(self,dataset):
        return self.runner.evaluate(dataset)
