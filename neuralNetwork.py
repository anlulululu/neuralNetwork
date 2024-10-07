import numpy as np
import scipy.special

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activate_function=lambda x:scipy.special.expit(x)
        pass

        #更新权重得到最正确的权重
    def train(self,inputs_list,targets_list):
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activate_function(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activate_function(final_inputs)
        output_errors=targets-final_outputs
        hidden_errors=np.dot(self.who.T,output_errors)
        #原来的答案每次增一点靠近正确的答案
        self.who+=self.lr*np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))
        self.wih+=self.lr*np.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),np.transpose(inputs))
        pass
    
    def query(self,inputs_list):
        inputs=np.array(inputs_list,ndmin=2).T
        hidden_inputs=np.dot(self.wih,inputs)
        hidden_outputs=self.activate_function(hidden_inputs)
        final_inputs=np.dot(self.who,hidden_outputs)
        final_outputs=self.activate_function(final_inputs)
        return final_outputs

input_nodes=784
hidden_nodes=200
#0-9一共十个数字
output_nodes=10
learning_rate=0.1
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file=open("D:/VsCode/code/mnist_train.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

count=5
for i in range(count):
    for record in training_data_list:
        all_values=record.split(',')
        #像素点作为输入介于[0.01-1]
        inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01
        #输出答案介于[0.01-0.99]
        targets=np.zeros(output_nodes)+0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass

test_data_file=open("D:/VsCode/code/mnist_test.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

score=[]
for record in test_data_list:
    all_values=record.split(',')
    correct_label=int(all_values[0])
    inputs=(np.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs=n.query(inputs)
    label=np.argmax(outputs)
    if(label==correct_label):
        score.append(1)
    else:
        score.append(0)
        pass
    pass

score_array=np.asarray(score)
print("performance",score_array.sum()/score_array.size)