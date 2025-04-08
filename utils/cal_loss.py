import torch.nn.functional as F
import torch

def cal_loss(outputs,labels,loss_func):
    loss = 0
    if isinstance(outputs,list):
        for i in outputs:
            loss += loss_func(i,labels)
        loss = loss/len(outputs)
    else:
        loss = loss_func(outputs,labels)
    return loss

def cal_kl_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        # 确保在评估模式下不追踪梯度，以提高性能
        with torch.no_grad():
            soft_outputs2 = [torch.softmax(out2, dim=1) for out2 in outputs2]
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1), soft_outputs2[i])
        loss /= len(outputs)
    else:
        # 同样，在评估模式下不追踪梯度
        with torch.no_grad():
            soft_outputs2 = torch.softmax(outputs2, dim=1)
        loss = loss_func(F.log_softmax(outputs, dim=1), soft_outputs2)
    return loss

def cal_js_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        # 确保在评估模式下不追踪梯度，以提高性能
        with torch.no_grad():
            soft_outputs1 = [torch.softmax(out1, dim=1) for out1 in outputs]
            soft_outputs2 = [torch.softmax(out2, dim=1) for out2 in outputs2]
            soft_outputs = []
            for i in range(len(soft_outputs1)):
                soft_outputs.append((soft_outputs1[i]+soft_outputs2[i])/2)
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1), soft_outputs[i])
        loss /= len(outputs)
    else:
        # 同样，在评估模式下不追踪梯度
        with torch.no_grad():
            soft_outputs1 = torch.softmax(outputs, dim=1)
            soft_outputs2 = torch.softmax(outputs2, dim=1)
            soft_outputs = (soft_outputs1+soft_outputs2)/2
        loss = loss_func(F.log_softmax(outputs, dim=1), soft_outputs)
    return loss

def cal_mmd_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        outputs2 = [out2 for out2 in outputs2]
        for i in range(len(outputs)):
            loss += loss_func(outputs[i], outputs2[i])
        loss /= len(outputs)
    else:
        loss = loss_func(outputs, outputs2)
    return loss

def cal_mse_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        outputs2 = [out2 for out2 in outputs2]
        for i in range(len(outputs)):
            loss += loss_func(outputs[i], outputs2[i])
        loss /= len(outputs)
    else:
        loss = loss_func(outputs, outputs2)
    return loss

def cal_triplet_loss(outputs,outputs2,labels,loss_func):
    loss = 0
    if isinstance(outputs,list):
        for i in range(len(outputs)):
            #将输出在B通道进行拼接，labels也进行相应拼接
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels,labels),dim=0)
            loss += loss_func(out_concat,labels_concat)
        loss = loss/len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels,labels),dim=0)
        loss = loss_func(out_concat,labels_concat)
    return loss


def cal_InfoNCE_loss(output,output2,loss_func):

    pass

def cal_MS_loss(outputs,outputs2,labels,loss_func,miner):
    loss = 0
    if isinstance(outputs,list):
        for i in range(len(outputs)):
            #将输出在B通道进行拼接，labels也进行相应拼接
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels,labels),dim=0)
        if miner is not None:
            miner_outputs = miner(out_concat,labels_concat)
            loss += loss_func(out_concat, labels_concat, miner_outputs)
        else:
            loss += loss_func(out_concat, labels_concat)
        # loss = loss/len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels,labels),dim=0)
        if miner is not None:
            miner_outputs = miner(out_concat, labels_concat)
            loss = loss_func(out_concat, labels_concat, miner_outputs)
        else:
            loss = loss_func(out_concat, labels_concat)

    return loss