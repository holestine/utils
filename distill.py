import torch
import torch.nn.functional as F
import numpy as np

def KL_Div(student_preds, teacher_preds, T=1.0):
    """ Compute the knowledge-distillation (KD) loss given teacher and student predictions """
    x = F.log_softmax(student_preds/T, dim=1)
    y = F.softmax(teacher_preds/T, dim=1)
    KLDiv = F.kl_div(x,y, reduction='batchmean')
    return KLDiv

def KD_Loss(ground_truth, student_preds, teacher_preds, criterion, T=1.0, alpha=0.7):
    student_classification_loss = criterion(student_preds, ground_truth) 
    teacher_distillation_loss = KL_Div(student_preds, teacher_preds, T)
    KDLoss = (1.0 - alpha) * student_classification_loss + (alpha * T * T) * teacher_distillation_loss
    return KDLoss

shape = (5,5,5)
ground_truth  = torch.randn(shape, requires_grad=False)
student_preds = torch.randn(shape, requires_grad=True)
teacher_preds = torch.randn(shape, requires_grad=True)
kd_loss = KD_Loss(ground_truth, student_preds, teacher_preds, torch.nn.MSELoss())
kd_loss.backward()
print(kd_loss)