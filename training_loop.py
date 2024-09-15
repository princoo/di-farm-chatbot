import torch
import torch.utils
import torch.utils.data
from helper_functions import accuracy_fn

def train_step(model: torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device: torch.device,
               ):
    
    train_loss,train_acc = 0,0
    model.train()

    for (words,labels) in dataloader:
        words,labels = words.to(device),labels.to(device)

        #  forward pass
        label_logits = model(words)
        labels = labels.long()
        label_preds = torch.softmax(label_logits,dim=1).argmax(dim=1)
        loss = loss_fn(label_logits,labels)
        train_loss +=loss
        train_acc +=  accuracy_fn(y_true=labels, y_pred=label_preds)

        #  optimizer zero grad
        optimizer.zero_grad()

        #  loss backward
        loss.backward()

        #  optimizer step
        optimizer.step()
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    # print(f"Train loss:{train_loss:.5f} | Train acc:{train_acc:.2f}%")
    return train_loss,train_acc

