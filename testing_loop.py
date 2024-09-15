import torch
import torch.utils
import torch.utils.data
from helper_functions import accuracy_fn

def test_step(model: torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device,
               ):
    
    test_loss,test_acc = 0,0
    model.train()

    for (words,labels) in dataloader:
        words,labels = words.to(device),labels.to(device)

        #  forward pass
        label_logits = model(words)
        labels = labels.long()
        label_preds = torch.softmax(label_logits,dim=1).argmax(dim=1)
        loss = loss_fn(label_logits,labels)
        test_loss +=loss
        test_acc +=  accuracy_fn(y_true=labels, y_pred=label_preds)
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    # print(f"Train loss:{test_loss:.5f} | Train acc:{test_acc:.2f}%")
    return test_loss,test_acc

