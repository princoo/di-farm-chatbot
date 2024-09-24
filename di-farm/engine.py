import torch
import torch.nn.functional as F
from sklearn import metrics
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from helper_functions import accuracy_fn

# hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Engine:
    @staticmethod
    def train(
        model:torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        loss_fn:torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_batch_size:int,
        epoch:int
    ):
        model.to(DEVICE)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=train_batch_size
        )

        train_loss, train_acc = 0,0
        scores = {}

        for batch_idx, batch_data in enumerate(dataloader):
            data = batch_data["token_type_ids"].to(DEVICE)
            target = batch_data["tag"].to(DEVICE)
            model.train()
            y_logits = model(data,target)
            y_preds = torch.argmax(y_logits,dim=1)
            loss = loss_fn(y_logits,target)
            train_loss +=loss
            train_acc = accuracy_fn(y_true=target,y_pred=y_preds)
            # scores = Engine.compute_metrics(y_logits,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(dataloader)
        print(
            "[TRAIN]: epoch : {0} | batch: [{1}/{2}] | loss : {3:.5f} | Acc : {4:.2f}%".format(
                epoch,
                batch_idx + 1,
                len(dataloader),
                train_loss,
                train_acc,
            )
        )
    
    @staticmethod
    def test(
        model:torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        loss_fn:torch.nn.Module,
        train_batch_size:int,
        epoch:int
    ):
        model.to(DEVICE)
        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
        )
        test_loss, test_acc = 0,0
        predictions = []

        model.eval()
        with torch.inference_mode():
            for batch_idx, tdata in enumerate(dataloader):
                data = tdata["token_type_ids"].to(DEVICE)
                target = tdata["tag"].to(DEVICE)
                y_logits = model(data, target)
                y_preds  = torch.argmax(y_logits,dim=1)
                loss = loss_fn(y_logits,target)
                test_loss +=loss
                test_acc = accuracy_fn(y_true=target,y_pred=y_preds)
                # scores = Engine.compute_metrics(ypreds_raw, target)
                ypreds = (
                    torch.softmax(y_logits, dim=1)
                    .argmax(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                predictions.append(ypreds)
            test_loss /= len(dataloader)
            test_acc /= len(dataloader)
            print(
                    "[TEST]: batch: {0}/{1} | epoch: {4} | loss: {2:.5f} | acc: {3:.2f}%".format(
                        batch_idx, len(dataloader),test_loss , test_acc, epoch
                    )
                )

        return predictions

    @staticmethod
    def compute_metrics(yout, target=None):
        if target is None:
            return {}
        yout = yout.detach().cpu().numpy().argmax(axis=1)
        target = target.detach().cpu().numpy()
        print(f"-----\n y_Logits: {yout} targets: {target}  \n------ \n ")
        return {
            "accuracy_score": metrics.accuracy_score(target, yout),
            "f1": metrics.f1_score(target, yout, average="macro"),
            "precision": metrics.precision_score(target, yout, average="macro"),
            "recall": metrics.recall_score(target, yout, average="macro"),
        }

