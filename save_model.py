from pathlib import Path
import torch

def save_model_version(model:torch.nn.Module,
                all_words:int,
                tags:int,
                input_size:int,
                hidden_units:int,
                output_size:int
                ):
    
    #  create a model directory path
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)

    #  Create a model save path
    MODEL_NAME = "chatbot_difarm_V1.pth"
    MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME
    data = {
        'model_state': model.state_dict(),
        'input_size': input_size,
        'hidden_units': hidden_units,
        'output_size': output_size,
        'all_words': all_words,
        'tags': tags,
    }
    torch.save(obj=data,f= MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

