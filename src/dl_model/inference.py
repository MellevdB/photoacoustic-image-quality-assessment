import torch
from .model_definition import PhotoacousticQualityNet

def load_model_checkpoint(checkpoint_path, device='cpu'):
    """
    Loads a saved model checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = PhotoacousticQualityNet(in_channels=checkpoint['in_channels'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def run_inference(model, dataloader, device='cpu'):
    """
    Runs inference using a trained model on a given dataloader.
    Returns all predictions in a list (or you can modify to return a CSV).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds.extend(outputs.cpu().numpy().flatten().tolist())

    return preds