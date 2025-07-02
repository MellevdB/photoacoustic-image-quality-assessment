import torch
from .model_definition import (
    PhotoacousticQualityNet,
    PhotoacousticQualityNetBN,
    PhotoacousticQualityNetMulti,
    IQDCNN,
    IQDCNNMulti,
    EfficientNetIQA,
    EfficientNetIQAMulti
)

def load_model_checkpoint(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_variant = checkpoint.get('model_variant', 'dropout')
    conv_filters = checkpoint.get('conv_filters', [32, 64, 128, 256])
    num_fc_units = checkpoint.get('num_fc_units', 128)
    dropout_rate = checkpoint.get('dropout_rate', 0.0)
    num_outputs = checkpoint.get('num_outputs', 1)
    in_channels = checkpoint.get('in_channels', 1)

    if model_variant == "dropout":
        model = PhotoacousticQualityNet(
            in_channels=in_channels,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units
        )
    elif model_variant == "batchnorm":
        model = PhotoacousticQualityNetBN(
            in_channels=in_channels,
            conv_filters=conv_filters,
            num_fc_units=num_fc_units
        )
    elif model_variant == "multi":
        model = PhotoacousticQualityNetMulti(
            in_channels=in_channels,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units,
            num_outputs=num_outputs
        )
    elif model_variant == "iqdcnn":
        model = IQDCNN(
            in_channels=in_channels,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units
        )
    elif model_variant == "iqdcnn_multi":
        model = IQDCNNMulti(
            in_channels=in_channels,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units,
            num_outputs=num_outputs
        )
    elif model_variant == "efficientnet":
        model = EfficientNetIQA(pretrained=False)
    elif model_variant == "efficientnet_multi":
        model = EfficientNetIQAMulti(pretrained=False, num_outputs=num_outputs)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")

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
            preds.extend(outputs.cpu().numpy())

    return preds 