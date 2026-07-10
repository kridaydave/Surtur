# Freeze the models idk
import config
from transformers import AutoModelForCausalLM

def load_model(name, dtype=None, device_map=None):
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if device_map is not None:
        kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)

def layer_numbers(spec, total_layers):
    return config.do_trainable_layers(spec, total_layers)

def get_layers(model):
    base = model
    if hasattr(model, "model"):
        base = model.model
    elif hasattr(model, "transformer"):
        base = model.transformer
    elif hasattr(model, "gpt_neox"):
        base = model.gpt_neox
        
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    if hasattr(base, "h"):
        return base.h
    raise AttributeError("Could not locate decoder layers in this model architecture.")

def apply_freeze(model, trainable_layers):
    for p in model.parameters():
        p.requires_grad_(False)
    decoder = get_layers(model)
    for idx in trainable_layers:
        for p in decoder[idx].parameters():
            p.requires_grad_(True)
    return model

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
