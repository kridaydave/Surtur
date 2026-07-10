# Freeze the models idk
import config
from transformers import AutoModelForCausalLM

def load_model(name):
    return AutoModelForCausalLM.from_pretrained(name)

def layer_numbers(spec,total_layers) :
    return config.do_trainable_layers(spec,total_layers)

def get_layers(model):
    base = model.model
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    raise AttributeError("could not locate decoder layers in this model")

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
