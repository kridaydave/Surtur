# Surtur config - where the fire goes
import json

# Only does x layers from right - may need custom layer setup later (Periphery isnt confined to back few)
def do_trainable_layers(spec,total_layers) :
    if spec == "all" :
        return list(range(total_layers))
    if str(spec).startswith("last_") :
        count = int(spec[5:])
        if count > total_layers :
            raise ValueError("Too many layers")
    elif "," in spec :
        parts = spec.split(",")
        result = []
        for part in parts :
            result.append(int(part))
        return result
    else :
        raise ValueError("Spec should be last_N or a list")

    start = total_layers - count # 12 layers - train last 4 = 8 layers (Need to be frozen)
    return list(range(start,total_layers))


def frozen_calc(total_layers,trainable):
    frozen = []
    for layer in range(total_layers) :
        if layer not in trainable:
            frozen.append(layer)
    return frozen

if __name__ == "__main__" :
    with open("surtur_config.json") as f :
        config = json.load(f)
    trainable = do_trainable_layers(config["spec"],config["total_layers"])
    frozen = frozen_calc(config["total_layers"],trainable)
    print(f"Trainable : {trainable} , Frozen : {frozen}")
