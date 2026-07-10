# Surtur config - where the fire goes
import json

# Only does x layers from right - may need custom layer setup later (Periphery isnt confined to back few)
def do_trainable_layers(spec, total_layers):
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, int):
        return [spec]
    
    spec_str = str(spec).strip()
    if spec_str == "all":
        return list(range(total_layers))
        
    if spec_str.startswith("last_"):
        count = int(spec_str[5:])
        if count > total_layers:
            raise ValueError(f"Too many layers: {count} requested, but total is {total_layers}")
        start = total_layers - count
        return list(range(start, total_layers))
        
    if "-" in spec_str and "," not in spec_str:
        parts = spec_str.split("-")
        if len(parts) == 2:
            start = int(parts[0])
            end = int(parts[1])
            return list(range(start, end + 1))
            
    if "," in spec_str:
        parts = spec_str.split(",")
    else:
        parts = spec_str.split()
        
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            subparts = part.split("-")
            if len(subparts) == 2:
                result.extend(range(int(subparts[0]), int(subparts[1]) + 1))
        else:
            result.append(int(part))
    if result:
        return result
        
    raise ValueError(f"Could not parse layer spec: {spec}")


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
