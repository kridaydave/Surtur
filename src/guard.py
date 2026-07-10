def assert_frozen_clean(model):
    broken = []
    for name, p in model.named_parameters():
        if not p.requires_grad and p.grad is not None:
            broken.append(name)
    if broken:
        raise RuntimeError(
            "PREMISE BREAK: frozen parameters received a gradient: "
            + ", ".join(broken)
        )


from freeze import get_layers

def assert_only_expected_train(model, expected_trainable):
    decoder = get_layers(model)
    expected_params = set()
    for idx in expected_trainable:
        for p in decoder[idx].parameters():
            expected_params.add(p)
    actual_params = {p for p in model.parameters() if p.requires_grad}
    unexpected = actual_params - expected_params
    if unexpected:
        names = [n for n, p in model.named_parameters() if p in unexpected]
        raise RuntimeError(
            "UNEXPECTED TRAINABLE PARAMS: " + ", ".join(sorted(names))
        )
