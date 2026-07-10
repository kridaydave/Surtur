from freeze import load_model, apply_freeze, count_params, layer_numbers, get_layers
from guard import assert_frozen_clean, assert_only_expected_train
import torch

MODEL = "facebook/opt-125m"

model = load_model(MODEL)
total_layers = len(get_layers(model))
print("total layers:", total_layers)

band = layer_numbers("last_4", total_layers)
print("trainable band:", band)

apply_freeze(model, band)
trainable, total = count_params(model)
print(f"trainable: {trainable:,} / total: {total:,}")

assert_only_expected_train(model, band)
print("guard 1 OK: only the band is trainable")

input_ids = torch.randint(0, 1000, (1, 8))
out = model(input_ids=input_ids, labels=input_ids)
loss = out.loss
loss.backward()

opt = torch.optim.SGD(model.parameters(), lr=0.01)
opt.step()

assert_frozen_clean(model)
print("guard 2 OK: no frozen layer received a gradient")
print("ALL CHECKS PASSED")
