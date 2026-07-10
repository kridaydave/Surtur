from transformers import TrainerCallback


class GradientInsulationCallback(TrainerCallback):
    """Hard guardrail: after the first backward pass, every frozen parameter
    must have a ~0.0 gradient norm. If any frozen layer leaked a gradient,
    the Surtur premise is broken, so we abort the run."""

    def __init__(self, model=None, epsilon: float = 1e-6):
        self.model = model
        self.epsilon = epsilon
        self.verified_once = False

    def _verify(self, model) -> None:
        if model is None:
            return
        frozen_leaked = []
        trainable_count = 0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_norm = float(param.grad.norm().item())
            if param.requires_grad:
                trainable_count += 1
            elif grad_norm > self.epsilon:
                frozen_leaked.append((name, grad_norm))

        if frozen_leaked:
            sample = ", ".join(n for n, _ in frozen_leaked[:5])
            raise RuntimeError(
                f"GRADIENT INSULATION FAILED: {len(frozen_leaked)} frozen params "
                f"received gradients (e.g. {sample}). Surtur premise broken."
            )

        if not self.verified_once:
            self.verified_once = True
            print(
                f"[Surtur] Insulation verified: {trainable_count} trainable params, "
                f"all frozen params at 0.0 grad norm."
            )

    def on_substep_end(self, args, state, control, model=None, **kwargs):
        self._verify(model or self.model or kwargs.get("model"))

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self._verify(model or self.model or kwargs.get("model"))
