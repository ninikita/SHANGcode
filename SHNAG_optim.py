import torch
from torch.optim.optimizer import Optimizer

class SHANG(Optimizer):
    def __init__(self, params, alpha=0.5, time_scale=1, weight_decay=0):
        defaults = dict(alpha=alpha, time_scale=time_scale, weight_decay=weight_decay)
        super(SHANG, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            alpha = group['alpha']
            gamma = group['time_scale']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if wd != 0:
                    grad.add_(p.data, alpha=wd)
                state = self.state.setdefault(p, {})
                if 'vel_prev' not in state:
                    state['vel_prev'] = p.data.clone()
                    continue
                vel_prev = state['vel_prev']
                alpha_ = alpha / (1 + alpha)
                beta = alpha / gamma
                # v_n update
                vel_prev.add_(grad, alpha=-beta)
                # x_{n+1} update
                p.data.mul_(1 - alpha_).add_(vel_prev, alpha=alpha_).add_(grad, alpha=-beta * alpha_)
        return loss

# SHANG++
class ISHANG(Optimizer):
    def __init__(self, params, alpha=0.5, time_scale=2, rho=1, weight_decay=0):
        defaults = dict(alpha=alpha, time_scale=time_scale, rho =rho, weight_decay=weight_decay)
        super(ISHANG, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            alpha = group['alpha']
            gamma = group['time_scale']
            rho = group['rho']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if wd != 0:
                    grad.add_(p.data, alpha=wd)
                state = self.state.setdefault(p, {})
                if 'vel_prev' not in state:
                    state['vel_prev'] = p.data.clone()
                    continue
                vel_prev = state['vel_prev']
                mod_alpha = alpha/(1+ rho * alpha)
                beta = alpha / gamma
                # v_n update
                vel_prev.add_(grad, alpha=-beta)
                # x_{n+1}
                p.data.mul_(1/(1+mod_alpha)).add_(vel_prev, alpha=(mod_alpha/(1+mod_alpha))).add_(grad, alpha=-beta*(mod_alpha/(1+mod_alpha)))
        return loss

