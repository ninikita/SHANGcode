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


class SHANGPlus(Optimizer):
    """
    SHANG++
    """
    def __init__(self, params, alpha=0.5, time_scale=1,rho=1.5, weight_decay=0):
        defaults = dict(alpha=alpha, time_scale=time_scale, rho =rho, weight_decay=weight_decay)
        super(SHANGPlus, self).__init__(params, defaults)

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


class SNAG(Optimizer):
    """
    from paper: Algorithm 3
    Julien Hermant, Marien Renaud, Jean-François Aujol, Charles Dossal, and Aude Ronde-
    pierre. Gradient correlation is a key ingredient to accelerate SGD with momentum.
    (ICLR), 2025.
    """
    def __init__(self, params, lr=1e-3, momentum=0.99, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SNAG, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']; m = group['momentum']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if wd != 0:
                    g.add_(p.data, alpha=wd)

                state = self.state.setdefault(p, {})
                b = state.get('b')
                if b is None:
                    b = torch.zeros_like(p.data)
                    state['b'] = b

                # b_n = m * b_{n-1} + g
                b.mul_(m).add_(g)

                # x <- x - lr * ( m * b_n + g )   （注意：不要再次就地改 b）
                res = b.mul(m).add_(g)   # 这里用的是 out-of-place 的 mul
                p.data.add_(res, alpha=-lr)
        return loss


class AGNES(Optimizer):
    """
    from paper:
    Kanan Gupta, Jonathan W. Siegel, and Stephan Wojtowytsch. Nesterov acceleration despite
    very noisy gradients. (NeurIPS),2024.
    https://github.com/kanangupta/AGNES

    When lr = correction, AGNES is NAG
    """

    def __init__(self, params, lr=1e-3, correction=0.1, momentum=0.99, weight_decay=0):
        defaults = dict(correction=correction, lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(AGNES, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AGNES, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr = group['lr']
            momentum = group['momentum']
            correction = group['correction']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                state = self.state[p]  # this contains the sequence of auxiliary variables we need: x'_n, v_n, v'_n
                if 'velocity' not in state:
                    state['velocity'] = torch.zeros_like(p.data)  # initialize v_0 as zero
                vel = state['velocity']

                p.data.add_(d_p, alpha=-correction)
                vel.add_(d_p, alpha=-1)
                vel.mul_(momentum)
                p.data.add_(vel, alpha=lr)

        return loss
