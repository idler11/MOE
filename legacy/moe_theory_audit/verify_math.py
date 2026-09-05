"""Reproducible checks for a proposed MoE quantization research direction.

All models here are small RANDOM synthetic networks, not pretrained LLMs.
The candidate-selection experiment uses exact endpoint route corrections;
it validates algebra and an oracle score, NOT the efficiency of a proposed
approximate implementation. Run: python verify_math.py
Requires numpy, scipy and torch. CPU is sufficient.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def empirical_vs_population() -> dict:
    # None of these empirical points is on a routing boundary at theta=0.20337.
    x = np.linspace(-0.999, 0.999, 1000)
    theta, eps = 0.20337, 1e-7
    def loss(t: float) -> float:
        return float(np.mean(((x > t).astype(float) - (x > 0))**2))
    fd = (loss(theta+eps)-loss(theta-eps))/(2*eps)
    return {'finite_sample_difference_quotient': fd,
            'uniform_population_derivative': 0.5,
            'explanation': 'Different objectives: finite empirical risk is locally constant in this example.'}


def boundary_geometry(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n, d, e, k = 2048, 24, 8, 2
    h = rng.standard_normal((n, d))
    delta = 0.35*rng.standard_normal((n, d))
    g = np.exp(0.2*rng.standard_normal(d))
    r = rng.standard_normal((e, d))/np.sqrt(d)
    b = r*g[None, :]
    def top(z: np.ndarray) -> np.ndarray:
        return np.sort(np.argsort(z, axis=1)[:, -k:], axis=1)
    def normalized_scores(z: np.ndarray) -> np.ndarray:
        return (z*g) @ r.T / np.sqrt(np.mean(z*z, axis=1, keepdims=True)+1e-5)
    raw0, raw1 = h @ b.T, (h+delta) @ b.T
    s0, s1 = top(raw0), top(raw1)
    exact_normalization = bool(np.array_equal(s0, top(normalized_scores(h))))
    predicted, numerical = [], []
    for t in range(n):
        active = s0[t]
        inactive = np.setdiff1d(np.arange(e), active)
        margins = raw0[t, active][:, None] - raw0[t, inactive][None, :]
        slope = (delta[t] @ b.T)
        velocities = slope[active, None] - slope[None, inactive]
        roots = np.full_like(margins, np.inf)
        np.divide(margins, -velocities, out=roots, where=velocities < 0)
        predicted.append(float(np.min(roots)))
        numerical.append(not np.array_equal(s0[t], s1[t]))
    predicted = np.asarray(predicted)
    predicted_flips = predicted < 1
    c = np.eye(e)-np.ones((e,e))/e
    a = c@b
    _, sv, vh = np.linalg.svd(a, full_matrices=True)
    rank = int(np.sum(sv > 1e-10))
    null_basis = vh[rank:].T
    null_delta = rng.standard_normal((n, d-rank)) @ null_basis.T
    h_null = h + 0.4*null_delta
    null_routing_stable = bool(np.array_equal(top(normalized_scores(h)), top(normalized_scores(h_null))))
    def softmax(z: np.ndarray) -> np.ndarray:
        exp = np.exp(z-z.max(axis=1, keepdims=True))
        return exp/exp.sum(axis=1, keepdims=True)
    p_change = float(np.max(np.abs(softmax(normalized_scores(h))-softmax(normalized_scores(h_null)))))
    return {'tokens':n, 'input_dimension':d, 'experts':e, 'centered_router_rank':rank,
            'RMSNorm_preserves_topk_order':exact_normalization,
            'predicted_event_vs_actual_accuracy':float(np.mean(predicted_flips==np.asarray(numerical))),
            'actual_event_rate':float(np.mean(numerical)),
            'nullspace_preserves_topk':null_routing_stable,
            'nullspace_max_gate_probability_change':p_change}


def finite_step_experiment(seed: int, candidates: int = 32) -> dict:
    torch.manual_seed(seed)
    n, d, inp, experts, width, k = 512, 16, 12, 8, 24, 2
    x = torch.randn(n, inp)
    residual = 0.5*torch.randn(n, d)
    teacher_w = 0.5*torch.randn(d, inp)/np.sqrt(inp)
    router = torch.randn(experts, d)/np.sqrt(d)
    gamma = torch.exp(0.1*torch.randn(d))
    w_up = torch.randn(experts, width, d)/np.sqrt(d)
    w_down = torch.randn(experts, d, width)/np.sqrt(width)
    bits = 4
    max_code = 2**(bits-1)-1
    scale = teacher_w.abs().max()/max_code
    code0 = torch.clamp(torch.round(teacher_w/scale), -max_code, max_code)
    q0 = code0*scale

    def outputs(w: torch.Tensor, force: torch.Tensor | None = None):
        h = residual + x @ w.T
        hn = h*gamma/torch.sqrt((h*h).mean(dim=-1, keepdim=True)+1e-5)
        probs = torch.softmax(hn @ router.T, dim=-1)
        active = torch.topk(probs, k, dim=-1).indices if force is None else force
        all_hidden = torch.nn.functional.silu(torch.einsum('nd,emd->nem', hn, w_up))
        all_outputs = torch.einsum('nem,edm->ned', all_hidden, w_down)
        weights = probs.gather(1, active)
        # Renormalized top-k weights: a specified synthetic architecture.
        weights = weights/weights.sum(dim=-1, keepdim=True)
        out = all_outputs.gather(1, active[:,:,None].expand(-1,-1,d))
        out = h+(weights[:,:,None]*out).sum(dim=1)
        return out, active

    teacher, _ = outputs(teacher_w)
    teacher = teacher.detach()
    old_y, old_s = outputs(q0)
    old_s = old_s.detach()
    def sample_losses(y):
        return ((y-teacher)**2).mean(dim=1)
    old_l = sample_losses(old_y).mean().item()
    true_deltas, smooth_scores, corrected_scores = [], [], []
    route_terms, identity_errors, event_rates, signs = [], [], [], []
    # Purely random, feasible, fixed-grid perturbations; candidates are not fitted to the scores.
    for _ in range(candidates):
        mask = torch.rand_like(q0)<0.15
        direction = 2*(torch.rand_like(q0)>0.5).to(q0.dtype)-1
        code1 = torch.clamp(code0+mask*direction, -max_code, max_code)
        q1 = code1*scale
        dw = q1-q0
        t = torch.zeros((), requires_grad=True)
        fy, _ = outputs(q0+t*dw, old_s)
        fl = sample_losses(fy).mean()
        grad = torch.autograd.grad(fl, t, create_graph=True)[0]
        hess = torch.autograd.grad(grad, t)[0]
        taylor = (grad+0.5*hess).item()
        with torch.no_grad():
            fixed_y, _ = outputs(q1, old_s)
            new_y, new_s = outputs(q1)
            lf = sample_losses(fixed_y)
            ln = sample_losses(new_y)
            flip = torch.any(torch.sort(new_s, dim=1).values != torch.sort(old_s, dim=1).values, dim=1)
            d_out = new_y-fixed_y
            r_out = fixed_y-teacher
            j_identity = (2*r_out*d_out+d_out*d_out).mean(dim=1)
            j = ln-lf
            exact_delta = ln.mean().item()-old_l
            fixed_delta = lf.mean().item()-old_l
            jump = j.mean().item()
            identity_errors.append(abs(exact_delta-fixed_delta-jump))
            identity_errors.append(float((j-j_identity).abs().max()))
            if bool(flip.any()):
                signs.extend(j[flip].tolist())
            event_rates.append(flip.double().mean().item())
            true_deltas.append(exact_delta)
            smooth_scores.append(taylor)
            corrected_scores.append(taylor+jump)
            route_terms.append(jump)
    truth=np.array(true_deltas); smooth=np.array(smooth_scores); corrected=np.array(corrected_scores)
    sf=np.array(signs)
    return {'seed':seed, 'tokens':n, 'candidates':candidates,
        'max_identity_error':max(identity_errors),
        'mean_event_rate':float(np.mean(event_rates)),
        'fraction_negative_endpoint_route_terms_given_event':float(np.mean(sf<0)),
        'mean_absolute_Taylor_error':float(np.mean(np.abs(smooth-truth))),
        'mean_absolute_corrected_error':float(np.mean(np.abs(corrected-truth))),
        'spearman_Taylor_vs_actual':float(spearmanr(smooth,truth).statistic),
        'spearman_corrected_vs_actual':float(spearmanr(corrected,truth).statistic),
        'candidate_regret_Taylor':float(truth[np.argmin(smooth)]-truth.min()),
        'candidate_regret_corrected':float(truth[np.argmin(corrected)]-truth.min()),
        'actual_delta_range':[float(truth.min()),float(truth.max())],
        'warning':'Endpoint route terms evaluated exactly; computational savings NOT tested.'}


def rounding_search(seed: int, p: int = 8) -> dict:
    """Exhaustive floor/ceiling choices at p most ambiguous weights.

    Coordinates selected only by fractional rounding distance to 0.5.
    This uses exact Hessians on a tiny model and exact route correction,
    so this is an oracle feasibility check, not an efficient LLM algorithm.
    """
    torch.manual_seed(seed)
    n, d, inp, experts, width, k = 256, 12, 8, 6, 16, 2
    x = torch.randn(n, inp)
    residual = 0.5*torch.randn(n, d)
    wt = 0.5*torch.randn(d, inp)/np.sqrt(inp)
    router = torch.randn(experts, d)/np.sqrt(d)
    up = torch.randn(experts, width, d)/np.sqrt(d)
    down = torch.randn(experts, d, width)/np.sqrt(width)
    maxcode=3 # 3-bit symmetric code grid, omitting -4.
    scale=wt.abs().max()/maxcode
    u=wt/scale
    q0=torch.clamp(u.round(),-maxcode,maxcode)*scale
    floor=u.floor(); ceil=u.ceil()
    ambiguity=(u-floor-0.5).abs()
    admissible=(floor>=-maxcode)&(ceil<=maxcode)&(ceil>floor)
    ambiguity[~admissible]=float('inf')
    ids=torch.argsort(ambiguity.flatten())[:p]
    directions=torch.zeros(p,d,inp)
    base_code=(q0/scale).round().flatten()
    fl=floor.flatten(); ce=ceil.flatten()
    for j,ind in enumerate(ids):
        alternative=ce[ind] if base_code[ind]==fl[ind] else fl[ind]
        directions[j].view(-1)[ind]=(alternative-base_code[ind])*scale

    def forward(w, force=None):
        h=residual+x@w.T
        hn=h/torch.sqrt((h*h).mean(-1,keepdim=True)+1e-5)
        prob=torch.softmax(hn@router.T,-1)
        active=torch.topk(prob,k,-1).indices if force is None else force
        allh=torch.nn.functional.silu(torch.einsum('nd,emd->nem',hn,up))
        allout=torch.einsum('nem,edm->ned',allh,down)
        pw=prob.gather(1,active); pw=pw/pw.sum(-1,keepdim=True)
        out=allout.gather(1,active[:,:,None].expand(-1,-1,d))
        return h+(pw[:,:,None]*out).sum(1),active
    teacher,_=forward(wt); teacher=teacher.detach()
    y0,s0=forward(q0); s0=s0.detach()
    def ll(y): return ((y-teacher)**2).mean(1)
    l0=ll(y0).mean().item()
    def fixed_loss(z):
        y,_=forward(q0+torch.einsum('p,pdi->di',z,directions),s0)
        return ll(y).mean()
    z0=torch.zeros(p,requires_grad=True)
    g=torch.autograd.functional.jacobian(fixed_loss,z0).detach()
    H=torch.autograd.functional.hessian(fixed_loss,z0).detach()
    Z=((torch.arange(2**p)[:,None]>>torch.arange(p)[None,:])&1).double()
    smooth=(Z@g+0.5*torch.einsum('bp,pq,bq->b',Z,H,Z)).numpy()
    actual=[]; jumps=[]; rate=[]; err=[]
    with torch.no_grad():
        for z in Z:
            q1=q0+torch.einsum('p,pdi->di',z,directions)
            y1,s1=forward(q1)
            yf,_=forward(q1,s0)
            a=ll(y1).mean().item()-l0
            j=(ll(y1)-ll(yf)).mean().item()
            actual.append(a); jumps.append(j)
            rate.append(float((torch.sort(s1,dim=1).values!=torch.sort(s0,dim=1).values).any(1).double().mean()))
            err.append(abs(a-(ll(yf).mean().item()-l0)-j))
    actual=np.asarray(actual); corrected=smooth+np.asarray(jumps)
    best=int(np.argmin(actual)); bs=int(np.argmin(smooth)); bc=int(np.argmin(corrected))
    return {'seed':seed,'tokens':n,'binary_rounding_choices':p,'candidates':int(2**p),
            'baseline_nearest_rounding_loss':l0,
            'oracle_best_loss':float(l0+actual[best]),
            'Taylor_selected_actual_loss':float(l0+actual[bs]),
            'corrected_selected_actual_loss':float(l0+actual[bc]),
            'Taylor_regret':float(actual[bs]-actual[best]),
            'corrected_regret':float(actual[bc]-actual[best]),
            'mean_absolute_Taylor_error':float(np.mean(abs(smooth-actual))),
            'mean_absolute_corrected_error':float(np.mean(abs(corrected-actual))),
            'spearman_Taylor_vs_actual':float(spearmanr(smooth,actual).statistic),
            'spearman_corrected_vs_actual':float(spearmanr(corrected,actual).statistic),
            'mean_event_rate':float(np.mean(rate)),
            'max_identity_error':max(err),
            'warning':'Exact full p-dimensional Hessian and exact endpoint corrections on random toy model. No pretrained model result, runtime improvement, or generalization claim.'}


def scalar_example() -> dict:
    # Fully specified counterexample, not fitted experimental data.
    # F(h)=h+1{h>0}; old h=-0.01, target=-0.01, next h=0.01.
    h0, h1 = -0.01, 0.01
    target=h0
    old=h0
    forced_new=h1
    actual_new=h1+1
    residual=forced_new-target
    jump=actual_new-forced_new
    return {'smooth_prediction':residual**2,
            'signed_endpoint_correction':2*residual*jump+jump**2,
            'actual_loss_increase':(actual_new-target)**2-(old-target)**2}


if __name__=='__main__':
    result={'disclaimer':'CPU synthetic verification only. No pretrained model weights or benchmarks used.',
            'finite_vs_population':empirical_vs_population(),
            'routing_geometry':boundary_geometry(),
            'scalar_crossing_example':scalar_example(),
            'finite_step_trials':[finite_step_experiment(s) for s in (0,1,2)],
            'exhaustive_rounding_trials':[rounding_search(s) for s in (0,1,2)]}
    dest=Path(__file__).with_name('results.json')
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
