import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
from sklearn.linear_model import RidgeCV
import random
from captum.attr import ShapleyValueSampling

from algonauts.utils.utils import evaluate_corr
from algonauts.utils import logger
from algonauts.training.losses import masked_negative_pearson_loss


def shapley_captum(model, val_loader, device, modalities,
                   n_samples: int = 2048, keep_frac: float = 1.0):
    """
    Return one scalar Shapley value ϕ per modality in ``modalities``.

    Parameters
    ----------
    model : torch.nn.Module
        Must accept positional args
            (features_dict, subject_ids, run_ids, attention_mask, *extra)
        where ``features_dict`` is {modality → Tensor[B,T,D]}.
    val_loader : torch.utils.data.DataLoader
        Yields the dict produced by ``collate_fn`` in data.py.
    device : torch.device or str
        Place tensors & model on this device.
    modalities : list[str]
        Keys inside each batch that you want to attribute.
        Order matters – it is preserved in the output dict.
    exact_cutoff : int
        If ``len(modalities)`` ≤ this cutoff use exact enumeration
        (`captum.attr.ShapleyValues`); otherwise use Monte‑Carlo
        sampling (`ShapleyValueSampling`).
    n_samples : int
        Number of permutations for the Monte‑Carlo estimator.
    keep_frac : float
        Fraction of validation batches to **randomly** keep (between 0 and 1].
        Use e.g. 0.1 to sample roughly 10 % of batches for attribution.

    Returns
    -------
    dict
        ``{ modality : float(phi) }`` – mean attribution over the
        chosen validation batch.
    """
    model.eval().requires_grad_(False)

    # ---- Wrap model.forward so Captum gets tensors/tuples ----------
    def forward_packed(*args):
        """
        Captum calls this with
            (*packed_inputs, subject_ids, run_ids, attention_mask, fmri_true)

        We:
          1. Move modality tensors and extras to `device`.
          2. Run the model to get voxel predictions  (B, V).
          3. Compute Pearson r with the ground‑truth fMRI for each sample.
          4. Return the vector of r's  – one scalar per batch element.
        """
        n_mod = len(modalities)
        feats_cpu   = args[:n_mod]           # tuple of modality tensors
        fmri_true   = args[n_mod]
        attn_mask   = args[n_mod + 1]
        subj_ids    = args[n_mod + 2]
        run_ids     = args[n_mod + 3]

        feats_gpu = {m: feats_cpu[i].to(device, non_blocking=True)
                     for i, m in enumerate(modalities)}
        subj_ids  = subj_ids
        run_ids   = run_ids
        attn_mask = attn_mask.to(device)
        fmri_true = fmri_true.to(device)     # (B, V)

        preds = model(feats_gpu, subj_ids, run_ids, attn_mask)  # (B, V)

        # -------- per‑sample Pearson correlation ---------------------
        r_batch = -masked_negative_pearson_loss(
            preds, fmri_true, attn_mask,
            zero_center=True, network_mask=None
        )
        return r_batch

    attr_engine = ShapleyValueSampling(forward_packed)

    # -------- iterate over the entire validation loader --------------
    phi_sum = {m: 0.0 for m in modalities}
    n_batches = 0
    n_samples = round(n_samples / len(modalities))  # samples per modality

    for batch in val_loader:
        # probabilistic subsampling of batches
        if keep_frac < 1.0 and random.random() > keep_frac:
            continue
        packed_inputs = tuple(batch[m].to(device, non_blocking=True) for m in modalities)
        baselines     = tuple(torch.zeros_like(t) for t in packed_inputs)

        # gather any extra batch‑specific tensors the model expects
        extras = [
            batch["fmri"].to(device, non_blocking=True),  # ground‑truth fMRI
            batch["attention_masks"].to(device, non_blocking=True),  # attention mask
            batch["subject_ids"],  # subject IDs
            batch["run_ids"]  # run IDs
        ]
        additional_forward_args = tuple(extras)

        feature_mask = tuple(
            torch.tensor(i, dtype=torch.long, device=device)
            for i in range(len(modalities))
        )

        attributions = attr_engine.attribute(
            inputs=packed_inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            show_progress=True,
            perturbations_per_eval=1,
        )

        # accumulate mean attribution per modality
        for m, a in zip(modalities, attributions):
            phi_sum[m] += a.mean().item()
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("keep_frac too small ‒ no batches sampled.")

    # average across batches
    phi = {m: v / n_batches for m, v in phi_sum.items()}

    # ---- W&B bar chart of per‑modality importance -------------------
    bar_table = wandb.Table(data=[[k, v] for k, v in phi.items()],
                            columns=["modality", "phi"])
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(list(phi.keys()), list(phi.values()))
    ax.set_ylabel("ϕ (Δ masked Pearson r)")
    ax.set_xticklabels(phi.keys(), rotation=45, ha="right")
    ax.set_title("Captum Shapley – contribution to masked r")
    plt.tight_layout()
    wandb.log({"shapley/bar_chart": wandb.Image(fig),
               "shapley/table":     bar_table})
    plt.close(fig)

    return phi


def feature_single_ablation(model, val_loader, device, base_r):
    """Leave‑one‑block Δr; returns delta_dict."""
    delta_dict = {}
    for name, proj in model.encoder.projections.items():
        W0, b0 = proj[0].weight.data.clone(), proj[0].bias.data.clone()
        proj[0].weight.zero_()
        proj[0].bias.zero_()
        with torch.no_grad():
            r = evaluate_corr(model, val_loader, device=device).mean().item()
        delta_dict[name] = r - base_r
        proj[0].weight.copy_(W0)
        proj[0].bias.copy_(b0)

    # bar‑plot
    abl_table = wandb.Table(data=[[k, v] for k, v in delta_dict.items()],
                            columns=["block", "delta_r"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(delta_dict.keys()), list(delta_dict.values()))
    ax.set_ylabel("Δ r")
    ax.set_xticklabels(delta_dict.keys(), rotation=45, ha="right")
    ax.set_title("Leave‑one‑block Δr (validation)")
    plt.tight_layout()
    wandb.log({"ablate/bar_chart": wandb.Image(fig), "ablate/table": abl_table})
    plt.close(fig)
    return delta_dict


def feature_pairwise_redundancy(model, val_loader, device, base_r, delta_dict, blocks):
    """Δr_AB – additive; logs heat‑map. Only if len(blocks) < cutoff."""
    n=len(blocks)
    red = torch.zeros(n,n)
    for i in range(n):
        for j in range(i+1,n):
            bi,bj=blocks[i],blocks[j]
            Wi,bi_b = model.encoder.projections[bi][0].weight.data.clone(), model.encoder.projections[bi][0].bias.data.clone()
            Wj,bj_b = model.encoder.projections[bj][0].weight.data.clone(), model.encoder.projections[bj][0].bias.data.clone()
            model.encoder.projections[bi][0].weight.zero_()
            model.encoder.projections[bi][0].bias.zero_()
            model.encoder.projections[bj][0].weight.zero_()
            model.encoder.projections[bj][0].bias.zero_()
            r_joint = evaluate_corr(model,val_loader,device=device).mean().item()-base_r
            model.encoder.projections[bi][0].weight.copy_(Wi)
            model.encoder.projections[bi][0].bias.copy_(bi_b)
            model.encoder.projections[bj][0].weight.copy_(Wj)
            model.encoder.projections[bj][0].bias.copy_(bj_b)
            red[i,j]=red[j,i]=r_joint-(delta_dict[bi]+delta_dict[bj])
    fig,ax=plt.subplots(figsize=(6,5))
    sns.heatmap(red.numpy(),ax=ax,xticklabels=blocks,yticklabels=blocks,square=True,cmap="rocket")
    ax.set_title("Pairwise redundancy Δr")
    wandb.log({"ablate/redundancy_heatmap":wandb.Image(fig)})
    plt.close(fig)


def feature_partial_r2(model, val_loader, device, blocks):
    """Variance partition; logs bar chart."""
    X_parts={b:[] for b in blocks}
    Y_parts=[]
    with torch.no_grad():
        for batch in val_loader:
            feats={k:batch[k].to(device) for k in val_loader.dataset.modalities}
            fused=model.encoder(feats,batch["subject_ids"],batch["run_ids"])
            attn=batch["attention_masks"].bool().to(device)
            fused=fused[attn]
            Y_parts.append(batch["fmri"][attn.cpu()])
            token_cnt=len(blocks)+1+int(model.encoder.use_run_embeddings)
            H_tok=fused.size(-1)//token_cnt
            for i,b in enumerate(blocks):
                col=slice(i*H_tok,(i+1)*H_tok)
                X_parts[b].append(fused[:,col].cpu())
    X_block={b:torch.cat(v).numpy() for b,v in X_parts.items()}
    Y=torch.cat(Y_parts).numpy()
    XK=np.hstack([X_block[b] for b in blocks])
    partial={}
    alphas=np.logspace(2,4,5)
    for b in blocks:
        red=np.hstack([X_block[bb] for bb in blocks if bb!=b])
        r_full=RidgeCV(alphas,cv=3,scoring="r2").fit(XK,Y).score(XK,Y)
        r_red =RidgeCV(alphas,cv=3,scoring="r2").fit(red,Y).score(red,Y)
        partial[b]=r_full-r_red
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(list(partial.keys()),list(partial.values()))
    ax.set_xticklabels(partial.keys(),rotation=45,ha="right")
    ax.set_ylabel("Partial R²")
    ax.set_title("Unique variance per block")
    plt.tight_layout()
    wandb.log({"partial_r2/bar_chart":wandb.Image(fig)})
    plt.close(fig)
    return partial


# ------------------------------------------------------------
def run_feature_analyses(model, val_loader, device):
    """
    Runs three lightweight post‑hoc analyses on the *validation* set:
        1. Leave‑one‑block ablation  (Δr)
        2. Permutation importance    (Δr)
        3. Linear SHAP values        (mean |ϕ|)
    Logs results to W&B under keys:
        ablate/<block>, permute/<block>, shap/hidden_<i>
    """
    model.eval().requires_grad_(False)

    blocks = list(model.encoder.projections.keys())
    EXACT_CUTOFF = 6

    # ---------- baseline ----------------------------------------------
    base_r = evaluate_corr(model, val_loader, device=device).mean().item()
    wandb.log({"diag/baseline_r": base_r})

    # ---------- analyses ----------------------------------------------
    with logger.step("🔹 Leave‑one‑block ablation"):
        delta_dict = feature_single_ablation(model, val_loader, device, base_r)

    with logger.step(f"🔹 Running Shapley analysis on {len(blocks)} blocks."):
        shapley_captum(model, val_loader, device, blocks, n_samples=512, keep_frac=0.5)

    if len(blocks) <= EXACT_CUTOFF:
        with logger.step("🔹 Pairwise redundancy"):
            feature_pairwise_redundancy(model, val_loader, device, base_r, delta_dict, blocks)

    # with logger.step("🔹 Partial R² variance partition"):
    #     feature_partial_r2(model, val_loader, device, blocks)