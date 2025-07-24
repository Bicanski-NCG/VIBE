from vibe.dataset.data import FMRI_Dataset
from scipy.stats import pearsonr
import os
from vibe.utils import ensure_paths_exist
import argparse
from pathlib import Path
import numpy as np
import torch
from vibe.models import load_model_from_ckpt
from vibe.models.ensemble import EnsembleAverager, ROIAdaptiveEnsemble
from tqdm import  tqdm



def generate_loss_masks(model,
                        config,
                        mask_filter='life',
                        mask_generation_function = None):


    if mask_generation_function is None:
        def mask_generation_function(preds, true):
            return generate_correlation_mask(preds,true)


    if isinstance(mask_filter,str):
        mask_filter = [mask_filter]

    device = config.device
    config.features = {modality: os.path.join(config.features_dir,path) for modality, path in config.features.items()}

    dataset = FMRI_Dataset(
    config.data_dir,
    feature_paths=config.features,
    input_dims=config.input_dims,
    modalities=config.modalities,
    noise_std=config.train_noise_std,
    normalization_stats=None,
    oversample_factor=config.oversample_factor,
    modality_dropout_mode = 'zeros',
    modality_dropout_prob = config.modality_dropout_prob,
    normalize_features = config.use_normalization)

    def filter_fn(sample):
        return sample["name"] in mask_filter
    
    dataset = dataset.filter_samples(filter_fn)


    loss_masks = {}

    for subject_id, run_id, features, fmri_response_tensor,loss_mask, dataset_name in tqdm(dataset):

        features = {modality: tens.unsqueeze(0).to(device) for modality,tens in features.items() }
        attention_mask = torch.ones((1, fmri_response_tensor.shape[-2]), dtype=torch.bool).to(device)
        subj_ids = torch.tensor([subject_id]).to(device)

        with torch.no_grad():
            preds = model(features, subj_ids, [0], attention_mask)

        mask = mask_generation_function(preds[0],fmri_response_tensor)
        
        loss_masks[(subject_id,dataset_name)] = mask



    return loss_masks



def generate_correlation_mask(pred,true,
                              smooth = False,
                              smooth_window  = 100,
                              mode = 'absolute',
                              cutoff = 0.15,
                              use_top_k_voxels = False):
    
    pred = pred.detach().cpu()
    true = true.detach().cpu()

    out_dim = int(true.shape[-1])
    if use_top_k_voxels:
        
        k = 15

        corrs = np.array([pearsonr(true[:,v], pred[:,v])[0] for v in range(true.shape[1])])

        sorted = np.argsort(corrs)

        pred = pred[:,sorted[-k:]]
        true = true[:,sorted[-k:]]


    r_t = np.array([pearsonr(true[t], pred[t])[0] for t in range(true.shape[0])])

    if smooth: 

        r_t = np.convolve(
        r_t, np.ones(smooth_window) / smooth_window, mode="same"
    )

    if mode == 'absolute':
        cutoff= cutoff
    elif mode =='relative':

        cutoff = np.quantile(r_t,cutoff)

    r_t[r_t<cutoff]=0.0

    r_t[r_t>= cutoff]=1.0

    mask = torch.tensor(r_t).float()

    mask = mask.unsqueeze(1).expand(-1,out_dim)

    return mask



def main():

    args = argparse.ArgumentParser(description="Make submission for fMRI predictions")
    group = args.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Checkpoint to load")
    args.add_argument("--roi_ensemble", action="store_true",
                      help="Use ROIAdaptiveEnsemble to select best models per ROI")
    group.add_argument("--ensemble", type=str, nargs="+",
                       help="List of checkpoints to load for ensemble averaging")
    args.add_argument("--mask_filter", type=str, default='life', help="For which movie to generate masks") #TODO: filter for more than one movie
    args.add_argument("--output_dir", default=None, type=str,
                      help="Root directory for outputs & checkpoints "
                           "(default $OUTPUT_DIR or runs)")

    # Arguments to modify how loss function is generated
    args.add_argument("--use_mask",type = str,default='pearsonr')
    args.add_argument("--smooth",type = bool,default=False)
    args.add_argument("--smooth_window",type = int,default=100)
    args.add_argument("--cutoff_mode",type = str,default='absolute', help = 'Determines whether cutoff is an absolute value or a relative value')
    args.add_argument("--cutoff",type = float, default = 0.15)
    args.add_argument("--use_top_k", action='store_true', default=False,
                   help="Set to False to not use top-k masking")


    args = args.parse_args()


    if args.use_mask == 'pearsonr':
        def mask_generation_function(pred, true):
            return generate_correlation_mask(pred, true, smooth=args.smooth, smooth_window=args.smooth_window,
                                             mode=args.cutoff_mode, cutoff=args.cutoff, use_top_k_voxels=args.use_top_k)
    else:
        raise NotImplementedError

    if not args.checkpoint and not args.ensemble:
        raise ValueError("Please provide a checkpoint to load or an ensemble list.")
    
    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}", flush=True)

    else:
        print(f"Using ensemble checkpoints: {args.ensemble}", flush=True)




    output_root =  Path(os.getenv("OUTPUT_DIR", "runs"))


    submission_dir = os.path.join(output_root, "submissions")
    checkpoint_dir = os.path.join(output_root, "checkpoints", args.checkpoint) if args.checkpoint else None
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt") if checkpoint_dir else None
    config_path = os.path.join(checkpoint_dir, "config.yaml") if checkpoint_dir else None

    ensure_paths_exist(
        (output_root, "output_dir"),
        *(([(checkpoint_dir, "checkpoint_dir")] if checkpoint_dir else [])),
        *(([(final_model_path, "final_model.pt")] if final_model_path else [])),
        *(([(config_path, "config.yaml")] if config_path else [])),
    )

    try:
        ensure_paths_exist(
            (submission_dir, "submission_dir"),
        )
    except FileNotFoundError:
        os.makedirs(submission_dir, exist_ok=True)

    # Build model according to --ensemble or single checkpoint, with optional ROI wrap
    device = "cuda"
    if args.ensemble:
        load_device = "cpu" if len(args.ensemble) > 25 else device
        # Ensemble averaging over provided run IDs
        checkpoint_names = args.ensemble
        # Load config from the first checkpoint
        first_ckpt_dir = output_root / "checkpoints" / checkpoint_names[0]
        _, config = load_model_from_ckpt(
            model_ckpt_path=str(first_ckpt_dir / "final_model.pt"),
            params_path=str(first_ckpt_dir / "config.yaml"),
        )
        # Load each model and collect
        models = []
        for chk in checkpoint_names:
            print(f"Loading model from checkpoint: {chk}", flush=True)
            checkpoint_dir = output_root / "checkpoints" / chk
            if args.roi_ensemble:
                m = ROIAdaptiveEnsemble(
                    roi_labels=torch.load(checkpoint_dir / "roi_names.pt", weights_only=False),
                    roi_to_epoch=torch.load(checkpoint_dir / "roi_to_epoch.pt", weights_only=False),
                    ckpt_dir=checkpoint_dir,
                    device=load_device,
                )
            else:
                m, _ = load_model_from_ckpt(
                    model_ckpt_path=str(checkpoint_dir / "final_model.pt"),
                    params_path=str(checkpoint_dir / "config.yaml"),
                )
            #m.to(load_device).eval()
            m.eval()
            models.append(m)
        model = EnsembleAverager(models=models, device=device, normalize=True)
    else:
        # Single checkpoint path
        print(f"Loading model from checkpoint: {args.checkpoint}", flush=True)
        checkpoint = args.checkpoint
        checkpoint_dir = output_root / "checkpoints" / checkpoint
        model, config = load_model_from_ckpt(
            model_ckpt_path=str(checkpoint_dir / "final_model.pt"),
            params_path=str(checkpoint_dir / "config.yaml"),
        )
        model.to(device)
        if args.roi_ensemble:
            # Wrap in ROIAdaptiveEnsemble for per-ROI best iters
            roi_labels = torch.load(checkpoint_dir / "roi_names.pt", weights_only=False)
            roi_to_epoch = torch.load(checkpoint_dir / "roi_to_epoch.pt", weights_only=False)
            model = ROIAdaptiveEnsemble(
                roi_labels=roi_labels,
                roi_to_epoch=roi_to_epoch,
                ckpt_dir=checkpoint_dir,
                device=device,
            )
    model.eval()

    masks = generate_loss_masks(model,
                                config,
                                mask_filter = args.mask_filter,
                                mask_generation_function= mask_generation_function)
    if args.output_dir is None:
        output_dir = checkpoint_dir
    else:
        output_dir = args.output_dir
    
    save_path = os.path.join(output_dir,'loss_masks.pt')
    torch.save(masks,save_path)

if __name__ == "__main__":
    main()

    