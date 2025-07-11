from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Config:
    # ─── CLI & I/O ───
    seed: int
    run_name: Optional[str]
    device: str
    
    features: Optional[Dict[str, str]]
    input_dims: Optional[Dict[str, int]]
    modalities: Optional[List[str]]

    features_dir: str = "Features"
    data_dir: str = "fmri"

    # ─── DataLoader ───
    batch_size: int = 4
    num_workers: int = 8
    prefetch_factor: int = 2
    persistent_workers: bool = False
    pin_memory: bool = False
    stratification_variable: Optional[str] = None
    val_name: str = "s06"
    val_run: str = "all"
    filter_name: List[str] = None
    train_noise_std: float = 0.0
    use_normalization: bool = False
    normalize_validation_bold: bool = False
    oversample_factor: int = 1

    # ─── Model I/O ───
    output_dim: int = 1000

    # ─── Fusion Transformer ───
    fusion_hidden_dim: int = 256
    fusion_layers: int = 1
    fusion_heads: int = 4
    fusion_dropout: float = 0.3
    subject_dropout_prob: float = 0.0
    use_fusion_transformer: bool = True
    use_run_embeddings: bool = False
    proj_layers: int = 1
    fuse_mode: str = "concat"
    subject_count: int = 4
    use_gmu: bool = False 

    # ─── Prediction Transformer ───
    pred_layers: int = 3
    pred_heads: int = 8
    pred_dropout: float = 0.3
    rope_pct: float = 1.0
    num_pre_tokens: int = 5
    n_prepend_zeros: int = 10

    # ─── Spatial Regularization ───
    normalize_pred_for_spatial_regularizer: bool = True
    spatial_sigma: float = 0.25
    lambda_net_adj: float = 0.0001
    lambda_sp_adj: float = 0.00001

    # ─── HRF ───
    use_hrf_conv: bool = False
    learn_hrf: bool = False
    hrf_size: int = 8
    tr_sec: float = 1.49

    # ─── Training ───
    mask_prob: float = 0.2
    modality_dropout_prob: float = 0.0
    modality_dropout_mode: str = 'zeros'
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_epochs: int = 3
    warmup_start_lr_factor: float = 0.1
    early_stop_patience: int = 5
    lambda_sample: float = 1.0
    lambda_roi: float = 1.0
    lambda_mse: float = 1.0
    lambda_hrf: float = 0.0
    roi_log_interval: int = 1
    pct_bads: float = 0.1
    max_scatter_points: int = 50000
    yeo_networks: int = 7
    
    loss_masks_path: str = None

    
    @staticmethod
    def from_yaml(features_path: str, params_path: str, seed: int, run_name: Optional[str] = None,
                  features_dir: str = "Features", data_dir: str = "fmri", device: str = "cuda") -> 'Config':
        """
        Load configuration from YAML files and return a Config object.
        
        Parameters:
        -----------
        features_path : str
            Path to the features YAML file.
        params_path : str
            Path to the parameters YAML file.
        features_dir : str
            Directory where features are stored.
        data_dir : str
            Directory where data is stored.
        seed : int
            Random seed for reproducibility.
        run_name : Optional[str]
            Name of the run, will be used on wandb.
        device : str
            Device to use for training (default is "cuda").
            
        
        Returns:
        --------
        Config
            A Config object populated with the parameters from the YAML files.
        """
        import yaml, random

        # load features.yaml
        with open(features_path, "r") as f:
            features = yaml.safe_load(f)

        # load params.yaml
        with open(params_path,"r") as f:
            params = yaml.safe_load(f)

        return Config(
            seed=seed,
            run_name=run_name,
            device=device,
            
            features=features["features"],
            input_dims=features["input_dims"],
            modalities=list(features["input_dims"].keys()),

            features_dir=features_dir,
            data_dir=data_dir,
            **params
        )
    
    def save(self, path: str):
        """
        Save the configuration to a YAML file.
        
        Parameters:
        -----------
        path : str
            Path to save the configuration file.
        """
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)

    def load(self, path: str):
        """
        Load the configuration from a YAML file.
        
        Parameters:
        -----------
        path : str
            Path to load the configuration file from.
        """
        import yaml
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
            self.__dict__.update(config_dict)