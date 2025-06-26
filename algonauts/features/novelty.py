from pathlib import Path
import argparse, torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAUS  = [10, 20, 40, 60, 120, 180, 300]  # s
LAM   = 5.0                              # λ for log branch

def novelty_raw(F: torch.Tensor, k: int) -> torch.Tensor:
    cs  = torch.cat([torch.zeros(1, F.size(1), device=F.device), torch.cumsum(F,0)], 0)
    m   = (cs[k:] - cs[:-k]) / k
    m   = torch.nn.functional.normalize(m, dim=1)
    cur = F[k-1:]
    nov = 1 - (cur * m).sum(1).clamp(-1,1)
    return torch.cat([torch.zeros(k-1, device=F.device), nov])   # (T,)

def process(path: Path, tr: float, in_root: Path, out_root: Path):
    F = torch.nn.functional.normalize(torch.load(path, map_location="cpu").float(), dim=1).to(DEVICE)
    raws = []
    for τ in TAUS:
        k = max(1, round(τ / tr))
        raws.append(novelty_raw(F, k))
    raw_mat = torch.stack(raws, dim=1)          # (T, 7)
    log_mat = torch.log1p(LAM * raw_mat)

    rel = path.relative_to(in_root).with_suffix(".pt")
    for name, mat in (("raw", raw_mat.cpu()), ("log", log_mat.cpu())):
        tgt = out_root / name / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mat, tgt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed_folder", required=True)
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--tr", type=float, default=1.49)
    args = ap.parse_args()

    in_root  = Path(args.embed_folder)
    out_root = Path(args.output_folder)

    print(f"Using device: {DEVICE}")

    for f in in_root.rglob("*.pt"):
        print("•", f.relative_to(in_root), flush=True)
        process(f, args.tr, in_root, out_root)

if __name__ == "__main__":
    main()