#!/usr/bin/env python3
"""Sub-experiment S1: Drafter latency on CPU vs GPU (PyTorch + ONNX Runtime).

Measures forward pass time for V5e-0-Cont and V5e-0-Cont2 across:
  - GPU (PyTorch CUDA)
  - CPU (PyTorch)
  - CPU (ONNX Runtime, after export)

Output: Table F.1 data — cross-hardware drafter latency.

Usage:
  python drafter_latency.py --vlm qwen2-2b --ckpt ./checkpoints/v5e0_qwen2-2b.pt
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import V5e0_Cont, V5e0_Cont2, VLM_CONFIGS


HIDDEN_DIMS = {"qwen2-2b": 1536, "qwen2-7b": 3584,
               "llava-1.5-7b": 4096, "llava-1.6-mistral-7b": 4096,
               "llava-onevision": 3584}


def benchmark_pytorch(head, h_t, root_emb, cont_emb, n_warmup=20, n_repeat=200, device="cuda"):
    """Measure forward latency (ms) with PyTorch on given device."""
    head = head.to(device).eval()
    h_t = h_t.to(device); root_emb = root_emb.to(device)
    if cont_emb is not None: cont_emb = cont_emb.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if cont_emb is None:
                _ = head(h_t, root_emb)
            else:
                _ = head(h_t, root_emb, cont_emb)
        if device == "cuda": torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_repeat):
            if device == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            if cont_emb is None:
                _ = head(h_t, root_emb)
            else:
                _ = head(h_t, root_emb, cont_emb)
            if device == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)   # ms
    return float(np.mean(times)), float(np.std(times)), float(np.median(times))


def export_onnx(head, head_name, dim, output_path, has_cont=False):
    """Export V5e-0 head to ONNX."""
    head = head.cpu().eval()
    h_t = torch.randn(1, dim)
    root_emb = torch.randn(1, dim)
    cont_emb = torch.randn(1, dim) if has_cont else None

    if has_cont:
        torch.onnx.export(
            head, (h_t, root_emb, cont_emb), output_path,
            input_names=["h_t", "root_emb", "cont_emb"],
            output_names=["z"],
            dynamic_axes={"h_t": {0: "batch"}, "root_emb": {0: "batch"},
                          "cont_emb": {0: "batch"}, "z": {0: "batch"}},
            opset_version=14,
        )
    else:
        torch.onnx.export(
            head, (h_t, root_emb), output_path,
            input_names=["h_t", "root_emb"],
            output_names=["z"],
            dynamic_axes={"h_t": {0: "batch"}, "root_emb": {0: "batch"},
                          "z": {0: "batch"}},
            opset_version=14,
        )
    return os.path.getsize(output_path) / 1e6   # MB


def benchmark_onnxruntime(onnx_path, dim, has_cont=False, n_warmup=20, n_repeat=200):
    """Measure ONNX Runtime CPU latency."""
    import onnxruntime as ort
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_opt, providers=["CPUExecutionProvider"])

    h_t = np.random.randn(1, dim).astype(np.float32)
    root_emb = np.random.randn(1, dim).astype(np.float32)
    cont_emb = np.random.randn(1, dim).astype(np.float32) if has_cont else None

    feed = {"h_t": h_t, "root_emb": root_emb}
    if has_cont: feed["cont_emb"] = cont_emb

    for _ in range(n_warmup): sess.run(None, feed)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times)), float(np.median(times))


def main(vlm_name, ckpt_path):
    D = HIDDEN_DIMS[vlm_name]
    print(f"[Drafter latency benchmark: {vlm_name} (D={D})]\n", flush=True)

    # Load checkpoint
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    alpha = state.get("alpha", 30.0)
    beta = state.get("beta", 30.0)

    # Build heads
    cont = V5e0_Cont(dim=D, alpha_init=alpha)
    cont.Q_proj.weight.data.copy_(state["W_Q1"].float())
    cont.Q_proj.bias.data.copy_(state["W_Q1_bias"].float())
    cont.eval()

    cont2 = V5e0_Cont2(dim=D, alpha_init=alpha, beta_init=beta)
    cont2.Q_proj.weight.data.copy_(state["W_Q2"].float())
    cont2.Q_proj.bias.data.copy_(state["W_Q2_bias"].float())
    cont2.eval()

    h_t = torch.randn(1, D)
    root_emb = torch.randn(1, D)
    cont_emb = torch.randn(1, D)

    results = {"vlm": vlm_name, "hidden_dim": D,
               "params_M_cont": sum(p.numel() for p in cont.parameters()) / 1e6,
               "params_M_cont2": sum(p.numel() for p in cont2.parameters()) / 1e6,
               "measurements": {}}

    # ---------- GPU PyTorch ----------
    if torch.cuda.is_available():
        print("[GPU PyTorch (CUDA)]")
        m, s, med = benchmark_pytorch(cont, h_t, root_emb, None, device="cuda")
        print(f"  V5e-0-Cont:  {m:.3f} ± {s:.3f} ms  (median {med:.3f})")
        results["measurements"]["gpu_cont"] = {"mean_ms": m, "std": s, "median_ms": med}
        m, s, med = benchmark_pytorch(cont2, h_t, root_emb, cont_emb, device="cuda")
        print(f"  V5e-0-Cont2: {m:.3f} ± {s:.3f} ms  (median {med:.3f})")
        results["measurements"]["gpu_cont2"] = {"mean_ms": m, "std": s, "median_ms": med}

    # ---------- CPU PyTorch ----------
    print("\n[CPU PyTorch]")
    m, s, med = benchmark_pytorch(cont, h_t, root_emb, None, device="cpu")
    print(f"  V5e-0-Cont:  {m:.3f} ± {s:.3f} ms  (median {med:.3f})")
    results["measurements"]["cpu_pytorch_cont"] = {"mean_ms": m, "std": s, "median_ms": med}
    m, s, med = benchmark_pytorch(cont2, h_t, root_emb, cont_emb, device="cpu")
    print(f"  V5e-0-Cont2: {m:.3f} ± {s:.3f} ms  (median {med:.3f})")
    results["measurements"]["cpu_pytorch_cont2"] = {"mean_ms": m, "std": s, "median_ms": med}

    # ---------- ONNX export ----------
    print("\n[ONNX export]")
    onnx_cont_path = f"./checkpoints/v5e0_cont_{vlm_name}.onnx"
    onnx_cont2_path = f"./checkpoints/v5e0_cont2_{vlm_name}.onnx"
    size_cont = export_onnx(cont, "Cont", D, onnx_cont_path, has_cont=False)
    size_cont2 = export_onnx(cont2, "Cont2", D, onnx_cont2_path, has_cont=True)
    print(f"  Cont ONNX:  {size_cont:.2f} MB")
    print(f"  Cont2 ONNX: {size_cont2:.2f} MB")
    results["onnx_size_MB"] = {"cont": size_cont, "cont2": size_cont2}

    # ---------- CPU ONNX Runtime ----------
    print("\n[CPU ONNX Runtime]")
    m, s, med = benchmark_onnxruntime(onnx_cont_path, D, has_cont=False)
    print(f"  V5e-0-Cont:  {m:.3f} ± {s:.3f} ms  (median {med:.3f})")
    results["measurements"]["cpu_onnx_cont"] = {"mean_ms": m, "std": s, "median_ms": med}
    m, s, med = benchmark_onnxruntime(onnx_cont2_path, D, has_cont=True)
    print(f"  V5e-0-Cont2: {m:.3f} ± {s:.3f} ms  (median {med:.3f})")
    results["measurements"]["cpu_onnx_cont2"] = {"mean_ms": m, "std": s, "median_ms": med}

    # ---------- Summary ----------
    print(f"\n{'='*70}")
    print(f"DRAFTER LATENCY SUMMARY — {vlm_name}")
    print(f"{'='*70}")
    print(f"{'Hardware':<20s} {'Cont (ms)':>12s} {'Cont2 (ms)':>12s} {'Total (ms)':>12s}")
    print(f"{'-'*60}")
    pairs = [
        ("GPU (PyTorch)", "gpu_cont", "gpu_cont2"),
        ("CPU (PyTorch)", "cpu_pytorch_cont", "cpu_pytorch_cont2"),
        ("CPU (ONNX RT)", "cpu_onnx_cont", "cpu_onnx_cont2"),
    ]
    for label, k1, k2 in pairs:
        if k1 in results["measurements"]:
            c = results["measurements"][k1]["median_ms"]
            c2 = results["measurements"][k2]["median_ms"]
            print(f"{label:<20s} {c:>12.3f} {c2:>12.3f} {c+c2:>12.3f}")

    save_path = f"./results/F_drafter_latency_{vlm_name}.json"
    with open(save_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\n[saved {save_path}]")
    print(f"[saved {onnx_cont_path}, {onnx_cont2_path}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--vlm", required=True, choices=list(HIDDEN_DIMS.keys()))
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    main(args.vlm, args.ckpt)
