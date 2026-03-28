from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

if TYPE_CHECKING:
    from core.config import Config
    from core.managers import ThumbnailManager

from core.managers.model_loader import get_lpips_metric


def _run_batched_lpips(
    pairs: List[Tuple[int, int]],
    all_frames_data: List[Dict[str, Any]],
    dedup_mask: np.ndarray,
    reasons: defaultdict,
    thumbnail_manager: "ThumbnailManager",
    output_dir: str,
    threshold: float,
    device: str = "cpu",
):
    if not pairs:
        return
    loss_fn = get_lpips_metric(device=device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 32
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        img1_batch, img2_batch, valid_indices = [], [], []
        for p_idx, c_idx in batch:
            p_path = Path(output_dir) / "thumbs" / all_frames_data[p_idx]["filename"]
            c_path = Path(output_dir) / "thumbs" / all_frames_data[c_idx]["filename"]
            img1, img2 = thumbnail_manager.get(p_path), thumbnail_manager.get(c_path)
            if img1 is not None and img2 is not None:
                img1_batch.append(transform(img1))
                img2_batch.append(transform(img2))
                valid_indices.append((p_idx, c_idx))
        if not valid_indices:
            continue
        img1_t, img2_t = torch.stack(img1_batch).to(device), torch.stack(img2_batch).to(device)
        with torch.no_grad():
            distances_t: torch.Tensor = loss_fn.forward(img1_t, img2_t).squeeze()  # type: ignore
            if distances_t.ndim == 0:
                distances_t = distances_t.unsqueeze(0)
            distances = distances_t.cpu().numpy()
        for j, (p_idx, c_idx) in enumerate(valid_indices):
            if float(distances[j]) <= threshold:
                p_score = all_frames_data[p_idx].get("metrics", {}).get("quality_score", 0)
                c_score = all_frames_data[c_idx].get("metrics", {}).get("quality_score", 0)
                if c_score > p_score:
                    if dedup_mask[p_idx]:
                        reasons[all_frames_data[p_idx]["filename"]].append("duplicate")
                    dedup_mask[p_idx] = False
                else:
                    if dedup_mask[c_idx]:
                        reasons[all_frames_data[c_idx]["filename"]].append("duplicate")
                    dedup_mask[c_idx] = False


def apply_deduplication_filter(
    all_frames_data: List[Dict[str, Any]],
    filters: Dict[str, Any],
    thumbnail_manager: Optional["ThumbnailManager"],
    config: "Config",
    output_dir: Optional[str],
) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    import imagehash

    num_frames = len(all_frames_data)
    filenames = [f["filename"] for f in all_frames_data]
    dedup_mask = np.ones(num_frames, dtype=bool)
    reasons = defaultdict(list)
    dedup_method = filters.get("dedup_method", "pHash")
    if filters.get("enable_dedup"):
        if dedup_method == "pHash" and imagehash and filters.get("dedup_thresh", -1) != -1:
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            hashes = {
                i: imagehash.hex_to_hash(all_frames_data[i]["phash"])
                for i in range(num_frames)
                if "phash" in all_frames_data[i]
            }
            hash_size = next(iter(hashes.values())).hash.size if hashes else 64
            kept_hash_matrix, kept_indices, kept_count = (
                np.zeros((num_frames, hash_size), dtype=bool),
                np.zeros(num_frames, dtype=int),
                0,
            )
            thresh = filters.get("dedup_thresh", 5)
            for i in sorted_indices:
                if i not in hashes:
                    continue
                curr_hash = hashes[i].hash.flatten()
                is_duplicate = False
                if kept_count > 0:
                    diffs = np.bitwise_xor(kept_hash_matrix[:kept_count], curr_hash).sum(axis=1)
                    matches = np.where(diffs <= thresh)[0]
                    if len(matches) > 0:
                        is_duplicate, match_pos = True, matches[0]
                        kept_idx = kept_indices[match_pos]
                        if all_frames_data[i].get("metrics", {}).get("quality_score", 0) > all_frames_data[
                            kept_idx
                        ].get("metrics", {}).get("quality_score", 0):
                            if dedup_mask[kept_idx]:
                                reasons[filenames[kept_idx]].append("duplicate")
                            dedup_mask[kept_idx], kept_hash_matrix[match_pos], kept_indices[match_pos] = (
                                False,
                                curr_hash,
                                i,
                            )
                        else:
                            if dedup_mask[i]:
                                reasons[filenames[i]].append("duplicate")
                            dedup_mask[i] = False
                if not is_duplicate:
                    kept_hash_matrix[kept_count], kept_indices[kept_count], kept_count = curr_hash, i, kept_count + 1
        elif dedup_method == "SSIM" and thumbnail_manager:
            threshold = filters.get("ssim_threshold", 0.95)

            def compare_fn(i1, i2):
                return ssim(cv2.cvtColor(i1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(i2, cv2.COLOR_RGB2GRAY)) >= threshold

            _generic_dedup(all_frames_data, dedup_mask, reasons, thumbnail_manager, output_dir, compare_fn)
        elif dedup_method == "LPIPS" and thumbnail_manager:
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            pairs = [(sorted_indices[i - 1], sorted_indices[i]) for i in range(1, len(sorted_indices))]
            _run_batched_lpips(
                pairs,
                all_frames_data,
                dedup_mask,
                reasons,
                thumbnail_manager,
                output_dir,
                filters.get("lpips_threshold", 0.1),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        elif dedup_method == "pHash then LPIPS" and thumbnail_manager and imagehash:
            sorted_indices = sorted(range(num_frames), key=lambda i: filenames[i])
            hashes = {
                i: imagehash.hex_to_hash(all_frames_data[i]["phash"])
                for i in range(num_frames)
                if "phash" in all_frames_data[i]
            }
            hash_size = next(iter(hashes.values())).hash.size if hashes else 64
            kept_hash_matrix, kept_indices, kept_count, p_hash_duplicates = (
                np.zeros((num_frames, hash_size), dtype=bool),
                np.zeros(num_frames, dtype=int),
                0,
                [],
            )
            thresh = filters.get("dedup_thresh", 5)
            for i in sorted_indices:
                if i not in hashes:
                    continue
                curr_hash = hashes[i].hash.flatten()
                is_duplicate = False
                if kept_count > 0:
                    diffs = np.bitwise_xor(kept_hash_matrix[:kept_count], curr_hash).sum(axis=1)
                    matches = np.where(diffs <= thresh)[0]
                    if len(matches) > 0:
                        is_duplicate, match_pos = True, matches[0]
                        kept_idx = kept_indices[match_pos]
                        p_hash_duplicates.append((kept_idx, i))
                if not is_duplicate:
                    kept_hash_matrix[kept_count], kept_indices[kept_count], kept_count = curr_hash, i, kept_count + 1
            if p_hash_duplicates:
                _run_batched_lpips(
                    p_hash_duplicates,
                    all_frames_data,
                    dedup_mask,
                    reasons,
                    thumbnail_manager,
                    output_dir,
                    filters.get("lpips_threshold", 0.1),
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
    return dedup_mask, reasons


def _generic_dedup(
    all_frames_data: List[Dict[str, Any]],
    dedup_mask: np.ndarray,
    reasons: defaultdict,
    thumbnail_manager: "ThumbnailManager",
    output_dir: Optional[str],
    compare_fn: Callable[[np.ndarray, np.ndarray], bool],
):
    num_frames = len(all_frames_data)
    sorted_indices = sorted(range(num_frames), key=lambda i: all_frames_data[i]["filename"])
    if not output_dir:
        return
    for i in range(1, len(sorted_indices)):
        c_idx, p_idx = sorted_indices[i], sorted_indices[i - 1]
        c_path = Path(output_dir) / "thumbs" / all_frames_data[c_idx]["filename"]
        p_path = Path(output_dir) / "thumbs" / all_frames_data[p_idx]["filename"]
        img1, img2 = thumbnail_manager.get(p_path), thumbnail_manager.get(c_path)
        if img1 is not None and img2 is not None:
            if compare_fn(img1, img2):
                if all_frames_data[c_idx].get("metrics", {}).get("quality_score", 0) > all_frames_data[p_idx].get(
                    "metrics", {}
                ).get("quality_score", 0):
                    if dedup_mask[p_idx]:
                        reasons[all_frames_data[p_idx]["filename"]].append("duplicate")
                    dedup_mask[p_idx] = False
                else:
                    if dedup_mask[c_idx]:
                        reasons[all_frames_data[c_idx]["filename"]].append("duplicate")
                    dedup_mask[c_idx] = False
