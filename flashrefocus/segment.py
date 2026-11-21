"""Subject masks for video with SAM 3 (text-prompted video segmentation and tracking).

SAM 3 segments and tracks every instance of a text prompt (for example "person") through a clip; one track is chosen
as the subject and the union of the others is returned too (for depth_edits.spotlight, which must not grow the
subject mask into a neighbour). Long clips are processed in chunks; the subject is carried from one chunk to the next
by the overlap of its last mask with the tracks of the next chunk.

Needs transformers >= 5.0 (Sam3VideoModel) and access to the gated checkpoint facebook/sam3 on Hugging Face. Stage I
pins an older transformers; the spotlight path (restoration off) does not use Stage I, so it can run in an
environment with a recent transformers.

  from flashrefocus.segment import track_subject
  subject, others = track_subject(frames, prompt="person", select="most_central")   # frames: (n, h, w, 3) uint8
"""
import numpy as np
import torch

_SAM3 = {}


def _load(device, dtype):
    try:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
    except ImportError as e:                                   # pragma: no cover
        raise ImportError("SAM 3 needs transformers >= 5.0 (pip install -U transformers) and access to "
                          "facebook/sam3 on Hugging Face.") from e
    key = (str(device), dtype)
    if key not in _SAM3:
        _SAM3.clear()
        _SAM3[key] = (Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=dtype).eval(),
                      Sam3VideoProcessor.from_pretrained("facebook/sam3"))
    return _SAM3[key]


def _tracks(frames, prompt, device, dtype):
    """All tracks of `prompt` in a chunk: list over frames of {track id: bool mask (h, w)}."""
    model, proc = _load(device, dtype)
    sess = proc.init_video_session(video=list(frames), inference_device=device, processing_device="cpu",
                                   video_storage_device="cpu", dtype=dtype)
    sess = proc.add_text_prompt(inference_session=sess, text=prompt)
    out = [dict() for _ in frames]
    with torch.no_grad():
        for o in model.propagate_in_video_iterator(inference_session=sess):
            p = proc.postprocess_outputs(sess, o)
            ids = p["object_ids"]
            ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
            if not ids:
                continue
            masks = p["masks"].cpu().numpy()
            masks = masks > 0.5 if masks.dtype != bool else masks
            out[o.frame_idx] = {int(i): m for i, m in zip(ids, masks)}
    return out


def _choose(tracks, select, point, previous):
    ids = sorted({i for f in tracks for i in f})
    if not ids:
        return None
    if previous is not None:                                   # continue the subject of the previous chunk
        first = next((f for f in tracks if f), {})
        iou = {i: (m & previous).sum() / max(1, (m | previous).sum()) for i, m in first.items()}
        if iou and max(iou.values()) > 0.1:
            return max(iou, key=iou.get)
    if select == "point":
        h, w = next(iter(next(f for f in tracks if f).values())).shape
        x, y = int(point[0] * (w - 1)), int(point[1] * (h - 1))
        for f in tracks:
            hit = [i for i, m in f.items() if m[y, x]]
            if hit:
                return hit[0]
        select = "most_central"
    score = dict.fromkeys(ids, 0.0)
    for f in tracks:
        for i, m in f.items():
            area = float(m.mean())
            if select == "largest":
                score[i] += area
            else:                                              # most_central: area weighted by closeness to the centre
                cols = np.nonzero(m.any(0))[0]
                cx = cols.mean() / m.shape[1] if len(cols) else 0.5
                score[i] += area * (1 - 2 * abs(cx - 0.5))
    return max(score, key=score.get)


def track_subject(frames, prompt="person", select="most_central", point=(0.5, 0.5), chunk=300, device="cuda",
                  dtype=torch.bfloat16):
    """frames: (n, h, w, 3) uint8 array or tensor. select: "most_central", "largest" or "point" (the track under
    `point`, fractions of width and height, on the first frame where a track covers it). Returns two bool tensors
    (n, h, w): the subject mask and the union of the other tracks (all False where the subject is not found)."""
    frames = frames.cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)
    n, h, w = frames.shape[:3]
    subject, others = np.zeros((n, h, w), bool), np.zeros((n, h, w), bool)
    previous = None
    for s in range(0, n, chunk):
        tracks = _tracks(frames[s:s + chunk], prompt, device, dtype)
        sid = _choose(tracks, select, point, previous)
        for k, f in enumerate(tracks):
            for i, m in f.items():
                if i == sid:
                    subject[s + k] = m
                else:
                    others[s + k] |= m
        previous = subject[s + len(tracks) - 1] if sid is not None and subject[s + len(tracks) - 1].any() else None
    return torch.from_numpy(subject), torch.from_numpy(others)
