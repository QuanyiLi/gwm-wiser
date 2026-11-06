"""
Differentiable robot-only renderer for the Panda used in WISER, in plain PyTorch.

- ``PandaMeshModel``: forward kinematics of ``panda_v3.urdf`` (base at (-0.615, 0, 0)) in
  torch, and the URDF's visual meshes (glb / stl via trimesh, vertex colours baked from
  their textures, per-link vertex-clustering decimation).
- ``soft_rasterize``: SoftRas-style soft rasterizer. Coverage of a pixel by a face is
  sigmoid(signed distance to the face boundary / sigma); a pixel's alpha is
  1 - prod(1 - coverage), its colour a depth-softmax over the covering faces, and the
  image is that colour composited "over" black with premultiplied alpha (no hard masks).
  Face/pixel candidate pairs are enumerated from face bounding boxes without gradient;
  everything that depends on vertex positions is differentiable.
- ``SoftRobotRenderer``: joint states -> images with the camera of ``RobotRenderer``
  (pose and intrinsics read from its SAPIEN camera), Lambert shading under the same
  ambient + two directional lights.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint

PANDA_BASE = (-0.615, 0.0, 0.0)
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(el):
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy_matrix([float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def _axis_angle(axis, q):
    """axis (3,), q (B,) -> (B, 3, 3) rotation matrices (Rodrigues)."""
    k = axis / axis.norm()
    K = torch.zeros(3, 3, dtype=q.dtype, device=q.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
    eye = torch.eye(3, dtype=q.dtype, device=q.device)
    return eye + s * K + (1 - c) * (K @ K)


def _decimate(verts, faces, colors, voxel):
    """Vertex clustering on a voxel grid: merges vertices within a voxel, drops degenerate faces."""
    if voxel <= 0:
        return verts, faces, colors
    key = np.floor(verts / voxel).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    nv = counts.shape[0]
    new_v = np.zeros((nv, 3))
    new_c = np.zeros((nv, colors.shape[1]))
    np.add.at(new_v, inv, verts)
    np.add.at(new_c, inv, colors)
    new_v /= counts[:, None]
    new_c /= counts[:, None]
    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]
    return new_v, f[np.sort(first)], new_c  # duplicates dropped, winding kept


class PandaMeshModel(torch.nn.Module):
    def __init__(self, urdf_path=None, voxel=0.0, device="cuda", dtype=torch.float32):
        super().__init__()
        if urdf_path is None:
            from mani_skill import PACKAGE_ASSET_DIR

            urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"
        import trimesh

        root = Path(urdf_path).parent
        urdf = ET.parse(urdf_path).getroot()
        self.joints = {}  # child link -> (parent, type, origin, axis, name)
        for j in urdf.findall("joint"):
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = (
                j.find("parent").get("link"), j.get("type"),
                torch.tensor(_origin(j.find("origin")), dtype=dtype),
                torch.tensor([float(v) for v in axis.get("xyz").split()], dtype=dtype) if axis is not None else None,
                j.get("name"),
            )
        self.links = [l.get("name") for l in urdf.findall("link")]
        verts, faces, cols, link_id = [], [], [], []
        n_before = 0
        offset = 0
        for li, link in enumerate(urdf.findall("link")):
            for vis in link.findall("visual"):
                mesh_el = vis.find("geometry/mesh")
                if mesh_el is None:
                    continue
                loaded = trimesh.load(root / mesh_el.get("filename"))
                geoms = loaded.dump() if hasattr(loaded, "dump") else [loaded]
                T = _origin(vis.find("origin"))
                v_l, f_l, c_l, base = [], [], [], 0
                for g in geoms:
                    vis = g.visual.to_color() if hasattr(g.visual, "to_color") else g.visual
                    c = np.asarray(vis.vertex_colors, dtype=np.float64)
                    if c.ndim == 1:  # a single colour for the whole mesh
                        c = np.tile(c[None], (len(g.vertices), 1))
                    c = c[:, :3] / 255.0
                    v_l.append(np.asarray(g.vertices)); f_l.append(np.asarray(g.faces) + base); c_l.append(c)
                    base += len(g.vertices)
                v = np.concatenate(v_l) @ T[:3, :3].T + T[:3, 3]
                f, c = np.concatenate(f_l), np.concatenate(c_l)
                n_before += len(f)
                v, f, c = _decimate(v, f, c, voxel)
                verts.append(v); faces.append(f + offset); cols.append(c)
                link_id.append(np.full(len(v), li))
                offset += len(v)
        self.n_faces_full = n_before
        self.register_buffer("verts", torch.tensor(np.concatenate(verts), dtype=dtype))
        self.register_buffer("faces", torch.tensor(np.concatenate(faces), dtype=torch.long))
        self.register_buffer("colors", torch.tensor(np.concatenate(cols), dtype=dtype))
        self.register_buffer("link_id", torch.tensor(np.concatenate(link_id), dtype=torch.long))
        self.register_buffer("base", torch.eye(4, dtype=dtype))
        self.base[:3, 3] = torch.tensor(PANDA_BASE, dtype=dtype)
        # area-weighted vertex normals in the link frame
        v, f = self.verts, self.faces
        fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=-1)
        vn = torch.zeros_like(v).index_add_(0, f.reshape(-1), fn.repeat_interleave(3, 0))
        self.register_buffer("normals", torch.nn.functional.normalize(vn, dim=-1))
        self.to(device)
        for k, (p, t, o, a, n) in list(self.joints.items()):
            self.joints[k] = (p, t, o.to(device), None if a is None else a.to(device), n)

    def link_poses(self, qpos):
        """qpos (B, 9): 7 arm joints + 2 finger joints -> {link: (B, 4, 4)} world poses."""
        B = qpos.shape[0]
        qmap = {n: qpos[:, i] for i, n in enumerate(ARM_JOINTS + FINGER_JOINTS)}
        poses = {}

        def pose(link):
            if link in poses:
                return poses[link]
            if link not in self.joints:  # root
                poses[link] = self.base.to(qpos.dtype).expand(B, 4, 4)
                return poses[link]
            parent, jtype, origin, axis, name = self.joints[link]
            T = pose(parent) @ origin.to(qpos.dtype)
            if jtype == "revolute" or jtype == "continuous":
                M = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(B, 1, 1)
                M[:, :3, :3] = _axis_angle(axis.to(qpos.dtype), qmap[name])
                T = T @ M
            elif jtype == "prismatic":
                M = torch.eye(4, dtype=qpos.dtype, device=qpos.device).repeat(B, 1, 1)
                M[:, :3, 3] = axis.to(qpos.dtype)[None] * qmap[name][:, None]
                T = T @ M
            poses[link] = T
            return T

        for l in self.links:
            pose(l)
        return poses

    def posed(self, qpos):
        """qpos (B, 9) -> world vertices (B, V, 3), world normals (B, V, 3)."""
        poses = self.link_poses(qpos)
        T = torch.stack([poses[l] for l in self.links], dim=1)[:, self.link_id]  # (B, V, 4, 4)
        v = (T[..., :3, :3] @ self.verts.to(qpos.dtype)[None, :, :, None])[..., 0] + T[..., :3, 3]
        n = (T[..., :3, :3] @ self.normals.to(qpos.dtype)[None, :, :, None])[..., 0]
        return v, n

    def tcp(self, qpos):
        return self.link_poses(qpos)["panda_hand_tcp"][:, :3, 3]


def _segment_softmax_weights(score, key, n_keys):
    mx = torch.full((n_keys,), -float("inf"), device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, key, score.detach(), reduce="amax", include_self=True)
    return torch.exp(score - mx[key])


def soft_rasterize(xy, inv_z, attrs, faces, H, W, sigma=0.5, gamma=1e-3, margin=2.0,
                   front_facing=None, max_pairs=80_000_000):
    """
    xy (N, V, 2) pixel coordinates (pixel i spans [i, i+1]); inv_z (N, V) inverse depth;
    attrs (N, V, C) per-vertex attributes; faces (F, 3).
    Returns (attr image (N, H, W, C) = depth-softmax colour, alpha (N, H, W)).
    """
    N, V, _ = xy.shape
    dev, dt = xy.device, xy.dtype
    tri = xy[:, faces]  # (N, F, 3, 2)
    with torch.no_grad():
        valid = (inv_z[:, faces] > 0).all(-1)
        if front_facing is not None:
            valid &= front_facing
        lo = torch.floor(tri.min(2).values - margin - 0.5).clamp_(min=0)
        hi = torch.ceil(tri.max(2).values + margin - 0.5)
        hi[..., 0].clamp_(max=W - 1)
        hi[..., 1].clamp_(max=H - 1)
        wh = (hi - lo + 1).clamp(min=0)
        valid &= (wh > 0).all(-1)
        n_idx, f_idx = valid.nonzero(as_tuple=True)
        lo_v, wh_v = lo[n_idx, f_idx].long(), wh[n_idx, f_idx].long()
        counts = wh_v[:, 0] * wh_v[:, 1]
        total = int(counts.sum())
        if total > max_pairs:
            raise RuntimeError(f"{total} face/pixel pairs > max_pairs {max_pairs}")
        rep = torch.repeat_interleave(torch.arange(len(counts), device=dev), counts)
        start = torch.cumsum(counts, 0) - counts
        off = torch.arange(total, device=dev) - start[rep]
        px = lo_v[rep, 0] + off % wh_v[rep, 0]
        py = lo_v[rep, 1] + off // wh_v[rep, 0]
        pn, pf = n_idx[rep], f_idx[rep]

    def barycentric(t, p):
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        def cr(u, v, w):  # cross(v - u, w - u)
            return (v[:, 0] - u[:, 0]) * (w[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])
        wa = cr(b, c, p) / area
        wb = cr(c, a, p) / area
        wc = cr(a, b, p) / area
        def safe_norm(x):  # finite gradient at 0 (a pixel centre exactly on an edge or vertex)
            return torch.sqrt((x * x).sum(-1) + 1e-12)
        lens = torch.stack([safe_norm(c - b), safe_norm(a - c), safe_norm(b - a)], -1)
        w = torch.stack([wa, wb, wc], -1)
        inside_d = (w * area.abs()[:, None] / lens.clamp(min=1e-8)).min(-1).values  # distance to boundary
        # outside: true distance to the triangle (nearest of the 3 segments), so the coverage
        # support is the triangle dilated by `margin` with rounded corners (no miter spikes at
        # sliver vertices that would reach past the enumeration window)
        def seg(u, v):
            e = v - u
            t = (((p - u) * e).sum(-1) / (e * e).sum(-1).clamp(min=1e-12)).clamp(0, 1)
            return safe_norm(p - (u + t[:, None] * e))
        out_d = torch.minimum(torch.minimum(seg(b, c), seg(c, a)), seg(a, b))
        sd = torch.where((w >= 0).all(-1), inside_d, -out_d)  # >0 inside, continuous at the boundary
        return w, sd

    p = torch.stack([px, py], -1).to(dt) + 0.5
    with torch.no_grad():  # drop pairs well outside the face before building the graph
        _, sd0 = barycentric(tri[pn, pf], p)
        keep = sd0 > -margin
        pn, pf, px, py, p = pn[keep], pf[keep], px[keep], py[keep], p[keep]
    t = tri[pn, pf]
    w, sd = barycentric(t, p)
    # compact-support coverage: smoothstep over [-margin, margin], exactly 0 at the window edge
    # (a sigmoid is not, and a far-in-front face entering the window would make the depth-softmax
    # colour jump); slope at the edge = 0.75 / margin, i.e. a sigmoid of width margin / 3
    tt = (sd / margin * 0.5 + 0.5).clamp(0, 1)
    D = tt * tt * (3 - 2 * tt)
    wc = w.clamp(min=0)
    wc = wc / wc.sum(-1, keepdim=True).clamp(min=1e-8)
    fv = faces[pf]  # (P, 3)
    iz = (wc * inv_z[pn[:, None], fv]).sum(-1)
    attr = (wc[..., None] * attrs[pn[:, None], fv]).sum(1)  # (P, C)
    key = (pn * H + py) * W + px
    n_keys = N * H * W
    log_trans = torch.zeros(n_keys, device=dev, dtype=dt).index_add(0, key, torch.log1p(-D.clamp(max=1 - 1e-6)))
    alpha = 1 - torch.exp(log_trans)
    zw = _segment_softmax_weights(iz / gamma, key, n_keys) * D
    den = torch.zeros(n_keys, device=dev, dtype=dt).index_add(0, key, zw)
    num = torch.zeros(n_keys, attr.shape[1], device=dev, dtype=dt).index_add(0, key, zw[:, None] * attr)
    img = num / den.clamp(min=1e-12)[:, None]
    with torch.no_grad():  # exact mesh silhouette (pixel centre inside any face), for parity checks
        hard = torch.zeros(n_keys, device=dev, dtype=torch.bool).index_fill_(0, key[sd.detach() > 0], True)
    soft_rasterize.last_hard = hard.view(N, H, W)
    return img.view(N, H, W, -1), alpha.view(N, H, W)


class SoftRobotRenderer(torch.nn.Module):
    """Joint states -> robot-only images on black, camera of ``RobotRenderer``'s SAPIEN camera."""

    AMBIENT = 0.3
    LIGHTS = ((1.0, 1.0, -1.0), (0.0, 0.0, -1.0))  # directions, colour 1 each (RobotRenderer._load_lighting)

    def __init__(self, extrinsic_cv, intrinsic_cv, H=224, W=448, voxel=0.0, sigma=0.5, gamma=1e-3,
                 cull_backfaces=False, srgb=True, device="cuda", dtype=torch.float32):
        super().__init__()
        self.mesh = PandaMeshModel(voxel=voxel, device=device, dtype=dtype)
        E = torch.as_tensor(np.asarray(extrinsic_cv), dtype=dtype, device=device).reshape(3, 4)
        K = torch.as_tensor(np.asarray(intrinsic_cv), dtype=dtype, device=device).reshape(3, 3)
        self.register_buffer("R", E[:, :3])
        self.register_buffer("t", E[:, 3])
        self.register_buffer("K", K)
        L = torch.tensor(self.LIGHTS, dtype=dtype, device=device)
        self.register_buffer("light_dirs", torch.nn.functional.normalize(L, dim=-1))
        self.H, self.W, self.sigma, self.gamma = H, W, sigma, gamma
        self.cull, self.srgb = cull_backfaces, srgb

    def forward(self, qpos, return_alpha=False, frames_per_call=None):
        """qpos (N, 9) -> images (N, 3, H, W) in [0, 1], premultiplied over black.

        Frames are rasterized in groups; with gradients each group is checkpointed, so the
        face/pixel pair tensors are rebuilt in backward instead of being kept alive.
        """
        if frames_per_call is None:  # face/pixel pairs grow with the (6 sigma) margin
            frames_per_call = 6 if self.sigma <= 0.5 else (3 if self.sigma <= 1.0 else 1)
        outs, alphas, hards = [], [], []
        for i in range(0, qpos.shape[0], frames_per_call):
            q = qpos[i: i + frames_per_call]
            if torch.is_grad_enabled() and q.requires_grad:
                o, a = torch.utils.checkpoint.checkpoint(self._render, q, use_reentrant=False)
            else:
                o, a = self._render(q)
            outs.append(o); alphas.append(a); hards.append(soft_rasterize.last_hard)
        self.last_hard = torch.cat(hards)
        out, alpha = torch.cat(outs), torch.cat(alphas)
        return (out, alpha) if return_alpha else out

    def _render(self, qpos):
        dt = self.R.dtype
        v, n = self.mesh.posed(qpos.to(dt))
        vc = v @ self.R.T + self.t  # OpenCV camera frame
        z = vc[..., 2]
        xy = torch.stack([self.K[0, 0] * vc[..., 0] / z + self.K[0, 2],
                          self.K[1, 1] * vc[..., 1] / z + self.K[1, 2]], -1)
        albedo = self.mesh.colors.to(dt)
        if self.srgb:
            albedo = torch.where(albedo <= 0.04045, albedo / 12.92, ((albedo + 0.055) / 1.055) ** 2.4)
        lam = sum(torch.relu((n * -l).sum(-1)) for l in self.light_dirs)  # world-frame Lambert
        shade = albedo[None] * (self.AMBIENT + lam[..., None])
        front = None
        if self.cull:
            f = self.mesh.faces
            fn = torch.cross(vc[:, f[:, 1]] - vc[:, f[:, 0]], vc[:, f[:, 2]] - vc[:, f[:, 0]], dim=-1)
            front = (fn * -vc[:, f].mean(2)).sum(-1) > 0
        img, alpha = soft_rasterize(xy, 1.0 / z, shade, self.mesh.faces, self.H, self.W,
                                    sigma=self.sigma, gamma=self.gamma, margin=3 * self.sigma, front_facing=front)
        if self.srgb:
            img = img.clamp(min=0)
            img = torch.where(img <= 0.0031308, img * 12.92, 1.055 * img.clamp(min=1e-8) ** (1 / 2.4) - 0.055)
        out = (alpha[..., None] * img.clamp(0, 1)).permute(0, 3, 1, 2)  # NHWC -> NCHW
        return out, alpha
