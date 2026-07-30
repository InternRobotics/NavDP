"""Training visualization helpers."""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_mpc_controller(refs, xs_pred, batch, save_path="", real_desired_v=None):
    """Plot reference vs predicted MPC trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8), dpi=80)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    real_v = np.asarray(real_desired_v).flatten() if real_desired_v is not None else None

    for i in range(batch):
        xr = refs[i, :, 0]
        yr = refs[i, :, 1]
        ax.plot(xr, yr, "--", color=colors[i % len(colors)], label=f"ref {i}", linewidth=2)
        xe = xs_pred[i, :, 0]
        ye = xs_pred[i, :, 1]
        ax.plot(xe, ye, "-", color=colors[i % len(colors)], label=f"pred {i}", linewidth=2)
        ax.plot(xe[0], ye[0], "o", color=colors[i % len(colors)], markersize=1, label=f"start {i}")
        ax.plot(xe[-1], ye[-1], "s", color=colors[i % len(colors)], markersize=1, label=f"end {i}")
        if real_v is not None and i < len(real_v):
            ax.annotate(f"v={real_v[i]:.3f}", (xr[-1], yr[-1]), fontsize=9, alpha=0.9, color="blue")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_title("Batch MPC: reference vs predicted trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=80)
    plt.close("all")


def get_grid_img_from_metadata(metadata: dict) -> np.ndarray:
    """Convert the occupancy grid stored in env metadata to a BGR image."""
    gm = metadata.get("grid_matrix", np.zeros((1, 1), dtype=np.int32))
    if gm is None or gm.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    gm = np.array(gm)
    if gm.ndim == 1:
        gm = gm.reshape(1, -1)
    gm = gm.T
    img = (1 - gm) * 255
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def overlay_on_corner(
    base_img,
    overlay_img,
    corner="top_right",
    scale_ratio=0.28,
    margin=8,
    border_color=(255, 255, 255),
    interpolation=cv2.INTER_LINEAR,
):
    """Scale overlay and paste it onto a corner of base_img."""
    overlay = np.array(overlay_img)
    if overlay.max() <= 1.0:
        overlay = (overlay * 255).astype(np.uint8)
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    base_h, base_w = base_img.shape[:2]
    max_h = max(1, int(base_h * scale_ratio))
    max_w = max(1, int(base_w * scale_ratio))
    oh, ow = overlay.shape[:2]
    scale = min(max_h / oh, max_w / ow)
    new_h = max(1, int(oh * scale))
    new_w = max(1, int(ow * scale))
    overlay_scaled = cv2.resize(overlay, (new_w, new_h), interpolation=interpolation)

    if corner == "top_right":
        x0 = base_w - new_w - margin
        y0 = margin
    elif corner == "top_left":
        x0 = margin
        y0 = margin
    else:
        raise ValueError(f"Unsupported corner: {corner}")

    x0 = max(0, min(x0, base_w - new_w))
    y0 = max(0, min(y0, base_h - new_h))
    cv2.rectangle(base_img, (x0 - 2, y0 - 2), (x0 + new_w + 1, y0 + new_h + 1), border_color, 2)
    base_img[y0:y0 + new_h, x0:x0 + new_w] = overlay_scaled
    return base_img


def append_square_map_right(rgb_img, map_img, gap=4, border_color=(255, 255, 255)):
    """Scale map to a square panel and append it to the right of rgb_img."""
    rgb_h = rgb_img.shape[0]
    square_map = cv2.resize(map_img, (rgb_h, rgb_h), interpolation=cv2.INTER_NEAREST)
    if gap > 0:
        separator = np.full((rgb_h, gap, 3), border_color, dtype=np.uint8)
        return np.hstack([rgb_img, separator, square_map])
    return np.hstack([rgb_img, square_map])


def visualize_grid_with_path_on_image(image, grid_image, pos_list, robot_point, end_point, score, birdeye_image=None):
    """Compose RGB with an optional bird-eye overlay and a square map panel."""
    grid_img = grid_image.copy()
    if len(pos_list) > 0 and len(pos_list) >= 2:
        cv2.polylines(grid_img, [np.array(pos_list, np.int32)], isClosed=False, color=(255, 0, 0), thickness=1)
    cv2.circle(grid_img, (int(robot_point[0]), int(robot_point[1])), 1, (0, 0, 255), -1)
    cv2.circle(grid_img, (int(end_point[0]), int(end_point[1])), 1, (0, 255, 0), -1)
    grid_img = np.flipud(grid_img)

    result_img = np.array(image).copy()
    if result_img.max() <= 1.0:
        result_img = (result_img * 255).astype(np.uint8)

    if birdeye_image is not None:
        result_img = overlay_on_corner(result_img, birdeye_image, corner="top_left", interpolation=cv2.INTER_LINEAR)
    result_img = append_square_map_right(result_img, grid_img)

    cv2.putText(result_img, f"trajectory score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return result_img
