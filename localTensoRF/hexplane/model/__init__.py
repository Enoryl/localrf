import torch

from hexplane.model.HexPlane import HexPlane
from hexplane.model.HexPlane_Slim import HexPlane_Slim
from hexplane.render.util.util import N_to_reso


def init_model(cfg, aabb, near_far, device):
    # 这里用传入的reso_cur，并且后面从cfg中取值，取值的方式也要改为字典
    # old:
    # reso_cur = N_to_reso(cfg.model.N_voxel_init, aabb, cfg.model.nonsquare_voxel)

    # old：
    # if cfg.systems.ckpt is not None:
    #     model = torch.load(cfg.systems.ckpt, map_location=device)
    # else:
    #     # There are two types of upsampling: aligned and unaligned.
    #     # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
    #     # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
    #     if cfg.model.upsampling_type == "aligned":
    #         reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
    #     model = eval(cfg.model.model_name)( # 默认的model_name为HexPlane_Slim
    #         aabb, reso_cur, device, cfg.model.time_grid_init, near_far, **cfg.model
    #     )
    # return model, reso_cur

    # new：
    if cfg["ckpt"] is not None:
        model = torch.load(cfg.systems.ckpt, map_location=device)
    else:
        # There are two types of upsampling: aligned and unaligned.
        # Aligned upsampling: N_t = 2 * N_t-1 - 1. Like: |--|--| ->upsampling -> |-|-|-|-|, where | represents sampling points and - represents distance.
        # Unaligned upsampling: We use linear_interpolation to get the new sampling points without considering the above rule.
        if cfg["upsampling_type"] == "aligned":
            reso_cur = [reso_cur[i] // 2 * 2 + 1 for i in range(len(reso_cur))]
        model = eval(cfg["model_name"])( # 默认的model_name为HexPlane_Slim
            aabb, cfg["reso_cur"], device, cfg["time_grid_init"], near_far, **cfg
        )
    return model, cfg["reso_cur"]
