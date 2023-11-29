# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2022 Anpei Chen

import os
import random

import numpy as np
import torch
import cv2
import re

from joblib import delayed, Parallel
from torch.utils.data import Dataset
from utils.utils import decode_flow
import json

def concatenate_append(old, new, dim):
    new = np.concatenate(new, 0).reshape(-1, dim)
    if old is not None:
        new = np.concatenate([old, new], 0)

    return new

class LocalRFDataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        frames_chunk=20,
        downsampling=-1,
        load_depth=False,
        load_flow=False,
        with_preprocessed_poses=False,
        n_init_frames=7, # 注意：这里传入的参数为5
        subsequence=[0, -1],
        test_frame_every=10, # 这里输入也为10，不变
        frame_step=1,
    ):
        self.root_dir = datadir
        self.split = split
        self.frames_chunk = max(frames_chunk, n_init_frames)
        self.downsampling = downsampling
        self.load_depth = load_depth
        self.load_flow = load_flow
        self.frame_step = frame_step

        if with_preprocessed_poses:
            with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
                self.transforms = json.load(f)
            self.image_paths = [os.path.basename(frame_meta["file_path"]) for frame_meta in self.transforms["frames"]]
            self.image_paths = sorted(self.image_paths)
            poses_dict = {os.path.basename(frame_meta["file_path"]): frame_meta["transform_matrix"] for frame_meta in self.transforms["frames"]}
            poses = []
            for idx, image_path in enumerate(self.image_paths):
                pose = np.array(poses_dict[image_path], dtype=np.float32)
                poses.append(pose)

            self.first_pose = np.array(poses_dict[self.image_paths[0]], dtype=np.float32)
            self.rel_poses = []
            for idx in range(len(poses)):
                if idx == 0:
                    pose = np.eye(4, dtype=np.float32)
                else:
                    pose = np.linalg.inv(poses[idx - 1]) @ poses[idx]
                self.rel_poses.append(pose)
            self.rel_poses = np.stack(self.rel_poses, axis=0) 

            self.pose_scale = 2e-2 / np.median(np.linalg.norm(self.rel_poses[:, :3, 3], axis=-1))
            self.rel_poses[:, :3, 3] *= self.pose_scale
            self.rel_poses = self.rel_poses[::frame_step]

        else:
            # 如果没有with_processed_poses，则总会进入这里
            self.image_paths = sorted(os.listdir(os.path.join(self.root_dir, "images")))
        if subsequence != [0, -1]: # 这个应该总是满足相等，因此不会进入这里
            self.image_paths = self.image_paths[subsequence[0]:subsequence[1]]

        # 分割图片数据集，输入framestep=1，割了但没完全割
        self.image_paths = self.image_paths[::frame_step]
        # 保存处理后的图片路径到all_image_paths
        # 这里是深拷贝还是浅拷贝?
        self.all_image_paths = self.image_paths

        self.test_mask = []
        self.test_paths = []
        # 应该是序号和路径进行迭代
        for idx, image_path in enumerate(self.image_paths):
            fbase = os.path.splitext(image_path)[0] # base路径（不包含后缀？
            index = int(fbase) if fbase.isnumeric() else idx # 如果fbase为纯数字则index以其为值，否则采用idx
            # test_frame_every=10
            # 如果设置了此参数值，且index与之区域为0，则加入测试集（test）
            if test_frame_every > 0 and index % test_frame_every == 0:
                self.test_paths.append(image_path)
                self.test_mask.append(1)
            else:
                # 否则不加入
                self.test_mask.append(0)
        # 最终的test_mask是一个和总数据长度相同的数组，如果序号对应的数据被添加到测试集中，则test_mask中的对应序号为1
        self.test_mask = np.array(self.test_mask)

        # split默认为“train”，训练集对应的split为“train”，测试集对应的split为“test”
        if split=="test":
            # 如果此数据集为测试集则用test_paths替换image_paths
            self.image_paths = self.test_paths
            # 以image_paths的长度为frames_chunk
            self.frames_chunk = len(self.image_paths)
        # num_images理所应当的是image_paths的长度
        self.num_images = len(self.image_paths)
        # 所有的fbase
        self.all_fbases = {os.path.splitext(image_path)[0]: idx for idx, image_path in enumerate(self.image_paths)}

        self.white_bg = False # 默认white_bg为False

        # 这里的near_far是直接自己指定的
        self.near_far = [0.1, 1e3] # Dummi
        # scene_bbox=[[-2, -2, -2], [2, 2, 2]]
        self.scene_bbox = 2 * torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

        # 初始化一些最后要返回的变量为None
        self.all_rgbs = None
        self.all_invdepths = None
        self.all_fwd_flow, self.all_fwd_mask, self.all_bwd_flow, self.all_bwd_mask = None, None, None, None
        self.all_loss_weights = None

        self.active_frames_bounds = [0, 0]
        self.loaded_frames = 0
        # n_init_frames=5
        self.activate_frames(n_init_frames)
        # 带着参数直接执行下方的函数↓


    def activate_frames(self, n_frames=1):
        # n_frames=5
        self.active_frames_bounds[1] += n_frames
        # 以第一次进入这里为例，[0, 0] -> [0, 5]
        self.active_frames_bounds[1] = min(
            self.active_frames_bounds[1], self.num_images
        )
        # 这个应该是防止获取数据时越界
        # [0, 5] -> [0, 5]

        # 如果active_frames_bounds较大一侧的边界大于loaded_frames
        # 也就是说，有还没有加载的可使用数据帧时，执行下面的if分支
        if self.active_frames_bounds[1] > self.loaded_frames:
            self.read_meta() # 跳转至read_meta函数



    def has_left_frames(self):
        return self.active_frames_bounds[1] < self.num_images

    def deactivate_frames(self, first_frame):
        n_frames = first_frame - self.active_frames_bounds[0]
        self.active_frames_bounds[0] = first_frame

        self.all_rgbs = self.all_rgbs[n_frames * self.n_px_per_frame:] 
        if self.load_depth:
            self.all_invdepths = self.all_invdepths[n_frames * self.n_px_per_frame:]
        if self.load_flow:
            self.all_fwd_flow = self.all_fwd_flow[n_frames * self.n_px_per_frame:]
            self.all_fwd_mask = self.all_fwd_mask[n_frames * self.n_px_per_frame:]
            self.all_bwd_flow = self.all_bwd_flow[n_frames * self.n_px_per_frame:]
            self.all_bwd_mask = self.all_bwd_mask[n_frames * self.n_px_per_frame:]
        self.all_loss_weights = self.all_loss_weights[n_frames * self.n_px_per_frame:]



    def read_meta(self):
        # 好家伙，上来直接定义一个函数是吧
        def read_image(i):
            # 根据序号i从image_paths中获取路径
            image_path = os.path.join(self.root_dir, "images", self.image_paths[i])
            # 获取motion_mask_path，这个应当可以不存在（测试代码时也没有这个masks文件）
            motion_mask_path = os.path.join(self.root_dir, "masks", 
                f"{os.path.splitext(self.image_paths[i])[0]}.png")
            # 如果对应mask不存在，则使用masks目录下的all.png
            # 但是我记得这个也没有
            if not os.path.isfile(motion_mask_path):
                motion_mask_path = os.path.join(self.root_dir, "masks/all.png")

            # 读取图片
            # 从RGB变成BGR通道
            img = cv2.imread(image_path)[..., ::-1]
            # 归一化到[0, 1]
            img = img.astype(np.float32) / 255
            # 这里输入的downsampling参数为-1，应当不会进入下面的分支
            if self.downsampling != -1: # 因该是图片降采样操作
                scale = 1 / self.downsampling
                img = cv2.resize(img, None, 
                    fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # loss_depth_weight_inital=0.1，为True会进入这个分支
            if self.load_depth:
                # 获取深度图的路径
                invdepth_path = os.path.join(self.root_dir, "depth", 
                    f"{os.path.splitext(self.image_paths[i])[0]}.png")
                # 参数-1表示全量读入，包括alpha通道，专为float32
                invdepth = cv2.imread(invdepth_path, -1).astype(np.float32)
                invdepth = cv2.resize(
                    # 这个作为新shape的tuple是图片去掉channel维度后，wh互换
                    # 不过实际上深度图的尺寸似乎并没有变化
                    invdepth, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
            else:
                invdepth = None

            # loss_flow_weight_inital=1，为True，会进入这个分支
            if self.load_flow:
                # 获取光流图的序号，这里又夹了两层
                glob_idx = self.all_image_paths.index(self.image_paths[i])
                if glob_idx+1 < len(self.all_image_paths):
                    # 如果此序号没有越界（不是数据集中最后一张图片），则记录路径到fwd_flow_path
                    fwd_flow_path = self.all_image_paths[glob_idx+1]
                else:
                    # 否则使用0对应的光流图，也就是flow中序号最小的（通常来说是一张纯色图片)
                    fwd_flow_path = self.all_image_paths[0]
                if self.frame_step != 1: # 如果frame_step不为1，这里输入的frame_step为1，因此应当不会进入这个分支
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_step{self.frame_step}_{os.path.splitext(fwd_flow_path)[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_step{self.frame_step}_{os.path.splitext(self.image_paths[i])[0]}.png")
                else:# ←应当会进入这里
                    # 获取fwd_flow_path和bwd_flow_path，区别在于前缀和splitext中的内容
                    # 这么些是为了保证bwd_flow_path始终比fwd_flow_path序号小1
                    # 居然只加载了flow_ds文件夹中的图片
                    fwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"fwd_{os.path.splitext(fwd_flow_path)[0]}.png")
                    bwd_flow_path = os.path.join(self.root_dir, "flow_ds", 
                        f"bwd_{os.path.splitext(self.image_paths[i])[0]}.png")
                # 读取fwd、bwd为encoded_fwd_flow和encoded_bwd_flow
                encoded_fwd_flow = cv2.imread(fwd_flow_path, cv2.IMREAD_UNCHANGED)
                encoded_bwd_flow = cv2.imread(bwd_flow_path, cv2.IMREAD_UNCHANGED)
                # 确定光流图和原图之间的尺寸比例，只需要用一个维度确定即可，另一维度比例相同
                # 以kitti为例子，image.shape[0]/encoded_fwd_flow.shape[0]=2
                flow_scale = img.shape[0] / encoded_fwd_flow.shape[0] 
                # 把光流图resize到原图大小
                encoded_fwd_flow = cv2.resize(
                    encoded_fwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
                encoded_bwd_flow = cv2.resize(
                    encoded_bwd_flow, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA)
                # 用utils中的decode_flow函数处理一下光流图得到fwd、bwd的flow和mask            
                fwd_flow, fwd_mask = decode_flow(encoded_fwd_flow)
                bwd_flow, bwd_mask = decode_flow(encoded_bwd_flow)
                # 然后还是乘一下scale
                fwd_flow = fwd_flow * flow_scale
                bwd_flow = bwd_flow * flow_scale
            else:
                # 没有加载光流图（光流误差的权值为0））
                fwd_flow, fwd_mask, bwd_flow, bwd_mask = None, None, None, None

            # 如果motion_mask_path所指的文件存在，kitti示例中并没有这个文件，因此也不会进入这里
            if os.path.isfile(motion_mask_path): 
                mask = cv2.imread(motion_mask_path, cv2.IMREAD_UNCHANGED)
                if len(mask.shape) != 2:
                    mask = mask[..., 0]
                mask = cv2.resize(mask, tuple(img.shape[1::-1]), interpolation=cv2.INTER_AREA) > 0
            else:
                # 如果不存在，那mask直接为None
                mask = None

            # 返回：
            # img       : 原图片
            # invdepth  : 深度图
            # fwd_flow  : 下列四个为经过decode_flow函数处理后的光流图得到的内容
            # fwd_mask  : ↑
            # bwd_flow  : ↑
            # bwd_mask  : ↑
            # mask      : None
            return {
                "img": img, 
                "invdepth": invdepth,
                "fwd_flow": fwd_flow,
                "fwd_mask": fwd_mask,
                "bwd_flow": bwd_flow,
                "bwd_mask": bwd_mask,
                "mask": mask,
            }
        # 真正的执行从这里开始
        # 这里的min应该也是为了避免越界，确定需要加载的数据帧数目
        n_frames_to_load = min(self.frames_chunk, self.num_images - self.loaded_frames)
        # 多线程调用read_image(i)，i为[loaded_frames, loaded_frames+n_frames_to_load]，回到上面的函数
        all_data = Parallel(n_jobs=-1, backend="threading")(
            delayed(read_image)(i) for i in range(self.loaded_frames, self.loaded_frames + n_frames_to_load) 
        )

        # 既然已经完成加载新帧信息了，那么修改一下loaded_frames
        self.loaded_frames += n_frames_to_load
        # 把加载到的信息全部放入all_XXX变量中保存
        all_rgbs = [data["img"] for data in all_data]
        all_invdepths = [data["invdepth"] for data in all_data]
        all_fwd_flow = [data["fwd_flow"] for data in all_data]
        all_fwd_mask = [data["fwd_mask"] for data in all_data]
        all_bwd_flow = [data["bwd_flow"] for data in all_data]
        all_bwd_mask = [data["bwd_mask"] for data in all_data]
        all_mask = [data["mask"] for data in all_data]

        # 这是在做什么？
        # 不过处理之后的图片除了少了channel的维度之外，另外两个维度值与原来的img相同
        all_laplacian = [
                np.ones_like(img[..., 0]) * cv2.Laplacian(
                            cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F
                        ).var()
            for img in all_rgbs
        ]
        # mask为None的相关操作，这里看不太懂
        all_loss_weights = [laplacian if mask is None else laplacian * mask for laplacian, mask in zip(all_laplacian, all_mask)]

        # 记录每张图片的wh（尺寸）
        self.img_wh = list(all_rgbs[0].shape[1::-1])
        # 每张图片的像素总数
        self.n_px_per_frame = self.img_wh[0] * self.img_wh[1]

        # 如果是训练集
        if self.split != "train": # 使用np.stack进行堆叠
            self.all_rgbs = np.stack(all_rgbs, 0)
            if self.load_depth:
                self.all_invdepths = np.stack(all_invdepths, 0)
            if self.load_flow:
                self.all_fwd_flow = np.stack(all_fwd_flow, 0)
                self.all_fwd_mask = np.stack(all_fwd_mask, 0)
                self.all_bwd_flow = np.stack(all_bwd_flow, 0)
                self.all_bwd_mask = np.stack(all_bwd_mask, 0)
        else: # 如果不是训练集
            # 使用concatenate_append进行堆叠
            self.all_rgbs = concatenate_append(self.all_rgbs, all_rgbs, 3)
            if self.load_depth:
                self.all_invdepths = concatenate_append(self.all_invdepths, all_invdepths, 1)
            if self.load_flow:
                self.all_fwd_flow = concatenate_append(self.all_fwd_flow, all_fwd_flow, 2)
                self.all_fwd_mask = concatenate_append(self.all_fwd_mask, all_fwd_mask, 1)
                self.all_bwd_flow = concatenate_append(self.all_bwd_flow, all_bwd_flow, 2)
                self.all_bwd_mask = concatenate_append(self.all_bwd_mask, all_bwd_mask, 1)
            self.all_loss_weights = concatenate_append(self.all_loss_weights, all_loss_weights, 1)


    def __len__(self):
        return int(1e10)

    def __getitem__(self, i):
        raise NotImplementedError
        idx = np.random.randint(self.sampling_bound[0], self.sampling_bound[1])

        return {"rgbs": self.all_rgbs[idx], "idx": idx}

    def get_frame_fbase(self, view_id):
        return list(self.all_fbases.keys())[view_id]

    def sample(self, batch_size, is_refining, optimize_poses, n_views=16):
        # 初次进行sample时，batch_size=4096（这个应该不变），is_refining=False，optimize_poses=True
        # 获取active_frames_bounds内的test_mask信息（对应frame是否位于test集合）
        active_test_mask = self.test_mask[self.active_frames_bounds[0] : self.active_frames_bounds[1]]
        # 总frame中位于test集合的比例
        test_ratio = active_test_mask.mean()
        if optimize_poses:# optimize_poses=True，应当进入这个分支
            # random.uniform(a,b)，生成一个取值范围在[a, b]的随机数
            # 也就是说哦train_test_poses是随机取值的，但大致和test_ratio相同
            train_test_poses = test_ratio > random.uniform(0, 1)
        else:# 不会进入这个分支
            train_test_poses = False

        # inclusion_mask是做什么的？
        # 如果train_test_poses的取值为true则直接以active_test_mask为值，否则为active_test_mask取反
        inclusion_mask = active_test_mask if train_test_poses else 1 - active_test_mask
        # sample_map为
        # 以self.active_frames_bounds[0]为起点，以self.active_frames_bounds[1]为终点
        # 步长为1，然后截取inclusion对应位置为1的元素
        sample_map = np.arange(
            self.active_frames_bounds[0], 
            self.active_frames_bounds[1], 
            dtype=np.int64)[inclusion_mask == 1]
        
        # raw_samples为n_views个范围在[0, inclusion.sum())的整型随机数，类型为int64
        raw_samples = np.random.randint(0, inclusion_mask.sum(), n_views, dtype=np.int64)

        # 这里的is_refining应当是会变化的，但初始输入为True，因为inclusion_mask的值较难达到4以上，因此也许通常不会进入这里？
        # Force having the last views during coarse optimization
        if not is_refining and inclusion_mask.sum() > 4:
            raw_samples[:2] = inclusion_mask.sum() - 1
            raw_samples[2:4] = inclusion_mask.sum() - 2
            raw_samples[4] = inclusion_mask.sum() - 3
            raw_samples[5] = inclusion_mask.sum() - 4

        view_ids = sample_map[raw_samples]
 
        # 处理idx
        # 首先生成idx为[0, n_px_per_frame)范围的batch_size个（4096）随机数
        idx = np.random.randint(0, self.n_px_per_frame, batch_size, dtype=np.int64)
        # 将idx reshape成16（n_views）*256维度的矩阵
        idx = idx.reshape(n_views, -1)
        # 给idx长度为16的维度依次加上view_ids（长度也为16）*n_px_per_frame的值
        idx = idx + view_ids[..., None] * self.n_px_per_frame
        # 又将idx展开成一维
        idx = idx.reshape(-1)

        # idx的每个元素减去active_frames_bounds[0] * self.n_px_per_frame的值
        idx_sample = idx - self.active_frames_bounds[0] * self.n_px_per_frame

        # 返回
        return {
            "rgbs": self.all_rgbs[idx_sample], 
            "loss_weights": self.all_loss_weights[idx_sample], 
            "invdepths": self.all_invdepths[idx_sample] if self.load_depth else None,
            "fwd_flow": self.all_fwd_flow[idx_sample] if self.load_flow else None,
            "fwd_mask": self.all_fwd_mask[idx_sample] if self.load_flow else None,
            "bwd_flow": self.all_bwd_flow[idx_sample] if self.load_flow else None,
            "bwd_mask": self.all_bwd_mask[idx_sample] if self.load_flow else None,
            "idx": idx,
            "view_ids": view_ids,
            "train_test_poses": train_test_poses,
        }