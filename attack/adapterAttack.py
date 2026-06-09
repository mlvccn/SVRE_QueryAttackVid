import torch
from ASMARTselection.AST_S import  AST_S
from ASMARTselection.AST_T import  AST_T
from ASMARTselection.Reward import reward_advantage, \
    target_reward_advantage, rep_frame, sparse_reward
from ASMARTselection.utils_ada import init_hidden

from ASMARTselection.edgebox import *
from ASMARTselection import edgebox as eb
import torch.nn.functional as F
from sklearn.utils import resample
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float


from Utils.utils import speed_up_process, process_grad

def transform_video(video, mode='forward'):

    dtype = video.dtype
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.as_tensor(mean, dtype=dtype, device='cuda')
    std = torch.as_tensor(std, dtype=dtype, device='cuda')
    if mode == 'forward':
        # [-mean/std, mean/std]
        video.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    elif mode == 'back':
        # [0, 1]
        video.mul_(std[:, None, None, None]).add_(mean[:, None, None, None])
    return video

def update_adv_video(adv_vid, unnorm_video, grad, eps, step):
    
    # adv_vid = transform_video(adv_vid.detach(), mode = 'back')
    # adv_vid = adv_vid + step * grad.sign()
    # bottom_bounded_adv = torch.where((unnorm_video - eps) > adv_vid, unnorm_video - eps, adv_vid)
    # bounded_adv = torch.where((unnorm_video + eps) < bottom_bounded_adv, unnorm_video + eps, bottom_bounded_adv)
    # clip_frame = torch.clamp(bounded_adv, 0., 1.)
    # clip_frame = transform_video(clip_frame.detach(), mode = 'forward')
    
    adv_vid = transform_video(adv_vid.detach(), mode='back') # [0, 1]
    adv_vid = adv_vid + step * grad.sign()      
    delta = torch.clamp(adv_vid - unnorm_video, min=-eps, max=eps)
    clip_frame = torch.clamp(unnorm_video + delta, min=0, max=1).detach()
    clip_frame = transform_video(clip_frame, mode='forward') # norm
    
    return clip_frame.detach()

def static_gt_random(fixed_set, static_num, random_num):

    # static_num = int(static_num)
    # random_num = int(random_num)
    # 1. 从 fixed_set 中随机选择 4 帧
    fixed_set = torch.tensor(fixed_set)
    selected_from_fixed = fixed_set[torch.randperm(len(fixed_set))[:static_num]]
    
    # 2. 构建全集 0~31
    all_indices = torch.arange(32)

    # 3. 从全集中移除已选的4帧（仅这4帧），得到剩余28帧
    mask = torch.ones(32, dtype=torch.bool)
    mask[selected_from_fixed] = False
    remaining_pool = all_indices[mask]  # shape: (28,)

    # 4. 从这28帧中随机选4帧
    selected_from_rest = remaining_pool[torch.randperm(len(remaining_pool))[:random_num]]  # shape: (4,)

    # 5. 合并成8帧（可选：打乱顺序）
    final_frames = torch.cat([selected_from_fixed, selected_from_rest])
    # 如果希望完全随机顺序，取消下一行注释：
    # final_frames = final_frames[torch.randperm(final_frames.shape[0])]

    # 6. 转为 (8, 1) 张量
    actions_test = final_frames.reshape(-1, 1)
    
    return actions_test

def get_slic_superpixel_masks(frame, n_segments=100, compactness=10):
    """
    对单帧 (C, H, W) 进行 SLIC 超像素分割，返回：
    - labels: (H, W) 超像素 ID
    - block_masks: List of (H, W) bool masks for each superpixel
    """
    # 转为 (H, W, C) numpy array in [0, 1]
    frame_np = frame.permute(1, 2, 0).cpu().numpy()
    frame_np = np.clip(frame_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)  # 反归一化
    segments = slic(img_as_float(frame_np), n_segments=n_segments, compactness=compactness, sigma=1)

    unique_labels = np.unique(segments)
    block_masks = []
    for label in unique_labels:
        mask = (segments == label)
        block_masks.append(torch.from_numpy(mask).bool().to(frame.device))
    return segments, block_masks

def nes_slic(
    model, vid, actions_t, n, target, sub_num,
    n_segments=100, compactness=10, key_pixel_mask=None, sigma=1e-3
):
    """
    在关键帧上使用 SLIC 超像素块进行 NES 攻击，块内梯度共享。
    
    Args:
        actions_t: 关键帧索引列表 (e.g., tensor([2, 6, 10]))
        n_segments: 每帧超像素数量
        key_pixel_mask: (B, C, T, H, W)，可选，用于进一步限制攻击区域
    
    Returns:
        grads: (len(actions_t), 3, H, W) —— 但实际是块常量
    """
    with torch.no_grad():
        B, C, T, H, W = vid.shape
        assert B == 1, "Only support batch size 1 for now"
        vid = vid.squeeze(0)  # (C, T, H, W)
        device = vid.device
        
        target = target.view(-1).long().to(device)
        assert target.numel() == 1

        # 提取关键帧并构建超像素块
        frame_block_masks = {}  # {t: [mask1, mask2, ...]}
        total_blocks = 0
        for t in actions_t:
            t = t.item()
            frame = vid[:, t, :, :]  # (C, H, W)
            _, block_masks = get_slic_superpixel_masks(frame, n_segments, compactness)
            frame_block_masks[t] = block_masks
            total_blocks += len(block_masks)

        # 初始化：每个关键帧的每个块有一个噪声向量
        # 我们将噪声组织为: (sub_num, num_total_blocks, C)
        noise_per_block = torch.randn((n // 2, total_blocks, C), device=device) * sigma
        all_noise_per_block = torch.cat([noise_per_block, -noise_per_block], dim=0)  # (n, total_blocks, C)

        batch_loss = []
        batch_idx = []
        all_full_noise_list = []

        for i in range(n // sub_num):
            start = i * sub_num
            end = start + sub_num
            noise_batch = all_noise_per_block[start:end]  # (sub_num, total_blocks, C)

            # 构建完整噪声视频 (sub_num, C, T, H, W)
            full_noise = torch.zeros(sub_num, C, T, H, W, device=device)

            block_idx = 0
            for t in actions_t:
                t = t.item()
                for b, mask in enumerate(frame_block_masks[t]):
                    # 将块噪声广播到空间位置
                    noise_val = noise_batch[:, block_idx, :].unsqueeze(-1).unsqueeze(-1)  # (sub_num, C, 1, 1)
                    full_noise[:, :, t, :, :] += noise_val * mask.float()  # (sub_num, C, H, W)
                    block_idx += 1

            # 可选：与频域关键像素掩码交集
            if key_pixel_mask is not None:
                full_noise = full_noise * key_pixel_mask.squeeze(0).float()

            adv_vid_rs = vid.unsqueeze(0).repeat(sub_num, 1, 1, 1, 1) + full_noise  # (sub_num, C, T, H, W)

            logits = model(adv_vid_rs)
            top_val, top_idx = logits.topk(1, 1, True, True)
            top_idx = top_idx.view(-1)
            top_val = top_val.view(-1)
            label = target.flatten()
            ground_preds = F.softmax(logits.detach(), 1).view(sub_num, -1)[:, label].flatten()
            top2_val = F.softmax(logits.detach(), 1).view(sub_num, -1).sort(1)[0][:, -2]
            hunxiao_probs = torch.zeros(sub_num, device=device)
            hunxiao_probs[top_idx == label] = top2_val[top_idx == label]
            hunxiao_probs[top_idx != label] = top_val[top_idx != label]
            loss = hunxiao_probs - ground_preds

            batch_loss.append(loss)
            batch_idx.append(top_idx)
            all_full_noise_list.append(full_noise)

        batch_loss = torch.cat(batch_loss)
        batch_idx = torch.cat(batch_idx)
        all_full_noise = torch.cat(all_full_noise_list, dim=0)

        # 计算权重（同原代码）
        # good_idx = (batch_idx == target).byte()
        # changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device=device))
        
        good_idx = (batch_idx == target[0])          # bool
        changed_loss = torch.where(
            good_idx,
            batch_loss,
            torch.full_like(batch_loss, 1000.0)
)
        
        # loss_order = torch.zeros_like(changed_loss)
        # sort_idx = changed_loss.sort()[1]
        # loss_order[sort_idx] = torch.arange(changed_loss.numel(), dtype=torch.float, device=device)
        # available = good_idx.sum().item()
        # unavailable = n - available
        # unavailable_weight = (loss_order[~good_idx].sum() / unavailable) if unavailable > 0 else 0.0
        # rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)

        # # 聚合梯度：按块加权平均
        # grad_blocks = torch.zeros(total_blocks, C, device=device)
        # block_start = 0
        # for t in actions_t:
        #     t = t.item()
        #     num_blocks_t = len(frame_block_masks[t])
        #     # 提取该帧所有块的噪声：(n, num_blocks_t, C)
        #     noise_t = all_noise_per_block[:, block_start:block_start + num_blocks_t, :]
        #     weight_t = rank_weight.unsqueeze(1).unsqueeze(2)  # (n, 1, 1)
        #     grad_t = (noise_t * weight_t).sum(dim=0) / sigma  # (num_blocks_t, C)
        #     grad_blocks[block_start:block_start + num_blocks_t] = grad_t
        #     block_start += num_blocks_t

        # # 将块梯度转回空间形式 (len(actions_t), C, H, W)
        # grads_list = []
        # block_idx = 0
        # for t in actions_t:
        #     t = t.item()
        #     grad_frame = torch.zeros(C, H, W, device=device)
        #     for mask in frame_block_masks[t]:
        #         block_grad = grad_blocks[block_idx]  # (C,)
        #         grad_frame += block_grad.view(C, 1, 1) * mask.float()
        #         block_idx += 1
        #     grads_list.append(grad_frame)
        # grads = torch.stack(grads_list, dim=0)  # (F, C, H, W)
        
        loss_order = torch.zeros_like(changed_loss)
        sort_idx = changed_loss.sort()[1]
        loss_order[sort_idx] = torch.arange(
            changed_loss.numel(), device=device, dtype=torch.float
        )

        available = good_idx.sum().item()
        unavailable = n - available
        if unavailable > 0:
            unavailable_weight = loss_order[~good_idx].sum() / unavailable
        else:
            unavailable_weight = loss_order.new_tensor(0.0)  

        rank_weight = torch.where(
            good_idx, loss_order, unavailable_weight
        ) / (n - 1)
        
        # rank_weight: (n,)
        weight = rank_weight.view(-1, 1, 1, 1, 1)   # (n,1,1,1,1)

        # all_full_noise: (n, C, T, H, W)
        # 注意：你在前面构建 full_noise 时，其实已经是空间形式了
        grad_full = torch.sum(all_full_noise / sigma * weight, dim=0)
        # grad_full: (C, T, H, W)

        grad_full = grad_full.permute(1, 0, 2, 3).contiguous().cuda()
        # → (T, C, H, W)
        
        actions_t = actions_t.view(-1).long().cuda()
        grads = grad_full.index_select(0, actions_t)
        # (F, C, H, W)

        return grads


def nes_patch_frame(model, vid, patch_size, mask_list, actions_t, n, target, sub_num, sigma=1e-3):
    with torch.no_grad():

        len_frame = len(actions_t)
        grads = torch.zeros((len_frame, 3, patch_size, patch_size), device='cuda')

        count_in = 0
        batch_loss = []
        batch_noise = []
        batch_idx = []
        vid = vid.squeeze(0)
        vid = vid.transpose(1, 0)
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=vid.dtype, device='cuda')
            std = std.view(1, 1, 3, 1, 1)
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma
            # noise_list = noise_list / std
            all_noise = torch.cat([noise_list, -noise_list], 0)

            label = target.flatten()


            mask_list_t = actions_t.view(-1)
            MASK = torch.zeros(adv_vid_rs.size())
            b = mask_list.numpy()
            c = b[mask_list_t]
            
            if c.ndim == 1:
                c = c[np.newaxis, :]  # 从 [4] → [1, 4]
            
            i = 0
            try:
                for x1, x2, y1, y2 in c:
                    key = mask_list_t[i]
                    x1 = torch.tensor(x1).to(torch.int64)
                    x2 = torch.tensor(x2).to(torch.int64)
                    y1 = torch.tensor(y1).to(torch.int64)
                    y2 = torch.tensor(y2).to(torch.int64)
                    MASK[:, key, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]
                    i = i+1
            except:
                print('c', c)


            adv_vid_rs += MASK.cuda()
            del MASK

            # top_val, top_idx, logits = model(adv_vid_rs)
            adv_vid_rs = adv_vid_rs.transpose(1, 2)
            logits = model(adv_vid_rs)
            # logits = outputs.logits
            top_val, top_idx = logits.topk(1, 1, True, True)
            top_idx = top_idx.view(-1)
            top_val = top_val.view(-1)
            ground_preds = F.softmax(logits.detach(), 1).view(sub_num, -1)[:, label].flatten()
            hunxiao_probs = torch.zeros(sub_num).cuda()
            top2_val = F.softmax(logits.detach(), 1).view(sub_num, -1).sort(1)[0][:, -2]
            hunxiao_probs[top_idx == label] = top2_val[top_idx == label]
            hunxiao_probs[top_idx != label] = top_val[top_idx != label]

            loss = hunxiao_probs - ground_preds
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)


        batch_noise = torch.cat(batch_noise, 0)
        batch_loss = torch.cat(batch_loss, 0)
        batch_idx = torch.cat(batch_idx)


        good_idx = torch.sum(batch_idx == target, 1).byte()
        changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))
        loss_order = torch.zeros(changed_loss.size(0), device='cuda')
        sort_index = changed_loss.sort()[1]
        loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)
        available_number = torch.sum(good_idx).item()
        count_in += available_number
        unavailable_number = n - available_number
        unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                    loss_order)) / unavailable_number if unavailable_number else torch.tensor(
            0., device='cuda')
        rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)
        grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)

        return grads
    
    
def nes_fre(model, vid, patch_size, mask_list, actions_t, n, target, sub_num, key_pixel_mask = None, sigma=1e-3):
    with torch.no_grad():
        len_frame = len(actions_t)
        grads = torch.zeros((len_frame, 3, patch_size, patch_size), device='cuda')
        count_in = 0
        batch_loss = []
        batch_noise = []
        batch_idx = []

        vid = vid.squeeze(0).transpose(1, 0)  # (T, C, H, W)
        assert n % sub_num == 0 and sub_num % 2 == 0

        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))  # (sub_num, T, C, H, W)
            std = torch.as_tensor([0.229, 0.224, 0.225], dtype=vid.dtype, device='cuda').view(1, 1, 3, 1, 1)
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma / std
            all_noise = torch.cat([noise_list, -noise_list], 0)  # (sub_num, F, C, H, W)

            # 构建全视频噪声掩码
            full_noise = torch.zeros_like(adv_vid_rs)  # (sub_num, T, C, H, W)
            mask_list_t = actions_t.view(-1)
            b = mask_list.numpy()
            c = b[mask_list_t]
            if c.ndim == 1:
                c = c[np.newaxis, :]

            i = 0
            for x1, x2, y1, y2 in c:
                key = mask_list_t[i]
                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                full_noise[:, key, :, y1:y2, x1:x2] = all_noise[:, i, :, :, :]
                i += 1

            # 只保留关键像素区域的噪声
            if key_pixel_mask is not None:
                # key_pixel_mask: (1, C, T, H, W) → 需要广播到 (sub_num, T, C, H, W)
                kp_mask = key_pixel_mask.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
                kp_mask = kp_mask.unsqueeze(0)  # (1, T, C, H, W)
                full_noise = full_noise * kp_mask.float()

            adv_vid_rs += full_noise
            adv_vid_rs = adv_vid_rs.transpose(1, 2)  # (sub_num, C, T, H, W)

            logits = model(adv_vid_rs)
            top_val, top_idx = logits.topk(1, 1, True, True)
            top_idx = top_idx.view(-1)
            top_val = top_val.view(-1)
            label = target.flatten()
            ground_preds = F.softmax(logits.detach(), 1).view(sub_num, -1)[:, label].flatten()
            hunxiao_probs = torch.zeros(sub_num, device='cuda')
            top2_val = F.softmax(logits.detach(), 1).view(sub_num, -1).sort(1)[0][:, -2]
            hunxiao_probs[top_idx == label] = top2_val[top_idx == label]
            hunxiao_probs[top_idx != label] = top_val[top_idx != label]
            loss = hunxiao_probs - ground_preds

            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(full_noise)  # 注意：现在是 full_noise

        # 后续梯度计算保持不变...
        batch_noise = torch.cat(batch_noise, 0)  # (n, T, C, H, W)
        batch_loss = torch.cat(batch_loss, 0)
        batch_idx = torch.cat(batch_idx)

        # 提取对应 actions_t 的梯度
        # grad_per_frame = []
        # for t in actions_t:
        #     t = t.item()
        #     noise_t = batch_noise[:, t, :, :, :]  # (n, C, H, W)
        #     # 这里简化：直接平均，或按 rank_weight 加权
        #     grad_t = torch.mean(noise_t, dim=0)
        #     grad_per_frame.append(grad_t)

        # grads = torch.stack(grad_per_frame, dim=0)  # (F, C, H, W)
        
        good_idx = torch.sum(batch_idx == target, 1).byte()
        changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))
        loss_order = torch.zeros(changed_loss.size(0), device='cuda')
        sort_index = changed_loss.sort()[1]
        loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)
        available_number = torch.sum(good_idx).item()
        count_in += available_number
        unavailable_number = n - available_number
        unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                    loss_order)) / unavailable_number if unavailable_number else torch.tensor(
            0., device='cuda')
        rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)
        
        # rank_weight: (n,)
        weight = rank_weight.view((-1, 1, 1, 1, 1))  # (n,1,1,1,1)

        # 加权求和 → (T, C, H, W)
        grad_full = torch.sum(batch_noise / sigma * weight, dim=0)

        # 只取 actions_t 对应的帧 → (F, C, H, W)
        grads += grad_full[mask_list_t]
        
        # grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)

        return grads
        
    

def  untargeted_video_attack( vid_model, vid, x0, ori_class, args, model_keyframes, K,  eps=0.05,
                        max_lr=0.01, min_lr= 1e-3, effect_num=20, sample_per_draw=60,  sub_num_sample=2, max_iter=15000):

    # ----------------------------------------------------初始化---------------------------------------------------------

    num_iter = 0
    T = 10
    cur_lr = max_lr
    input_frames = 32
    unnorm_video = transform_video(vid.clone().detach(), mode = 'back')
    nes_num = 0
    rein = True
    adv_vid = vid.clone()
    logits = vid_model(adv_vid)
    num_iter += 1
    static_num = 4
    random_num = K - static_num
    actions_list = [2, 6, 10, 14, 18, 22, 26, 30]
    # actions_test = torch.randperm(32)[:8].reshape(-1, 1)
    
    ## 数据依赖
    # actions_list = data_denpendcy_attack(vid_model, vid, x0, ori_class, args, model_keyframes, K)
    
    # fre_mask, _ = frequency_key_pixel_frame(vid, top_k_ratio=0.5, apply_spatial=True, apply_temporal=False)
    # fre_mask = fre_mask.cuda()
    

    while num_iter < max_iter:
        
        mask_test = torch.tensor([[0, 224, 0, 224]]).repeat(input_frames, 1)
        actions_test = static_gt_random(actions_list, static_num=8, random_num=0)
        # actions_test = static_gt_random(actions_list, static_num, random_num)
        # actions_test = torch.randperm(32)[:8].reshape(-1, 1)
        patch_size_test = 224
        patch_size = args.patch_size
        
        # patch+frame
        gs = nes_patch_frame(vid_model, adv_vid, patch_size_test, mask_test, actions_test,  sample_per_draw, ori_class, sub_num_sample, sigma=1e-3)
        # gs = nes_fre(vid_model, adv_vid, patch_size_test, mask_test, actions_test,  sample_per_draw, ori_class, sub_num_sample, key_pixel_mask=fre_mask, sigma=1e-3)
        
    #     gs = nes_slic(
    #     vid_model, adv_vid, actions_test,
    #     n=sample_per_draw, target=ori_class, sub_num=sub_num_sample,
    #     n_segments=80, compactness=10,
    #     key_pixel_mask=None, sigma=1e-3
    # )

        gs = torch.unsqueeze(gs, dim=0)
        num_iter += sample_per_draw
        nes_num += 1


        # ---------------------------------------------------噪声攻击-----------------------------------------------------
        proposed_adv_vid = adv_vid.clone()
        g = process_grad(gs, rein)
        g = g.transpose(1, 2)
        
        # 帧块空间采样
        mask_list_t = actions_test.view(-1)
        MASK = torch.zeros(adv_vid.size()).cuda()
        # b = mask.numpy()
        b = mask_test.numpy()
        c = b[mask_list_t]
        
        if c.ndim == 1:
            c = c[np.newaxis, :]  # 从 [4] → [1, 4]
        
        i = 0
        for x1, x2, y1, y2 in c:
            key = mask_list_t[i]
            x1 = torch.tensor(x1).to(torch.int64)
            x2 = torch.tensor(x2).to(torch.int64)
            y1 = torch.tensor(y1).to(torch.int64)
            y2 = torch.tensor(y2).to(torch.int64)
            MASK[:, :, key, y1:y2, x1:x2] = g[:, :, i, :, :]
            i = i + 1
        
        # MASK = MASK * fre_mask.float()
        
        proposed_adv_vid += cur_lr * MASK.cuda()
        clip_frame = update_adv_video(adv_vid, unnorm_video, MASK.cuda(), eps, cur_lr)
        
        with torch.no_grad():
            logits = vid_model(clip_frame)
        # logits = outputs.logits
        top_val, top_idx = logits.topk(1, 1, True, True)
        num_iter += 1

        if ori_class != top_idx[0][0]:
            adv_vid = clip_frame.clone()
            return True, num_iter, adv_vid

        adv_vid = clip_frame.detach()
        
        del clip_frame
        continue

    return False,  num_iter, adv_vid


# model_denpendcy

def data_denpendcy_attack( vid_model, vid, x0, ori_class,  args, model_keyframes, K,  eps=0.063,
                        max_lr=0.03, min_lr= 1e-3, effect_num=20, sample_per_draw=60,  sub_num_sample=2, max_iter=100):

    # ----------------------------------------------------初始化---------------------------------------------------------

    num_iter = 0
    T = 10
    cur_lr = max_lr
    input_frames = 32
    unnorm_video = transform_video(vid.clone().detach(), mode = 'back')
    nes_num = 0
    rein =True
    confidence_lsit = []
    adv_vid = vid.clone()
    logits = vid_model(adv_vid)
    top_val, top_idx = logits.topk(1, 1, True, True)
    num_iter += 1
    pre_confidence = top_val.view(-1).cpu()
    confidence_lsit.append(pre_confidence.cuda())
    
    # model_keyframes = [10, 28, 29]
    # K = 8
    
    # Step 1: 对每个 model_keyframe 使用 NES 攻击，计算 s_t^m 和 s_t^k
    sensitivity_scores = []  # 存储每帧的综合得分 s_t

    for t in range(input_frames):
        
        if t not in model_keyframes:
            continue  # 只处理 model_keyframes 中的帧
        
        adv_vid = vid.clone().detach()
        num_iter = 0
        
        while num_iter < max_iter:
            
            mask_test = torch.tensor([[0, 224, 0, 224]]).repeat(32, 1)
            target_frame = t  # 举例
            actions_test =  torch.tensor([target_frame], device='cuda', dtype=torch.long) 
            # actions_test = torch.arange(0, input_frames).reshape(input_frames, 1)
            patch_size_test = 224
            patch_size = args.patch_size
            
            # patch+frame
            gs = nes_patch_frame(vid_model, adv_vid, patch_size_test, mask_test, actions_test,  sample_per_draw, ori_class, sub_num_sample, sigma=1e-3)

            gs = torch.unsqueeze(gs, dim=0)
            num_iter += sample_per_draw
            nes_num += 1

            # ---------------------------------------------------噪声攻击-----------------------------------------------------
            proposed_adv_vid = adv_vid.clone()
            g = process_grad(gs, rein)
            g = g.transpose(1, 2)
            
            # 帧块空间采样
            mask_list_t = actions_test.view(-1)
            MASK = torch.zeros(adv_vid.size())
            # b = mask.numpy()
            b = mask_test.numpy()
            c = b[mask_list_t]
            
            if c.ndim == 1:
                c = c[np.newaxis, :]  # 从 [4] → [1, 4]
            
            i = 0
            for x1, x2, y1, y2 in c:
                key = mask_list_t[i]
                x1 = torch.tensor(x1).to(torch.int64)
                x2 = torch.tensor(x2).to(torch.int64)
                y1 = torch.tensor(y1).to(torch.int64)
                y2 = torch.tensor(y2).to(torch.int64)
                MASK[:, :, key, y1:y2, x1:x2] = g[:, :, i, :, :]
                i = i + 1

            proposed_adv_vid += cur_lr * MASK.cuda()
            
            clip_frame = update_adv_video(adv_vid, unnorm_video, MASK.cuda(), eps, cur_lr)

            with torch.no_grad():
                logits = vid_model(clip_frame)
            # logits = outputs.logits
            top_val, top_idx = logits.topk(1, 1, True, True)
            num_iter += 1

            adv_vid = clip_frame.detach()
            
            del clip_frame
            continue
        
        with torch.no_grad():
            logits_attacked = vid_model(adv_vid).cuda()
            logits_original = vid_model(vid).cuda()  # 原始输入的 logits

        # --- 提取真实类别 logits ---
        y = ori_class.squeeze().cuda()  # 真实类别
        z_orig = logits_original[:, y]  # z_y^(0)
        z_att = logits_attacked[:, y]   # z_y^(t)

        # --- 计算 m^0 和 m_t ---
        # max_other_orig = torch.max(logits_original[:, torch.arange(logits_original.size(1)) != y], dim=1)[0]
        # max_other_att = torch.max(logits_attacked[:, torch.arange(logits_attacked.size(1)) != y], dim=1)[0]
        
        device = logits_original.device
        indices_orig = torch.arange(logits_original.size(1), device=device)
        indices_att = torch.arange(logits_attacked.size(1), device=device)
        mask_orig = indices_orig != y
        mask_att = indices_att != y
        max_other_orig = torch.max(logits_original[:, mask_orig], dim=1)[0]
        max_other_att = torch.max(logits_attacked[:, mask_att], dim=1)[0]

        m0 = z_orig - max_other_orig  # m^0
        mt = z_att - max_other_att    # m_t

        sm = m0 - mt  # margin 下降

        # --- 计算 KL 敏感度 ---
        p = torch.zeros_like(logits_original[0])  # one-hot
        p[y] = 1.0
        q = F.softmax(logits_attacked[0], dim=0)
        sk = -torch.log(q[y])  # KL(p || q) = -log(q_y)

        # --- 综合得分 ---
        st = sm + sk
        sensitivity_scores.append((t, st.item()))
        
    # Step 2: 如果 model_keyframes 不足 K 帧，则从其余帧中随机填充
    if len(sensitivity_scores) < K:
        remaining_frames = [i for i in range(input_frames) if i not in [t for t, _ in sensitivity_scores]]
        need_fill = K - len(sensitivity_scores)
        filled_indices = resample(remaining_frames, replace=False, n_samples=need_fill)
        
        # 随机赋予一个较低的得分（如 0），避免影响排序
        for idx in filled_indices:
            sensitivity_scores.append((idx, 0.0))

    # Step 3: 按得分排序，取 Top-K
    sensitivity_scores.sort(key=lambda x: x[1], reverse=True)
    final_keyframes = [t for t, _ in sensitivity_scores[:K]]

    return final_keyframes


def model_denpendcy_attack(vid_model, vid, ori_class):
    
    logits = vid_model(vid)
    ori_class_logit = logits[0, ori_class].item()
    frames = vid.shape[2]
    mask_total_drop = torch.zeros(frames).to('cuda')
    channel_total_drop = torch.zeros(frames).to('cuda')
    alpha = 0.5
    
    for i in range(frames):
        
        masked_video = vid.clone().detach()
        masked_video = transform_video(masked_video, mode='back')
        masked_video[:, :, i, :, :] = 0.0
        masked_video = transform_video(masked_video, mode='forward')
        
        logits_masked = vid_model(masked_video)
        drop = max(0, ori_class_logit - logits_masked[0, ori_class].item())
        mask_total_drop[i] += drop
        
        perturbed_vid = vid.clone().detach()  # (1, 3, T, H, W)

        # 提取当前帧 (3, H, W)
        original_frame = perturbed_vid[0, :, i, :, :]  # (3, H, W)
        perm = torch.randperm(3).to('cuda')  # e.g., tensor([2, 0, 1])
        shuffled_frame = original_frame[perm, :, :]  # (3, H, W)
        mixed_frame = alpha * shuffled_frame + (1 - alpha) * original_frame
        perturbed_vid[0, :, i, :, :] = mixed_frame
        
        logits_channel = vid_model(perturbed_vid)
        drop = max(0, ori_class_logit - logits_channel[0, ori_class].item())
        channel_total_drop[i] = drop

    return mask_total_drop, channel_total_drop

    

# data dependency

def frequency_key_pixel_video(vid, top_k_ratio=0.1):
    """
    所有视频的前top-k像素
    
    Args:
        vid: 输入视频 (B, C, T, H, W)
        top_k_ratio: 保留 top-k% 的像素作为关键区域
    
    Returns:
        key_pixel_mask: 关键像素掩码 (B, C, T, H, W), bool 类型
        energy_map: 能量图 (T, H, W), float
    """
    co_s = 0.05
    co_t = 0.1

    # 获取时空高通滤波结果（保留高频关键信息）
    st_hpf_video = freqaug(
        vid, co_s, co_t,
        type_s=True,      # 高通
        type_t=True,      # 高通
        apply_spatial=True,
        apply_temporal=True
    )  # shape: (B, C, T, H, W)

    # 计算能量：对通道取 L2 范数，再对 batch 取平均
    energy = torch.norm(st_hpf_video, dim=1)  # (B, T, H, W)
    energy = energy.mean(dim=0)               # (T, H, W)

    # 展平并选 top-k 像素
    flat_energy = energy.view(-1)  # (T*H*W,)
    num_pixels = flat_energy.numel()
    k = int(top_k_ratio * num_pixels)
    
    if k == 0:
        k = 1

    _, top_indices = torch.topk(flat_energy, k, largest=True)
    
    # 创建掩码
    mask_flat = torch.zeros_like(flat_energy, dtype=torch.bool)
    mask_flat[top_indices] = True
    key_pixel_mask_3d = mask_flat.view_as(energy)  # (T, H, W)

    # 扩展为 (B, C, T, H, W)
    B, C, T, H, W = vid.shape
    key_pixel_mask = key_pixel_mask_3d[None, None, :, :, :].expand(B, C, -1, -1, -1)

    return key_pixel_mask, energy


def frequency_key_pixel_frame(vid, top_k_ratio=0.1, apply_spatial=True, apply_temporal=True):
    """
    每帧独立选择 top-k% 像素
    
    Args:
        vid: 输入视频 (B, C, T, H, W)
        top_k_ratio: 每帧保留 top-k% 的像素作为关键区域（0 < top_k_ratio <= 1）
    
    Returns:
        key_pixel_mask: 关键像素掩码 (B, C, T, H, W), bool 类型
        energy_map: 能量图 (B, T, H, W), float
    """
    co_s = 0.05
    co_t = 0.1

    # 获取时空高通滤波结果（保留高频关键信息）
    st_hpf_video = freqaug(
        vid, co_s, co_t,
        type_s=True,      # 空间高通
        type_t=True,      # 时间高通
        apply_spatial=apply_spatial,
        apply_temporal=apply_temporal
    )  # shape: (B, C, T, H, W)

    # 计算每帧的能量：L2 norm over channels → (B, T, H, W)
    energy = torch.norm(st_hpf_video, dim=1)  # (B, T, H, W)

    B, T, H, W = energy.shape
    key_pixel_mask_4d = torch.zeros_like(energy, dtype=torch.bool)  # (B, T, H, W)

    # 对每一帧独立处理
    for b in range(B):
        for t in range(T):
            frame_energy = energy[b, t]  # (H, W)
            flat_energy = frame_energy.view(-1)  # (H*W,)
            num_pixels = flat_energy.numel()
            k = max(1, int(top_k_ratio * num_pixels))  # 至少选1个像素

            _, top_indices = torch.topk(flat_energy, k, largest=True)

            mask_flat = torch.zeros_like(flat_energy, dtype=torch.bool)
            mask_flat[top_indices] = True
            key_pixel_mask_4d[b, t] = mask_flat.view(H, W)

    # 扩展到 (B, C, T, H, W)
    key_pixel_mask = key_pixel_mask_4d.unsqueeze(1).expand(-1, vid.size(1), -1, -1, -1)

    return key_pixel_mask, energy


def freqaug(
    x,
    co_s,
    co_t,
    type_s,
    type_t,
    apply_spatial=True,
    apply_temporal=True
):
    """
    Frequency-domain augmentation with optional spatial/temporal filtering.

    Args:
        x (torch.Tensor): Input video, shape (B, C, T, H, W)
        co_s (float): Spatial cutoff frequency (normalized, e.g., 0.1)
        co_t (float): Temporal cutoff frequency (normalized)
        type_s (bool): If True, spatial HPF; else LPF (ignored if apply_spatial=False)
        type_t (bool): If True, temporal HPF; else LPF (ignored if apply_temporal=False)
        apply_spatial (bool): Whether to apply spatial filtering
        apply_temporal (bool): Whether to apply temporal filtering

    Returns:
        torch.Tensor: Augmented or original video
    """

    B, C, T, H, W = x.shape
    device = x.device

    # Initialize full-pass masks (all True = keep all frequencies)
    pass_t = torch.ones(T, dtype=torch.bool, device=device)   # (T,)
    pass_h = torch.ones(H, dtype=torch.bool, device=device)   # (H,)
    pass_w = torch.ones(W, dtype=torch.bool, device=device)   # (W,)

    # --- Temporal filtering ---
    if apply_temporal:
        freq_t = torch.fft.fftfreq(T, d=1.0).to(device)
        pass_t = torch.abs(freq_t) < co_t
        if type_t:
            pass_t = torch.logical_not(pass_t)  # HPF

    # --- Spatial filtering ---
    if apply_spatial:
        freq_h = torch.fft.fftfreq(H, d=1.0).to(device)
        freq_w = torch.fft.fftfreq(W, d=1.0).to(device)
        pass_h = torch.abs(freq_h) < co_s
        pass_w = torch.abs(freq_w) < co_s

        pass_hw = torch.outer(pass_h, pass_w)  # (H, W)
        if type_s:
            pass_hw = torch.logical_not(pass_hw)  # HPF
    else:
        # If not applying spatial filter, create all-True mask
        pass_hw = torch.ones((H, W), dtype=torch.bool, device=device)

    # Build 3D filter mask: (T, H, W)
    filter_mask = pass_t[:, None, None] & pass_hw[None, :, :]  # broadcasting

    # Apply 3D FFT over (T, H, W)
    X = torch.fft.fftn(x, dim=(-3, -2, -1))  # (B, C, T, H, W)

    # Apply filter
    X_filtered = X * filter_mask[None, None, :, :, :]

    # Inverse FFT and take real part
    x_aug = torch.fft.ifftn(X_filtered, dim=(-3, -2, -1)).real

    return x_aug

def compute_model_key_frame(model_dependency, w, alpha):

    # 统计量
    mu = np.mean(model_dependency)
    sigma = np.std(model_dependency)
    threshold = mu + alpha * sigma

    print(f"μ = {mu:.4f}, σ = {sigma:.4f}, 阈值 = {threshold:.4f}")

    # 寻找局部最大值（仅用于判断是否为局部最大）
    def find_local_maxima(scores, w):
        local_max_indices = []
        N = len(scores)
        for t in range(w, N - w):
            window = scores[t-w:t+w+1]
            if scores[t] == np.max(window):
                local_max_indices.append(t)
        return local_max_indices

    # 找到所有满足任一条件的帧索引
    local_max_indices = find_local_maxima(model_dependency, w)
    keyframes = []

    for t in range(len(model_dependency)):
        is_local_max = (t in local_max_indices)
        is_significant = (model_dependency[t] > threshold)
        
        if is_local_max or is_significant:
            keyframes.append(t)

    print("关键帧索引（满足任一条件）:", keyframes)
    
    return keyframes 