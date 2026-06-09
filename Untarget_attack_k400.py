from Utils.utils import *
from configs.config import*
from model_wrapper.vid_model_top_k import C3D_K_Model, SLOWFAST_K_Model, TSN_K_Model, TSM_K_Model, C3D_K_Model_k400
import numpy as np
# from attack.attackZERO_test import untargeted_video_attack
from attack.adapterAttack import untargeted_video_attack, model_denpendcy_attack, data_denpendcy_attack, compute_model_key_frame

from utils import *
from gluoncv.torch.model_zoo import get_model


def main(args):

        args.model_name = 'i3d_resnet101'
        args.dataset_name = 'k400'
        model_name = args.model_name
        dataset_name = args.dataset_name
        gpus = args.tt_gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])
        print('load {} dataset'.format(dataset_name))
        print('load {} model'.format(model_name))
        # model = generate_model(model_name, dataset_name)
        print('Initialize model')
        attacked_ids = list(range(400))
            
        ### tt_test
        cfg_path = CONFIG_PATHS[model_name]
        cfg = get_cfg_custom(cfg_path, args.batch_size)
        
        vid_model = get_model(cfg).cuda()
        vid_model.eval()


        def GetPairs_ori( idx):
            x0 = torch.from_numpy(np.load('numpy_video/{}.npy'.format(idx)))
            #x0 = image_to_vector(model_name, dataset_name, x0)  # 将视频归一化在[0,1]之间

            if x0.size(0) == 3:
                x0 = x0.transpose(1, 0)
            return x0.cuda(), x0

        def GetPairs_ori_tt_ucf_test( idx):
            x0 = torch.from_numpy(np.load('/home/pangbo/reproduction/TT-master/output/nonlocal101_kinetics/Kinetics-i3d_resnet101-BIM_SBS-False-20-tb_bs20/{}-ori.npy'.format(idx)))
            # x0 = image_to_vector(model_name, dataset_name, x0)  # 将视频归一化在[0,1]之间
            x0 = torch.unsqueeze(x0, dim=0)
        
            if x0.size(0) == 3:
                x0 = x0.transpose(1, 0)
            return x0.cuda(), x0

        method = str('vis_k8')
        result_path =args.adv_path
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        result_root = os.path.join(result_path,'ASTii/{}_{}_{}'.format(method, args.model_name, args.dataset_name))
        if not os.path.exists(result_root):
           os.makedirs(result_root)
           
        result_npy = os.path.join(result_root, 'npy')
        if not os.path.exists(result_npy):
            os.makedirs(result_npy)
        
        total_iternum_average = 0
        total_pertubation_average = 0
        NUM = 0
        success_num = 0
        metric_path = os.path.join(result_root, 'metric.txt')
        
        # model_dependency
        all_frame_model_score = torch.zeros(32).to('cuda')
        
        all_mask_score = torch.zeros(32).to('cuda')
        all_channel_score = torch.zeros(32).to('cuda')
        
        model_keyframes = list(range(32))
        
        # frequence 
        # for idx in range(0, 1):
            
        #     idx = 35
            
        #     # vid, x0 = GetPairs_ori(attacked_ids[idx])
        #     vid, x0 = GetPairs_ori_tt_ucf_test(attacked_ids[idx])
        #     ori_vid_batch = vid
        #     # top_val, label, logits = vid_model(ori_vid_batch[None, :])
        #     logits = vid_model(ori_vid_batch)
        #     top_val, label = logits.topk(1, 1, True, True)
            
        #     '--------------------Attack-----------------------'
        #     print('THE {}th Attacking.....'.format(attacked_ids[idx]))
            
        #     spatial_hpf_video, temporal_hpf_video, st_hpf_video = data_denpendency_attack_v1(vid_model, vid, label)
        #     np.save(os.path.join(result_npy, 's-adv'.format(label.item())), spatial_hpf_video)
        #     np.save(os.path.join(result_npy, 't-adv'.format(label.item())), temporal_hpf_video)
        #     np.save(os.path.join(result_npy, 'st-adv'.format(label.item())), st_hpf_video)
            
            
        # 模型依赖
        # for idx in range(0, len(attacked_ids)):
            
        #     vid, x0 = GetPairs_ori_tt_ucf_test(attacked_ids[idx])
        #     ori_vid_batch = vid
        #     logits = vid_model(ori_vid_batch)
        #     top_val, label = logits.topk(1, 1, True, True)
            
        #     '--------------------Attack-----------------------'
        #     print('THE {}th Attacking.....'.format(attacked_ids[idx]))
            
        #     mask_score, channel_score = model_denpendcy_attack(vid_model, vid, label)
        #     all_mask_score += mask_score
        #     all_channel_score += channel_score
        #     print('Current Average Mask{}'.format(all_mask_score / (idx + 1)))
        #     print('Current Average Channel{}'.format(all_channel_score / (idx + 1)))
        
        # model_dependency = all_channel_score / len(attacked_ids)
        # model_keyframes = compute_model_key_frame(model_dependency, w = 1, alpah = 1)

        for idx in range(0, len(attacked_ids)):
            # vid, x0 = GetPairs_ori(attacked_ids[idx])
            vid, x0 = GetPairs_ori_tt_ucf_test(attacked_ids[idx])
            ori_vid_batch = vid
            # top_val, label, logits = vid_model(ori_vid_batch[None, :])
            logits = vid_model(ori_vid_batch)
            top_val, label = logits.topk(1, 1, True, True)
            
            '--------------------Attack-----------------------'
            print('THE {}th Attacking.....'.format(attacked_ids[idx]))
            res, iter_num, adv_vid = untargeted_video_attack(vid_model, vid, x0, label, args, model_keyframes, K = 8, max_iter=args.max_iter)
            
            NUM += 1
            '--------------------complete-----------------------'
            # AP = pertubation(vid, adv_vid)
            AP = tt_pertubation(vid, adv_vid)
            print('The average pertubation of video is: {}'.format(AP.cpu()))
            total_pertubation_average += AP.cpu()
            if res:
                # 成功
                total_iternum_average += iter_num
                f = open(metric_path, 'a')
                f.write(str('----------------{}-------------------'.format(attacked_ids[idx])))
                f.write('\n')
                f.write(str(iter_num))
                f.write('\n')
                f.write(str(AP.cpu()*255))
                f.write('\n')
                f.close()
                print('untargeted attack succeed using {} quries'.format(iter_num))
                success_num += 1
            else:
                # 失败
                total_iternum_average += iter_num
                metric_path = os.path.join(result_root, 'metric.txt')
                f = open(metric_path, 'a')
                f.write(str('----------------{}-------------------'.format(attacked_ids[idx])))
                f.write('\n')
                f.write(str('Attack Fails'))
                f.write('\n')
                f.write(str(AP.cpu()*255))
                f.write('\n')
                f.close()
                
                print('--------------------Attack Fails-----------------------')
                
            adv_vis = adv_vid.cpu().numpy()
            np.save(os.path.join(result_npy, '{}-adv'.format(label.item())), adv_vis)

            print('total iternum  is {} '.format(total_iternum_average))
            print('total average is {} '.format(total_pertubation_average))
            print('fail number is {} '.format(NUM-success_num))

        total_iternum_averages = total_iternum_average / (NUM)
        total_pertubation_averages = total_pertubation_average / (NUM)
        print('total iternum average is {:.4} '.format(total_iternum_averages))
        print('total pertubation average is {:.4} '.format(total_pertubation_averages*255))
        print('total success rate  is {:.4} '.format(success_num*100 / NUM))
        f = open(metric_path, 'a')
        f.write('\n')
        f.write(str('********************** total results ***********************'))
        f.write('\n')
        f.write(str('total iternum average is {:.4} '.format(total_iternum_averages)))
        f.write('\n')
        f.write(str('total pertubation average is {:.4} '.format(total_pertubation_averages*255)))
        f.write('\n')

        f.write(str('success rate is {:.4} '.format(success_num*100 / NUM)))
        f.write('\n')



if __name__ == '__main__':
    args = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)


class Obj:
    def __init__(self, info):
        self.info = info
