import torch
from lreid.tools import MultiItemAverageMeter
from lreid.evaluation import accuracy
# from IPython import embed
from collections import OrderedDict
import copy
from torchvision.transforms import transforms
import numpy as np
import random

'''
GCReID full mathod
(meta-learning)
'''


def random_transform(imgs, flag):
    if flag == 0:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==1:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomRotation(degrees=30), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif flag==2:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomRotation(degrees=30),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))
        ]
    else:
        transform_train = [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    transform_train =transforms.Compose(transform_train)
    
    imgs_aug = torch.Tensor(np.zeros_like(imgs.cpu()))
    for i in range(len(imgs)):
        imgs_aug[i] = transform_train(imgs[i])

    return imgs_aug

def train_p_s_an_epoch(config, base, loader, current_step, old_model, old_graph_model, current_epoch=None, output_featuremaps=True):
    base.set_all_model_train()
    meter = MultiItemAverageMeter()
    if old_model is None:
        print('****** training tasknet ******\n')
    else:
        print('****** training both tasknet and metagraph ******\n')
    heatmaps_dict = {}
    for _ in range(config.steps):
        base.set_model_and_optimizer_zero_grad()
        mini_batch = loader.continual_train_iter_dict[current_step].next_one()

        while mini_batch[0].size(0) != config.p * 2*config.k:
            mini_batch = loader.continual_train_iter_dict[current_step].next_one()
        
       
        if len(mini_batch) > 6:
            assert config.continual_step == 'task'

        imgs, global_pids, global_cids, dataset_name, local_pids, image_paths = mini_batch

        index_a = np.zeros(config.p)
        index_a[:(config.p//2)]=1 
        index_a = np.r_[index_a, index_a] # lens=16 [1,1,1,1,0,0,0,0, 1,1,1,1,0,0,0,0]
        index_a = np.r_[index_a, index_a] # lens=32
        index_a = np.r_[index_a, index_a] # lens=64
        index_b = 1-index_a
        index_a = index_a.astype(bool) # convert to Ture and False
        index_b = index_b.astype(bool)
        
       
      
        imgs_1, local_pids_1, = imgs[index_a], local_pids[index_a]
        imgs_2, local_pids_2, = imgs[index_b], local_pids[index_b]   
        
        imgs_1, local_pids_1 = imgs_1.to(base.device), local_pids_1.to(base.device)
        imgs_2, local_pids_2 = imgs_2.to(base.device), local_pids_2.to(base.device)
       
        imgs, local_pids = imgs_1, local_pids_1
                 
        flag = random.choice([0,1,2])
        imgs_aug_1 = random_transform(imgs_1, flag)      
        imgs_aug_1 = imgs_aug_1.to(base.device)
        local_pids_aug_1 = local_pids_1

        index1 = np.random.randint(0, 2, config.p*config.k) # lens=32,每个number代表一行      
        index2 = 1-index1
        index1, index2 = index1.astype(bool), index2.astype(bool)
        
        # mix
        imgs_aug = torch.cat((imgs_aug_1[index1], imgs_2[index2]), dim=0)
        local_pids_aug = torch.cat((local_pids_aug_1[index1], local_pids_2[index2]), dim=0)
        
        
        # final augemented samples
        imgs_aug, local_pids_aug = imgs_aug.to(base.device), local_pids_aug.to(base.device)
        del mini_batch, imgs_1, imgs_2
        torch.cuda.empty_cache()

        loss = 0
        loss_aug = 0 
 
        meta_lr = 0.001

        if old_model is None:  
            features, cls_score, feature_maps = base.model_dict['tasknet'](imgs, current_step)
            feature_temp = features

            features_a, cls_score_a, feature_maps_a = base.model_dict['tasknet'](imgs_aug, current_step)
            
            features = torch.cat((features, features_a), 0) #将原始样本特征和增强后的样本特征 放一起
            protos, protos_a, protos_k, _ = base.model_dict['metagraph'](features.detach())
            feature_fuse = feature_temp + protos_k

            ide_loss = config.weight_x * base.ide_criterion(cls_score, local_pids) 
            plasticity_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse, local_pids, local_pids, local_pids)
            loss += ide_loss
            loss += plasticity_loss

            grads = torch.autograd.grad(loss, base.model_dict['tasknet'].parameters(), retain_graph=True, allow_unused=True)
            tasknet_tmp = copy.deepcopy(base.model_dict['tasknet'])

            # step1
            for param_tmp, param, grad in zip(tasknet_tmp.parameters(), base.model_dict['tasknet'].parameters(), grads):
                if grad is not None:
                    param_tmp.data.copy_((param_tmp.data -meta_lr * grad).data)
                    param_tmp.grad = grad
 

            # step2
            features_aug, cls_score_aug, feature_maps_aug  = tasknet_tmp(imgs_aug, current_step)
            
            # base_1
            ide_loss_aug = config.weight_x * base.ide_criterion(cls_score_aug, local_pids_aug)                 
            loss_aug += ide_loss_aug  

            features_aug_new = features_aug + protos_k

            plasticity_loss_aug = config.weight_t * base.triplet_criterion(features_aug_new, features_aug_new, features_aug_new, local_pids_aug, local_pids_aug, local_pids_aug)   

            loss_aug += plasticity_loss_aug


            meter.update({
                'ide_loss': ide_loss.data,
                'ide_loss_aug':ide_loss_aug.data,
                'plasticity_loss': plasticity_loss.data,
                'plasticity_loss_aug': plasticity_loss_aug.data,
            })

        else:
            old_current_step = list(range(current_step))
            new_current_step = list(range(current_step + 1))
            features, cls_score_list, feature_maps = base.model_dict['tasknet'](imgs, new_current_step)
            cls_score = cls_score_list[-1]
            feature_temp = features

            features_a, cls_score_list_a, feature_maps_a = base.model_dict['tasknet'](imgs_aug, new_current_step)

            features = torch.cat((features, features_a), 0) #mix features
            protos, protos_a, protos_k, correlation = base.model_dict['metagraph'](features.detach())
            
            feature_fuse = feature_temp + protos_k
            
            ide_loss = config.weight_x * base.ide_criterion(cls_score, local_pids)
            plasticity_loss = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse,
                                                                    local_pids, local_pids, local_pids)
            
            loss +=ide_loss
            loss += plasticity_loss

         
            with torch.no_grad():
                old_features, old_cls_score_list, old_feature_maps = old_model(imgs, old_current_step)
                old_features_aug, old_cls_score_list_aug, old_feature_maps_aug = old_model(imgs_aug, old_current_step)
                old_vertex = old_graph_model.meta_graph_vertex

            
            new_logit = torch.cat(cls_score_list, dim=1)
            old_logit = torch.cat(old_cls_score_list, dim=1) 

            # base_2 Ld
            knowladge_distilation_loss = config.weight_kd * base.loss_fn_kd(new_logit, old_logit, config.kd_T)                    
            loss += knowladge_distilation_loss

                
            grads = torch.autograd.grad(loss, base.model_dict['tasknet'].parameters(), retain_graph=True, allow_unused=True)

            tasknet_tmp = copy.deepcopy(base.model_dict['tasknet'])
            for param_tmp, param, grad in zip(tasknet_tmp.parameters(), base.model_dict['tasknet'].parameters(), grads):
                if grad is not None:
                    param_tmp.data.copy_((param_tmp.data - meta_lr * grad).data)
                    param_tmp.grad = grad
            
            features_aug, cls_score_list_aug, feature_maps_aug = tasknet_tmp(imgs_aug, new_current_step)
                        
            cls_score_aug = cls_score_list_aug[-1]

            new_logit_aug = torch.cat(cls_score_list_aug, dim=1)
            knowladge_distilation_loss_aug = config.weight_kd * base.loss_fn_kd(new_logit_aug, old_logit, config.kd_T)                    
            loss_aug += knowladge_distilation_loss_aug

            feature_fuse = features_aug + protos_k
            # base_1
            ide_loss_aug = config.weight_x * base.ide_criterion(cls_score_aug, local_pids_aug)
            
            plasticity_loss_aug = config.weight_t * base.triplet_criterion(feature_fuse, feature_fuse, feature_fuse,
                                                                    local_pids_aug, local_pids_aug, local_pids_aug)

             # base_2 Ld
            knowladge_distilation_loss = config.weight_kd * base.loss_fn_kd(new_logit, old_logit, config.kd_T)                    
            loss += knowladge_distilation_loss

            # base_2 Ls
            stability_loss = config.weight_r * base.model_dict['metagraph'].StabilityLoss(old_vertex, base.model_dict['metagraph'].meta_graph_vertex)
            loss_aug += stability_loss
            
            loss_aug += ide_loss_aug
            loss_aug += plasticity_loss_aug
               
            meter.update({
                'ide_loss': ide_loss.data,
                'ide_loss_aug':ide_loss_aug.data,
                'plasticity_loss': plasticity_loss.data,
                'plasticity_loss_aug': plasticity_loss_aug.data,
            })


        # acc = accuracy(cls_score, local_pids, [1])[0]


        total_loss = loss + loss_aug 

        ### optimize
        base.optimizer_dict['tasknet'].zero_grad()
        base.optimizer_dict['metagraph'].zero_grad()
        if config.fp_16:  # we use optimier to backward loss
            with base.amp.scale_loss(total_loss, base.optimizer_list) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
            

        base.optimizer_dict['tasknet'].step() 
        base.optimizer_dict['metagraph'].step() 

    if config.re_init_lr_scheduler_per_step:
        _lr_scheduler_step = current_epoch
    else:
        _lr_scheduler_step = current_step * config.total_train_epochs + current_epoch
    base.lr_scheduler_dict['tasknet'].step(_lr_scheduler_step)
    base.lr_scheduler_dict['metagraph'].step(_lr_scheduler_step)

    if output_featuremaps and not config.output_featuremaps_from_fixed:
        heatmaps_dict['feature_maps_true'] = base.featuremaps2heatmaps(imgs.detach().cpu(), feature_maps.detach().cpu(),
                                                                       image_paths,
                                                                       current_epoch,
                                                                       if_save=config.save_heatmaps,
                                                                       if_fixed=False,
                                                                       if_fake=False
                                                                       )
        return (meter.get_value_dict(), meter.get_str(), heatmaps_dict)
    else:
        return (meter.get_value_dict(), meter.get_str())