from delivery import data_delivery
from torchvision import transforms
from loaders import CA_Loader
from mb_models import mb12_build_model
import torch, argparse
from torch.utils.data import DataLoader
import torch.nn as nn 
from trainings import dict_training_multi_branch
from torchreid import models, utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('calculation is on:',device)
torch.cuda.empty_cache()

def default_argument_parser():
    
    parser = argparse.ArgumentParser(description="SI Runner")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--feat-select", action="store_false", help="perform feature selection")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="learning rate")
    parser.add_argument("--num-epoch", type=int, default=30, help="epochs")
    parser.add_argument("--ver", default="V_1_mb12_featselect", help="version")
    parser.add_argument("--save-path", default="./results/", help="saving path")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    return parser

#batch_size = 32 #save_path = './results/' 'V_1_mb12_featselect' 30 #lr = 3.5e-4
#%%

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    main_path = './Market-1501-v15.09.15/gt_bbox/'
    path_attr = './attributes/new_total_attr.npy'

    attr = data_delivery(main_path,
                    path_attr=path_attr,
                    path_start=None,
                    only_id=False,
                    double = False,
                    need_collection=True,
                    need_attr=True,
                    mode = 'CA_Market')

    train_idx_path = './attributes/train_idx_full.pth' 
    test_idx_path = './attributes/test_idx_full.pth'
    train_idx = torch.load(train_idx_path)
    test_idx = torch.load(test_idx_path)

    for key , value in attr.items():
        try: print(key , 'size is: \t {} \n'.format((value.size())))
        except TypeError:
            print(key,'\n')
    #%%    
    train_transform = transforms.Compose([
                                transforms.RandomRotation(degrees=10),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(saturation=[1,3])
                                ])

    train_data = CA_Loader(img_path=main_path,
                            attr=attr,
                            resolution=(256,128),
                            transform=train_transform,
                            indexes=train_idx,
                            need_attr =False,
                            need_collection=True,
                            need_id = False,
                            two_transforms = False)

    test_data = CA_Loader(img_path=main_path,
                            attr=attr,
                            resolution=(256, 128),
                            indexes=test_idx,
                            need_attr = False,
                            need_collection=True,
                            need_id = False,
                            two_transforms = False,                          
                            ) 

    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=100,shuffle=False)
    #%%
    torch.cuda.empty_cache()

    model = models.build_model(
        name='osnet_x1_0',
        num_classes=751,
        loss='softmax',
        pretrained=False
    )

    weight_path = './osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

    utils.load_pretrained_weights(model, weight_path)

    def feat_selection_loader(nth):
        import numpy as np
        from functions import layers_num_corrector
        filenames = ['head', 'head_colour', 'body', 'body_type', 'body_colour', 'leg', 'leg_colour', 
                        'foot', 'foot_colour', 'bags', 'age', 'gender']
        feat_indices =[torch.from_numpy(np.load('./FS_results/layers'+fname+'.npy')[:nth]).to(device) for fname in filenames]
        final = []
        for i in range(len(feat_indices)):
            final.append(layers_num_corrector(feat_indices[i]))
        return final

    if args.feat_select == True: feat_indices = feat_selection_loader(nth=25)
    else: feat_indices = None

    # sep_fc = True and sep_clf = False is not possible

    attr_net = mb12_build_model(model = model,
                    main_cov_size = 512,
                    attr_dim = 64,
                    dropout_p = 0.3,
                    sep_conv_size = 128,
                    feature_selection=feat_indices)


    attr_net = attr_net.to(device)

    def get_n_params(model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    i = 0
    for child in attr_net.children():
        i += 1
        # print(i)
        # print(child)
        if i == 15:
        #     # print(child)
            a = get_n_params(child)
            print(child)

    a = get_n_params(attr_net)      

    #%%
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.BCEWithLogitsLoss()

    params = attr_net.parameters()

    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-08)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 17], gamma=0.1)
    #%%

    dict_training_multi_branch(num_epoch = args.num_epoch,
                        attr_net = attr_net,
                        train_loader = train_loader,
                        test_loader = test_loader,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        cce_loss = criterion1,
                        bce_loss = criterion2,
                        save_path = args.save_path,                    
                        device = device,
                        version = args.ver,
                        resume=False,
                        loss_train = None,
                        loss_test=None,
                        train_attr_F1=None,
                        test_attr_F1=None,
                        train_attr_acc=None,
                        test_attr_acc=None,  
                        stoped_epoch=None)
