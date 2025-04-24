import argparse



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        help='gpu device.', default='cuda')
    parser.add_argument('--project_name', type=str,
                        help='project_name.', default='new_project')    
    
    # about model
    parser.add_argument('--layer_num', type=int,
                        help='layers of model', default=24)
    parser.add_argument('--d_state', type=int,
                        help='state size of mamba', default=512)
    parser.add_argument("-c", "--continue", action="store_true")
    
    # about training
    parser.add_argument('--batch', type=int,
                        help='batch size', default=4)
    parser.add_argument('--accumulation_step', type=int,
                        help='accumulation_step', default=4)

    # parser.add_argument('--ckpt', type=int,
    #                     help='ckpt epoch.', default=50)
    args = parser.parse_args()
    return args

opt = parse_opt()
print(opt)