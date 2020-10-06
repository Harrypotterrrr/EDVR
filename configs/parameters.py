import argparse


def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='edvr', choices=['edvr, defcnn'])
    parser.add_argument('--version', type=str, default='')

    # Network settings
    parser.add_argument('--nframes', type=int, default=5)
    parser.add_argument('--filter_num', type=int, default=128)
    parser.add_argument('--front_rb', type=int, default=5)
    parser.add_argument('--back_rb', type=int, default=40)
    parser.add_argument('--deform_groups', type=int, default=8)

    # Training setting
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=1e-3)

    # Learning rate scheduler
    parser.add_argument('--lr_schr', type=str, default='const', choices=['const', 'exp', 'step', 'multi', 'reduce'])
    parser.add_argument('--lr_exp_step', type=float, default=1000)
    parser.add_argument('--lr_exp_decay', type=float, default=0.95)
    parser.add_argument('--lr_boundary', default=[2000, 6000, 8000], nargs='+', type=float, help='lr constant decay boundary')
    parser.add_argument('--lr_boundary_value', default=[4e-4, 1e-4, 5e-5], nargs='+', type=float, help='lr constant value of decay boundary')

    # parser.add_argument('--total_sample', type=int, default=24000)

    # using pretrained
    parser.add_argument('--pretrained_model', action='store_true')

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('-g', '--gpus', default=['0', '1', '2', '3'], nargs='+', type=str, help='Specify GPU ids.')
    parser.add_argument('--dataset', type=str, default='reds', choices=['reds', 'vimeo'])
    parser.add_argument('--shuffle_ratio', type=int, default=20)
    parser.add_argument('--buffer_size', type=int, default=32) # TODO potential shuffle insufficiently bug
    parser.add_argument('--prefetch_buffer_size', type=int, default=32)

    # Path
    parser.add_argument('--root_path', type=str, default='/home/lz/Disk/jhl/REDS/')
    parser.add_argument('--tfrecord_path', type=str, default='./tfrecord')
    parser.add_argument('--model_save_path', type=str, default='./save')
    parser.add_argument('--log_path', type=str, default='./logs/')
    parser.add_argument('--log_train_path', type=str, default='train/')
    parser.add_argument('--log_val_path', type=str, default='val/')

    # Loop
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--val_step', type=int, default=20)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--log_block_size', type=int, default=100)
    parser.add_argument('--log_epoch', type=int, default=1)
    parser.add_argument('--log_sec', type=int, default=5)
    parser.add_argument('--model_save_step', type=int, default=500)

    # Dataloader
    parser.add_argument('--norm_value', type=int, default=255)
    parser.add_argument('--no_mean_norm', action='store_true', default=True)
    parser.add_argument('--std_norm', action='store_true', default=False)
    parser.add_argument('--mean_dataset', type=str, default='activitynet')
    parser.add_argument('--train_crop', type=str, default='corner')
    parser.add_argument('--sample_size', type=int, default=64)

    parser.add_argument('--initial_scale', type=float, default=1.0)
    parser.add_argument('--n_scales', type=int, default=5)
    parser.add_argument('--scale_step', type=float, default=0.84089641525)

    config = parser.parse_args()
    return vars(config) # convert into dict