import argparse


def get_train_config():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str,
                        help='Directory path to a batch of content images',
                        required=False, default="./content")
    parser.add_argument('--style_dir', type=str,
                        help='Directory path to a batch of style images',
                        required=False, default="./style")
    parser.add_argument('--vgg', type=str, required=False, default="./vgg_normalised.pth",
                        help='Path to the VGG model')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='Directory to save the intermediate samples')
    parser.add_argument('--quality', type=str, default="mid", choices=["low", "mid", "high"])

    # training options
    parser.add_argument('--save_dir', default='./exp',
                        help='Directory to save the models')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--style_weight', type=float, default=3.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--SSC_weight', type=float, default=3.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--resume', action='store_true', help='train the model from the checkpoint')
    parser.add_argument('--checkpoints', default='./checkpoints',
                        help='Directory to save the checkpoint')
    args = parser.parse_args()


    return args


def get_compile_config():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, required=False, default="./example_content.jpg")
    parser.add_argument('--style', type=str, required=False, default="./example_style.jpg")
    parser.add_argument('--pretrained_folder', type=str, required=False, default="./pretrained")
    parser.add_argument('--quality', type=str, default="mid", choices=["low", "mid", "high"])

    args = parser.parse_args()

    return args
