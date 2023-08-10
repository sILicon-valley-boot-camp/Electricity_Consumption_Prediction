import argparse

def args_for_main(parser):
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--window_size', type=int, default=24, help='window size')
    parser.add_argument('--seed', type=int, default=42, help='seed for random')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')

def args_for_train(parser):
    parser.add_argument('--data_path', type=str, default='data/new_train.csv', help='data path')
    parser.add_argument('--info_path', type=str, default='data/new_building_info.csv', help='info path')
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for the optimizer')

    parser.add_argument('--input_dim', type=int, default=19, help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')

def args_for_test(parser):
    parser.add_argument('--data_path', type=str, default='data/new_test.csv', help='data path')
    parser.add_argument('--info_path', type=str, default='data/new_building_info.csv', help='info path')
    parser.add_argument('--model_path', type=str, default='model.pt', help='model path')

    parser.add_argument('--input_dim', type=int, default=19, help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')

def get_args():
    parser = argparse.ArgumentParser()
    args_for_main(parser)

    if parser.parse_args().mode == 'train':
        args_for_train(parser)
    else:
        args_for_test(parser)

    args = parser.parse_args()
    return args
