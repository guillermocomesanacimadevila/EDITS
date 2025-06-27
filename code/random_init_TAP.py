import torch
from pdb import set_trace
import tarrow
from tarrow.models import TimeArrowNet


def reinitialize_weights(model):
    import torch.nn.init as init
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            # Only reinitialize if the weight has 2 or more dimensions
            if len(layer.weight.shape) >= 2:
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                # Handle 1D or fewer dimensions (e.g., BatchNorm, LayerNorm, etc.)
                init.normal_(layer.weight, mean=0.0, std=1.0)
        if hasattr(layer, 'bias') and layer.bias is not None:
            # Reinitialize bias (if it exists)
            init.zeros_(layer.bias)


def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--TAP_model_save_dir", type=str)
    parser.add_argument("--model_seed", type=int)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--model_load_dir")
    args = parser.parse_args()


    devicestring = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(devicestring)
    print("Running on "+devicestring)
    model_save_dir = os.path.join(args.TAP_model_save_dir, args.model_id)
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'{args.model_id}.pth')

    model = tarrow.models.TimeArrowNet.from_folder(model_folder=args.model_load_dir)
    model.to(device)

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    reinitialize_weights(model)

    torch.save(model.state_dict(), model_save_path)
    print(f'model saved to {model_save_path}')


if __name__ == '__main__':
    main()