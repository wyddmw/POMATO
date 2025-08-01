import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from dust3r.POMATO import AsymmetricMASt3R as PairwiseModel
from dust3r.POMATO_temp import AsymmetricMASt3R as TemporalModel
from dust3r.tracking import tracking

def load_model(args):
    if 'pairwise' in args.model_type:
        model = PairwiseModel.from_pretrained(args.weights).to(args.device)
    else:
        model = TemporalModel.from_pretrained(args.weights).to(args.device)
    
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking Evaluation Script')
    parser.add_argument('--model_type', type=str, required=True, help='Type of model to load')
    parser.add_argument('--weights', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device to load the model on')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--overlap', type=int, default=3)
    
    args = parser.parse_args()

    model = load_model(args)
    
    tracking(model, args)

   