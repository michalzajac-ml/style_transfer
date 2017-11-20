import argparse
import numpy as np
import scipy

from model import StyleTransfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Style transfer')
    parser.add_argument('--content', type=str, required=True, help='Content file')
    parser.add_argument('--style', type=str, required=True, help='Style file')
    parser.add_argument('--output', type=str, default='output.png', help='Output file')
    parser.add_argument('--gif', type=str, help='Save gif with history of creating the image')
    parser.add_argument('--iters', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--save-every-n', type=int, help='Save intermediate result every n iterations')
    parser.add_argument('--style-weight', type=float, default=100.0,
                        help='Weight of the style part in loss function, relative to the content part')
    parser.add_argument('--denoising-weight', type=float, default=50.0,
                        help='Weight of the denoising part in loss function, relative to the content part')
    parser.add_argument('--lr', type=float, default=2.0, help='Learning rate')
    args = parser.parse_args()

    content = scipy.misc.imread(args.content).astype(np.float32)
    style = scipy.misc.imread(args.style).astype(np.float32)

    style_transfer = StyleTransfer(
        content=content,
        style=style,
        style_weight=args.style_weight,
        denoising_weight=args.denoising_weight,
        learning_rate=args.lr
    )
    style_transfer.generate_image(
        num_steps=args.iters,
        output_file=args.output,
        gif_file=args.gif,
        save_every_n=args.save_every_n
    )
