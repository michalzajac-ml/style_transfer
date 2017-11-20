import imageio


# if Keras model has one output, it returns a tensor, otherwise a list of tensors
def make_list(x):
    if isinstance(x, list):
        return x
    return [x]


# auxiliary function to inverse ImageNet preprocessing
def postprocess(x):
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68

    x = x[..., ::-1]
    return x


def create_gif(images, output_file, duration=0.1):
    imageio.mimsave(output_file, images, duration=duration)
