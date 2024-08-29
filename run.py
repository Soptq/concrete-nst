import os
import time

import torch
from torchvision import transforms
from torchvision.utils import save_image
import torch.utils.data as data

from concrete.ml.torch.compile import compile_brevitas_qat_model
from concrete.fhe import Configuration

from config import get_compile_config
from sampler import InfiniteSamplerWrapper

import quant_model as net
from PIL import Image

from tqdm import tqdm


def find_pretrained_files(folder, quality):
    files = {"content_encoder": None, "style_encoder": None, "modulator": None, "decoder": None}
    for file in os.listdir(folder):
        if f"content_encoder_{quality}" in file:
            files["content_encoder"] = os.path.join(folder, file)
        elif f"style_encoder_{quality}" in file:
            files["style_encoder"] = os.path.join(folder, file)
        elif f"modulator_{quality}" in file:
            files["modulator"] = os.path.join(folder, file)
        elif f"decoder_{quality}" in file:
            files["decoder"] = os.path.join(folder, file)
    for k, v in files.items():
        if v is None:
            raise FileNotFoundError(f"Cannot find {k} in {folder}")
    return files


if __name__ == '__main__':
    args = get_compile_config()

    if args.quality == "high":
        image_size = 192
    elif args.quality == "mid":
        image_size = 128
    elif args.quality == "low":
        image_size = 64
    else:
        raise ValueError(f"Unknown quality size: {args.quality}")

    def compile_transform():
        transform_list = [
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

    content_encoder = net.Encoder()
    style_encoder = net.Encoder()
    modulator = net.Modulator()
    decoder = net.Decoder()

    content_encoder.eval()
    style_encoder.eval()
    modulator.eval()
    decoder.eval()

    pretrained_files = find_pretrained_files(args.pretrained_folder, args.quality)
    content_encoder.load_state_dict(torch.load(pretrained_files["content_encoder"]), strict=False)
    style_encoder.load_state_dict(torch.load(pretrained_files["style_encoder"]), strict=False)
    modulator.load_state_dict(torch.load(pretrained_files["modulator"]), strict=True)
    decoder.load_state_dict(torch.load(pretrained_files["decoder"]), strict=True)
    print(f"Successfully loaded pretrained models from {pretrained_files}")
    network = net.TestNet(content_encoder, style_encoder, modulator, decoder)

    print("Preparing compiling inputs...")
    tf = compile_transform()
    content_image = Image.open(str(args.content)).convert('RGB')
    style_image = Image.open(str(args.style)).convert('RGB')
    content_input = tf(content_image).unsqueeze(0)
    style_input = tf(style_image).unsqueeze(0)

    print("Running the compiled model in clear mode...")
    t_begin = time.time()
    with torch.no_grad():
        stylized_results = network(content_input, style_input)

    visualized_imgs = torch.cat([content_input, style_input, stylized_results])
    save_image(visualized_imgs, "output_clear.jpg", nrow=3)
    t_end = time.time()
    print(f"Successfully run the model in clear mode in {t_end - t_begin:.2f}s")
    print("Check the output_clear.jpg file for the result")

    print("Compiling...")
    t_begin = time.time()
    config = Configuration(
        enable_tlu_fusing=True,
        print_tlu_fusing=False,
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location="~/.cml_keycache",
        show_progress=True,
        use_gpu=False
    )
    quantized_network = compile_brevitas_qat_model(
        network,
        (content_input, style_input),
        rounding_threshold_bits={"n_bits": 8, "method": "approximate"},
        configuration=config,
        verbose=True
    )
    t_end = time.time()
    print(f"Successfully compiled the model in {t_end - t_begin:.2f}s")
    print("maximum_integer_bit_width: ", quantized_network.fhe_circuit.graph.maximum_integer_bit_width())
    print("statistics: ", quantized_network.fhe_circuit.statistics)
    print("Running the compiled model in FHE mode...")
    t_begin = time.time()
    with torch.no_grad():
        stylized_results = quantized_network.forward(content_input.numpy(), style_input.numpy(), fhe="execute")
    visualized_imgs = torch.cat([content_input, style_input, torch.tensor(stylized_results)])
    save_image(visualized_imgs, "output_fhe.jpg", nrow=3)
    t_end = time.time()
    print(f"Successfully run the model in FHE mode in {t_end - t_begin:.2f}s")
    print("Check the output_fhe.jpg file for the result")
