import argparse
import csv
import io
import json
import os
import random
import time
import urllib.parse
import urllib.request
import gc
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOW = ROOT / "workflows" / "wallpaper_api.json"
DEFAULT_PRESETS = ROOT / "prompts" / "wallpaper_presets.json"
DEFAULT_OUTPUT = ROOT / "output" / "wallpapers"
QUEUE_PATH = ROOT / "output" / "upload_queue.csv"

RESOLUTION_PRESETS = {
    "desktop-4k": ("1024x576", "3840x2160"),
    "desktop-8k": ("1024x576", "7680x4320"),
    "phone-4k": ("576x1024", "2160x3840"),
    "phone-qhd": ("576x1024", "1440x2560"),
}


def load_env_file(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_node(workflow, class_type):
    matches = [node_id for node_id, node in workflow.items() if node.get("class_type") == class_type]
    if not matches:
        raise ValueError(f"Workflow is missing a {class_type} node")
    return matches[0]


def aspect_prompt(width, height):
    if height > width:
        return "designed as a 9:16 vertical phone wallpaper, lock screen composition, final upscale for mobile"
    if width > height:
        return "designed as a 16:9 desktop wallpaper, final upscale to 4K"
    return "designed as a square wallpaper, final upscale for high resolution display"


def set_workflow_inputs(workflow, theme, subject, style, negative, width, height, seed, checkpoint):
    workflow = deepcopy(workflow)
    checkpoint_id = find_node(workflow, "CheckpointLoaderSimple")
    latent_id = find_node(workflow, "EmptyLatentImage")
    sampler_id = find_node(workflow, "KSampler")
    save_id = find_node(workflow, "SaveImage")
    text_nodes = [node_id for node_id, node in workflow.items() if node.get("class_type") == "CLIPTextEncode"]
    if len(text_nodes) < 2:
        raise ValueError("Workflow needs positive and negative CLIPTextEncode nodes")

    positive_id, negative_id = text_nodes[0], text_nodes[1]
    prompt = f"{subject}, {style}, {aspect_prompt(width, height)}"

    workflow[checkpoint_id]["inputs"]["ckpt_name"] = checkpoint
    workflow[latent_id]["inputs"]["width"] = width
    workflow[latent_id]["inputs"]["height"] = height
    workflow[latent_id]["inputs"]["batch_size"] = 1
    workflow[sampler_id]["inputs"]["seed"] = seed
    workflow[positive_id]["inputs"]["text"] = prompt
    workflow[negative_id]["inputs"]["text"] = negative
    workflow[save_id]["inputs"]["filename_prefix"] = f"wallpapers/raw/{theme}/{width}x{height}/wallpaper"

    return workflow, prompt


def request_json(url, payload=None):
    if payload is None:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def queue_prompt(server, workflow):
    return request_json(f"http://{server}/prompt", {"prompt": workflow})["prompt_id"]


def wait_for_history(server, prompt_id, timeout_seconds):
    deadline = time.time() + timeout_seconds
    url = f"http://{server}/history/{prompt_id}"
    while time.time() < deadline:
        history = request_json(url)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def download_image(server, image_ref):
    params = urllib.parse.urlencode(
        {
            "filename": image_ref["filename"],
            "subfolder": image_ref["subfolder"],
            "type": image_ref["type"],
        }
    )
    with urllib.request.urlopen(f"http://{server}/view?{params}") as response:
        return response.read()


def save_final_image(image_bytes, output_path, target_size=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    image.save(output_path, "PNG", optimize=True)
    return image.size


def resolve_realesrgan_weights(weights):
    path = Path(weights)
    candidates = [
        path,
        ROOT / weights,
        ROOT / "models" / "upscale_models" / weights,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find RealESRGAN weights {weights!r}. Put the .pth file in "
        f"{ROOT / 'models' / 'upscale_models'} or pass --realesrgan-weights."
    )


def upscale_with_realesrgan(
    image_bytes,
    output_path,
    target_size=None,
    scale=4,
    weights="RealESRGAN_x4plus.pth",
    tile=128,
):
    # basicsr 1.4.2 expects an older torchvision module path.
    import sys
    import types

    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms.functional as torchvision_functional

            functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
            functional_tensor.rgb_to_grayscale = torchvision_functional.rgb_to_grayscale
            sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
        except ImportError:
            pass

    try:
        import numpy as np
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as exc:
        raise RuntimeError(
            "RealESRGAN is not installed for this Python. Use --upscale-engine pillow, "
            "or install it in the same Python that runs this script."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path = resolve_realesrgan_weights(weights)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=str(weights_path),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    try:
        output, _ = upsampler.enhance(np.array(image), outscale=scale)
        image = Image.fromarray(output)
        if target_size and image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        image.save(output_path, "PNG", optimize=True)
        return image.size
    finally:
        del upsampler
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_metadata(path, metadata):
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def append_queue_row(row):
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = QUEUE_PATH.exists()
    with QUEUE_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "created_at",
                "status",
                "theme",
                "resolution",
                "file_path",
                "caption",
                "prompt",
                "seed",
                "model",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_size(value):
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def main():
    load_env_file()

    parser = argparse.ArgumentParser(description="Batch-generate organized ComfyUI wallpapers.")
    parser.add_argument("--server", default=os.getenv("COMFYUI_SERVER", "127.0.0.1:8188"), help="ComfyUI server address.")
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--presets", type=Path, default=DEFAULT_PRESETS)
    parser.add_argument("--theme", default="all", help="Theme name from presets, or all.")
    parser.add_argument("--count", type=int, default=1, help="Images per selected theme.")
    parser.add_argument(
        "--resolution-preset",
        choices=sorted(RESOLUTION_PRESETS),
        default=None,
        help="Shortcut for source and target sizes.",
    )
    parser.add_argument("--source-size", default="1024x576", help="ComfyUI generation size.")
    parser.add_argument("--target-size", default="3840x2160", help="Final resized output size. Use none to skip.")
    parser.add_argument("--upscale-engine", choices=["pillow", "realesrgan"], default=os.getenv("WALLPAPER_UPSCALE_ENGINE", "pillow"))
    parser.add_argument("--realesrgan-weights", default=os.getenv("REALESRGAN_WEIGHTS", "RealESRGAN_x4plus.pth"))
    parser.add_argument("--realesrgan-scale", type=int, default=int(os.getenv("REALESRGAN_SCALE", "4")))
    parser.add_argument("--realesrgan-tile", type=int, default=int(os.getenv("REALESRGAN_TILE", "128")), help="Use 0 for no tiling.")
    parser.add_argument("--checkpoint", default=os.getenv("WALLPAPER_CHECKPOINT", "dreamshaper_8.safetensors"))
    parser.add_argument("--seed", type=int, default=None, help="Base seed. Defaults to random.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned prompts without queueing.")
    args = parser.parse_args()

    workflow = load_json(args.workflow)
    presets = load_json(args.presets)
    if args.resolution_preset:
        args.source_size, args.target_size = RESOLUTION_PRESETS[args.resolution_preset]
    source_width, source_height = parse_size(args.source_size)
    target_size = None if args.target_size.lower() == "none" else parse_size(args.target_size)
    selected_styles = [
        style for style in presets["styles"] if args.theme == "all" or style["theme"] == args.theme
    ]
    if not selected_styles:
        raise ValueError(f"No preset theme matched {args.theme!r}")

    for style in selected_styles:
        for index in range(args.count):
            seed = args.seed + index if args.seed is not None else random.randint(1, 2**31 - 1)
            subject = style["subjects"][index % len(style["subjects"])]
            run_workflow, prompt = set_workflow_inputs(
                workflow,
                style["theme"],
                subject,
                style["style"],
                presets["negative_prompt"],
                source_width,
                source_height,
                seed,
                args.checkpoint,
            )
            caption = f"{style['caption']} {style['tags']}"
            print(f"[{style['theme']}] seed={seed} subject={subject}")
            if args.dry_run:
                print(prompt)
                continue

            prompt_id = queue_prompt(args.server, run_workflow)
            history = wait_for_history(args.server, prompt_id, timeout_seconds=900)
            image_refs = [
                image
                for output in history.get("outputs", {}).values()
                for image in output.get("images", [])
            ]
            if not image_refs:
                raise RuntimeError(f"Prompt {prompt_id} completed without image outputs")

            for image_index, image_ref in enumerate(image_refs, start=1):
                image_bytes = download_image(args.server, image_ref)
                resolution = f"{target_size[0]}x{target_size[1]}" if target_size else args.source_size
                name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}_{image_index}.png"
                final_path = DEFAULT_OUTPUT / "final" / style["theme"] / resolution / name
                if args.upscale_engine == "realesrgan":
                    final_size = upscale_with_realesrgan(
                        image_bytes,
                        final_path,
                        target_size=target_size,
                        scale=args.realesrgan_scale,
                        weights=args.realesrgan_weights,
                        tile=args.realesrgan_tile,
                    )
                    upscale_note = f"Final upscale uses RealESRGAN scale {args.realesrgan_scale}, tile {args.realesrgan_tile}."
                else:
                    final_size = save_final_image(image_bytes, final_path, target_size=target_size)
                    upscale_note = "Final resize uses Pillow Lanczos. Add an ESRGAN/RealESRGAN model for AI upscaling."
                metadata = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "theme": style["theme"],
                    "subject": subject,
                    "prompt": prompt,
                    "negative_prompt": presets["negative_prompt"],
                    "seed": seed,
                    "checkpoint": args.checkpoint,
                    "source_size": f"{source_width}x{source_height}",
                    "final_size": f"{final_size[0]}x{final_size[1]}",
                    "upscale_engine": args.upscale_engine,
                    "caption": caption,
                    "source_comfy_image": image_ref,
                    "note": upscale_note,
                }
                write_metadata(final_path, metadata)
                append_queue_row(
                    {
                        "created_at": metadata["created_at"],
                        "status": "ready",
                        "theme": style["theme"],
                        "resolution": metadata["final_size"],
                        "file_path": str(final_path),
                        "caption": caption,
                        "prompt": prompt,
                        "seed": seed,
                        "model": args.checkpoint,
                    }
                )
                print(f"saved {final_path}")


if __name__ == "__main__":
    main()
