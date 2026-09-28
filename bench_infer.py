import glob, torch
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
device = torch.device("cuda")
model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512").to(device).eval()
lit = MultiViewDUSt3RLitModule.load_for_inference(model); lit.eval()
files = sorted(glob.glob("bench_frames/*.jpg"))
images = load_images(files, size=512, verbose=False)
with torch.no_grad():
    out, prof = inference(images, model, device, dtype=torch.float32, verbose=False, profiling=True)
print("views", len(out["preds"]))
