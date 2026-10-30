#!/usr/bin/env python3
"""Local orbit video -> COLMAP -> optional Brush Gaussian splat training."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

def run(args, dry=False):
    print("+", " ".join(map(str,args)), flush=True)
    if not dry:
        subprocess.run(list(map(str,args)), check=True)

def require(name):
    exe=shutil.which(name)
    if not exe:
        raise RuntimeError(f"Missing executable: {name}. Install it or add it to PATH.")
    return exe

def duration_of(video):
    result=subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration",
         "-of","default=noprint_wrappers=1:nokey=1",str(video)],
        capture_output=True,text=True,check=True)
    duration=float(result.stdout.strip())
    if not math.isfinite(duration) or duration<=0:
        raise RuntimeError("Could not determine positive video duration")
    return duration

def main():
    p=argparse.ArgumentParser(description="Extract orbit frames, run CPU COLMAP, optionally train with Brush.")
    p.add_argument("video",type=Path,help="Input orbit video")
    p.add_argument("--output",type=Path,default=None,help="Project directory (default: ./splats/<video-stem>)")
    p.add_argument("--frames",type=int,default=96,help="Target uniformly sampled frames")
    p.add_argument("--resolution",type=int,default=1280,help="Max extracted image dimension")
    p.add_argument("--matcher",choices=["exhaustive","sequential"],default="exhaustive")
    p.add_argument("--camera-model",choices=["SIMPLE_RADIAL","PINHOLE","OPENCV"],default="SIMPLE_RADIAL")
    p.add_argument("--skip-colmap",action="store_true",help="Reuse COLMAP sparse/0")
    p.add_argument("--skip-extract",action="store_true",help="Reuse images")
    p.add_argument("--train",choices=["none","brush"],default="none")
    p.add_argument("--brush-bin",default="brush",help="Path to local Brush executable")
    p.add_argument("--brush-viewer",action="store_true")
    p.add_argument("--brush-iters",type=int,default=10000)
    p.add_argument("--brush-resolution",type=int,default=1280)
    p.add_argument("--brush-max-splats",type=int,default=1000000)
    p.add_argument("--brush-sh-degree",type=int,default=2,choices=range(4),metavar="{0,1,2,3}")
    p.add_argument("--brush-growth-stop",type=int,default=6000)
    p.add_argument("--brush-refine-every",type=int,default=200)
    p.add_argument("--brush-export-every",type=int,default=5000)
    p.add_argument("--brush-cache",default="2GiB")
    p.add_argument("--brush-render-mode",choices=["default","mip"],default="default")
    p.add_argument("--brush-alpha-mode",choices=["masked","transparent"])
    p.add_argument("--brush-invert-masks",action="store_true")
    p.add_argument("--brush-export-path",type=Path,default=None)
    p.add_argument("--brush-option",action="append",default=[],metavar="ARG",help="Extra Brush argument token, repeat as needed")
    p.add_argument("--brush-args",nargs=argparse.REMAINDER,default=[],help="Pass remaining args to Brush; must be last")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    if a.frames<3 or a.resolution<128: p.error("--frames must be >=3; --resolution >=128")
    if a.brush_iters<1 or a.brush_resolution<128 or a.brush_max_splats<1 or a.brush_refine_every<1 or a.brush_export_every<1 or a.brush_growth_stop<0:
        p.error("Invalid Brush numeric options")
    video=a.video.expanduser().resolve()
    if not video.is_file(): p.error(f"Video not found: {video}")
    root=(a.output or Path("splats")/video.stem).expanduser().resolve()
    images=root/"images"; sparse=root/"sparse"; model=sparse/"0"; database=root/"database.db"
    if not a.dry_run:
        images.mkdir(parents=True,exist_ok=True)
        sparse.mkdir(parents=True,exist_ok=True)
    if not a.skip_extract:
        require("ffmpeg"); require("ffprobe")
        duration=duration_of(video)
        if not a.dry_run and list(images.glob("frame_*.jpg")):
            print(f"Reusing existing extracted frames in {images}")
            a.skip_extract=True
        fps=a.frames/duration
        vf=f"fps={fps:.10f},scale={a.resolution}:{a.resolution}:force_original_aspect_ratio=decrease:force_divisible_by=2"
        if not a.skip_extract: run(["ffmpeg","-hide_banner","-loglevel","error","-i",video,"-vf",vf,"-frames:v",a.frames,"-q:v","2",images/"frame_%05d.jpg"],a.dry_run)
    elif not a.dry_run and not any(images.glob("*.jpg")):
        p.error(f"No JPEG frames in {images}")
    if not a.skip_colmap:
        require("colmap")
        if database.exists() and not a.dry_run:
            if model.is_dir():
                print(f"Reusing existing COLMAP reconstruction at {model}")
                a.skip_colmap=True
            else:
                print(f"Removing incomplete COLMAP database {database} to restart feature extraction")
                database.unlink()
        if not a.skip_colmap: run(["colmap","feature_extractor","--database_path",database,"--image_path",images,"--ImageReader.single_camera","1","--ImageReader.camera_model",a.camera_model,"--SiftExtraction.use_gpu","0"],a.dry_run)
        if not a.skip_colmap: run(["colmap",f"{a.matcher}_matcher","--database_path",database,"--SiftMatching.use_gpu","0"],a.dry_run)
        if not a.skip_colmap: run(["colmap","mapper","--database_path",database,"--image_path",images,"--output_path",sparse],a.dry_run)
    if not a.dry_run:
        if not model.is_dir():
            raise RuntimeError(f"No COLMAP model at {model}; inspect mapper logs")
        run(["colmap","model_analyzer","--path",model])
        (root/"manifest.json").write_text(json.dumps({"video":str(video),"images":str(images),"colmap_model":str(model),"frames_target":a.frames,"resolution":a.resolution,"matcher":a.matcher},indent=2)+"\n")
    if a.train=="brush":
        brush=shutil.which(a.brush_bin) or (str(Path(a.brush_bin).expanduser().resolve()) if Path(a.brush_bin).expanduser().is_file() else None)
        if not brush: raise RuntimeError("Brush not found; use --brush-bin /path/to/brush")
        extra=[*a.brush_option,*a.brush_args]
        if extra and extra[0]=="--": extra=extra[1:]
        export_path=(a.brush_export_path or root/"exports").expanduser().resolve()
        if not a.dry_run: export_path.mkdir(parents=True,exist_ok=True)
        cmd=[brush,str(root),
             "--total-train-iters",str(a.brush_iters),
             "--max-resolution",str(a.brush_resolution),
             "--max-splats",str(a.brush_max_splats),
             "--sh-degree",str(a.brush_sh_degree),
             "--growth-stop-iter",str(min(a.brush_growth_stop,a.brush_iters)),
             "--refine-every",str(a.brush_refine_every),
             "--export-every",str(a.brush_export_every),
             "--max-scene-batch-cache-size",a.brush_cache,
             "--render-mode",a.brush_render_mode,
             "--export-path",str(export_path)]
        if a.brush_viewer: cmd.append("--with-viewer")
        if a.brush_alpha_mode: cmd.extend(["--alpha-mode",a.brush_alpha_mode])
        if a.brush_invert_masks: cmd.append("--invert-masks")
        cmd.extend(extra)
        run(cmd,a.dry_run)
    print(f"\nDataset: {root}\nImages: {images}\nCOLMAP model: {model}")

if __name__=="__main__":
    try: main()
    except (subprocess.CalledProcessError,RuntimeError) as e:
        print(f"ERROR: {e}",file=sys.stderr)
        sys.exit(1)
