"""读取 crave_report/index.html, 将所有外部资源(img src / video src)
替换为 base64 内嵌, 输出单文件 self-contained.html。"""
import base64, re, sys
from pathlib import Path

report_dir = Path("/vePFS/tim/workspace/deepdive_kai0/web/showcase/reports/crave_report")
html = (report_dir / "index.html").read_text(encoding="utf-8")

# 1) 先用新 pipeline 图替换 recurrence_milestone 图引用
#    把 recurrence_milestone.png 替换为 milestone_repr_frames.png + milestone_repr_decode.png 更合理,
#    但先加新 pipeline 图。用户要求把新图放在 §2.1 附近, 我们插一张大图。
#    找个合适位置: 在 "2.1 灵感来源" 段落之后插入 pipeline 图

# 先把新的 pipeline 图往展示目录拷一份
import shutil
pipeline_src = Path("/vePFS/tim/workspace/deepdive_kai0/web/showcase/content/img/crave_milestone_pipeline.png")
if pipeline_src.exists():
    shutil.copy(pipeline_src, report_dir / "assets" / "crave_milestone_pipeline.png")

# 2) 找到所有 src="assets/..." 和 src="assets/...mp4", 替换为 base64
def embed_file(m):
    src = m.group(2)
    path = report_dir / src
    if not path.exists():
        print(f"  MISSING {src}", file=sys.stderr)
        return m.group(0)
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = path.suffix.lower()
    if ext == ".png":
        mimetype = "image/png"
        tag = f'<img src="data:{mimetype};base64,{b64}"'
    elif ext == ".mp4":
        mimetype = "video/mp4"
        tag = f'<video src="data:{mimetype};base64,{b64}"'
    elif ext == ".jpg" or ext == ".jpeg":
        mimetype = "image/jpeg"
        tag = f'<img src="data:{mimetype};base64,{b64}"'
    else:
        tag = m.group(0)  # leave as-is
    print(f"  EMBED {src} ({len(data)/1e6:.1f}MB)", file=sys.stderr)
    return tag

# Match <img src="assets/..."> or <video src="assets/...">
# Handle both with and without trailing attributes
# Also handle src inside <source> if any
html2 = re.sub(r'<(img|video)\s+src="(assets/[^"]+)"', embed_file, html)

# 3) 也处理可能存在的 <source src="assets/...">
html2 = re.sub(r'<source\s+src="(assets/[^"]+)"', lambda m: f'<source src="data:video/mp4;base64,{base64.b64encode((report_dir / m.group(1)).read_bytes()).decode("ascii")}"', html2)

out = report_dir / "self_contained.html"
out.write_text(html2, encoding="utf-8")
print(f"\nDONE: {out} ({len(html2)/1e6:.1f}MB)", file=sys.stderr)
