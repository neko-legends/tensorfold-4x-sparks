# Images

New since the G13 publication (G15a-G16). DeepSeek-V4.1-Flash sees images: the EXL3 packs keep its ViT, and the
engine runs DeepSeek's tower and aligner (BF16, ~0.97 GB) **on rank 0 only**; the image rows reach rank 1 through
the embedding exchange both ranks already do. Image positions route experts with the release's `gate.bias_vl` and get
no Engram contribution, as in DeepSeek's reference (`inference/vision.py`, `image_processor.py`, `model.py`).

```bash
IMG=$(base64 -w0 photo.png)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "DeepSeek-V4.1-Flash-TF",
  "reasoning_effort": "none", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What does the sign in this photo say?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG"'"}}]}]}'
```

**Config** (on in `config/prod.env.example`):

```
TF_DSV41_IMAGES=native                    # placeholder (engine default) | reject | native
TF_DSV41_BIAS_VL=/cache/dsv41-bias-vl     # folder holding bias_vl.safetensors, in the cache volume
TF_DSV41_VISION_PREP_MB=64                # preprocessed images cached by data: URL digest
```

**The image routing bias.** Both EXL3 2.9 bpw packs dropped the 43 `*.ffn.gate.bias_vl` tensors (40 layers + 3
DSpark blocks). `scripts/serve.sh prebuild` (or `scripts/serve.sh cache` alone) fetches them into the cache volume on
both nodes, 66 KB read by HTTP range requests from `deepseek-ai/DeepSeek-V4.1-Flash` (each node needs network access
once). By hand, inside the image with the volume mounted:

```bash
python -m tensorfold.families.deepseek_v41.cuda.bias_vl_fetch /cache/dsv41-bias-vl        # or --src <local release dir>
```

The file records the source revision and each tensor's SHA-256. Without it, native mode still runs and image rows
route with the text bias (what the vLLM kit does); the boot log says so. Users who do not want images on the GPU at
all set **`TF_DSV41_IMAGES=placeholder`** (the engine's default, no tower loaded, no bias needed): each image part
becomes a short notice telling the model it cannot see the image, so agent turns with screenshots still work;
`reject` answers HTTP 400 instead.

**Accepted and refused.** Image parts in user, tool and system messages: OpenAI `image_url` (a `data:` URL or a
public `https` URL, fetched with a 10 s / 30 s timeout), `input_image`, `image`, and Anthropic `source` blocks. At
most **8 images a request** (`TF_DSV41_VISION_MAX_IMAGES`; long agent histories should drop old screenshots), 20 MB
and 64 M pixels an image, at most 1,024 positions an image; `http`, private and link-local addresses are refused. A
`<｜deepseek_image｜>` token typed into text is escaped to plain text, never treated as an image.

**Video.** With `TF_DSV41_IMAGES=native`, a `video_url` part whose URL is a base64 `data:` URL (a string or `{"url"}`)
is decoded on the head into at most `TF_VIDEO_MAX_FRAMES` (8) PNG frames, each preceded by a text part
`Video frame at <t> s:`, in place among the message's other parts; the model sees ordinary images. Remote video URLs
are refused. The frames share the request's image budget (`TF_DSV41_VISION_MAX_IMAGES`, 8): images plus frames over it
are refused (HTTP 400), so more frames need that limit raised too. Also bounded: `TF_VIDEO_MAX_DURATION` (300 s),
`TF_VIDEO_FPS` (2), `TF_VIDEO_MAX_SIDE` (768 px), `TF_VIDEO_MAX_OUTPUT_BYTES` (16 MiB), `TF_VIDEO_MAX_INPUT_BYTES`
(32 MiB) and `TF_VIDEO_TIMEOUT` (30 s for the one-shot `python3 -I` child with PyAV and Pillow, `TF_VIDEO_PYTHON`; the
zig image's venv has both, checked at image build). One decoder run per video part per request, shared by the history and full renders;
`/tokenize` counts the same tokens as chat. With `placeholder` or `reject`, video parts are refused and nothing runs.

The live profile deployed 2026-10-26 raises the limits to `TF_VIDEO_MAX_DURATION=900` (15 minutes),
`TF_VIDEO_MAX_FRAMES=32`, `TF_DSV41_VISION_MAX_IMAGES=32`, `TF_VIDEO_TIMEOUT=120` and
`TF_VIDEO_MAX_OUTPUT_BYTES=64 MiB`; decoded input and the HTTP body cap remain 32 MiB. It was verified with a real
six-minute MP4 (red first half, blue second half), which the model answered `red, blue`, and a normal text request.

**Measured** (G15b-G16): visual questions 20 / 20; text-only replies byte-identical with images on, off or
placeholder (4 / 4 HTTP text probes, and the speed suite's replies); hit == cold, batched == alone and turn-2 replay
exact with images in the prompt; 4 image requests during a 299K prefill all answered with the worker at >= 5.1 GiB.
Cost: the head's MemAvailable at boot -0.8 to -1.4 GiB. The duplicate-image bug (the same screenshot twice in one
request killed both ranks) is fixed in this engine (see [Changes since G13](CHANGELOG.md)).
