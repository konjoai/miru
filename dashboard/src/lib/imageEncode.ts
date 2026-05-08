/**
 * Image encoding utilities for miru's transport quirk.
 *
 * miru's /api/analyze decodes `image_b64` as RAW RGB BYTES (3 channels),
 * not a PNG. To send an image, we:
 *
 *   1. Render it to a small canvas (typically 16×16 to match the attention grid).
 *   2. Read RGBA, drop alpha → RGB.
 *   3. Base64-encode the bytes.
 */

/**
 * Render a Canvas to raw uint8 RGB at the given resolution (default 16×16),
 * then base64-encode the bytes. Matches the demo workaround in
 * /Users/wesleyscholl/miru/demo/index.html.
 */
export function canvasToRawRgbBase64(
  source: HTMLCanvasElement | HTMLImageElement,
  size = 16,
): string {
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  const ctx = tmp.getContext("2d");
  if (!ctx) throw new Error("2D context unavailable");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "medium";
  ctx.drawImage(source, 0, 0, size, size);
  const img = ctx.getImageData(0, 0, size, size);
  const rgba = img.data;
  const rgb = new Uint8Array(size * size * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j]     = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return uint8ToBase64(rgb);
}

function uint8ToBase64(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  // btoa is widely available in browsers and jsdom.
  return btoa(s);
}

/** Decode a File (image/*) and draw it into the supplied canvas at native size. */
export function loadFileIntoCanvas(file: File, canvas: HTMLCanvasElement): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        URL.revokeObjectURL(url);
        reject(new Error("2D context unavailable"));
        return;
      }
      const W = canvas.width;
      const H = canvas.height;
      // contain-fit
      const ratio = Math.min(W / img.width, H / img.height);
      const dw = img.width * ratio;
      const dh = img.height * ratio;
      const dx = (W - dw) / 2;
      const dy = (H - dh) / 2;
      ctx.fillStyle = "#0a0c12";
      ctx.fillRect(0, 0, W, H);
      ctx.drawImage(img, dx, dy, dw, dh);
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("image decode failed"));
    };
    img.src = url;
  });
}
