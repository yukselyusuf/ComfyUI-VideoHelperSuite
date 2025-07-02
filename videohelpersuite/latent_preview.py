from PIL import Image
import time
import io
from threading import Thread
import torch.nn.functional as F
import torch

import latent_preview
from .utils import hook

rates_table = {
    'Mochi': 24 // 6,
    'LTXV': 24 // 8,
    'HunyuanVideo': 24 // 4,
    'Cosmos1CV8x8x8': 24 // 8,
    'Wan21': 16 // 4
}


class WrappedPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, previewer, rate=8):
        self.first_preview = True
        self.last_time = 0
        self.c_index = 0
        self.rate = rate
        if hasattr(previewer, 'taesd'):
            self.taesd = previewer.taesd
        elif hasattr(previewer, 'latent_rgb_factors'):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
        else:
            raise Exception('Unsupported preview type for VHS animated previews')

    def decode_latent_to_preview_image(self, preview_format, x0):
        if x0.ndim == 5:
            x0 = x0.movedim(2, 1)
            x0 = x0.reshape((-1,) + x0.shape[-3:])
        num_images = x0.size(0)
        new_time = time.time()
        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews / self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None

        self.first_preview = False
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]

        Thread(target=self.process_previews, args=(x0, self.c_index, num_images)).start()
        self.c_index = (self.c_index + num_previews) % num_images
        return None

    def process_previews(self, image_tensor, ind, leng):
        image_tensor = self.decode_latent_to_preview(image_tensor)
        if image_tensor.size(1) > 512 or image_tensor.size(2) > 512:
            image_tensor = image_tensor.movedim(-1, 0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (512 * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height, 512), mode='bilinear')
            else:
                width = (512 * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (512, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0, -1)

        previews_ubyte = (((image_tensor + 1.0) / 2.0).clamp(0, 1)
                          .mul(0xFF)).to(device="cpu", dtype=torch.uint8)

        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big') * 2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            i.save(message, format="JPEG", quality=95, compress_level=1)

            # WebSocket veya başka gönderim yapılmıyor (PromptServer kaldırıldı)
            # Bu noktada istersen 'message.getvalue()' dosyaya yazılabilir/loglanabilir

            ind = (ind + 1) % leng

    def decode_latent_to_preview(self, x0):
        if hasattr(self, 'taesd'):
            x_sample = self.taesd.decode(x0).movedim(1, 3)
            return x_sample
        else:
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(
                x0.movedim(1, -1),
                self.latent_rgb_factors,
                bias=self.latent_rgb_factors_bias
            )
            return latent_image


@hook(latent_preview, 'get_previewer')
def get_latent_video_previewer(device, latent_format, *args, **kwargs):
    # Preview ayarları artık prompt üzerinden alınmadığı için sabit değer kullanıyoruz
    rate_setting = rates_table.get(latent_format.__class__.__name__, 8)

    previewer = get_latent_video_previewer.__wrapped__(device, latent_format, *args, **kwargs)

    # VHS_latentpreview desteği kaldırıldı, her zaman default previewer döner
    if not hasattr(previewer, "decode_latent_to_preview"):
        return previewer

    return WrappedPreviewer(previewer, rate_setting)