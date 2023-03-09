import math
import sys 
sys.path.insert(0, './')
from io import BytesIO 
import numpy as np
import cv2  
from PIL import Image, ImageOps, ImageDraw 
import PIL.ImageEnhance
import PIL.ImageOps
import skimage as sk
from skimage import color
from skimage.filters import gaussian 
import torchvision.transforms as transforms 
from wand.image import Image as WandImage
from libs.ops import disk, plasma_fractal 
from pkg_resources import resource_filename  
 
class Stretch:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = np.asarray(img)
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        srcpt.append([p, h_50])
        x = self.rng.uniform(0, frac) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt.append([p + x, p])
        dstpt.append([p + x, h - p])
        dstpt.append([p + x, h_50])

        # 2nd left-most
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        dstpt.append([p + w_33 + x, p])
        dstpt.append([p + w_33 + x, h - p])

        # 3rd left-most
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        dstpt.append([p + w_66 + x, p])
        dstpt.append([p + w_66 + x, h - p])

        # right-most
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        srcpt.append([w - p, h_50])
        x = self.rng.uniform(-frac, 0) * w_33  # if self.rng.uniform(0,1) > 0.5 else 0
        dstpt.append([w - p + x, p])
        dstpt.append([w - p + x, h - p])
        dstpt.append([w - p + x, h_50])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


class Distort:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = np.asarray(img)
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0
        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = len(b) - 1
        else:
            index = mag
        frac = b[index]

        # top pts
        srcpt.append([p, p])
        x = self.rng.uniform(0, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + x, p + y])

        srcpt.append([p + w_33, p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + w_33 + x, p + y])

        srcpt.append([p + w_66, p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([p + w_66 + x, p + y])

        srcpt.append([w - p, p])
        x = self.rng.uniform(-frac, 0) * w_33
        y = self.rng.uniform(0, frac) * h_50
        dstpt.append([w - p + x, p + y])

        # bottom pts
        srcpt.append([p, h - p])
        x = self.rng.uniform(0, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + x, h - p + y])

        srcpt.append([p + w_33, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + w_33 + x, h - p + y])

        srcpt.append([p + w_66, h - p])
        x = self.rng.uniform(-frac, frac) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([p + w_66 + x, h - p + y])

        srcpt.append([w - p, h - p])
        x = self.rng.uniform(-frac, 0) * w_33
        y = self.rng.uniform(-frac, 0) * h_50
        dstpt.append([w - p + x, h - p + y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        return img


class Curve:
    def __init__(self, square_side=224, rng=None):
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.side = square_side
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        orig_w, orig_h = img.size

        if orig_h != self.side or orig_w != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        isflip = self.rng.uniform(0, 1) > 0.5
        if isflip:
            img = ImageOps.flip(img)
            # img = TF.vflip(img)

        img = np.asarray(img)
        w = self.side
        h = self.side
        w_25 = 0.25 * w
        w_50 = 0.50 * w
        w_75 = 0.75 * w

        b = [1.1, .95, .8]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        rmin = b[index]

        r = self.rng.uniform(rmin, rmin + .1) * h
        x1 = (r ** 2 - w_50 ** 2) ** 0.5
        h1 = r - x1

        t = self.rng.uniform(0.4, 0.5) * h

        w2 = w_50 * t / r
        hi = x1 * t / r
        h2 = h1 + hi

        sinb_2 = ((1 - x1 / r) / 2) ** 0.5
        cosb_2 = ((1 + x1 / r) / 2) ** 0.5
        w3 = w_50 - r * sinb_2
        h3 = r - r * cosb_2

        w4 = w_50 - (r - t) * sinb_2
        h4 = r - (r - t) * cosb_2

        w5 = 0.5 * w2
        h5 = h1 + 0.5 * hi
        h_50 = 0.50 * h

        srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
                 (0, h_50), (w, h_50)]
        dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
                 (w - w4, h4), (w5, h5), (w - w5, h5)]

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if isflip:
            # img = TF.vflip(img)
            img = ImageOps.flip(img)
            rect = (0, self.side // 2, self.side, self.side)
        else:
            rect = (0, 0, self.side, self.side // 2)

        img = img.crop(rect)
        img = img.resize((orig_w, orig_h), Image.BICUBIC)
        return img



class Contrast:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [0.4, .3, .2, .1, .05]
        c = [0.4, .3, .2]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = np.asarray(img) / 255.
        means = np.mean(img, axis=(0, 1), keepdims=True)
        img = np.clip((img - means) * c + means, 0, 1) * 255

        return Image.fromarray(img.astype(np.uint8))


class Brightness:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # W, H = img.size
        # c = [.1, .2, .3, .4, .5]
        c = [.1, .2, .3]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = sk.color.rgb2hsv(img)
        img[:, :, 2] = np.clip(img[:, :, 2] + c, 0, 1)
        img = sk.color.hsv2rgb(img)

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img
        # if isgray:
        # if isgray:
        #    img = color.rgb2gray(img)

        # return Image.fromarray(img.astype(np.uint8))


class JpegCompression:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = [25, 18, 15, 10, 7]
        c = [25, 18, 15]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        output = BytesIO()
        img.save(output, 'JPEG', quality=c)
        return Image.open(output)


class Pixelate:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # c = [0.6, 0.5, 0.4, 0.3, 0.25]
        c = [0.6, 0.5, 0.4]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        img = img.resize((int(w * c), int(h * c)), Image.BOX)
        return img.resize((w, h), Image.BOX)



class GaussianBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # kernel = [(31,31)] prev 1 level only
        ksize = int(min(w, h) / 2) // 4
        ksize = (ksize * 2) + 1
        kernel = (ksize, ksize)
        sigmas = [.5, 1, 2]
        if mag < 0 or mag >= len(sigmas):
            index = self.rng.integers(0, len(sigmas))
        else:
            index = mag

        sigma = sigmas[index]
        return transforms.GaussianBlur(kernel_size=kernel, sigma=sigma)(img)


class DefocusBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]
        c = [(2, 0.1), (3, 0.1), (4, 0.1)]  # , (6, 0.5)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        img = np.asarray(img) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
            n_channels = 3
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(n_channels):
            channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
        channels = np.asarray(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        # if isgray:
        #    img = img[:,:,0]
        #    img = np.squeeze(img)

        img = np.clip(channels, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class MotionBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        n_channels = len(img.getbands())
        isgray = n_channels == 1
        # c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]
        c = [(10, 3), (12, 4), (14, 5)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        output = BytesIO()
        img.save(output, format='PNG')
        img = WandImage(blob=output.getvalue())

        img.motion_blur(radius=c[0], sigma=c[1], angle=self.rng.uniform(-45, 45))
        img = cv2.imdecode(np.frombuffer(img.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img.astype(np.uint8))

        if isgray:
            img = ImageOps.grayscale(img)

        return img


class GlassBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        # c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        c = [(0.45, 1, 1), (0.6, 1, 2), (0.75, 1, 2)]  # , (1, 2, 3)] #prev 2 levels only
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag

        c = c[index]

        img = np.uint8(gaussian(np.asarray(img) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for y in range(h - c[1], c[1], -1):
                for x in range(w - c[1], c[1], -1):
                    dx, dy = self.rng.integers(-c[1], c[1], size=(2,))
                    y_prime, x_prime = y + dy, x + dx
                    # swap
                    img[y, x], img[y_prime, x_prime] = img[y_prime, x_prime], img[y, x]

        img = np.clip(gaussian(img / 255., sigma=c[0], multichannel=True), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ZoomBlur:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [np.arange(1, 1.11, .01),
             np.arange(1, 1.16, .01),
             np.arange(1, 1.21, .02)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag

        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        uint8_img = img
        img = (np.asarray(img) / 255.).astype(np.float32)

        out = np.zeros_like(img)
        for zoom_factor in c:
            zw = int(w * zoom_factor)
            zh = int(h * zoom_factor)
            zoom_img = uint8_img.resize((zw, zh), Image.BICUBIC)
            x1 = (zw - w) // 2
            y1 = (zh - h) // 2
            x2 = x1 + w
            y2 = y1 + h
            zoom_img = zoom_img.crop((x1, y1, x2, y2))
            out += (np.asarray(zoom_img) / 255.).astype(np.float32)

        img = (img + out) / (len(c) + 1)

        img = np.clip(img, 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))

        return img



class Shrink:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        self.tps = cv2.createThinPlateSplineShapeTransformer()
        self.translateXAbs = TranslateXAbs(self.rng)
        self.translateYAbs = TranslateYAbs(self.rng)

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        img = np.asarray(img)
        srcpt = []
        dstpt = []

        w_33 = 0.33 * w
        w_50 = 0.50 * w
        w_66 = 0.66 * w

        h_50 = 0.50 * h

        p = 0

        # frac = 0.4

        b = [.2, .3, .4]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        frac = b[index]

        # left-most
        srcpt.append([p, p])
        srcpt.append([p, h - p])
        x = self.rng.uniform(frac - .1, frac) * w_33
        y = self.rng.uniform(frac - .1, frac) * h_50
        dstpt.append([p + x, p + y])
        dstpt.append([p + x, h - p - y])

        # 2nd left-most 
        srcpt.append([p + w_33, p])
        srcpt.append([p + w_33, h - p])
        dstpt.append([p + w_33, p + y])
        dstpt.append([p + w_33, h - p - y])

        # 3rd left-most 
        srcpt.append([p + w_66, p])
        srcpt.append([p + w_66, h - p])
        dstpt.append([p + w_66, p + y])
        dstpt.append([p + w_66, h - p - y])

        # right-most 
        srcpt.append([w - p, p])
        srcpt.append([w - p, h - p])
        dstpt.append([w - p - x, p + y])
        dstpt.append([w - p - x, h - p - y])

        n = len(dstpt)
        matches = [cv2.DMatch(i, i, 0) for i in range(n)]
        dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
        src_shape = np.asarray(srcpt).reshape((-1, n, 2))
        self.tps.estimateTransformation(dst_shape, src_shape, matches)
        img = self.tps.warpImage(img)
        img = Image.fromarray(img)

        if self.rng.uniform(0, 1) < 0.5:
            img = self.translateXAbs(img, val=x)
        else:
            img = self.translateYAbs(img, val=y)

        return img


class Rotate:
    def __init__(self, square_side=224, rng=None):
        self.side = square_side
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, iscurve=False, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        if h != self.side or w != self.side:
            img = img.resize((self.side, self.side), Image.BICUBIC)

        b = [15, 30, 45]
        if mag < 0 or mag >= len(b):
            index = 1
        else:
            index = mag
        rotate_angle = b[index]

        angle = self.rng.uniform(rotate_angle - 20, rotate_angle)
        if self.rng.uniform(0, 1) < 0.5:
            angle = -angle

        img = img.rotate(angle=angle, resample=Image.BICUBIC, expand=not iscurve)
        img = img.resize((w, h), Image.BICUBIC)

        return img


class Perspective:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size

        # upper-left, upper-right, lower-left, lower-right
        src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        # low = 0.3

        b = [.05, .1, .15]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        low = b[index]

        high = 1 - low
        if self.rng.uniform(0, 1) > 0.5:
            topright_y = self.rng.uniform(low, low + .1) * h
            bottomright_y = self.rng.uniform(high - .1, high) * h
            dest = np.float32([[0, 0], [w, topright_y], [0, h], [w, bottomright_y]])
        else:
            topleft_y = self.rng.uniform(low, low + .1) * h
            bottomleft_y = self.rng.uniform(high - .1, high) * h
            dest = np.float32([[0, topleft_y], [w, 0], [0, bottomleft_y], [w, h]])
        M = cv2.getPerspectiveTransform(src, dest)
        img = np.asarray(img)
        img = cv2.warpPerspective(img, M, (w, h))
        img = Image.fromarray(img)

        return img


class TranslateX:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        b = [.03, .06, .09]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = self.rng.uniform(v - 0.03, v)

        v = v * img.size[0]
        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateY:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        b = [.07, .14, .21]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        v = b[index]
        v = self.rng.uniform(v - 0.07, v)

        v = v * img.size[1]
        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))


class TranslateXAbs:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, val=0, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        v = self.rng.uniform(0, val)

        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))


class TranslateYAbs:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, val=0, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        v = self.rng.uniform(0, val)

        if self.rng.uniform(0, 1) > 0.5:
            v = -v
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))



class GaussianNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(.08, .38)
        b = [.06, 0.09, 0.12]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + 0.03)
        img = np.asarray(img) / 255.
        img = np.clip(img + self.rng.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ShotNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng
        # Create a dedicated rng for the Poisson noise
        self.noise = np.random.Generator(self.rng.bit_generator.jumped())

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(3, 60)
        b = [13, 8, 3]
        if mag < 0 or mag >= len(b):
            index = 2
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + 7)
        img = np.asarray(img) / 255.
        img = np.clip(self.noise.poisson(img * c) / float(c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class ImpulseNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(.03, .27)
        b = [.03, .07, .11]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + .04)
        # sk.util.random_noise() uses legacy np.random.* functions.
        # We can't pass an rng instance so we specify the seed instead.
        # np.random.seed() accepts 32-bit integers only,
        # generate 4 to simulate a 128-bit state/seed.
        s = self.rng.integers(2 ** 32, size=4)
        img = sk.util.random_noise(np.asarray(img) / 255., mode='s&p', seed=s, amount=c) * 255
        return Image.fromarray(img.astype(np.uint8))


class SpeckleNoise:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # c = self.rng.uniform(.15, .6)
        b = [.15, .2, .25]
        if mag < 0 or mag >= len(b):
            index = 0
        else:
            index = mag
        a = b[index]
        c = self.rng.uniform(a, a + .05)
        img = np.asarray(img) / 255.
        img = np.clip(img + img * self.rng.normal(size=img.shape, scale=c), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8)) 



class Posterize:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        c = [6, 3, 1]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        bit = self.rng.integers(c, c + 2)
        img = PIL.ImageOps.posterize(img, bit)

        return img


class Solarize:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        c = [192, 128, 64]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        thresh = self.rng.integers(c, c + 64)
        img = PIL.ImageOps.solarize(img, thresh)

        return img


class Invert:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = PIL.ImageOps.invert(img)

        return img


class Equalize:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = PIL.ImageOps.equalize(img)

        return img


class AutoContrast:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = PIL.ImageOps.autocontrast(img)

        return img


class Sharpness:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        c = [.1, .7, 1.3]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = self.rng.uniform(c, c + .6)
        img = PIL.ImageEnhance.Sharpness(img).enhance(magnitude)

        return img


class Color:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        c = [.1, .5, .9]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]
        magnitude = self.rng.uniform(c, c + .6)
        img = PIL.ImageEnhance.Color(img).enhance(magnitude)

        return img 


class Fog:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(1.5, 2), (2., 2), (2.5, 1.7)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img) / 255.
        max_val = img.max()
        # Make sure fog image is at least twice the size of the input image
        max_size = 2 ** math.ceil(math.log2(max(w, h)) + 1)
        fog = c[0] * plasma_fractal(mapsize=max_size, wibbledecay=c[1], rng=self.rng)[:h, :w][..., np.newaxis]
        # x += c[0] * plasma_fractal(wibbledecay=c[1])[:224, :224][..., np.newaxis]
        # return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
        if isgray:
            fog = np.squeeze(fog)
        else:
            fog = np.repeat(fog, 3, axis=2)

        img += fog
        img = np.clip(img * max_val / (max_val + c[0]), 0, 1) * 255
        return Image.fromarray(img.astype(np.uint8))


class Frost:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(0.78, 0.22), (0.64, 0.36), (0.5, 0.5)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        filename = [resource_filename('resources/frost1.png'),
                    resource_filename('resources/frost2.png'),
                    resource_filename('resources/frost3.png'),
                    resource_filename('resources/frost4.jpg'),
                    resource_filename('resources/frost5.jpg'),
                    resource_filename('resources/frost6.jpg')]
        index = self.rng.integers(0, len(filename))
        filename = filename[index]
        # Some images have transparency. Remove alpha channel.
        frost = Image.open(filename).convert('RGB')

        # Resize the frost image to match the input image's dimensions
        f_w, f_h = frost.size
        if w / h > f_w / f_h:
            f_h = round(f_h * w / f_w)
            f_w = w
        else:
            f_w = round(f_w * h / f_h)
            f_h = h
        frost = np.asarray(frost.resize((f_w, f_h)))

        # randomly crop
        y_start, x_start = self.rng.integers(0, f_h - h + 1), self.rng.integers(0, f_w - w + 1)
        frost = frost[y_start:y_start + h, x_start:x_start + w]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img)

        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        img = np.clip(np.round(c[0] * img + c[1] * frost), 0, 255)
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class Snow:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        w, h = img.size
        c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
             (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
             (0.55, 0.3, 4, 0.9, 12, 8, 0.7)]
        if mag < 0 or mag >= len(c):
            index = self.rng.integers(0, len(c))
        else:
            index = mag
        c = c[index]

        n_channels = len(img.getbands())
        isgray = n_channels == 1

        img = np.asarray(img, dtype=np.float32) / 255.
        if isgray:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)

        snow_layer = self.rng.normal(size=img.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

        # snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
        snow_layer[snow_layer < c[3]] = 0

        snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        output = BytesIO()
        snow_layer.save(output, format='PNG')
        snow_layer = WandImage(blob=output.getvalue())

        snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=self.rng.uniform(-135, -45))

        snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                                  cv2.IMREAD_UNCHANGED) / 255.

        # snow_layer = cv2.cvtColor(snow_layer, cv2.COLOR_BGR2RGB)

        snow_layer = snow_layer[..., np.newaxis]

        img = c[6] * img
        gray_img = (1 - c[6]) * np.maximum(img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
        img += gray_img
        img = np.clip(img + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
        img = Image.fromarray(img.astype(np.uint8))
        if isgray:
            img = ImageOps.grayscale(img)

        return img


class Rain:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1
        line_width = self.rng.integers(1, 2)

        c = [50, 70, 90]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        n_rains = self.rng.integers(c, c + 20)
        slant = self.rng.integers(-60, 60)
        fillcolor = 200 if isgray else (200, 200, 200)

        draw = ImageDraw.Draw(img)
        max_length = min(w, h, 10)
        for i in range(1, n_rains):
            length = self.rng.integers(5, max_length)
            x1 = self.rng.integers(0, w - length)
            y1 = self.rng.integers(0, h - length)
            x2 = x1 + length * math.sin(slant * math.pi / 180.)
            y2 = y1 + length * math.cos(slant * math.pi / 180.)
            x2 = int(x2)
            y2 = int(y2)
            draw.line([(x1, y1), (x2, y2)], width=line_width, fill=fillcolor)

        return img


class Shadow:
    def __init__(self, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(self, img, mag=-1, prob=1.):
        if self.rng.uniform(0, 1) > prob:
            return img

        # img = img.copy()
        w, h = img.size
        n_channels = len(img.getbands())
        isgray = n_channels == 1

        c = [64, 96, 128]
        if mag < 0 or mag >= len(c):
            index = 0
        else:
            index = mag
        c = c[index]

        img = img.convert('RGBA')
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        transparency = self.rng.integers(c, c + 32)
        x1 = self.rng.integers(0, w // 2)
        y1 = 0

        x2 = self.rng.integers(w // 2, w)
        y2 = 0

        x3 = self.rng.integers(w // 2, w)
        y3 = h - 1

        x4 = self.rng.integers(0, w // 2)
        y4 = h - 1

        draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], fill=(0, 0, 0, transparency))

        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB")
        if isgray:
            img = ImageOps.grayscale(img)

        return img
    
 
def apply_func_distorsion(image, vertical, horizontal, max_offset, func):  
    if not vertical and not horizontal:
        return image

    rgb_image = image.convert('RGBA')
    
    img_arr = np.array(rgb_image)

    vertical_offsets = [func(i) for i in range(img_arr.shape[1])]
    horizontal_offsets = [
        func(i)
        for i in range(
            img_arr.shape[0] + (
                (max(vertical_offsets) - min(min(vertical_offsets), 0)) if vertical else 0
            )
        )
    ]

    new_img_arr = np.zeros((
                        img_arr.shape[0] + (2 * max_offset if vertical else 0),
                        img_arr.shape[1] + (2 * max_offset if horizontal else 0),
                        4
                    ))

    new_img_arr_copy = np.copy(new_img_arr)
    
    if vertical:
        column_height = img_arr.shape[0]
        for i, o in enumerate(vertical_offsets):
            column_pos = (i + max_offset) if horizontal else i
            new_img_arr[max_offset+o:column_height+max_offset+o, column_pos, :] = img_arr[:, i, :]

    if horizontal:
        row_width = img_arr.shape[1]
        for i, o in enumerate(horizontal_offsets):
            if vertical:
                new_img_arr_copy[i, max_offset+o:row_width+max_offset+o,:] = new_img_arr[i, max_offset:row_width+max_offset, :]
            else:
                new_img_arr[i, max_offset+o:row_width+max_offset+o,:] = img_arr[i, :, :]

    return Image.fromarray(np.uint8(new_img_arr_copy if horizontal and vertical else new_img_arr)).convert('RGBA')

def Distortion_Sin(image, vertical=False, horizontal=False):
    """
        Apply a sine distorsion on one or both of the specified axis
    """ 
    max_offset = int(image.height ** 0.5)

    return apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.sin(math.radians(x)) * max_offset)))

def Distortion_Cos(self, image, vertical=False, horizontal=False):
    """
        Apply a cosine distorsion on one or both of the specified axis
    """

    max_offset = int(image.height ** 0.5)

    return apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: int(math.cos(math.radians(x)) * max_offset)))

def random(self, image, vertical=False, horizontal=False):
    """
        Apply a random distorsion on one or both of the specified axis
    """

    max_offset = int(image.height ** 0.4)

    return apply_func_distorsion(image, vertical, horizontal, max_offset, (lambda x: rnd.randint(0, max_offset)))
 