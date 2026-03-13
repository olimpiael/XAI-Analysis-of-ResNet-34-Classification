import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
from torchvision.models.resnet import BasicBlock
import math
from torch.nn.utils import fuse_conv_bn_eval
from pathlib import Path
import random
from captum.attr import DeepLiftShap, LayerDeepLiftShap
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum")
import time
import csv
from skimage.metrics import structural_similarity as ssim


# ======= ŚCIEŻKI =======
PROJECT_DIR = Path(r"C:\Users\lukas\OneDrive\Pulpit\Praktyki")

MODEL_PATH  = PROJECT_DIR / "Modele/Model_lisc/resnet34_last.pth"
CLASS_PATH  = PROJECT_DIR / "Modele/Model_lisc/class_index.json"
VAL_ROOT    = PROJECT_DIR / "Baza_lisc/val"
typ = True  # True - lisc, false - skora

# ======= ŚCIEŻKA ZAPISU =======
OUT_DIR = MODEL_PATH.parent / "Analiza"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======= PARAMETRY WYBORU =======
# Ile poprawnych obrazów z różnych klas analizować (domyślnie 1)
num_samples = 1               # zmień na 2, 3, ... aby zwiększyć liczbę wizualizacji
max_baselines = 10            # DeepSHAP: liczba baseline’ów z val

# ======= TRANSFORMACJA DLA OBRAZU =======
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ======= WŁĄCZANE MODUŁY =======

LRP_     = True     # włącz/wyłącz
GradCAM_ = True     # włącz/wyłącz (obejmuje też Score-CAM i Layer-CAM)
DeepSHAP_= True     # włącz/wyłącz

# ======= WCZYTANIE KLAS =======
with open(CLASS_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes  = len(class_to_idx)

# ======= MODELE =======
def disable_inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = False

# Bezpieczny BasicBlock.forward bez operacji in-place (naprawa błędu hooków)
def safe_basicblock_forward(self, x):
    identity = x
    if self.downsample is not None:
        identity = self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=False)

    out = self.conv2(out)
    out = self.bn2(out)

    # krytyczna zmiana: brak in-place +=
    out = out + identity
    out = F.relu(out, inplace=False)
    return out

# Podmieniamy forward w BasicBlock (to nie zmienia logiki wyników, tylko unika in-place)
BasicBlock.forward = safe_basicblock_forward

# Główny model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.apply(disable_inplace_relu)
model.to(device)

# ======= PRZETWARZANIE WSZYSTKICH OBRAZKÓW (statystyki / t-SNE / CM) =======
total = 0
correct = 0
features_list = []
labels_list = []
wrong_images = []
wrong_preds = []
wrong_labels = []
y_true = []
y_pred = []

feature_extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})

for class_name in os.listdir(VAL_ROOT):
    class_dir = os.path.join(VAL_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
            continue
        img_path = os.path.join(class_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = feature_extractor(input_tensor)["features"].squeeze()
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_class = idx_to_class[pred_idx]
            features_list.append(feats.view(-1).cpu().numpy())
            labels_list.append(class_to_idx[class_name])
            total += 1
            if pred_class == class_name:
                correct += 1
            else:
                wrong_images.append(img)
                wrong_preds.append(pred_class)
                wrong_labels.append(class_name)
            y_true.append(class_to_idx[class_name])
            y_pred.append(pred_idx)
        except Exception as e:
            print(f"Błąd przy {img_path}: {e}")


# ======= (opcjonalne) Fuzja BN -> Conv do LRP (zostawiamy jak w Twoim kodzie) =======
def fuse_resnet_bn(m):
    for name, module in m.named_children():
        if isinstance(module, nn.Sequential):
            fuse_resnet_bn(module)
        elif isinstance(module, BasicBlock):
            if isinstance(module.conv1, nn.Conv2d) and isinstance(module.bn1, nn.BatchNorm2d):
                module.conv1 = fuse_conv_bn_eval(module.conv1, module.bn1)
                module.bn1 = nn.Identity()
            if isinstance(module.conv2, nn.Conv2d) and isinstance(module.bn2, nn.BatchNorm2d):
                module.conv2 = fuse_conv_bn_eval(module.conv2, module.bn2)
                module.bn2 = nn.Identity()
            if module.downsample is not None:
                layers = []
                conv = None
                for sub in module.downsample:
                    if isinstance(sub, nn.Conv2d):
                        conv = sub
                    elif isinstance(sub, nn.BatchNorm2d):
                        bn = sub
                        conv = fuse_conv_bn_eval(conv, bn)
                        layers.append(conv)
                module.downsample = nn.Sequential(*layers)
        else:
            fuse_resnet_bn(module)

# Fuzja tylko dla kopii do LRP, aby nie zaburzać CAM-ów
model_lrp = models.resnet34(weights=None)
model_lrp.fc = torch.nn.Linear(model_lrp.fc.in_features, len(class_to_idx))
model_lrp.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_lrp.eval()
model_lrp.apply(disable_inplace_relu)
model_lrp.to(device)
fuse_resnet_bn(model_lrp)

# ======= LRP KLASA (Twoja) =======
class LRPResNet34(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.layer_list = [
            ('conv1', self.model.conv1),
            ('bn1', self.model.bn1),
            ('relu', self.model.relu),
            ('maxpool', self.model.maxpool),
            ('layer1', self.model.layer1),
            ('layer2', self.model.layer2),
            ('layer3', self.model.layer3),
            ('layer4', self.model.layer4),
            ('avgpool', self.model.avgpool),
            ('flatten', nn.Flatten()),
            ('fc', self.model.fc)
        ]

    def forward(self, x):
        acts = []
        for name, layer in self.layer_list:
            x = layer(x)
            acts.append((name, x))
        return acts

    def lrp(self, x, class_idx, eps=1e-6, rule="epsilon", gamma=0.25, alpha=1.0, beta=0.0, return_all_layers=False):
        x = x.clone().detach().to(self.device).requires_grad_(False)
        acts = [x]
        tmp = x
        for name, layer in self.layer_list:
            tmp = layer(tmp)
            acts.append(tmp)
        one_hot = torch.zeros_like(acts[-1])
        one_hot[0, class_idx] = acts[-1][0, class_idx]
        relevance = one_hot

        rel_maps = {}
        for i in reversed(range(len(self.layer_list))):
            name, layer = self.layer_list[i]
            inp = acts[i]
            out = acts[i+1]
            relevance = self._relprop(layer, inp, out, relevance, eps, rule, gamma, alpha, beta)
            if name in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                rel_maps[name] = relevance.clone().detach()
        return rel_maps if return_all_layers else relevance

    def _relprop(self, layer, input_act, output_act, relevance, eps, rule, gamma, alpha, beta):
        if relevance is None:
            # zabezpieczenie: jeśli poprzednia warstwa zwróciła None, zwracamy zerową mapę
            return torch.zeros_like(input_act)
        if isinstance(layer, nn.Linear):
            w = layer.weight.data
            b = None
            z = F.linear(input_act, w, b)
            if rule == "epsilon":
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance / z
                c = torch.matmul(s, w)
                return input_act * c
            elif rule == "gamma":
                w_gamma = w + gamma * torch.clamp(w, min=0)
                z = F.linear(input_act, w_gamma)
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance / z
                c = torch.matmul(s, w_gamma)
                return input_act * c
            elif rule == "alphabeta":
                w_pos = torch.clamp(w, min=0)
                w_neg = torch.clamp(w, max=0)
                z_pos = F.linear(input_act, w_pos)
                z_neg = F.linear(input_act, w_neg)
                z_pos += eps
                z_neg -= eps
                s_pos = relevance / z_pos
                s_neg = relevance / z_neg
                c = alpha * torch.matmul(s_pos, w_pos) + beta * torch.matmul(s_neg, w_neg)
                return input_act * c

        elif isinstance(layer, nn.Conv2d):
            w = layer.weight
            stride, padding, kernel_size = layer.stride, layer.padding, layer.kernel_size
            B, C_in, H, W = input_act.shape
            C_out = w.shape[0]
            x_unfold = F.unfold(input_act, kernel_size=kernel_size, stride=stride, padding=padding)
            w_flat = w.view(C_out, -1)

            if rule == "epsilon":
                z = torch.einsum("oc,bcl->bol", w_flat, x_unfold)
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance.view(B, C_out, -1) / z
                c = torch.einsum("oc,bol->bcl", w_flat, s)
            elif rule == "gamma":
                w_gamma = w + gamma * torch.clamp(w, min=0)
                w_flat = w_gamma.view(C_out, -1)
                z = torch.einsum("oc,bcl->bol", w_flat, x_unfold)
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance.view(B, C_out, -1) / z
                c = torch.einsum("oc,bol->bcl", w_flat, s)
            elif rule == "alphabeta":
                w_pos = torch.clamp(w, min=0)
                w_neg = torch.clamp(w, max=0)
                w_pos_flat = w_pos.view(C_out, -1)
                w_neg_flat = w_neg.view(C_out, -1)
                z_pos = torch.einsum("oc,bcl->bol", w_pos_flat, x_unfold) + eps
                z_neg = torch.einsum("oc,bcl->bol", w_neg_flat, x_unfold) - eps
                R = relevance.view(B, C_out, -1)
                s_pos = R / z_pos
                s_neg = R / z_neg
                c_pos = torch.einsum("oc,bol->bcl", w_pos_flat, s_pos)
                c_neg = torch.einsum("oc,bol->bcl", w_neg_flat, s_neg)
                c = alpha * c_pos + beta * c_neg

            R_in = F.fold(c, output_size=(H, W), kernel_size=kernel_size, stride=stride, padding=padding)
            return input_act * R_in

        elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU)):
            return relevance
        elif isinstance(layer, nn.MaxPool2d):
            return F.interpolate(relevance, size=input_act.shape[2:], mode="nearest")
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            return F.interpolate(relevance, size=input_act.shape[2:], mode="nearest")
        elif isinstance(layer, nn.Flatten):
            return relevance.view(input_act.shape)
        elif isinstance(layer, nn.Sequential):
            xs = [input_act]
            x = input_act
            for sublayer in layer:
                x = sublayer(x)
                xs.append(x)
            for i in reversed(range(len(layer))):
                sub_rel = self._relprop(layer[i], xs[i], xs[i+1], relevance, eps, rule, gamma, alpha, beta)
                if sub_rel is not None:
                    relevance = sub_rel
            return relevance
        elif isinstance(layer, BasicBlock):
            x = input_act
            if layer.downsample is not None:
                skip = layer.downsample(x)
            else:
                skip = x
            m1 = layer.conv1(x)
            m2 = layer.bn1(m1)
            m3 = layer.relu(m2)
            m4 = layer.conv2(m3)
            main = layer.bn2(m4)
            out = main + skip
            out = layer.relu(out)
            Z_main = main.abs()
            Z_skip = skip.abs()
            Z_sum = Z_main + Z_skip + 1e-12
            R_main = relevance * (Z_main / Z_sum)
            R_skip = relevance * (Z_skip / Z_sum)
            if layer.downsample is not None:
                R_skip = self._relprop(layer.downsample, x, skip, R_skip, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.bn2, m4, main, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.conv2, m3, m4, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.relu, m2, m3, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.bn1, m1, m2, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.conv1, x, m1, R_main, eps, rule, gamma, alpha, beta)
            return R_main + R_skip
        return relevance
# ======= CAM KLASY =======
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients
        activations = self.activations
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha_numer = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, eps)
        alpha = alpha_numer / alpha_denom
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        weighted_activations = weights * activations
        heatmap = weighted_activations.sum(dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy()

class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        cam = self.activations * self.gradients
        cam = cam.sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.target_layer.register_forward_hook(forward_hook)
    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        _ = self.model(input_tensor)
        with torch.no_grad():
            acts = self.activations
            B, C, H, W = acts.shape
            input_up = input_tensor.clone()
            act_norm = (acts - acts.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0])
            act_norm /= (act_norm.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
            scores = []
            for c in range(C):
                upsample = F.interpolate(act_norm[:, c:c+1, :, :], size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
                masked_input = input_up * upsample
                score = self.model(masked_input)[0, class_idx].item()
                scores.append(score)
            weights = torch.tensor(scores, device=acts.device).view(C, 1, 1)
            cam = (weights * acts.squeeze(0)).sum(0)
            cam = torch.relu(cam)
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)
            return cam.cpu().numpy()

# ======= POMOCNICZE =======
def get_real_image_baselines(val_root, transform, num_baselines=10):
    baselines = []
    all_imgs = []
    for cls in os.listdir(val_root):
        cls_dir = os.path.join(val_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.png', '.tif', '.bmp')):
                all_imgs.append(os.path.join(cls_dir, fname))
    chosen_imgs = random.sample(all_imgs, min(num_baselines, len(all_imgs)))
    for img_path in chosen_imgs:
        img = Image.open(img_path).convert('RGB')
        baselines.append(transform(img))
    baselines = torch.stack(baselines)
    return baselines

def tensor_to_uint8_rgb(input_tensor):
    # input [1,3,224,224] -> HWC uint8
    orig = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    orig = (orig * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    orig = np.clip(orig, 0, 1)
    return (orig * 255).astype(np.uint8)

# ======= FUNKCJE POMOCNICZE DO NORMALIZACJI I METRYK =======

def normalize_map(arr, signed=False, out_size=(224, 224)):
    """
    arr: 2D numpy (H, W)
    signed=False  -> [0,1]
    signed=True   -> [-1,1] z przycięciem do 99 percentyla |x|
    """
    arr = np.array(arr, dtype=np.float32)

    if signed:
        v = np.abs(arr)
        vmax = np.percentile(v, 99) if np.any(v > 0) else 1.0
        arr = np.clip(arr, -vmax, vmax) / (vmax + 1e-8)   # [-1,1]
    else:
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - mn) / (mx - mn + 1e-8)           # [0,1]

    if out_size is not None:
        arr = cv2.resize(arr, out_size, interpolation=cv2.INTER_LINEAR)
    return arr


def normalize_to_01(arr, signed=False, out_size=(224,224)):
    """
    Normalize any 2D map to [0,1].
    If signed=True, treat input like signed relevance: clip to 99th pct of |x|, scale to [-1,1] then map to [0,1] by (x+1)/2.
    If signed=False, min-max to [0,1].
    """
    a = np.array(arr, dtype=np.float32)
    if signed:
        v = np.abs(a)
        vmax = np.percentile(v, 99) if np.any(v > 0) else 1.0
        a = np.clip(a, -vmax, vmax) / (vmax + 1e-8)
        unit = (a + 1.0) / 2.0
    else:
        mn = a.min()
        mx = a.max()
        if mx - mn < 1e-12:
            unit = np.zeros_like(a, dtype=np.float32)
        else:
            unit = (a - mn) / (mx - mn + 1e-8)
    if out_size is not None:
        unit = cv2.resize(unit, out_size, interpolation=cv2.INTER_LINEAR)
    return unit


def compute_entropy(hm):
    """
    Entropia mapy po normalizacji do rozkładu prawdopodobieństwa.
    hm: 2D, może być signed lub unsigned – używamy |hm|.
    """
    p = np.abs(hm).astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def energy_concentration(hm, top_percent=5.0):
    """
    Udział energii (sumy |x|) w top X% pikseli o najwyższej wartości.
    hm: 2D (signed/unsigned) – używamy |hm|.
    """
    v = np.abs(hm).flatten()
    if v.size == 0 or np.all(v == 0):
        return 0.0
    total = v.sum()
    k = max(1, int(len(v) * (top_percent / 100.0)))
    idx = np.argpartition(-v, k-1)[:k]
    return float(v[idx].sum() / (total + 1e-8))


def pearson_corr(a, b):
    """
    Korelacja Pearsona między dwiema mapami 2D (po flatten).
    """
    x = a.flatten().astype(np.float64)
    y = b.flatten().astype(np.float64)
    if x.size != y.size:
        raise ValueError("Różne rozmiary map w pearson_corr")
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2)) + 1e-12
    return float(num / den)


def cosine_sim(a, b):
    """
    Cosine similarity między dwiema mapami 2D (po flatten).
    """
    x = a.flatten().astype(np.float64)
    y = b.flatten().astype(np.float64)
    num = np.dot(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
    return float(num / den)


def ssim_map(a, b, signed_a=False, signed_b=False):
    """
    SSIM dla dwóch map 2D.
    Jeśli signed, przekładamy [-1,1] -> [0,1] przez (x+1)/2.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if signed_a:
        a = (a + 1.0) / 2.0
    if signed_b:
        b = (b + 1.0) / 2.0
    # Zakres danych [0,1]
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return float(ssim(a, b, data_range=1.0))


def occlusion_drop(input_tensor, heatmap, model, class_idx, device, k_frac=0.1):
    """
    Prosty sanity-check: usuwamy top k_frac pikseli wg |heatmap|
    i mierzymy spadek logitu dla danej klasy.
    input_tensor: [1,3,224,224] (na device)
    heatmap: 2D 224x224 (signed/unsigned)
    """
    hm = np.abs(heatmap).astype(np.float32)
    flat = hm.flatten()
    n_pix = flat.size
    k = max(1, int(n_pix * k_frac))
    # indeksy najważniejszych pikseli
    idx_sorted = np.argpartition(-flat, k-1)[:k]
    mask = np.ones_like(flat, dtype=np.float32)
    mask[idx_sorted] = 0.0
    mask = mask.reshape(1, 1, 224, 224)  # broadcast na kanały

    mask_t = torch.from_numpy(mask).to(device)
    x = input_tensor.clone().to(device)
    x_occluded = x * mask_t  # wyzerowanie najważniejszych pikseli

    with torch.no_grad():
        logits_orig = model(x)[0, class_idx].item()
        logits_occ  = model(x_occluded)[0, class_idx].item()

    return float(logits_orig - logits_occ)  # im większa różnica, tym mapa bardziej "faithful"


# ======= WYBÓR POPRAWNYCH PRZYKŁADÓW Z RÓŻNYCH KLAS =======
correct_samples = {}  # {class_name: input_tensor_cpu}
for cls in os.listdir(VAL_ROOT):
    cls_dir = os.path.join(VAL_ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.tif', '.bmp')):
            continue
        img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
        inp = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.argmax(model(inp), dim=1).item()
        if pred == class_to_idx[cls]:
            correct_samples[cls] = inp.detach().cpu()  # zapis tensor CPU
            break

# Ustal listę klas do analizy
selected_classes = list(correct_samples.keys())[:max(1, num_samples)]
print(f"Analizowanych obrazów: {len(selected_classes)}")

# Przygotuj instancje metod
methods = []
if LRP_:
    print("Inicjalizacja LRP...")
    lrp_model = LRPResNet34(model_lrp, device)
    methods.append("LRP")
    
if GradCAM_:
    print("Inicjalizacja CAM...")
    target_layer = model.layer1[-1].conv2
    gc = GradCAM(model, target_layer)
    sc = ScoreCAM(model, target_layer)
    lc = LayerCAM(model, target_layer)
    methods.extend(["Grad-CAM", "Score-CAM", "Layer-CAM"])

if DeepSHAP_:
    print("Inicjalizacja DeepSHAP...")
    deep_shap = DeepLiftShap(model)
    layer_deepshap = LayerDeepLiftShap(model, model.layer1)
    methods.append("DeepSHAP")

# Układ rysunku: na wiersz jeden liść; pierwsza kolumna oryginał; potem wybrane metody
# ======= BUFOR NA ZNORMALIZOWANE MAPY I CZASY =======
normalized_maps = {m: [] for m in methods}  # { "LRP": [map_2D, ...], ... }
method_times    = {m: [] for m in methods}  # czasy generowania map (sekundy)

n_rows = len(selected_classes)
n_cols = 1 + len(methods)  # oryginał + każda aktywna metoda
# teraz dwie fizyczne linie na próbkę: raw (pierwsza) i normalized (druga)
n_rows_vis = max(1, n_rows * 2)
fig, axes = plt.subplots(n_rows_vis, n_cols, figsize=(4*n_cols, 3*n_rows_vis))
if n_rows_vis == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows_vis == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# ======= GŁÓWNA PĘTLA WIZUALIZACJI (ZNORMALIZOWANE MAPY) =======
for r, cls in enumerate(selected_classes):
    inp_cpu = correct_samples[cls]          # [1,3,224,224] CPU
    inp = inp_cpu.to(device)

    # dwa wiersze wizualizacji: raw (r_raw) i normalized (r_norm)
    r_raw = 2 * r
    r_norm = r_raw + 1

    # kolumna 0: oryginał (pokazany w obu wierszach dla wygody porównania)
    orig = tensor_to_uint8_rgb(inp)         # HWC uint8
    ax0 = axes[r_raw, 0]
    ax0.imshow(orig[..., ::-1])             # tylko podgląd RGB
    ax0.set_title(f"Oryginał: {cls}")
    ax0.axis('off')
    ax0b = axes[r_norm, 0]
    ax0b.imshow(orig[..., ::-1])
    ax0b.set_title(f"Oryginał (kopią): {cls}")
    ax0b.axis('off')

    c = 1  # aktualna kolumna (metody)

    # LRP
    if LRP_:
        print(f"Generowanie LRP dla klasy: {cls}")
        t0 = time.perf_counter()
        rel_maps = lrp_model.lrp(
            inp,
            class_to_idx[cls],
            rule="alphabeta",
            alpha=1.0,
            beta=0.0,
            return_all_layers=True
        )
        t1 = time.perf_counter()
        method_times["LRP"].append(t1 - t0)

        # bierzemy layer1, sumujemy po kanałach
        R = rel_maps['layer1'][0].detach().cpu().numpy().sum(0)
        # surowa mapa (przeskalowana do rozmiaru obrazu) i jednorodna normalizacja do [0,1]
        R_raw = cv2.resize(R.astype(np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Remove negative values (keep only positive relevance) to reduce constant negative noise
        R_pos = np.clip(R, 0.0, None)
        # percentile clipping to reduce outliers (use 99th percentile)
        vmax_pos = np.percentile(R_pos, 99) if np.any(R_pos > 0) else 0.0
        if vmax_pos > 0:
            R_pos_clip = np.clip(R_pos, 0.0, vmax_pos) / (vmax_pos + 1e-8)
        else:
            R_pos_clip = R_pos
        # optional smoothing to remove high-frequency noise
        R_pos_smooth = cv2.GaussianBlur((R_pos_clip).astype(np.float32), (5, 5), 0)
        # resize to image size to ensure consistent dimensions
        R_unit = cv2.resize(R_pos_smooth, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        normalized_maps["LRP"].append(R_unit.copy())

        # Rysuj raw na górnym wierszu (symetryczne vmin/vmax)
        vmax = max(np.abs(R_raw).max(), 1e-8)
        im = axes[r_raw, c].imshow(R_raw, cmap='bwr', vmin=-vmax, vmax=+vmax)
        axes[r_raw, c].set_title("LRP (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        # Rysuj znormalizowane [0,1] na dolnym wierszu
        im2 = axes[r_norm, c].imshow(R_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("LRP (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

    if GradCAM_:
        print(f"Generowanie CAM dla klasy: {cls}")

        # Grad-CAM
        t0 = time.perf_counter()
        h_grad = gc.generate(inp, class_to_idx[cls])  # 2D
        t1 = time.perf_counter()
        method_times["Grad-CAM"].append(t1 - t0)

        h_grad_raw = cv2.resize(np.array(h_grad, dtype=np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        h_grad_unit = normalize_to_01(h_grad, signed=False, out_size=(orig.shape[1], orig.shape[0]))
        normalized_maps["Grad-CAM"].append(h_grad_unit.copy())

        im = axes[r_raw, c].imshow(h_grad_raw, cmap='jet')
        axes[r_raw, c].set_title("Grad-CAM (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(h_grad_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("Grad-CAM (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

        # Score-CAM
        print(f"Generowanie Score-CAM dla klasy: {cls}")
        t0 = time.perf_counter()
        h_score = sc.generate(inp, class_to_idx[cls])
        t1 = time.perf_counter()
        method_times["Score-CAM"].append(t1 - t0)

        h_score_raw = cv2.resize(np.array(h_score, dtype=np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        h_score_unit = normalize_to_01(h_score, signed=False, out_size=(orig.shape[1], orig.shape[0]))
        normalized_maps["Score-CAM"].append(h_score_unit.copy())

        im = axes[r_raw, c].imshow(h_score_raw, cmap='jet')
        axes[r_raw, c].set_title("Score-CAM (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(h_score_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("Score-CAM (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

        # Layer-CAM
        print(f"Generowanie Layer-CAM dla klasy: {cls}")
        t0 = time.perf_counter()
        h_layer = lc.generate(inp, class_to_idx[cls])
        t1 = time.perf_counter()
        method_times["Layer-CAM"].append(t1 - t0)

        h_layer_raw = cv2.resize(np.array(h_layer, dtype=np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        h_layer_unit = normalize_to_01(h_layer, signed=False, out_size=(orig.shape[1], orig.shape[0]))
        normalized_maps["Layer-CAM"].append(h_layer_unit.copy())

        im = axes[r_raw, c].imshow(h_layer_raw, cmap='jet')
        axes[r_raw, c].set_title("Layer-CAM (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(h_layer_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("Layer-CAM (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

    if DeepSHAP_:
        print(f"Generowanie DeepSHAP dla klasy: {cls}")
        baselines = get_real_image_baselines(VAL_ROOT, transform, num_baselines=max_baselines).to(device)

        t0 = time.perf_counter()
        explainer = locals().get('layer_deepshap', None)
        if explainer is None:
            explainer = LayerDeepLiftShap(model, model.layer1)

        attribution = explainer.attribute(inp, baselines=baselines, target=class_to_idx[cls])
        t1 = time.perf_counter()
        method_times["DeepSHAP"].append(t1 - t0)

        # uśredniamy po kanałach
        attr = attribution.squeeze().cpu().detach().mean(0).numpy()
        attr_raw = cv2.resize(attr.astype(np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        # For DeepSHAP, zero negative contributions to focus on positive support (like baseline-zeroing)
        attr_pos = np.clip(attr, 0.0, None)
        vmax_attr = np.percentile(attr_pos, 99) if np.any(attr_pos > 0) else 0.0
        if vmax_attr > 0:
            attr_clip = np.clip(attr_pos, 0.0, vmax_attr) / (vmax_attr + 1e-8)
        else:
            attr_clip = attr_pos
        attr_smooth = cv2.GaussianBlur(attr_clip.astype(np.float32), (5, 5), 0)
        # resize to image size
        attr_unit = cv2.resize(attr_smooth, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        normalized_maps["DeepSHAP"].append(attr_unit.copy())

        vmax = max(np.abs(attr_raw).max(), 1e-8)
        im = axes[r_raw, c].imshow(attr_raw, cmap='bwr', vmin=-vmax, vmax=+vmax)
        axes[r_raw, c].set_title("DeepSHAP (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(attr_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("DeepSHAP (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1
plt.tight_layout()
save_path = OUT_DIR / f"wizualizacje_XAI_{len(selected_classes)}obrazy.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Zapisano: {save_path}")

# ======= ANALIZA LICZBOWA MAP XAI =======

if len(selected_classes) > 0:
    print("\n=== ANALIZA LICZBOWA MAP XAI (ZNORMALIZOWANE) ===")

    active_methods = [m for m in methods if len(normalized_maps[m]) > 0]

    # 1) Entropia i koncentracja energii
    metrics_summary = []  # lista wierszy do CSV

    for m in active_methods:
        maps_m = normalized_maps[m]  # lista 2D
        ents = [compute_entropy(hm) for hm in maps_m]
        nec1  = [energy_concentration(hm, top_percent=1.0)  for hm in maps_m]
        nec5  = [energy_concentration(hm, top_percent=5.0)  for hm in maps_m]
        nec10 = [energy_concentration(hm, top_percent=10.0) for hm in maps_m]

        avg_time = np.mean(method_times[m]) if len(method_times[m]) > 0 else 0.0

        row = {
            "method": m,
            "entropy_mean": float(np.mean(ents)),
            "entropy_std":  float(np.std(ents)),
            "nec1_mean":    float(np.mean(nec1)),
            "nec5_mean":    float(np.mean(nec5)),
            "nec10_mean":   float(np.mean(nec10)),
            "time_mean_s":  float(avg_time)
        }
        metrics_summary.append(row)

    # 2) Macierze podobieństwa (Pearson i SSIM)
    n_m = len(active_methods)
    pearson_mat = np.zeros((n_m, n_m), dtype=np.float32)
    ssim_mat    = np.zeros((n_m, n_m), dtype=np.float32)

    # Wszystkie normalized_maps są teraz w skali [0,1], więc traktujemy je jako unsigned
    signed_method = {m: False for m in active_methods}

    for i, mi in enumerate(active_methods):
        for j, mj in enumerate(active_methods):
            vals_p = []
            vals_s = []
            # bierzemy wspólną liczbę obrazów (może być różna w teorii)
            n_i = len(normalized_maps[mi])
            n_j = len(normalized_maps[mj])
            n_use = min(n_i, n_j)
            for k in range(n_use):
                hi = normalized_maps[mi][k]
                hj = normalized_maps[mj][k]
                vals_p.append(pearson_corr(hi, hj))
                vals_s.append(
                    ssim_map(
                        hi, hj,
                        signed_a=signed_method[mi],
                        signed_b=signed_method[mj]
                    )
                )
            pearson_mat[i, j] = float(np.mean(vals_p)) if len(vals_p) > 0 else 0.0
            ssim_mat[i, j]    = float(np.mean(vals_s)) if len(vals_s) > 0 else 0.0

    # 3) Prosty pixel-flipping sanity check (k=10%)
    k_frac = 0.10
    drop_summary = []
    for m in active_methods:
        drops = []
        for idx, cls in enumerate(selected_classes):
            if idx >= len(normalized_maps[m]):
                continue
            hm = normalized_maps[m][idx]  # 224x224
            inp_cpu = correct_samples[cls]
            inp = inp_cpu.to(device)
            class_idx = class_to_idx[cls]
            try:
                dlogit = occlusion_drop(inp, hm, model, class_idx, device, k_frac=k_frac)
            except Exception as e:
                print(f"Błąd occlusion dla metody {m}, klasa {cls}: {e}")
                dlogit = 0.0
            drops.append(dlogit)
        drop_summary.append({
            "method": m,
            f"logit_drop_top{int(k_frac*100)}%_mean": float(np.mean(drops) if len(drops) > 0 else 0.0)
        })

    # 4) Zapis CSV
    metrics_csv = OUT_DIR / "xai_metrics_per_method.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "entropy_mean", "entropy_std", "nec1_mean", "nec5_mean", "nec10_mean", "time_mean_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_summary:
            writer.writerow(row)

    drops_csv = OUT_DIR / "xai_pixel_flipping.csv"
    with open(drops_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(drop_summary[0].keys()) if drop_summary else ["method", "logit_drop"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in drop_summary:
            writer.writerow(row)

    pearson_csv = OUT_DIR / "xai_similarity_pearson.csv"
    with open(pearson_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + active_methods)
        for i, mi in enumerate(active_methods):
            writer.writerow([mi] + list(pearson_mat[i, :]))

    ssim_csv = OUT_DIR / "xai_similarity_ssim.csv"
    with open(ssim_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + active_methods)
        for i, mi in enumerate(active_methods):
            writer.writerow([mi] + list(ssim_mat[i, :]))

    print(f"Zapisano metryki do: {metrics_csv}")
    print(f"Zapisano pixel-flipping do: {drops_csv}")
    print(f"Zapisano macierz Pearsona do: {pearson_csv}")
    print(f"Zapisano macierz SSIM do: {ssim_csv}")

    # ---- Czytelne wypisanie wartości w terminalu ----
    print("\n=== PODSUMOWANIE METRYK (czytelne w terminalu) ===")

    # Metryki per metoda (tabela)
    if metrics_summary:
        header = ["method", "entropy_mean", "entropy_std", "nec1_mean", "nec5_mean", "nec10_mean", "time_mean_s"]
        print("\nMetryki per method:")
        print(" | ".join([h.center(12) for h in header]))
        print("-" * (12 * len(header) + 3 * (len(header)-1)))
        for row in metrics_summary:
            vals = [
                str(row.get("method", "")),
                f"{row.get('entropy_mean',0):.4f}",
                f"{row.get('entropy_std',0):.4f}",
                f"{row.get('nec1_mean',0):.4f}",
                f"{row.get('nec5_mean',0):.4f}",
                f"{row.get('nec10_mean',0):.4f}",
                f"{row.get('time_mean_s',0):.4f}"
            ]
            print(" | ".join([v.center(12) for v in vals]))

    # Pixel-flipping drops
    if drop_summary:
        print("\nPixel-flipping (logit drop top k%):")
        for row in drop_summary:
            # print all keys except method in a compact form
            method = row.get("method", "")
            others = [f"{k}: {v:.4f}" for k, v in row.items() if k != "method"]
            print(f" - {method}: " + ", ".join(others))

    # Macierze podobieństwa (czytelnie)
    def _print_matrix(mat, name):
        print(f"\n{name}:")
        if len(active_methods) == 0:
            print("  (brak aktywnych metod)")
            return
        # header
        line = "       "+" ".join([m.center(8) for m in active_methods])
        print(line)
        for i, m in enumerate(active_methods):
            row_vals = " ".join([f"{mat[i,j]:.4f}".center(8) for j in range(mat.shape[1])])
            print(f"{m.ljust(6)} {row_vals}")

    _print_matrix(pearson_mat, "Pearson similarity")
    _print_matrix(ssim_mat, "SSIM similarity")

    # Zapis JSON z podsumowaniem (opcjonalnie czytelne do dalszego użytku)
    try:
        metrics_json = OUT_DIR / "xai_metrics_summary.json"
        to_dump = {
            "metrics_summary": metrics_summary,
            "drop_summary": drop_summary,
            "active_methods": active_methods,
            "pearson": pearson_mat.tolist(),
            "ssim": ssim_mat.tolist()
        }
        with open(metrics_json, "w", encoding="utf-8") as jf:
            json.dump(to_dump, jf, ensure_ascii=False, indent=2)
        print(f"Zapisano JSON z podsumowaniem: {metrics_json}")
    except Exception as e:
        print(f"Nie udało się zapisać JSON podsumowania: {e}")

    # 5) Prosty wykres słupkowy: entropia i czas
    plt.figure(figsize=(8, 4))
    x = np.arange(len(active_methods))
    ent_vals = [row["entropy_mean"] for row in metrics_summary]
    plt.bar(x, ent_vals)
    plt.xticks(x, [row["method"] for row in metrics_summary], rotation=45)
    plt.ylabel("Entropy (mean)")
    plt.title("Entropia znormalizowanych map XAI")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "xai_entropy_bar.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    time_vals = [row["time_mean_s"] for row in metrics_summary]
    plt.bar(x, time_vals)
    plt.xticks(x, [row["method"] for row in metrics_summary], rotation=45)
    plt.ylabel("Czas [s] (średni na mapę)")
    plt.title("Koszt obliczeniowy metod XAI")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "xai_time_bar.png", dpi=200)
    plt.show()

    # 6) Wizualizacja macierzy podobieństwa (Pearson)
    fig_sim, ax_sim = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax_sim[0].imshow(pearson_mat, vmin=-1, vmax=1, cmap="bwr")
    ax_sim[0].set_xticks(np.arange(n_m))
    ax_sim[0].set_yticks(np.arange(n_m))
    ax_sim[0].set_xticklabels(active_methods, rotation=45)
    ax_sim[0].set_yticklabels(active_methods)
    ax_sim[0].set_title("Pearson similarity")
    fig_sim.colorbar(im1, ax=ax_sim[0], fraction=0.046, pad=0.04)

    im2 = ax_sim[1].imshow(ssim_mat, vmin=0, vmax=1, cmap="viridis")
    ax_sim[1].set_xticks(np.arange(n_m))
    ax_sim[1].set_yticks(np.arange(n_m))
    ax_sim[1].set_xticklabels(active_methods, rotation=45)
    ax_sim[1].set_yticklabels(active_methods)
    ax_sim[1].set_title("SSIM similarity")
    fig_sim.colorbar(im2, ax=ax_sim[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "xai_similarity_matrices.png", dpi=200)
    plt.show()
