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
t_SNE_   = False
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

if t_SNE_:
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n Poprawnie sklasyfikowanych: {correct}/{total} obrazów")
    print(f" Dokładność: {accuracy:.2f}%")
    # t-SNE
    print("Obliczanie t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(np.array(features_list))
    plt.figure(figsize=(8, 6))
    for class_idx in np.unique(labels_list):
        idxs = np.where(np.array(labels_list) == class_idx)
        plt.scatter(tsne_result[idxs, 0], tsne_result[idxs, 1], label=idx_to_class[class_idx], alpha=0.6)
    plt.legend()
    plt.title("t-SNE Visualization of Feature Embeddings")
    plt.savefig(os.path.join(OUT_DIR, "tsne_features.png"))
    plt.show()

    # Macierz pomyłek
    print("Rysowanie macierzy pomyłek...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_class[i] for i in range(len(class_to_idx))])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.show()

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
n_rows = len(selected_classes)
n_cols = 1 + len(methods)  # oryginał + każda aktywna metoda
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# Główna pętla wizualizacji
for r, cls in enumerate(selected_classes):
    inp_cpu = correct_samples[cls]
    inp = inp_cpu.to(device)

    # kolumna 0: oryginał
    orig = tensor_to_uint8_rgb(inp)
    ax0 = axes[r, 0]
    ax0.imshow(orig[..., ::-1])  # cv2 overlayy używają BGR, tu tylko pokazujemy oryginał
    ax0.set_title(f"Oryginał: {cls}")
    ax0.axis('off')

    c = 1  # aktualna kolumna

    if LRP_:
        print(f"Generowanie LRP dla klasy: {cls}")
        # LRP na layer1 (jak w Twoim kodzie)
        rel_maps = lrp_model.lrp(
            inp,
            class_to_idx[cls],
            rule="alphabeta",
            alpha=1.0,
            beta=0.0,
            return_all_layers=True
        )
        R = rel_maps['layer1'][0].detach().cpu().numpy().sum(0)
        R -= R.min(); R /= (R.max() + 1e-8)
        heat = cv2.resize(R, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        im = axes[r, c].imshow(heat, cmap='jet')
        axes[r, c].set_title("LRP (layer1)")
        axes[r, c].axis('off')
        cb = fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        cb.set_label("Relevance")
        c += 1

    if GradCAM_:
        print(f"Generowanie CAM dla klasy: {cls}")
        # Grad-CAM
        h = gc.generate(inp, class_to_idx[cls])
        h_res = cv2.resize(h, (orig.shape[1], orig.shape[0]))
        im = axes[r, c].imshow(h_res, cmap='jet')
        axes[r, c].set_title("Grad-CAM")
        axes[r, c].axis('off')
        cb = fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        cb.set_label("Intensity")
        c += 1

        # Score-CAM
        print(f"Generowanie Score-CAM dla klasy: {cls}")
        h = sc.generate(inp, class_to_idx[cls])
        h_res = cv2.resize(h, (orig.shape[1], orig.shape[0]))
        im = axes[r, c].imshow(h_res, cmap='jet')
        axes[r, c].set_title("Score-CAM")
        axes[r, c].axis('off')
        cb = fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        cb.set_label("Intensity")
        c += 1

        # Layer-CAM
        print(f"Generowanie Layer-CAM dla klasy: {cls}")
        h = lc.generate(inp, class_to_idx[cls])
        h_res = cv2.resize(h, (orig.shape[1], orig.shape[0]))
        im = axes[r, c].imshow(h_res, cmap='jet')
        axes[r, c].set_title("Layer-CAM")
        axes[r, c].axis('off')
        cb = fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        cb.set_label("Intensity")
        c += 1

    if DeepSHAP_:
        print(f"Generowanie DeepSHAP dla klasy: {cls}")
        # Use LayerDeepLiftShap on `model.layer1` to match `optymalne_XAI_lisc.py` behavior
        baselines = get_real_image_baselines(VAL_ROOT, transform, num_baselines=max_baselines).to(device)
        # prefer layer_deepshap (initialized earlier) if available, otherwise fallback
        explainer = locals().get('layer_deepshap', None)
        if explainer is None:
            explainer = LayerDeepLiftShap(model, model.layer1)

        attribution = explainer.attribute(inp, baselines=baselines, target=class_to_idx[cls])
        attr = attribution.squeeze().cpu().detach().mean(0).numpy()
        vmax = np.percentile(np.abs(attr), 99)
        im = axes[r, c].imshow(attr, cmap='bwr', vmin=-vmax, vmax=+vmax)
        axes[r, c].set_title("DeepSHAP (layer1)")
        axes[r, c].axis('off')
        cb = fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.04)
        cb.set_label("SHAP value")
        c += 1



plt.tight_layout()
save_path = OUT_DIR / f"wizualizacje_XAI_{len(selected_classes)}obrazy.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Zapisano: {save_path}")
