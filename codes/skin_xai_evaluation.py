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
from captum.attr import DeepLiftShap,LayerDeepLiftShap
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum")
import torchvision.models.resnet as resnet_mod

  
# ======= ŚCIEŻKI =======
PROJECT_DIR = Path(r"C:\Users\lukas\OneDrive\Pulpit\Praktyki")
MODEL_PATH  = PROJECT_DIR / "Modele/Model_skora/resnet34_last.pth"
CLASS_PATH  = PROJECT_DIR / "Modele/Model_skora/class_index.json"
VAL_ROOT    = PROJECT_DIR / "Baza_skora/val"
typ = False #True - lisc, false - skora
# ======= ŚCIEŻKA ZAPISU WERYFIKACJI =======
OUT_DIR = MODEL_PATH.parent / "XAI_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ======= TRANSFORMACJA DLA OBRAZU =======
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
    
])
t_SNE_ = False
LRP_ = True
GradCAM_ = False
DeepSHAP_ = False


# ======= WCZYTANIE KLAS =======
with open(CLASS_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes  = len(class_to_idx)

# ======= WCZYTANIE MODELU =======
model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ======= PRZETWARZANIE WSZYSTKICH OBRAZKÓW =======
total = 0
correct = 0
features_list = []
labels_list = []
wrong_images = []
wrong_preds = []
wrong_labels = []
y_true = []
y_pred = []

# Dla ekstrakcji cech
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
            input_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                feats = feature_extractor(input_tensor)["features"].squeeze()
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_class = idx_to_class[pred_idx]

            # Zbieranie cech i etykiet
            features_list.append(feats.view(-1).numpy())
            labels_list.append(class_to_idx[class_name])

            # Jeśli błędna klasyfikacja — zbierz dla Grad-CAM
            if pred_class != class_name:
                wrong_images.append(img)
                wrong_preds.append(pred_class)
                wrong_labels.append(class_name)


            total += 1
            if pred_class == class_name:
                correct += 1
            y_true.append(class_to_idx[class_name])
            y_pred.append(pred_idx)

        except Exception as e:
            print(f"Błąd przy {img_path}: {e}")
if t_SNE_:
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\n Poprawnie sklasyfikowanych: {correct}/{total} obrazów")
    print(f" Dokładność: {accuracy:.2f}%")
    # ======= t-SNE =======
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

    # ======= MACIERZ POMYŁEK =======
    print("Rysowanie macierzy pomyłek...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_class[i] for i in range(len(class_to_idx))])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.show()

def fuse_resnet_bn(model):
    for name, module in model.named_children():
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

model.eval()
fuse_resnet_bn(model)
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
                relevance = self._relprop(layer[i], xs[i], xs[i+1], relevance, eps, rule, gamma, alpha, beta)
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





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
def disable_inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = False
model.apply(disable_inplace_relu)

model.to(device)


# ======= Przygotowanie przykładów =======
correct_samples = {}
for cls in os.listdir(VAL_ROOT):
    cls_dir = os.path.join(VAL_ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    for fname in os.listdir(cls_dir):
        if not fname.lower().endswith(('.jpg','.png','.tif','.bmp')):
            continue
        img = Image.open(os.path.join(cls_dir, fname)).convert('RGB')
        inp = transform(img).unsqueeze(0)
        with torch.no_grad():
            pred = torch.argmax(model(inp.to(device)), dim=1).item()
        if pred == class_to_idx[cls]:
            correct_samples[cls] = inp  # zapisujemy tensor na CPU
            break
if LRP_:
# ======= Wizualizacja LRP dla wybranych warstw =======
    lrp_model = LRPResNet34(model, device)
    # Jedno poprawnie sklasyfikowane wejście na klasę
    max_per_figure = 7  
    layer_name = 'layer1'
    R_max_all = 0

    selected_classes = ["Actinic Keratoses and Intraepithelial Carcinoma", "Basal Cell Carcinoma", "Benign Keratosis-like Lesions", "Dermatofibroma", "Melanocytic Nevi", "Melanoma", "Vascular Lesions"]
    n_classes = len(selected_classes)
    classes_per_row = 2 
    rows = math.ceil(n_classes / classes_per_row)
    cols = 2 * classes_per_row

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.6))


    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, cls in enumerate(selected_classes):
        row = idx // classes_per_row
        col_block = idx % classes_per_row
        col_heat = col_block * 2
        col_overlay = col_heat + 1

        inp = correct_samples[cls].to(device)
        rel_maps = lrp_model.lrp(
            inp,
            class_to_idx[cls],
            rule="alphabeta",
            alpha=1.0,
            beta=0.0,
            return_all_layers=True
        )
        R = rel_maps[layer_name][0].detach().cpu().numpy().sum(0)
        R -= R.min(); R /= (R.max() + 1e-8)

        orig = inp.squeeze().permute(1,2,0).cpu().numpy()
        orig = (orig * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        orig = np.clip(orig, 0, 1); orig = (orig*255).astype(np.uint8)

        heat = cv2.resize(R, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        H_full = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        cmap = cv2.applyColorMap((255 * H_full).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig, 0.5, cmap, 0.5, 0)

        im = axes[row, col_heat].imshow(H_full, cmap='jet', vmin=0.0, vmax=1.0)

        # axes[row, col_heat].imshow(H_full, cmap='jet', vmin=0.0, vmax=1.0)
        axes[row, col_heat].set_title(f"{cls} – heatmap")
        axes[row, col_heat].axis('off')
        cb = fig.colorbar(im, ax=axes[row, col_heat], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7) 

        axes[row, col_overlay].imshow(overlay[..., ::-1])
        axes[row, col_overlay].set_title(f"{cls}")
        axes[row, col_overlay].axis('off')


    for idx in range(len(selected_classes), rows * classes_per_row):
        row = idx // classes_per_row
        col_block = idx % classes_per_row
        axes[row, col_block * 2].axis('off')
        axes[row, col_block * 2 + 1].axis('off')

    fig.suptitle(f"LRP", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"lrp_{layer_name}_7klas.png"
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.show()


# ======= GradCAM - rozne rodzaje =======


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

if GradCAM_:
    print("Porównanie GradCamow")
    methods = {
        "Grad-CAM": GradCAM(model, model.layer1[-1].relu),
        "Score-CAM": ScoreCAM(model, model.layer1[-1].relu),
        "Layer-CAM": LayerCAM(model, model.layer1[-1].relu)
    }

    selected_classes = list(correct_samples.keys())
    rows = len(selected_classes)
    cols = len(methods)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

    for row_idx, cls in enumerate(selected_classes):
        input_tensor = correct_samples[cls].to(device)
        class_idx = class_to_idx[cls]

        orig = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        orig = (orig * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        orig = np.clip(orig, 0, 1)
        orig = (orig * 255).astype(np.uint8)

        for col_idx, (name, cam_method) in enumerate(methods.items()):
            heatmap = cam_method.generate(input_tensor, class_idx)
            heat_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
            heat_color = cv2.applyColorMap((255 * heat_resized).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig, 0.5, heat_color, 0.5, 0)

            ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
            ax.imshow(overlay[..., ::-1])
            if row_idx == 0:
                ax.set_title(name, fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(cls, fontsize=10)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_DIR / "gradcam_all_methods.png")
    plt.show()

if DeepSHAP_:
    print("Generowanie DeepSHAP...")
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

    def new_relu(*args, **kwargs):
        return original_relu(inplace=False)

    def new_forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x); out = self.bn1(out)
        out = F.relu(out, inplace=False)        # pierwszy ReLU (funkcyjny)

        out = self.conv2(out); out = self.bn2(out)
        out += identity
        out = F.relu(out, inplace=False)        # drugi ReLU (funkcyjny)

        return out
    BasicBlock.forward = new_forward 
   
    # === Klasy ===
    original_relu = torch.nn.ReLU
    torch.nn.ReLU = new_relu

    torch.nn.ReLU = original_relu

    # === Zbieranie poprawnych przykładów ===
    correct_samples = {}
    for cls in os.listdir(VAL_ROOT):
        cls_dir = os.path.join(VAL_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.tif', '.bmp')):
                continue
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            if pred == class_to_idx[cls]:
                correct_samples[cls] = (img, input_tensor)
                break

    # === DeepSHAP ===
    deep_shap = DeepLiftShap(model)

    results = []
    for cls, (img_pil, inp) in correct_samples.items():
        num_baselines = 10
        baselines = get_real_image_baselines(VAL_ROOT, transform, num_baselines=num_baselines).to(device)
        attribution = deep_shap.attribute(inp, baselines=baselines, target=class_to_idx[cls])

        results.append((cls, img_pil, attribution))

    n = 7  

    all_classes = list(class_to_idx.keys())[:n] 
    sampled = []
    for kl in all_classes:
        if kl in correct_samples:
            sampled.append((kl, correct_samples[kl][0], correct_samples[kl][1]))
        else:
            print(f"Brak poprawnego dla klasy {kl}")

    fig, axes = plt.subplots(n, n, figsize=(n*2, n*2))  # całość: duży grid
    for row, (kl, img_pil, inp) in enumerate(sampled):
        baselines = get_real_image_baselines(VAL_ROOT, transform, num_baselines=10).to(device)
        for col, klasa_kolumny in enumerate(all_classes):
            explainer = LayerDeepLiftShap(model, model.layer1)
            target_idx = class_to_idx[klasa_kolumny]
            attribution = explainer.attribute(inp, baselines=baselines, target=target_idx)
            attr = attribution.squeeze().cpu().detach().mean(0).numpy()
            vmax = np.percentile(np.abs(attr), 99)
            ax = axes[row, col]
            # Bez oryginału, tylko mapa:
            im = ax.imshow(attr, cmap='bwr', alpha=1.0, vmin=-vmax, vmax=+vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(klasa_kolumny, fontsize=6, rotation=0, ha="left")
            if col == 0:
                ax.set_ylabel(kl, fontsize=6,rotation=60)
    plt.tight_layout()
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="SHAP value")
    plt.show()
    fig.savefig(OUT_DIR / "deepshap_full.png", dpi=300, bbox_inches="tight")
