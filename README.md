# ComfyUI-VAE-Noise-Fix

A lightweight ComfyUI custom node that auto-detects and repairs SDXL VAE high-frequency artifacts (fireflies / dead pixels) using traditional computer vision — Laplacian + CCA + Telea inpainting. **Zero neural network dependency.**

輕量級 ComfyUI 自訂節點，利用傳統電腦視覺自動偵測並修復 SDXL VAE 產生的高頻噪點。不需要任何額外的神經網路模型，即插即用。

---

## Why This Exists / 為什麼需要這個節點

SDXL 的 VAE 在解碼潛在空間時，受限於 4 通道 × 8 倍壓縮率的先天頻寬限制，當搭配高 CFG Scale 或祖先採樣器使用時，經常在暗部背景或高對比邊緣爆出孤立的純白/純黑/純色死點。社群常稱之為「螢火蟲 (Fireflies)」。

現有的解決方案大多依賴神經網路重繪（Hires. fix、Inpaint with U-Net），不僅極度消耗 VRAM，還會破壞原圖已經滿意的細節。本節點反其道而行，用傳統電腦視覺在毫秒級完成精準的微觀修補，完全不動用 GPU 算力。

---

## Features / 功能特色

- **三階段智慧偵測**：Laplacian 梯度提取 → CCA 連通域面積過濾 → 鄰域色彩方差驗證
- **星空友善**：透過衝激訊號 vs 高斯漸層的色彩距離判定，避免誤殺星星、高光、金屬反射等自然細節
- **零 NN 相依**：不需要下載任何 `.safetensors` / `.pth` 模型權重
- **Batch 支援**：正確處理 ComfyUI 4D Tensor `[B, H, W, C]` 的批次維度，支援影片幀序列
- **Debug 預覽**：一鍵切換半透明紅色遮罩疊加，直觀調參
- **獨立 GUI 工具**：不需要 ComfyUI 也能使用，提供互動式視窗、即時滑桿調參、放大鏡、前後對比、噪點統計

---

## Installation / 安裝

### Prerequisites / 前置需求

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (已安裝並可正常執行)
- Python 3.10+
- `opencv-python` (通常 ComfyUI 環境已內建)

### Install / 安裝步驟

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/<your-username>/ComfyUI-VAE-Noise-Fix.git
```

如果 ComfyUI 環境缺少 OpenCV：

```bash
pip install opencv-python
```

安裝完成後重啟 ComfyUI，節點會自動出現在 **Add Node → image → postprocessing → VAE Noise Fix (Traditional CV)**。

---

## Usage / 使用方式

### Standalone GUI Preview / 獨立 GUI 預覽工具

不需要 ComfyUI，只要有 Python + OpenCV 即可啟動互動式預覽：

```bash
# 開啟檔案對話框選擇圖片
python gui_preview.py

# 直接指定圖片
python gui_preview.py path/to/image.png

# 載入整個資料夾，用 A/D 鍵切換
python gui_preview.py path/to/folder/
```

**GUI 功能一覽：**

| 操作 | 說明 |
|------|------|
| 滑桿 `Sensitivity` | 即時調整梯度敏感度，畫面同步更新 |
| 滑桿 `Max Noise Size` | 即時調整最大噪點面積 |
| 按鍵 `1` | 顯示原圖 |
| 按鍵 `2` | 顯示紅色遮罩疊加（偵測結果） |
| 按鍵 `3` | 顯示修補後影像 |
| 按鍵 `4` | 左右並排對比（原圖 \| 修補） |
| 按鍵 `Z` | 開關放大鏡（跟隨滑鼠，局部像素級檢視） |
| 滾輪 | 調整放大鏡倍率（2x–16x） |
| 按鍵 `S` | 儲存修補結果到原圖旁（`_fixed` 後綴） |
| 按鍵 `A` / `D` | 上一張 / 下一張圖片 |
| 按鍵 `Q` / `ESC` | 退出 |

畫面左下角會即時顯示噪點統計：偵測到的噪點數量、總像素數、面積分佈範圍、處理耗時等。

---

### ComfyUI Node Interface / ComfyUI 節點介面

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `image` | IMAGE | — | 輸入影像（ComfyUI PyTorch Tensor） |
| `gradient_sensitivity` | FLOAT (0.01–1.0) | 0.35 | Laplacian 梯度閾值。數值越低越敏感，越高則只抓極端斷層 |
| `max_noise_size` | INT (1–100) | 6 | 最大噪點面積（像素）。超過此面積的斑塊視為正常物件 |
| `preview_mask` | BOOLEAN | False | Debug 模式：輸出紅色半透明遮罩疊加圖，用於視覺化調參 |

### Basic Workflow / 基本工作流

```
[KSampler] → [VAE Decode] → [VAE Noise Fix] → [Save Image]
```

1. 將節點連接在 VAE Decode 之後、Save Image 之前
2. 先開啟 `preview_mask = True` 觀察紅色標記是否精準覆蓋噪點
3. 調整 `gradient_sensitivity` 和 `max_noise_size` 直到標記只覆蓋噪點、不涵蓋正常細節
4. 關閉 `preview_mask`，節點自動輸出修補後的影像

### Parameter Tuning Guide / 調參指南

**gradient_sensitivity（梯度敏感度）**

- `0.15–0.25`：高敏感，適合乾淨的暗背景肖像，會抓出所有微小噪點
- `0.30–0.45`：中等敏感（推薦起點），適合大多數場景
- `0.50–0.80`：低敏感，適合高頻紋理密集的場景（城市夜景、星空），避免誤判

**max_noise_size（最大噪點面積）**

- `1–4`：僅修補單一像素級的死點，最保守
- `5–10`：涵蓋小群聚噪點，適合一般使用
- `10–30`：處理 VAE 嚴重崩壞時的群聚網格噪點

---

## System Architecture / 系統架構

### Pipeline Overview / 管線總覽

```
ComfyUI IMAGE Tensor [B, H, W, C]
        │
        │  (per frame)
        ▼
┌─ TensorBridge ──────────────────────┐
│  GPU RGB float32 → CPU BGR uint8    │
│  BT.709 perceptual grayscale        │
└─────────────────────────────────────┘
        │
        ▼
┌─ GradientNoiseDetector ────────────────────────────┐
│                                                     │
│  Stage 1: Laplacian Energy Extraction               │
│    Laplacian kernel (CV_16S, ksize=3) on grayscale  │
│    → absolute energy map                            │
│                                                     │
│  Stage 2: Binarisation + CCA Area Filter            │
│    threshold = gradient_sensitivity × 255           │
│    connectedComponentsWithStats (8-connectivity)    │
│    retain components with area ≤ max_noise_size     │
│                                                     │
│  Stage 3: Neighbourhood Variance Verification       │
│    per component: mean colour vs annular ring mean  │
│    Euclidean colour distance ≥ threshold → noise    │
│    (rejects stars / specular highlights)             │
│                                                     │
└─────────────── binary mask [H, W] ─────────────────┘
        │
        ├── preview_mask=True ──► DebugOverlayRenderer
        │                          red semi-transparent blend
        │
        └── preview_mask=False ─► TeleaInpainter
                                   cv2.inpaint (INPAINT_TELEA)
                                   radius = clamp(ceil(√area), 2, 7)
        │
        ▼
┌─ TensorBridge ──────────────────────┐
│  CPU BGR uint8 → GPU RGB float32    │
└─────────────────────────────────────┘
        │
        ▼
ComfyUI IMAGE Tensor [B, H, W, C]
```

### Module Breakdown / 模組拆解

程式碼遵循 SOLID 原則，拆分為 5 個獨立 Class：

**1. TensorBridge**
負責 PyTorch Tensor 與 OpenCV NumPy Array 之間的格式轉換。GPU → CPU 的資料搬移僅發生一次，後續的量化、通道轉換皆在 NumPy 層完成，最小化跨裝置傳輸開銷。灰階計算採用 ITU-R BT.709 加權（`Y = 0.2126R + 0.7152G + 0.0722B`），比 OpenCV 預設的 BT.601 更符合 sRGB 色域的 SDXL 生成內容。

**2. GradientNoiseDetector**
三階段偵測管線的核心。Stage 1 用 Laplacian 算子提取二階梯度能量，同時捕捉亮點、暗點與色塊噪點（因為它們都具備極高的局部對比度）。Stage 2 透過 CCA 連通域分析提取斑塊面積，丟棄超過 `max_noise_size` 的正常高頻紋理。Stage 3 計算每個候選斑塊與周圍環形鄰域的歐幾里得色彩距離：VAE 噪點呈現斷崖式跳變（衝激訊號），自然高光則有階梯式漸層過渡，藉此精準區分。

**3. TeleaInpainter**
封裝 OpenCV 的 Fast Marching Method（Telea 2004）。修補半徑由 `max_noise_size` 自適應計算（`ceil(√area)`），限制在 `[2, 7]` 區間，防止過度模糊。相較於 PatchMatch，Telea 在修補 1–5 像素的微小破洞時速度極快且邊緣過渡自然，不會引入遠處不相關的紋理。

**4. DebugOverlayRenderer**
將偵測到的噪點遮罩以 45% 不透明度的紅色疊加在原圖上，讓使用者直觀確認偵測結果並調整參數。

**5. VAENoiseFixNode**
ComfyUI 節點封裝層。處理 `INPUT_TYPES` / `RETURN_TYPES` 註冊、Batch 維度的 for 迴圈迭代，以及 `preview_mask` 分支邏輯。

### Data Flow & Memory Boundary / 資料流與記憶體邊界

```
GPU (PyTorch)                          CPU (OpenCV + NumPy)
─────────────                          ────────────────────
IMAGE [B,H,W,C] float32 RGB
       │
       │── tensor.detach().cpu() ──►   ndarray [H,W,C] uint8 BGR
       │   (single transfer)           │
       │                               ├─ Laplacian → energy map
       │                               ├─ threshold → binary mask
       │                               ├─ CCA → area filter
       │                               ├─ colour distance → verify
       │                               └─ cv2.inpaint (Telea)
       │                               │
       │◄── torch.from_numpy().to() ── repaired [H,W,C] uint8 BGR
       │    (single transfer)
IMAGE [B,H,W,C] float32 RGB
```

每幀僅發生兩次 GPU↔CPU 資料搬移（一進一出），OpenCV 的所有運算皆在 CPU 完成。

---

## Test Dataset / 測試資料集

`dataset/` 目錄包含 4 組不同場景的 SDXL 生成影像，用於驗證演算法在各種邊界情況的表現：

| Group | 場景描述 | 解析度 | 挑戰 |
|-------|---------|--------|------|
| A | 賽博龐克少女肖像，純黑背景 | 768×1280 / 2K | 暗背景易出現死白點 |
| B | 賽博龐克夜街，霓虹燈 | 768×1280 / 1024² / 2K | 大量高頻紋理（霓虹、反射、濕地面） |
| C | 銀河星空，前景山脈剪影 | 768×1280 / 1024² / 2K | **關鍵邊界案例**：必須保留數千顆星星 |
| D | 動漫風格少女肖像，乾淨暗背景 | 768×1280 / 1024² / 2K | 平面上色風格，噪點較少 |

### Standalone Test / 獨立測試

不需要啟動 ComfyUI 即可驗證管線邏輯：

```bash
cd ComfyUI-VAE-Noise-Fix/
python test_vae_noise_fix.py
```

測試腳本會讀取 `dataset/` 中的影像，輸出遮罩疊加圖 (`_mask.png`) 和修補結果 (`_fixed.png`) 到 `test_output/` 目錄。

---

## File Structure / 檔案結構

```
ComfyUI-VAE-Noise-Fix/
├── __init__.py                  # ComfyUI 套件註冊入口
├── vae_noise_fix.py             # 主要節點實作（5 個 Class）
├── gui_preview.py               # 獨立 GUI 互動預覽工具
├── test_vae_noise_fix.py        # 批次測試腳本
├── .gitignore
├── README.md
├── dataset/                     # 測試影像（4 組場景）
│   ├── Group A/                 #   肖像 × 純黑背景
│   ├── GroupB/                  #   賽博龐克夜街
│   ├── GroupC/                  #   銀河星空（邊界案例）
│   └── GroupD/                  #   動漫風格肖像
├── 系統架構.txt                  # 系統架構設計文件
├── Related Work聚焦.txt          # 相關工作參考
└── 針對-SDXL-VAE-高頻噪點修復之
    傳統電腦視覺自動化節點開發
    評估報告.md                   # 技術評估報告
```

---

## Algorithm Deep Dive / 演算法深入解析

### Why Laplacian Instead of Luminance Threshold / 為什麼用 Laplacian 而非亮度閾值

SDXL 的 VAE 噪點不只有純白死點，也可能出現純黑、純紫等色塊。這些噪點的共同特徵不是「亮度高」，而是「與周圍像素的梯度跳變極大」。Laplacian 算子計算的是二階導數，能同時捕捉所有方向的劇烈數值變化，因此比單純的亮度閾值更加通用。

### Why Telea Over PatchMatch / 為什麼選擇 Telea 而非 PatchMatch

| 維度 | Telea (FMM) | PatchMatch |
|------|-------------|------------|
| 機制 | PDE 水平集擴散，從邊界向內推進 | 隨機最近鄰域區塊搜尋 |
| 最佳場景 | 極小破洞、細長刮痕 | 大面積物件移除 |
| 微小噪點表現 | 完美融入周邊漸層 | 易引入遠處不相關紋理 |
| 速度 | 毫秒級（僅計算邊界鄰域） | 需全圖搜尋與迭代 |
| 決定性 | 每次結果相同 | 具隨機性 |

對 1–5 像素的 VAE 噪點而言，Telea 是降維打擊。

### Neighbourhood Variance: Stars vs Noise / 鄰域方差：星星 vs 噪點

```
Star (gradual falloff)          VAE Noise (impulse)
┌───────────────┐               ┌───────────────┐
│  80 150 200   │               │  10  10  10   │
│ 150 250 200   │               │  10 255  10   │
│  80 150  80   │               │  10  10  10   │
└───────────────┘               └───────────────┘
colour distance: LOW            colour distance: HIGH
→ rejected (not noise)          → confirmed as noise
```

---

## Limitations / 已知限制

- 當 VAE 崩壞極其嚴重（大面積網格狀噪點），單一 `max_noise_size` 可能不足以覆蓋，需手動調高
- 極高解析度（4K+）下的 CCA 迴圈在 CPU 上需要額外時間（通常仍在 1 秒內）
- 本節點僅處理像素空間的後處理修補，無法從根本上解決潛在空間的 VAE 頻寬限制

---

## License

MIT

---

## Acknowledgements / 致謝

本專案為國立陽明交通大學「影像編修技術與特效合成」課程期末專題。
