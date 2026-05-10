# ComfyUI-VAE-Noise-Fix

A lightweight ComfyUI custom node that **auto-detects and repairs SDXL VAE Spurious Bright Pixels** (isolated high-frequency artifacts) using traditional computer vision —  Laplacian + Median Residual + CCA + LAB-cliff verification + Telea inpainting.  **Zero neural-network dependency.**

輕量級 ComfyUI 自訂節點，利用傳統電腦視覺自動偵測並修復 SDXL VAE 產生的孤立高頻噪點（異常像素）。不需要任何額外的神經網路模型，即插即用。

---

## Why This Exists / 為什麼需要這個節點

SDXL 的 VAE 在解碼潛在空間時，受限於 4 通道 × 8 倍壓縮率的先天頻寬限制，當搭配高 CFG Scale 使用時，經常在暗部背景或高對比邊緣爆出孤立的純白/純黑/純色死點。學界與相關文獻將此類現象稱為**異常像素 (Spurious Bright Pixels)** 或 **孤立高頻噪點 (Isolated High-Frequency Artifacts)**。

現有的解決方案大多依賴神經網路重繪（Hires. fix、Inpaint with U-Net），不僅極度消耗 VRAM，還會破壞原圖已經滿意的細節。本節點反其道而行：用傳統電腦視覺在毫秒級完成修補，完全不動用 GPU 算力。

---

## Features / 功能特色

- **五階段偵測**：雙路能量提取（Laplacian + Median Residual）→ 雙閥值二值化（seed / context）→ 結構過濾（面積 + 形狀 + 孤立性）→ LAB 色度懸崖驗證 → 形態膨脹
- **解析度自適應**：`max_noise_size` 以 1024² 為基準自動依當前解析度等比例放大
- **星空友善**：以 LAB 空間的「懸崖陡峭度比值」+ 色度距離區分人造噪點與自然高光，避免誤殺星星、高光、反射等正常圖片內容
- **形狀感知**：取得物體真實長寬比，避免細毛、髮絲等細長結構被誤抓
- **零 NN 相依**：不需要下載任何模型權重
- **Batch 支援**：正確處理 ComfyUI 4D Tensor `[B, H, W, C]`，支援影片/序列幀
- **11 種 Preview 模式**：除了原圖、紅色遮罩、修補結果、並排對比，還能直接觀察 Laplacian / Median / Seed / Context / Filtered / Verified 等任一中間結果，極大簡化調參過程
- **MASK 輸出通道**：ComfyUI 端額外輸出偵測遮罩，供下游節點（合成、條件、任意 inpaint）重用
- **獨立 GUI 工具**：不需要 ComfyUI 也能使用，含即時滑桿、放大鏡、Canvas 縮放、HUD 統計

---

## Installation / 安裝

### Prerequisites / 前置需求

- Python 3.10+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（若僅使用獨立 GUI 工具則非必需）
- `opencv-python`、`numpy`、`torch`（ComfyUI 環境通常已內建）

### As a ComfyUI Custom Node / 作為 ComfyUI 節點安裝

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/megrez33281/ComfyUI-VAE-Noise-Fix.git
```

如果環境缺少 OpenCV：

```bash
pip install opencv-python numpy
```

重啟 ComfyUI 後，節點會自動出現在
**Add Node → image → postprocessing → VAE Noise Fix (Traditional CV)**。

### Standalone (No ComfyUI) / 純獨立模式安裝

純粹使用 GUI 工具或執行批次測試時，只要：

```bash
git clone https://github.com/<your-username>/ComfyUI-VAE-Noise-Fix.git
cd ComfyUI-VAE-Noise-Fix
pip install opencv-python numpy torch
```

> **Note**：獨立模式下 `torch` 仍是必要依賴，因為 `comfyui_node.py` 與向後相容 shim 會匯入。若僅使用 GUI 預覽，可改安裝最小的 CPU 版 torch（`pip install torch --index-url https://download.pytorch.org/whl/cpu`）。

---

## Usage / 使用方式

### A. ComfyUI 節點 / ComfyUI Node

#### 節點輸入 / Inputs

| 參數 | 類型 | 預設 | 說明 |
|------|------|------|------|
| `image` | IMAGE | — | 輸入影像（4D Tensor，自動處理 batch） |
| `gradient_sensitivity` | FLOAT (0.01–1.0) | 0.35 | Laplacian + Median 兩條路徑的共用敏感度。越高代表與周圍差異越大才會被抓出來 |
| `max_noise_size` | INT (1–100) | 6 | 最大噪點面積（以 1024² 為基準）。超過者視為正常物件；實作會依當前解析度等比放大 |
| `mask_dilate` | INT (0–10) | 2 | 偵測完成後對遮罩做幾像素的膨脹，避免 Telea 取樣到尚未清乾淨的邊界（等同於Telea的mask要畫多大，實際從GUI看覆蓋noise即可） |
| `preview_mode` | enum | `Repaired` | 11 選 1，決定 IMAGE 輸出顯示哪個中間結果（見下表） |

#### 節點輸出 / Outputs

| 名稱 | 類型 | 說明 |
|------|------|------|
| `image` | IMAGE | 由 `preview_mode` 決定的視圖（原圖、紅色遮罩、修補後、並排…） |
| `mask` | MASK | 偵測完成後（含 dilation）的二值遮罩，可餵給其他 ComfyUI 節點 |

#### 11 種 preview_mode

| 模式 | 用途 | 推薦時機 |
|------|------|----------|
| `Original` | 原圖直通 | 與下游節點比對用 |
| `Mask Overlay` | 紅色半透明遮罩疊在原圖上 | **調參時的主要視圖** |
| `Mask Only` | 黑底白點 + 綠色定位圈 | 快速統計 / 找出最小的點 |
| `Repaired` | 修補後影像 | **正式輸出** |
| `Side-by-Side` | 原圖 \| 修補 並排 | 用於對比 |
| `Laplacian Energy Map (Binary)` | Laplacian 能量二值圖，用於抓出特別亮、邊緣銳利的點 | 確認亮邊類噪點是否被抓 |
| `Median Residual Map (Binary)` | 中值殘差二值圖，用於抓出突兀色塊 | 確認色塊類噪點（有時會以純色塊出現）是否被抓 |
| `Combined Seed Mask` | 結合Laplacian與Median，即可能是noise的候選 | 觀察「核心」候選 |
| `Context Mask (Low Thresh)` | 抓出noise的準確範圍 | 觀察暈影完整邊界是否覆蓋noise |
| `Filtered Candidates (Shape/Iso)` | 過濾掉面積過大或者細長的結構 | 確認 aspect-ratio / 面積規則是否正確 |
| `Final Verified Mask (LAB)` | 剔除天然的反光或者星星 | 找出哪些被當成自然高光剔除 |

#### 基本工作流 / Basic Workflow

```
[KSampler] → [VAE Decode] → [VAE Noise Fix] → [Save Image]
                                  │
                                  └─ mask ─► [其他需要 MASK 的節點]
```

調參流程：

1. 先把 `preview_mode` 切到 `Mask Overlay`，觀察紅色標記是否精準覆蓋噪點
2. 若漏抓 → 降低 `gradient_sensitivity`；若誤抓正常細節 → 提高它
3. 若噪點面積較大 → 提高 `max_noise_size`
4. 若修補後殘留淡色光暈 → 提高 `mask_dilate`
5. 切回 `Repaired` 取得最終結果

### B. Standalone GUI / 獨立 GUI 預覽工具

不需要 ComfyUI，用 Python + OpenCV 即可啟動：

```bash
python gui_preview.py                         # 開啟檔案對話框
python gui_preview.py path/to/image.png       # 直接開啟單張圖
python gui_preview.py path/to/folder/         # 載入整個資料夾，A/D 切換
```

**完整快捷鍵：**

| 操作 | 說明 |
|------|------|
| 滑桿 `Sensitivity (×100)` | 即時調整 `gradient_sensitivity`（0.01–1.0） |
| 滑桿 `Max Noise Size` | 即時調整 `max_noise_size` |
| 滑桿 `Mask Dilate` | 即時調整膨脹半徑（0–10） |
| `1` | Original — 原圖 |
| `2` | Mask Overlay — 紅色遮罩疊加 |
| `3` | Mask Only — 黑底白點 |
| `4` | Repaired — 修補後 |
| `5` | Side-by-Side — 並排對比 |
| `6` | Laplacian binary 能量圖 |
| `7` | Median residual binary 圖 |
| `8` | Seed mask（高敏感） |
| `9` | Context mask（低敏感） |
| `0` | Filtered candidates（結構過濾後） |
| `-` | Final verified mask（LAB 過濾後） |
| `Z` | 切換放大鏡（跟隨滑鼠） |
| 滾輪 | 調整放大鏡倍率（2×–16×） |
| `Ctrl + 滾輪` | Canvas 整體縮放（1×–16×，以滑鼠位置為錨點） |
| `R` | Canvas 縮放歸零 |
| `S` | 將目前視圖存檔到原圖旁（自動加後綴） |
| `A` / `D` | 上一張 / 下一張圖 |
| `Q` / `ESC` | 退出 |

畫面左下角即時顯示 HUD 統計：解析度、噪點數、總像素、覆蓋率、面積分佈、偵測耗時、Canvas 倍率。

### C. Batch Test Script / 批次測試腳本

驗證所有 dataset 圖片的偵測 / 修補管線：

```bash
python test_vae_noise_fix.py
```

會在 `test_output/` 產出每張圖片的 `*_mask.png`（紅色遮罩）與 `*_fixed.png`（修補結果）。

---

## How an Image Gets Repaired / 圖片修補流程詳解

### Pipeline Flow Diagram / 完整流程圖

```
┌───────────────────────────────────────────────────────────────────┐
│  ComfyUI IMAGE Tensor  [B, H, W, 3]  float32  RGB  in [0, 1]      │
└──────────────────────────┬────────────────────────────────────────┘
                           │  iterate over batch dim B
                           ▼
            ┌───────────────────────────────┐
            │      TensorBridge             │   single GPU→CPU transfer
            │  detach → cpu → numpy         │
            │  RGB f32 → BGR u8             │
            └──────────────┬────────────────┘
                           │  bgr_u8  [H, W, 3]
                           ▼
   ╔══════════════════════════════════════════════════════════════╗
   ║   ▼  Stage 1 — DUAL-PATH ENERGY EXTRACTION  ▼                ║
   ║                                                              ║
   ║   ┌─────────────────────────┐   ┌─────────────────────────┐  ║
   ║   │ LaplacianEnergyExtractor│   │ MedianResidualExtractor │  ║
   ║   │  • BT.709 grayscale     │   │ • dynamic kernel        │  ║
   ║   │  • cv2.Laplacian(CV_16S)│   │   (≥√max_noise_size)    │  ║
   ║   │  • |·| → uint8 energy   │   │ • |src - median|        │  ║
   ║   │   抓亮邊 / 暗邊死點      │   │ • per-pixel max-channel │  ║
   ║   │                         │   │   抓平緩色塊型異常像素     │  ║
   ║   └────────────┬────────────┘   └────────────┬────────────┘  ║
   ║                │                             │                ║
   ║                ▼                             ▼                ║
   ╠══════════════════════════════════════════════════════════════╣
   ║   ▼  Stage 2 — DUAL-THRESHOLD BINARISATION  ▼                ║
   ║                                                              ║
   ║   DualPathMaskGenerator(sensitivity)  →   seed_mask          ║
   ║       T_lap = sens·255  ,  T_med = 20 + sens·80              ║
   ║       OR-merge two paths                                     ║
   ║                                                              ║
   ║   DualPathMaskGenerator(0.25·sensitivity) → context_mask     ║
   ║       (low threshold; captures full halo extent)             ║
   ║                                                              ║
   ╠══════════════════════════════════════════════════════════════╣
   ║   ▼  Stage 3 — STRUCTURAL FILTER (CCA)  ▼                    ║
   ║                                                              ║
   ║   for each connected component in context_mask:              ║
   ║      Rule ①  isolation : ctx_area ≤ 5·max_noise_size         ║
   ║              else "part of a larger object" → reject         ║
   ║      Rule ②  must contain a SEED pixel                       ║
   ║              and seed_area ≤ max_noise_size                  ║
   ║              else "no real high-energy core" → reject        ║
   ║      Rule ③  rotated-bbox aspect ratio ≤ 3.0                 ║
   ║              (cv2.minAreaRect — catches 45° hair)            ║
   ║              else "elongated structure" → reject             ║
   ║                                                              ║
   ║                          ↓                                    ║
   ║                    filtered_mask                              ║
   ║                                                              ║
   ╠══════════════════════════════════════════════════════════════╣
   ║   ▼  Stage 4 — LAB CHROMATIC VERIFICATION  ▼                 ║
   ║                                                              ║
   ║   Convert ROI to CIE-LAB                                     ║
   ║   for each surviving blob:                                   ║
   ║      total_drop          = peak_L*  − mean_bg_L*             ║
   ║      internal_drop_ratio = (peak_L* − mean_comp_L*)          ║
   ║                            / total_drop                      ║
   ║                                                              ║
   ║      cliff?    internal_drop_ratio > 0.6 − 0.2·sens          ║
   ║                (≈1.0 = impulse cliff,  ≈0.5 = soft star)     ║
   ║                                                              ║
   ║      colored?  ‖a*b*_comp − a*b*_bg‖ ≥ 3 + 15·sens           ║
   ║                (catches purple/green VAE bleed)              ║
   ║                                                              ║
   ║      keep iff  cliff  OR  colored                            ║
   ║                                                              ║
   ║                          ↓                                    ║
   ║                    verified_mask                              ║
   ║                                                              ║
   ╠══════════════════════════════════════════════════════════════╣
   ║   ▼  Stage 5 — MORPHOLOGICAL DILATION  ▼                     ║
   ║                                                              ║
   ║   MaskDilator(radius = mask_dilate)                          ║
   ║      isotropic ellipse SE                                    ║
   ║      pads boundary so Telea samples clean pixels             ║
   ║                                                              ║
   ║                          ↓                                    ║
   ║                     final_mask                                ║
   ╚══════════════════════════════════════════════════════════════╝
                              │
                              ├─────────────────────────────┐
                              │                             │
                              ▼                             ▼
            ┌───────────────────────────────┐   any(final_mask) == 0 ?
            │  TeleaInpainter.inpaint       │   ──► skip, copy original
            │   cv2.INPAINT_TELEA           │       (fast-path)
            │   radius = clip(⌈√A⌉, 2, 7)   │
            │   Fast Marching Method:       │
            │    propagate from boundary    │
            │    → integrate gradient       │
            │    → fill with weighted mean  │
            └──────────────┬────────────────┘
                           │  repaired_bgr
                           ▼
            ┌───────────────────────────────┐
            │  PreviewMode dispatch         │   pick 1 of 11 views
            │   ORIGINAL / OVERLAY / SOLO   │
            │   REPAIRED / SIDE-BY-SIDE     │
            │   LAPLACIAN / MEDIAN / SEED   │
            │   CONTEXT / FILTERED / VERIFIED│
            └──────────────┬────────────────┘
                           │  view_bgr
                           ▼
            ┌───────────────────────────────┐
            │      TensorBridge             │   single CPU→GPU transfer
            │  BGR u8 → RGB f32             │
            │  numpy → torch.from_numpy().to│
            └──────────────┬────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│  IMAGE  [B, H, W, 3]  float32 RGB  +  MASK  [B, H, W]  float32    │
└───────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Walkthrough

當一張帶有 VAE 異常像素的影像進入節點，會經歷以下 8 個階段。  
每個階段都對應到一個獨立的模組，每個模組只做一件事，並把成果交給下一個模組。  

#### Step 0 — GPU 影像搬到 CPU

ComfyUI 內部以 PyTorch tensor 的形式在 GPU 上傳遞影像，但 OpenCV 的傳統 CV 演算法是 CPU 工具。  
`TensorBridge` 模組負責把影像從 GPU 搬到 CPU、轉成 OpenCV 慣用的 BGR uint8 格式。  
整個 batch 在 CPU 上跑完後再一次性回灌 GPU，避免反覆搬移浪費頻寬。

#### Step 1 — 同時用兩種濾鏡找出可疑像素

VAE 的異常像素有兩種常見型態：  
(a) 邊緣**銳利**的單點亮/暗點  
(b) 看起來顏色**平整**、但與周遭格格不入的小色塊  
一種濾鏡很難同時抓到這兩種，所以這一步同時跑兩條互補的偵測路徑：

- **Laplacian 路徑**
  Laplacian 是一種衡量「某個像素的值與四周相差有多劇烈」的二階微分濾鏡。  
  當一個像素跟鄰居的差距極大（不論是極亮、極暗、還是純色塊的硬邊），Laplacian 在該位置就會給出強烈反應；反之，連續漸層的區域反應微弱。  
  亮的死白點與暗的死黑點之所以都會被它抓到，原因不是因為它們「亮」或「暗」，而是因為它們**和周圍像素的差異劇烈**。  
  所以這條路徑專門擒住**邊緣銳利**型的異常像素。

- **Median 殘差路徑**
  中值濾波是一種「以鄰域中位數取代當前像素值」的濾鏡，一兩顆極端像素無法影響整體中位數，因此**對極端值幾乎免疫**。  
  把原圖**減去**自己的中值濾波結果，得到的「殘差」就只會在「與周遭格格不入的單點/小色塊」上有明顯數值，平緩漸層區域則會是接近 0。  
  這條路徑專門擒住**邊緣平緩、但顏色明顯錯開背景**的異常像素（純色塊）。

兩張結果並聯起來，得到一張「哪裡有可疑像素」的能量圖。  

#### Step 2 — 用兩個閾值產生兩張遮罩

光是知道「這裡有可疑像素」還不夠，還需要分別問兩個問題：  

- 「這裡是不是真的有強烈異常的核心？」 → 用較**嚴格**的敏感度閾值，得到一張只標出最強烈像素的 **核心遮罩 (seed mask)**。
- 「異常區域的整體輪廓有多大？」 → 用較**寬鬆**的敏感度閾值，得到一張連暈影邊界都框進來的 **輪廓遮罩 (context mask)**。

之所以要分成兩張，是因為下一步的篩選邏輯同時需要「這個區域是否確實有強烈異常的核心」和「這個區域長什麼樣」兩種資訊。

#### Step 3 — 連通域分析 + 形狀過濾，剔除大物件與細長線條

到這一步，輪廓遮罩已經把所有可疑區域都標出來了，但裡面難免混入「不是異常像素但長得有點像」的東西，例如：頭髮中的一根、衣服上的細格紋、極小的反光高光等。

這一步用**連通域分析 (Connected Component Analysis, CCA)**  
這是一種把「相鄰相連的同色像素」歸為同一群（稱為一個 *blob*）的演算法，把每個 blob 拆出來，然後逐一檢查三道條件：  

1. **大小檢查**：候選 blob 的尺寸不能比使用者設定的最大噪點尺寸大太多；明顯比設定值大的視為大物件，不符合噪點特性。
2. **核心檢查**：blob 內部必須含有「核心遮罩」中的種子像素（也就是真的存在強烈異常的核心），否則只是個低能量的微小區塊，不算真噪點。
3. **形狀檢查**：對 blob 計算最小外接旋轉矩形，量它的長寬比；若長寬比過於懸殊（例如 3:1 以上），表示它是頭髮、細線、刮痕這類**細長結構**，而不是緊湊的小點，捨棄。

過完三道才算是「候選異常像素」，進入下一階段精細驗證。  

#### Step 4 — 用 LAB 色彩空間區分異常像素與自然亮點

這是整個流程中最棘手的問題：  
**星星、燈光、金屬高光**這類自然亮點，外形跟異常像素幾乎一樣  
同樣是小、亮、孤立，如果直接修補就會把星空抹掉。  
要把兩者分開，得從**亮度的變化型態**下手。

兩者間最可能的區別是：  
自然亮點是**漸層的山丘**，中心最亮，向外逐步降回背景亮度，過渡平滑；異常像素則是**斷崖式的脈衝**，中心極亮，鄰近像素直接就是背景亮度，沒有過渡。  
要量化這個差異，影像會先轉換到 **CIE-LAB 色彩空間**  
這是一種把「亮度（L\*）」與「色彩（a\*、b\*）」分開儲存的色彩表示法，比 RGB 更接近人眼感知，分析亮度變化時更穩定。  
在 LAB 空間裡，每個候選 blob 都做兩道檢查：  

- **亮度變化型態檢查**：  
  比較 blob 內部的亮度分布是「平緩爬坡」還是「急轉直上的懸崖」。  
  如果是平緩爬坡（內部亮度與峰值之間有明顯落差，呈現過渡），判定為自然亮點 → 排除  
  如果是懸崖（從邊緣到中心都接近峰值，沒有過渡），判定為異常像素 → 保留  
- **色彩偏移檢查**：  
  比較 blob 平均色彩與周圍背景色彩。  
  如果色相偏離背景過大（例如純紫、純綠飄在膚色上），即使沒有亮度差也判定為異常像素。

只要任一檢查成立，就確認為要修補的像素。  
  

#### Step 5 — 把遮罩往外擴張幾個像素

異常像素的「污染」常常不只是核心那一點，邊緣可能殘留一些尚未清乾淨的異常色彩。  
如果直接把核心遮罩交給修補階段，修補演算法會在遮罩**外側**取樣作為填色來源  
結果取到的就是還沒清乾淨的污染像素，反而把髒東西重新塗回去，形成淡色的暈圈。

這一步用**形態學膨脹 (dilation)**把遮罩往外推幾個像素，確保修補邊界停在乾淨區域。  

#### Step 6 — Telea 修補：從邊界向內擴散填色

完成異常核心的定位與範圍鎖定後，可以開始進行修補 
本節點選用 **Telea 快速行進法 (Fast Marching Method, FMM)**進行修補  
一種把「填補破洞」當作「擴散方程」來解的傳統影像修補演算法。  
它的運作方式由外向內滲透，從遮罩邊界開始一圈一圈往內推進；每個內部像素的填色是周圍已知像素的加權平均，並沿著影像中亮度相同的曲線（**等照度線 / isophote**）平滑延伸，避免突兀的邊緣。
  
選用 Telea 而不是 PatchMatch 或擴散模型 inpainting，是因為對於 1–5 像素的微小破洞，Telea 速度極快（毫秒級）、結果可重現，而且不會像 PatchMatch 那樣有從遠處抓到不相關紋理的風險。  
修補半徑會依使用者設定的最大噪點尺寸自動換算，並限制在合理範圍內，避免半徑過小退化、半徑過大引入不必要的模糊。

如果這張圖根本沒偵測到異常像素（遮罩完全是空的），這一步直接跳過、原圖原封不動傳出。  

#### Step 7 — 挑出使用者要看的視圖

所有中間結果（11 種視圖：原圖、紅色遮罩、Mask Solo、修補後、並排對比、Laplacian 二值圖、中值殘差圖、核心遮罩、輪廓遮罩、結構過濾後、LAB 驗證後）都會被儲存起來。  
使用者可以透過切換不同的視圖進行參數調整以及效果預覽


#### Step 8 — CPU 結果搬回 GPU

修補後的影像從 BGR uint8 轉回 RGB float32，搭配偵測遮罩一起回灌 GPU，包裝成 ComfyUI 的 IMAGE + MASK 輸出 tensor。  
整個 batch 結束。


---

## File Structure / 檔案結構

```
ComfyUI-VAE-Noise-Fix/
├── __init__.py                    # ComfyUI 套件註冊入口
├── comfyui_node.py                # ComfyUI 節點綁定（無演算法）
├── vae_noise_fix.py               # 向後相容 shim（舊 import 路徑仍可用）
├── gui_preview.py                 # GUI 入口腳本
├── test_vae_noise_fix.py          # 批次測試腳本
│
├── core/                          # === 演算法層（11 模組）===
│   ├── __init__.py                #   公開 API 再匯出
│   ├── tensor_bridge.py           #   ComfyUI tensor ↔ OpenCV ndarray
│   ├── energy.py                  #   Laplacian / Median 能量提取
│   ├── thresholding.py            #   雙路能量 → 二值遮罩
│   ├── structural_filter.py       #   CCA + 面積 + aspect-ratio + 孤立性
│   ├── chromatic_filter.py        #   LAB cliff steepness + 色度距離
│   ├── morphology.py              #   遮罩膨脹（保護 Telea 邊界）
│   ├── detector.py                #   GradientNoiseDetector 編排器
│   ├── inpainter.py               #   Telea / FMM 修補
│   ├── overlay.py                 #   Overlay / MaskSolo / SideBySide render
│   ├── statistics.py              #   DetectionStats DTO + 計算器
│   └── pipeline.py                #   NoiseFixPipeline + PreviewMode + DetectionResult
│
├── gui/                           # === GUI 層（5 模組）===
│   ├── __init__.py
│   ├── image_io.py                #   Windows 非 ASCII 路徑安全 I/O
│   ├── zoom_lens.py               #   局部放大鏡狀態 + render
│   ├── canvas_zoom.py             #   整體縮放/平移狀態 + render
│   ├── statistics_hud.py          #   HUD 面板 render
│   └── preview_app.py             #   PreviewApp 主控（事件 + dispatch）
│
├── dataset/                       # 測試影像（4 組場景）
│   ├── GroupA/                    #   肖像 × 純黑背景
│   ├── GroupB/                    #   賽博龐克夜街
│   ├── GroupC/                    #   銀河星空（邊界案例）
│   └── GroupD/                    #   動漫風格肖像
│
├── 系統架構.txt                    # 原始系統架構設計文件
├── Related Work/                  # 相關文獻 PDF
└── 針對-SDXL-VAE-高頻噪點修復之傳統電腦視覺自動化節點開發評估報告.md
```

---

## Module Cheat Sheet / 模組速查表

每個 class 單一職責、可獨立測試：

| 類別 | 位置 | 職責 |
|------|------|------|
| `TensorBridge` | `core/tensor_bridge.py` | ComfyUI Tensor ↔ OpenCV ndarray、BT.709 灰階 |
| `LaplacianEnergyExtractor` | `core/energy.py` | 二階梯度能量 |
| `MedianResidualExtractor` | `core/energy.py` | 中值殘差（動態核心大小） |
| `DualPathMaskGenerator` | `core/thresholding.py` | 雙閥值 → 二值遮罩 |
| `StructuralFilter` | `core/structural_filter.py` | CCA + 面積 + 形狀 + 孤立性 |
| `ChromaticFilter` | `core/chromatic_filter.py` | LAB cliff steepness + 色度 |
| `MaskDilator` | `core/morphology.py` | 等向膨脹 |
| `GradientNoiseDetector` | `core/detector.py` | 編排上述六個 stage，回傳 `IntermediateMaps` |
| `TeleaInpainter` | `core/inpainter.py` | FMM 修補（半徑 clamp 到 [2, 7]） |
| `DebugOverlayRenderer` | `core/overlay.py` | 紅色半透明遮罩疊加 |
| `MaskSoloRenderer` | `core/overlay.py` | 黑底白點 + 綠色定位圈 |
| `SideBySideRenderer` | `core/overlay.py` | 並排對比 |
| `NoiseStatistics` | `core/statistics.py` | 統計聚合 |
| `NoiseFixPipeline` | `core/pipeline.py` | 一次跑完 + 11 種視圖 + 統計 |
| `PreviewMode` | `core/pipeline.py` | 11 種視圖列舉值 |
| `DetectionResult` | `core/pipeline.py` | 不可變 DTO，存所有中間結果 |
| `VAENoiseFixNode` | `comfyui_node.py` | ComfyUI 綁定（不含演算法） |
| `PreviewApp` | `gui/preview_app.py` | OpenCV HighGUI controller |
| `ZoomLens` / `CanvasZoom` / `StatisticsHUD` | `gui/*` | GUI 視覺輔助 |

---

## Test Dataset / 測試資料集

`dataset/` 目錄含 4 組不同場景的 SDXL 生成影像：

| Group | 場景描述 | 解析度 | 挑戰點 |
|-------|---------|--------|--------|
| A | 賽博龐克少女肖像，純黑背景 | 768×1280 / 2K | 暗背景大量死白點 |
| B | 賽博龐克夜街，霓虹燈 | 768×1280 / 1024² / 2K | 高頻紋理（霓虹、反射、濕地面），易誤判 |
| C | 銀河星空，前景山脈剪影 | 768×1280 / 1024² / 2K | **關鍵邊界案例** — 必須保留數千顆星星 |
| D | 動漫風格少女肖像，乾淨暗背景 | 768×1280 / 1024² / 2K | 平面上色，噪點較少 |

---

## Algorithm Notes / 演算法重點筆記

### Why Laplacian + Median Together / 為什麼雙路並用

單純用亮度閾值無法處理 SDXL 會出現的純黑、純紫、純綠異常像素，它們的亮度未必比背景高。  
Laplacian 擅長抓「**邊緣銳利**的單點變化」，但對於「形狀平緩的小色塊」（中心與邊緣只差幾個像素寬，內部還相對均勻）反應就會偏弱。  
Median 殘差則反其道而行：對形狀平緩、但**顏色與背景錯開**的色塊極度敏感，但對細邊銳利的單點反應反而平淡。  
把兩條路徑的結果並聯起來，剛好把兩種型態的異常像素都抓到。

### Why Telea Over PatchMatch

| 維度 | Telea (FMM) | PatchMatch |
|------|-------------|------------|
| 機制 | PDE 水平集從邊界向內推進 | 隨機 nearest-neighbor patch 搜尋 |
| 最佳場景 | 極小破洞、細長刮痕 | 大面積物件移除 |
| 微小噪點表現 | 完美融入邊緣漸層 | 易引入遠處不相關紋理 |
| 速度 | 毫秒級（只算邊界鄰域） | 需全圖搜尋與迭代 |
| 決定性 | 結果可重現 | 隨機起始 |

對 1–5 像素的 VAE 異常像素而言，Telea 在速度、品質、可重現性三個維度上同時勝出。

### Star vs Noise Discrimination / 星星與異常像素如何區分

下圖是同樣大小的兩個亮斑在 5×5 鄰域內的亮度分布。星星（自然亮點）從邊緣到中心是逐步爬升的山丘；異常像素則是中心一根針聳立在平地上的脈衝：

```
Natural Star (gradual falloff)       Spurious Bright Pixel (impulse)
┌───────────────────┐                ┌───────────────────┐
│  80 150 200 150 80│                │  10  10 255  10 10│
│ 150 250 255 250 150│               │  10  10 255  10 10│
│  80 150 200 150 80│                │  10  10  10  10 10│
└───────────────────┘                └───────────────────┘
中心與內部平均差距大                    中心與內部平均差距小
（內部本來就有過渡）                    （內部從邊緣到中心都接近峰值）
背景與峰值差距大                        背景與峰值差距大
→ 平緩山丘，視為自然亮點                → 斷崖式脈衝，視為異常像素
   ✗ 排除                                 ✓ 接受並修補
```

實作上再多搭配一道色彩偏移檢查，即可把「沒有亮度差但有色差」的純色異常像素也抓到。  

---

## Limitations / 已知限制

- 當 VAE 崩壞極其嚴重（大面積網格狀噪點），單一 `max_noise_size` 可能不足覆蓋，需手動拉高
- 極高解析度（4K+）下的 CCA 與 LAB 迴圈在 CPU 上需要額外時間（典型 1–2 秒內）
- 本節點僅處理像素空間後處理，無法從根本上解決潛在空間的 VAE 頻寬限制
- 對於星星或反光的保護僅限於其本身存在漸進的漸層，若是邊緣銳利（例如 2x2 pixel 的純白星星）也會被當成噪點砍掉

---

## License

MIT

---

## Acknowledgements / 致謝

本專案為國立陽明交通大學「影像編修技術與特效合成」課程期末專題。
