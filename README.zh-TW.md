# ComfyUI-VAE-Noise-Fix

[![CI](https://github.com/megrez33281/ComfyUI-VAE-Noise-Fix/actions/workflows/ci.yml/badge.svg)](https://github.com/megrez33281/ComfyUI-VAE-Noise-Fix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[English](README.md) | **繁體中文**

輕量級 ComfyUI 自訂節點，利用傳統電腦視覺自動偵測並修復 SDXL VAE 產生的孤立高頻噪點（異常像素，俗稱 "fireflies"）：Laplacian + 中值殘差 → 連通域形狀過濾 → CIE-LAB 懸崖驗證 → Telea 修補。
**不需要任何神經網路、不必下載模型，在 CPU 上以毫秒到約一秒完成。**

![修補前後對比：全圖、原圖局部、紅色為偵測到的像素、修補結果](docs/images/before_after.png)

<sub>取自 `dataset/GroupA/000.png` 的局部，4 倍最近鄰放大，預設參數。可用 `python scripts/make_readme_figure.py` 重新產生。</sub>

---

## 為什麼需要這個節點

SDXL 的 VAE 在解碼潛在空間時，受限於 4 通道 × 8 倍壓縮率的先天頻寬限制，搭配高 CFG Scale 使用時，經常在暗部背景或高對比邊緣爆出孤立的純白／純色死點。相關文獻將此現象稱為**異常像素 (Spurious Bright Pixels)** 或**孤立高頻噪點 (Isolated High-Frequency Artifacts)**。

現有的解決方案大多依賴神經網路重繪（Hires. fix、U-Net inpaint），不僅消耗 VRAM，還可能破壞原圖已經滿意的細節。本節點反其道而行：用傳統電腦視覺找出少數壞點、只修補那些像素，完全不動用 GPU。

## 特色

- **五階段偵測**：雙路能量提取（Laplacian + Median Residual）→ 雙閾值二值化（seed / context）→ 結構過濾（面積 + 形狀 + 孤立性）→ LAB 色度懸崖驗證 → 形態膨脹
- **解析度自適應**：`max_noise_size` 以 1024² 為基準，依當前解析度等比例放大
- **高光感知**：具有漸層衰減的柔和亮點，以及髮絲、細線等細長結構，會被過濾掉而不會被塗掉
- **零 NN 相依**：只需要 OpenCV 與 NumPy
- **Batch 支援**：正確處理 ComfyUI 4D Tensor `[B, H, W, C]`，支援影片／序列幀
- **11 種 Preview 模式**：可直接在 ComfyUI 觀察 Laplacian / Median / Seed / Context / Filtered / Verified 任一中間結果，簡化調參
- **MASK 輸出**：偵測遮罩可供下游任何節點使用
- **手動修整**：拆分為 Detector / Mask Editor / Inpainter 節點，內建彈出式遮罩編輯器
- **獨立 GUI**：不需 ComfyUI 也能對自己的圖片調參（tkinter，不需要 PyTorch）

---

## 安裝

### ComfyUI-Manager（推薦）

開啟 **Manager → Custom Nodes Manager**，搜尋 **VAE Noise Fix**，按 *Install* 後重啟 ComfyUI。

### comfy-cli

```bash
comfy node install vae-noise-fix
```

### 手動安裝

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/megrez33281/ComfyUI-VAE-Noise-Fix.git
pip install -r ComfyUI-VAE-Noise-Fix/requirements.txt   # opencv-python、numpy
```

請使用執行 ComfyUI 的同一個 Python 環境（可攜版為 `python_embeded\python.exe -m pip install -r ...`）。PyTorch 與 Pillow 由 ComfyUI 提供。

重啟後節點會出現在 **Add Node → image → postprocessing**。

---

## 節點

| 節點 | 輸出 | 使用時機 |
|------|------|----------|
| **VAE Noise Fix (Traditional CV)** | `image`、`mask` | 偵測與修補一步完成 |
| **VAE Noise Detector** | `image`、`mask` | 修補前想先檢查或編輯遮罩 |
| **VAE Noise Mask Editor** | `image`、`mask` | 想手動補畫／擦除遮罩（彈出式編輯器） |
| **VAE Noise Inpainter (Telea)** | `image` | 已經有遮罩（來自 Detector、編輯器或其他節點） |

### 參數

| 參數 | 類型 | 預設 | 說明 |
|------|------|------|------|
| `image` | IMAGE | — | 輸入影像 `[B, H, W, C]`，自動處理 batch |
| `gradient_sensitivity` | FLOAT 0.01–1.0 | 0.35 | Laplacian 與 Median 兩條路徑共用的閾值。越高代表與周圍差異越大才會被抓出來 |
| `max_noise_size` | INT 1–100 | 6 | 最大噪點面積（以 1024² 為基準）。超過者視為正常物件；會依解析度等比放大 |
| `mask_dilate` | INT 0–10 | 2 | 對最終遮罩膨脹的像素數，避免 Telea 取樣到尚未清乾淨的邊界。實際從 GUI 確認紅色遮罩覆蓋到光暈即可 |
| `preview_mode` | enum | `Repaired` | 決定 `image` 輸出顯示哪個視圖（見下表），不影響 `mask` 輸出 |

Inpainter 只有 `image`、`mask`、`max_noise_size` 三個輸入；`max_noise_size` 用來決定 Telea 半徑（⌈√size⌉，限制在 2–7），請與 Detector 設定一致。

### 11 種 preview_mode

| 模式 | 用途 | 推薦時機 |
|------|------|----------|
| `Original` | 原圖直通 | 與下游節點比對 |
| `Mask Overlay` | 紅色半透明遮罩疊在原圖上 | **調參時的主要視圖** |
| `Mask Only` | 黑底白點 + 綠色定位圈 | 快速找出最小的點 |
| `Repaired` | 修補後影像 | **正式輸出** |
| `Side-by-Side` | 原圖 \| 修補 並排（寬度兩倍） | 前後對比 |
| `Laplacian Energy Map (Binary)` | 邊緣銳利的亮／暗點 | 確認銳利型噪點是否被抓 |
| `Median Residual Map (Binary)` | 與周遭格格不入的平整色塊 | 確認色塊型噪點是否被抓 |
| `Combined Seed Mask` | 在設定敏感度下結合 Laplacian 與 Median | 觀察「核心」候選 |
| `Context Mask (Low Thresh)` | 同上，但敏感度為 ¼ | 觀察光暈邊界是否完整覆蓋 |
| `Filtered Candidates (Shape/Iso)` | 經面積／長寬比／孤立性過濾後 | 確認形狀規則是否正確 |
| `Final Verified Mask (LAB)` | LAB 驗證後、膨脹前 | 找出哪些被當成自然高光剔除 |

Detector 節點不做修補，因此沒有 `Repaired` 與 `Side-by-Side`。

### 工作流

**全自動**

```
[KSampler] → [VAE Decode] → [VAE Noise Fix] → [Save Image]
                                   │
                                   └─ mask ─► [其他需要 MASK 的節點]
```

**手動修整**

```
[VAE Decode] → [VAE Noise Detector] ─ image, mask ─► [VAE Noise Mask Editor] ─ image, mask ─► [VAE Noise Inpainter] → [Save Image]
```

1. 先 Queue 一次。Mask Editor 會暫存第一幀，並把偵測到的遮罩原樣傳出。
2. 點 Mask Editor 節點上的 **Edit Mask**。可補畫、擦除，並調整筆刷大小、遮罩透明度，支援復原／重做（Ctrl+Z / Ctrl+Shift+Z）、清除、反轉、縮放（Ctrl+滾輪）、平移（Space+拖曳）。完成後按 *Save & Close*。
3. 再 Queue 一次，編輯後的遮罩會套用到整個 batch。

`workflow.json` 是一個範例工作流，並排比較數組 `VAE Noise Fix` 參數設定。

### 調參流程

1. 把 `preview_mode` 切到 `Mask Overlay`，觀察紅色標記是否精準覆蓋噪點
2. 漏抓 → 降低 `gradient_sensitivity`；誤抓正常細節 → 提高它
3. 噪點面積較大 → 提高 `max_noise_size`
4. 修補後殘留淡色光暈 → 提高 `mask_dilate`
5. 切回 `Repaired` 取得最終結果

---

## 獨立 GUI 工具

不需要 ComfyUI 即可對任意圖片調參。需要含 tkinter 的 Python 3.10+，以及：

```bash
pip install -r requirements.txt Pillow      # 不需要 PyTorch
python gui_preview.py                       # 開啟檔案對話框
python gui_preview.py path/to/image.png     # 直接開啟單張圖
python gui_preview.py path/to/folder/       # 載入整個資料夾（A / D 切換）
```

右側面板包含三個參數滑桿、視圖列表、筆刷編輯、縮放工具、存檔按鈕，以及即時統計（解析度、噪點數、總像素、覆蓋率、面積範圍、偵測耗時）。

| 按鍵／滑鼠 | 說明 |
|------------|------|
| `1`–`5` | 原圖 / 紅色遮罩疊加 / 黑底白點 / 修補後 / 並排對比 |
| `6` `7` `8` `9` `0` `-` | Laplacian / Median / Seed / Context / Filtered / Verified 除錯視圖 |
| `E` | 切換筆刷編輯模式：左鍵拖曳補畫、右鍵拖曳擦除、滾輪調整筆刷大小 |
| `[` / `]` | 筆刷半徑 − / + |
| `C` | 清除所有筆刷編輯 |
| `Z` | 切換放大鏡；滾輪調整倍率 |
| `Ctrl` + 滾輪 | 以滑鼠位置為中心縮放畫布 |
| `R` | 畫布縮放歸零 |
| `S` | 將目前視圖存到原圖旁（若有編輯，一併存遮罩） |
| *Apply Edits & Save Repaired* 按鈕 | 套用筆刷編輯、以原解析度修補並存檔 |
| `A` / `D` | 上一張 / 下一張 |
| `Q` / `Esc` | 退出 |

---

## 圖片修補流程詳解

完整流程圖請見 [英文版 README](README.md#how-it-works)。

當一張帶有 VAE 異常像素的影像進入節點，會經歷以下 8 個步驟。每個步驟對應一個獨立模組，每個模組只做一件事，並把成果交給下一個模組。

#### Step 0 — GPU 影像搬到 CPU

ComfyUI 內部以 PyTorch tensor 在 GPU 上傳遞影像，但 OpenCV 是 CPU 工具。`TensorBridge` 負責把影像搬到 CPU、轉成 OpenCV 慣用的 BGR uint8 格式。

#### Step 1 — 同時用兩種濾鏡找出可疑像素

VAE 的異常像素有兩種常見型態：(a) 邊緣**銳利**的單點亮／暗點；(b) 顏色**平整**、但與周遭格格不入的小色塊。一種濾鏡很難同時抓到兩者，所以同時跑兩條互補路徑：

- **Laplacian 路徑**：Laplacian 是衡量「像素與四周相差多劇烈」的二階微分濾鏡。極亮、極暗或純色塊的硬邊都會產生強烈反應，連續漸層則反應微弱。這條路徑專門抓**邊緣銳利**型的異常像素。
- **Median 殘差路徑**：中值濾波以鄰域中位數取代像素值，對一兩顆極端像素幾乎免疫。原圖減去中值濾波結果，殘差只會在「格格不入的單點／小色塊」上明顯。這條路徑專門抓**邊緣平緩、但顏色明顯錯開背景**的異常像素。

#### Step 2 — 用兩個閾值產生兩張遮罩

- 較**嚴格**的閾值 → **核心遮罩 (seed mask)**：這裡是否真有強烈異常的核心？
- 較**寬鬆**的閾值 → **輪廓遮罩 (context mask)**：異常區域連同光暈有多大？

下一步的篩選同時需要這兩種資訊。

#### Step 3 — 連通域分析 + 形狀過濾

用**連通域分析 (CCA)** 把相連的像素歸為同一個 blob，逐一檢查：

1. **大小**：輪廓面積不可超過最大噪點尺寸太多，否則視為大物件。
2. **核心**：必須含有核心遮罩的種子像素，且種子面積不超過最大噪點尺寸。
3. **形狀**：最小外接旋轉矩形的長寬比若超過 3:1，視為頭髮、細線、刮痕等**細長結構**，捨棄。

#### Step 4 — 用 LAB 色彩空間區分異常像素與自然亮點

星星、燈光、金屬高光等自然亮點，外形跟異常像素幾乎一樣：小、亮、孤立。兩者主要的差別在**亮度變化型態**：自然亮點是**漸層的山丘**，由中心逐步降回背景；異常像素則是**斷崖式的脈衝**，鄰近像素直接就是背景亮度。

影像先轉到 **CIE-LAB**（亮度 L\* 與色彩 a\*、b\* 分開），對每個候選 blob 做兩道檢查，任一成立即判定為異常像素：

- **亮度型態**：比較峰值、blob 平均亮度與周圍環帶亮度，判斷是斷崖還是山丘。
- **色彩偏移**：blob 平均色彩與周圍差距過大（例如純紫、純綠飄在膚色上），即使沒有亮度差也判定為異常。

```
自然亮點（漸層衰減）                  異常像素（脈衝）
  80 150 200 150  80                    10  10 255  10  10
 150 250 255 250 150                    10  10 255  10  10
  80 150 200 150  80                    10  10  10  10  10
→ 山丘：保留                           → 斷崖：修補
```

#### Step 5 — 遮罩向外膨脹

異常像素的污染常不只核心一點。若直接修補，演算法會在遮罩外側取樣到尚未清乾淨的像素，形成淡色暈圈。**形態學膨脹**把遮罩往外推幾個像素，確保修補邊界停在乾淨區域。

#### Step 6 — Telea 修補

**Telea 快速行進法 (FMM)** 從遮罩邊界一圈圈往內推進，每個像素以周圍已知像素加權平均填色，並沿**等照度線 (isophote)** 平滑延伸。對 1–5 像素的微小破洞，Telea 速度極快（毫秒級）、結果可重現，且不會像 PatchMatch 從遠處抓到不相關紋理。修補半徑依最大噪點尺寸換算並限制在 2–7。遮罩為空時直接跳過。

#### Step 7 — 挑出使用者要看的視圖

11 種視圖都會保留，依 `preview_mode` 輸出其中之一。

#### Step 8 — CPU 結果搬回 GPU

修補結果轉回 RGB float32，連同遮罩包裝成 ComfyUI 的 IMAGE + MASK 輸出。

---

## 限制

- VAE 崩壞極其嚴重（大面積網格狀噪點）時，可能需要手動拉高 `max_noise_size`
- 4K 以上解析度，CCA 與 LAB 迴圈需要額外 CPU 時間（通常 1–2 秒內）
- 本節點僅做像素空間後處理，無法從根本解決潛在空間的 VAE 頻寬限制
- 高光保護是啟發式的：較小或較亮的星星主要是靠大小與孤立性規則避開，而非 LAB 檢查。邊緣銳利的亮點（例如 2×2 純白星星）與異常像素無法區分，會被修掉
- 亮背景上的暗點（純黑死點）雖會被 Laplacian 路徑抓到，但 LAB 階段只接受比周圍亮的 blob，因此目前不會被修補

---

## 檔案結構與測試資料

見[英文版 README](README.md#project-structure)。`dataset/` 含 4 組 SDXL 生成影像：

| Group | 場景 | 解析度 | 挑戰點 |
|-------|------|--------|--------|
| A | 賽博龐克少女肖像，純黑背景 | 768×1280 / 2K | 暗背景大量死白點 |
| B | 賽博龐克夜街，霓虹燈 | 768×1280 / 1024² / 2K | 高頻紋理（霓虹、反射、濕地面），易誤判 |
| C | 銀河星空，前景山脈剪影 | 768×1280 / 1024² / 2K | 邊界案例：滿天真實星星 |
| D | 動漫風格少女肖像，乾淨暗背景 | 768×1280 / 1024² / 2K | 平面上色，噪點較少 |

## 開發

```bash
pip install opencv-python numpy pillow pytest torch
pytest                         # 單元測試 + 節點測試
python scripts/run_dataset.py  # 對 dataset/ 批次輸出 *_mask.png / *_fixed.png 到 test_output/ 供目視檢查
```

上架 Comfy Registry 的步驟見[英文版 README](README.md#publishing-to-the-comfy-registry-maintainer)。

---

## 授權

[MIT](LICENSE) © 2026 Phecda

## 致謝

本專案源自國立陽明交通大學「影像編修技術與特效合成」課程期末專題。課程繳交的簡報、報告與相關素材封存於 tag [`term-project-spring-2026`](https://github.com/megrez33281/ComfyUI-VAE-Noise-Fix/tree/term-project-spring-2026)。
