# dnc_crane_detection2

## OCR倍率連携の概要

本リポジトリには、YOLOv5ベースの危険検知処理と、倍率表示をOCRで取得する仕組みが含まれています。

- `read_multiplier.py` は、映像内の「×」記号の右隣にある倍率値を高精度に読み取るための専用スクリプトです。固定ROIや自動キャリブレーション、色マスク、テンプレートマッチを組み合わせて、1〜30倍の整数を抽出します。
- `detect_videos_xoko.py` は、別仮想環境で稼働する `read_magnification_video_complete_fixed.py` をサブプロセス起動し、得られた `ocr_value` を焦点距離に変換してクレーン危険検知ロジックに反映します。

## 代表的な実行例

### 倍率読み取り（ROI手動指定）
```bash
python read_multiplier.py --video "/mnt/data/ズーム動画.mp4" --fps 5 --roi 1450 960 130 60 --digit-offset 5 --min-digit-width 60 --save-annotated
```
- `--roi` : 「×」記号左上の座標と、右側の数字が収まる幅・高さをピクセルで指定します。
- `--digit-offset` : 「×」の右端から何ピクセル空けて数字帯域を開始するかを設定します。
- `--min-digit-width` : 1桁/2桁どちらでも読めるよう最低限確保する帯域幅を指定します。
- `--save-annotated` : yellow=ROI、orange=数字帯域で描画した画像を `out_frames/` に保存します。

### YOLO推論＋倍率連携
```bash
python detect_videos_xoko.py --weights yolov5s.pt --source cam.mp4 \
    --ocr-python "C:/dnc/AI_env_check/ocr_venv/Scripts/python.exe" \
    --ocr-script "C:/dnc/AI_env_check/read_magnification_video_complete_fixed.py" \
    --ocr-extra-args "--summary" --ocr-cache-ttl 5
```
- `--ocr-python` : OCR用仮想環境のPython実行ファイル。
- `--ocr-script` : `ocr_value` を出力するOCRスクリプトパス。
- `--ocr-cache-ttl` : OCR呼び出し間隔の最小秒数。値をキャッシュしてサブプロセス起動を抑制します。
- 取得した倍率は `focal_length = 6 + (180-6)*(ocr_value-1)/29` で焦点距離へ換算され、像高推定に利用されます。

## 参照
- `ocr_system.py` : OCRの設定例や補助関数。
- `detect_videos_beep.py` : 元になったYOLO推論スクリプト。
