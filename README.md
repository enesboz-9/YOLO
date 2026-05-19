---
title: YOLOv8 Araç Tespiti
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
license: mit
tags:
  - object-detection
  - yolov8
  - computer-vision
  - gradio
  - turkish
short_description: YOLOv8m ile gerçek zamanlı araç tespiti (araba, kamyon, otobüs…)
---

# 🚗 YOLOv8 Gerçek Zamanlı Nesne Tespiti

Özel eğitilmiş **YOLOv8m** modeliyle 5 araç sınıfını tespit eden Gradio uygulaması.

## Tespit Edilen Sınıflar

| Sınıf | Renk |
|-------|------|
| 🔵 araba | Mavi |
| 🟠 motorsiklet | Turuncu |
| 🔴 kamyon | Kırmızı |
| 🟢 otobüs | Yeşil |
| 🟣 bisiklet | Mor |

## Özellikler

- **Görüntü Tespiti** — JPG/PNG/WEBP yükle, anında sonuç al
- **Video Tespiti** — MP4/AVI/MOV (max 30 sn), işlenmiş video indir
- Ayarlanabilir güven eşiği (0.1 – 0.9)
- Sınıf bazlı renkli bounding box
- Tespit özet tablosu (sınıf, sayı, ortalama güven)

## Model Bilgileri

- **Mimari:** YOLOv8m
- **Framework:** Ultralytics 8.x, PyTorch 2.0
- **mAP50:** [skorunu buraya yaz]%
- **Eğitim Verisi:** Özel veri seti (5 sınıf, Türkçe etiket)

## Nasıl Kullanılır

1. **Görüntü Tespiti** sekmesine geç
2. Bir araç görseli yükle (ya da örnek görsellerden birini seç)
3. Güven eşiğini ayarla (varsayılan: 0.40)
4. **Tespit Et** butonuna tıkla

## Teknik Detaylar

- Inference: CPU (Hugging Face ücretsiz tier)
- Video: Frame atlama destekli (hız optimizasyonu)
- Dil: Türkçe arayüz

## Linkler

- 📂 Model: `KULLANICI_ADIN/MODEL_REPO_ADIN`
- 💻 Kaynak Kod: [GitHub](https://github.com/KULLANICI/REPO)
