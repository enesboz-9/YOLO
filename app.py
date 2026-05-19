"""
YOLOv8 Gerçek Zamanlı Nesne Tespiti - Gradio Uygulaması
========================================================
Bu uygulama, özel eğitilmiş YOLOv8m modeliyle araç tespiti yapar.
Tespit edilen sınıflar: araba, motorsiklet, kamyon, otobüs, bisiklet

Hugging Face Spaces (CPU tier) üzerinde çalışacak şekilde optimize edilmiştir.
"""

import gradio as gr
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import time
from collections import defaultdict
from huggingface_hub import hf_hub_download

# ─────────────────────────────────────────────
# YAPILANDIRMA — buraya kendi bilgilerini gir
# ─────────────────────────────────────────────
HF_REPO_ID    = "KULLANICI_ADIN/MODEL_REPO_ADIN"   # ← değiştir
MODEL_FILE    = "best.pt"
MAP50_SKORU   = "XX.X"                              # ← değiştir (örn. "87.4")
GITHUB_LINK   = "https://github.com/KULLANICI/REPO" # ← değiştir
MAX_VIDEO_SN  = 30   # Maksimum video süresi (saniye)
FRAME_ATLAMA  = 2    # Video işlemede her kaçıncı frame işlensin (1 = hepsi)

# ─────────────────────────────────────────────
# SINIF TANIMLAMALARI
# ─────────────────────────────────────────────
SINIFLAR = {
    0: {"ad": "araba",       "renk": (59,  130, 246)},  # mavi
    1: {"ad": "motorsiklet", "renk": (249, 115,  22)},  # turuncu
    2: {"ad": "kamyon",      "renk": (239,  68,  68)},  # kırmızı
    3: {"ad": "otobüs",      "renk": ( 34, 197,  94)},  # yeşil
    4: {"ad": "bisiklet",    "renk": (168,  85, 247)},  # mor
}

# ─────────────────────────────────────────────
# MODEL YÜKLEME (global, tek seferlik)
# ─────────────────────────────────────────────
model = None

def modeli_yukle():
    """
    YOLOv8 modelini Hugging Face Hub'dan indirir ve yükler.
    Uygulama başlarken bir kez çalışır; hata durumunda None döner.
    """
    global model
    try:
        from ultralytics import YOLO
        print(f"[BİLGİ] Model indiriliyor: {HF_REPO_ID}/{MODEL_FILE}")
        model_yolu = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
        model = YOLO(model_yolu)
        print("[BİLGİ] Model başarıyla yüklendi ✓")
    except Exception as hata:
        print(f"[HATA] Model yüklenemedi: {hata}")
        model = None

# Uygulama başlangıcında yükle
modeli_yukle()

# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def kutu_ciz(goruntu_np: np.ndarray, tespitler) -> np.ndarray:
    """
    NumPy görüntüsü üzerine renkli bounding box ve etiket çizer.

    Args:
        goruntu_np: BGR formatında NumPy dizisi
        tespitler : Ultralytics inference sonucu

    Returns:
        Çizilmiş BGR NumPy dizisi
    """
    cizim = goruntu_np.copy()

    for kutu in tespitler[0].boxes:
        sinif_id  = int(kutu.cls[0])
        guven     = float(kutu.conf[0])
        x1, y1, x2, y2 = map(int, kutu.xyxy[0])

        sinif_bilgi = SINIFLAR.get(sinif_id, {"ad": str(sinif_id), "renk": (255, 255, 255)})
        renk  = sinif_bilgi["renk"]         # (R, G, B)
        bgr   = (renk[2], renk[1], renk[0]) # OpenCV → BGR
        etiket = f"{sinif_bilgi['ad']}  {guven:.0%}"

        # Kutu çiz
        kalinlik = max(2, int((x2 - x1) / 150))
        cv2.rectangle(cizim, (x1, y1), (x2, y2), bgr, kalinlik)

        # Etiket arka planı
        (genislik, yukseklik), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        etiket_y = max(y1 - 10, yukseklik + 6)
        cv2.rectangle(cizim, (x1, etiket_y - yukseklik - 6),
                      (x1 + genislik + 6, etiket_y + 2), bgr, -1)

        # Etiket metni
        cv2.putText(cizim, etiket, (x1 + 3, etiket_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return cizim


def tespit_tablosu_olustur(tespitler) -> pd.DataFrame:
    """
    Tespit sonuçlarından okunabilir bir özet tablo üretir.

    Returns:
        Sınıf adı | Tespit Sayısı | Ort. Güven Skoru içeren DataFrame
    """
    sayac      = defaultdict(list)
    for kutu in tespitler[0].boxes:
        sinif_id = int(kutu.cls[0])
        guven    = float(kutu.conf[0])
        ad       = SINIFLAR.get(sinif_id, {"ad": str(sinif_id)})["ad"]
        sayac[ad].append(guven)

    satirlar = []
    for ad, guvenler in sorted(sayac.items()):
        satirlar.append({
            "🚗 Sınıf":              ad,
            "📦 Tespit Sayısı":      len(guvenler),
            "📊 Ort. Güven Skoru":   f"{np.mean(guvenler):.1%}",
        })

    if not satirlar:
        return pd.DataFrame(columns=["🚗 Sınıf", "📦 Tespit Sayısı", "📊 Ort. Güven Skoru"])
    return pd.DataFrame(satirlar)


# ─────────────────────────────────────────────
# SEKME 1 — GÖRÜNTÜ TESPİTİ
# ─────────────────────────────────────────────

def goruntu_tespiti(goruntu: np.ndarray, confidence: float):
    """
    Yüklenen görüntü üzerinde nesne tespiti yapar.

    Args:
        goruntu   : Gradio'dan gelen RGB NumPy dizisi
        confidence: Minimum güven eşiği (0.1 – 0.9)

    Returns:
        (sonuc_goruntu_pil, tablo_df, durum_metni)
    """
    # Girdi doğrulama
    if goruntu is None:
        raise gr.Error("❌ Lütfen bir görüntü yükleyin.")

    if model is None:
        raise gr.Error("❌ Model yüklenemedi. Hugging Face Hub bağlantısını kontrol edin.")

    # RGB → BGR (OpenCV formatı)
    bgr = cv2.cvtColor(goruntu, cv2.COLOR_RGB2BGR)

    # Çıkarım
    baslangic = time.time()
    tespitler = model(bgr, conf=confidence, verbose=False)
    sure_ms   = (time.time() - baslangic) * 1000

    tespit_sayisi = len(tespitler[0].boxes)

    # Kutu çizimi
    sonuc_bgr = kutu_ciz(bgr, tespitler)
    sonuc_rgb = cv2.cvtColor(sonuc_bgr, cv2.COLOR_BGR2RGB)
    sonuc_pil = Image.fromarray(sonuc_rgb)

    # Tablo
    tablo = tespit_tablosu_olustur(tespitler)

    # Durum mesajı
    if tespit_sayisi == 0:
        durum = f"ℹ️ Hiç nesne tespit edilemedi. Güven eşiğini ({confidence:.0%}) düşürmeyi deneyin."
    else:
        durum = f"✅ {tespit_sayisi} nesne tespit edildi — İşlem süresi: {sure_ms:.0f} ms"

    return sonuc_pil, tablo, durum


# ─────────────────────────────────────────────
# SEKME 2 — VİDEO TESPİTİ
# ─────────────────────────────────────────────

def video_isle(video_yolu: str, confidence: float, frame_atla: bool):
    """
    Verilen video dosyasını frame frame işler, üzerine tespit kutuları çizer.

    Args:
        video_yolu: Geçici video dosyasının yolu
        confidence: Minimum güven eşiği
        frame_atla: True ise her 2. frame işlenir (hız artırır)

    Returns:
        (cikti_video_yolu, durum_metni)
    """
    # Girdi doğrulama
    if video_yolu is None:
        raise gr.Error("❌ Lütfen bir video yükleyin.")

    if model is None:
        raise gr.Error("❌ Model yüklenemedi. Hugging Face Hub bağlantısını kontrol edin.")

    cap = cv2.VideoCapture(video_yolu)
    if not cap.isOpened():
        raise gr.Error("❌ Video dosyası açılamadı. Desteklenen formatlar: mp4, avi, mov")

    # Video meta bilgileri
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    toplam_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    genislik     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    yukseklik    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sure_sn      = toplam_frame / fps

    # Süre kontrolü
    if sure_sn > MAX_VIDEO_SN:
        cap.release()
        raise gr.Error(
            f"❌ Video çok uzun ({sure_sn:.0f} saniye). "
            f"Maksimum izin verilen süre {MAX_VIDEO_SN} saniyedir."
        )

    # Çıktı video dosyası
    cikti_dosya = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    cikti_yolu  = cikti_dosya.name
    cikti_dosya.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(cikti_yolu, fourcc, fps, (genislik, yukseklik))

    atlama_adimi = FRAME_ATLAMA if frame_atla else 1
    toplam_tespit = 0
    islenen_frame = 0
    onceki_frame  = None  # Atlanan frame için önceki sonucu tekrar yaz
    baslangic     = time.time()

    while True:
        basarili, frame = cap.read()
        if not basarili:
            break

        if islenen_frame % atlama_adimi == 0:
            tespitler    = model(frame, conf=confidence, verbose=False)
            sonuc_frame  = kutu_ciz(frame, tespitler)
            toplam_tespit += len(tespitler[0].boxes)
            onceki_frame  = sonuc_frame
        else:
            # Atlanan frame: önceki işlenmiş frame'i kullan
            sonuc_frame = onceki_frame if onceki_frame is not None else frame

        writer.write(sonuc_frame)
        islenen_frame += 1

    cap.release()
    writer.release()

    sure_toplam = time.time() - baslangic
    durum = (
        f"✅ Video işlendi! "
        f"⏱️ Süre: {sure_toplam:.1f}s | "
        f"🎬 Frame: {islenen_frame} | "
        f"📦 Toplam tespit: {toplam_tespit}"
    )
    return cikti_yolu, durum


# ─────────────────────────────────────────────
# GRADIO ARAYÜZÜ
# ─────────────────────────────────────────────

# CSS — mobil uyumlu, temiz görünüm
OZEL_CSS = """
.baslik { text-align: center; margin-bottom: 8px; }
.aciklama { text-align: center; color: #6b7280; margin-bottom: 16px; }
.footer { text-align: center; font-size: 0.85rem; color: #9ca3af; margin-top: 20px; }
.durum-kutu { padding: 10px; border-radius: 8px; background: #f0fdf4; }
"""

ACIKLAMA_METNI = f"""
<div class="baslik"><h2>🚗 YOLOv8 Araç Tespit Sistemi</h2></div>
<div class="aciklama">
Özel eğitilmiş <b>YOLOv8m</b> modeliyle 5 araç sınıfını gerçek zamanlı tespit eder.<br>
<b>Sınıflar:</b> 
<span style='color:#3b82f6'>■ araba</span> &nbsp;
<span style='color:#f97316'>■ motorsiklet</span> &nbsp;
<span style='color:#ef4444'>■ kamyon</span> &nbsp;
<span style='color:#22c55e'>■ otobüs</span> &nbsp;
<span style='color:#a855f7'>■ bisiklet</span>
</div>
"""

FOOTER_METNI = f"""
<div class="footer">
📊 mAP50: <b>{MAP50_SKORU}%</b> &nbsp;|&nbsp;
🤖 Model: YOLOv8m &nbsp;|&nbsp;
<a href="{GITHUB_LINK}" target="_blank">GitHub ↗</a>
</div>
"""

# Örnek görseller (varsa; yoksa boş liste bırak)
ORNEK_GORSELLER = []
# Örnek: ORNEK_GORSELLER = [["examples/araba.jpg", 0.4], ["examples/kamyon.jpg", 0.5]]

with gr.Blocks(css=OZEL_CSS, title="YOLOv8 Araç Tespiti") as demo:

    gr.HTML(ACIKLAMA_METNI)

    with gr.Tabs():

        # ── SEKME 1: Görüntü ─────────────────────
        with gr.Tab("🖼️ Görüntü Tespiti"):
            with gr.Row():
                with gr.Column(scale=1):
                    goruntu_girdi = gr.Image(
                        label="Görüntü Yükle (JPG / PNG / WEBP)",
                        type="numpy",
                        image_mode="RGB",
                    )
                    conf_slider_g = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                        label="Güven Eşiği (Confidence Threshold)",
                        info="Düşük değer → daha fazla tespit (daha fazla yanlış pozitif riski)",
                    )
                    tespit_btn = gr.Button("🔍 Tespit Et", variant="primary")

                with gr.Column(scale=1):
                    goruntu_cikti = gr.Image(label="Tespit Sonucu", type="pil")
                    durum_goruntu = gr.Textbox(label="Durum", interactive=False)
                    tablo_cikti   = gr.Dataframe(
                        label="Tespit Özeti",
                        headers=["🚗 Sınıf", "📦 Tespit Sayısı", "📊 Ort. Güven Skoru"],
                    )

            tespit_btn.click(
                fn=goruntu_tespiti,
                inputs=[goruntu_girdi, conf_slider_g],
                outputs=[goruntu_cikti, tablo_cikti, durum_goruntu],
            )

            if ORNEK_GORSELLER:
                gr.Examples(
                    examples=ORNEK_GORSELLER,
                    inputs=[goruntu_girdi, conf_slider_g],
                    label="📂 Örnek Görseller",
                )

        # ── SEKME 2: Video ───────────────────────
        with gr.Tab("🎬 Video Tespiti"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_girdi = gr.Video(
                        label=f"Video Yükle (MP4 / AVI / MOV — max {MAX_VIDEO_SN} sn)",
                    )
                    conf_slider_v = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.4, step=0.05,
                        label="Güven Eşiği (Confidence Threshold)",
                    )
                    frame_atla_chk = gr.Checkbox(
                        label="⚡ Hızlı Mod (her 2. frame — CPU için önerilir)",
                        value=True,
                    )
                    video_btn = gr.Button("▶️ Videoyu İşle", variant="primary")
                    gr.Markdown(
                        f"> ⚠️ CPU üzerinde video işleme **{MAX_VIDEO_SN} saniyeye** kadar "
                        "desteklenir. Uzun videolar reddedilecektir."
                    )

                with gr.Column(scale=1):
                    video_cikti  = gr.Video(label="İşlenmiş Video")
                    durum_video  = gr.Textbox(label="Durum", interactive=False)

            video_btn.click(
                fn=video_isle,
                inputs=[video_girdi, conf_slider_v, frame_atla_chk],
                outputs=[video_cikti, durum_video],
            )

    gr.HTML(FOOTER_METNI)

# ─────────────────────────────────────────────
# BAŞLATMA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
