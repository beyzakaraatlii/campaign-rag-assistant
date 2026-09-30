# Yerel doğrulama raporu

Tarih: 29 Eylül 2026. Çalışma kapsamı: yalnızca `campaign-rag-assistant` proje klasörü. GitHub push, publish, upload veya paylaşım yapılmadı.

## Değişiklikler

| Dosya | Değişiklik |
|---|---|
| `app.py` | Çoklu PDF yükleme, yeniden indeksleme, durum, hata yönetimi, kalıcı yanıt ve açılabilir kaynak parçaları |
| `config.py` | Projeye bağlı yollar, PDF limitleri ve proje içi model cache ayarları |
| `ingest.py` | PDF doğrulama, sayfa numarası, güvenli okuma ve parçalama |
| `database.py` | Eski şemayla uyumluluk, işlem içinde indeks değiştirme ve belge manifesti |
| `retrieval.py` | Ortak embedding, boş indeks davranışı, boyut kontrolü, sıralama/eşik |
| `vector_store.py` | Embedding modelinin tembel ve paylaşılan yüklenmesi |
| `llm.py` | Foundry/Phi-4-mini cache'i, kaynaklı prompt, boş yanıt kontrolü |
| `models.py` | Import sırasında model başlatılmasının kaldırılması |
| `uploads.py` | Yeni: ad doğrulama, hash ile tekrar kontrolü, üzerine yazmadan kayıt |
| `.streamlit/config.toml` | Yeni: localhost, yükleme limiti, telemetry kapalı, tema |
| `requirements.txt` | Yeni: bu ortamda kullanılan doğrudan bağımlılık sürümleri |
| `.gitignore` | Yeni: belge, indeks, cache, yedek ve sırların dışlanması |
| `README.md` | Yeni: kurulum, mimari, kullanım, test ve sınırlamalar |
| `tests/__init__.py`, `tests/helpers.py`, `tests/test_rag.py` | Yeni: sentetik PDF/istemci ve 24 deterministik test |
| `tests/smoke_local.py` | Yeni: gerçek embedding ve isteğe bağlı Phi-4-mini testi |
| `data/documents/.gitkeep` | Yeni: boş belge klasörü için yer tutucu |
| `data/campaign_rag.db` | Üç mevcut PDF'den 141 parça ile yeniden indekslendi |

Eski Python dosyaları, veritabanı ve üç PDF `.local-backup/` altında tutuldu. Üç PDF'nin yedekleriyle SHA-256 değerleri aynı: kullanıcı belgelerinin içerikleri değişmedi.

## Geçen kontroller

- `python -m unittest discover -s tests -v`: **24/24 geçti**.
- Gerçek PDF ayrıştırma; boş, bozuk, şifreli, aşırı büyük ve fazla sayfalı PDF reddi.
- Güvensiz dosya adları, proje dışına çıkan yollar, yinelenen içerik ve aynı adlı farklı içerik.
- Sayfa/örtüşme aktarımı; yeni PDF → indeks → retrieval → üretim istemcisine bağlam aktarımı.
- Tekrar indekslemenin kayıt çoğaltmaması; embedding/SQL/PDF hatasında eski indeksin korunması.
- Eski SQLite şemasının okunması ve yeni şemaya yükseltilmesi; embedding model uyumsuzluğu.
- Eksik/boş indeks, boş soru, boş model yanıtı ve model cache davranışı.
- Streamlit AppTest: yükleme → indeksleme → soru → cevap/kaynak → rerun sonrası yanıtın korunması; hata gösterimi.
- `python -m tests.smoke_local`: gerçek embedding ile İngilizce ve Türkçe sorularda yeni sentetik PDF ilk sırada bulundu. İngilizce sorgu benzerliği yaklaşık **0.7781**.
- Mevcut PDF'ler: **3 belge, 79 metin içeren sayfa, 141 parça**; indeks güncel.
- `python -m compileall`: sözdizimi kontrolü geçti.
- `python -m pip check`: bozuk bağımlılık yok. Windows için requirements pinleri kurulu sürümlerle eşleşiyor.
- Yerel Streamlit: `http://127.0.0.1:8502`, `/_stcore/health` HTTP **200**. Tarayıcıda arayüz ve güncel indeks görüntülendi. 8501 kullanımda olduğundan 8502 seçildi.

## Test kapsamının sınırları

- Hızlı testlerde embedding ve üretim istemcisi kontrollü test nesneleridir. Gerçek embedding ayrıca smoke testiyle sınandı.
- Streamlit dosya yükleme sınırı AppTest'te enjekte edildi. In-app tarayıcı otomasyonu dosya seçici olayını alamadı; tarayıcıdan gerçek dosya seçme eylemi otomatik doğrulanamadı.
- OCR, ağır/zararlı PDF'ler için süreç izolasyonu, çok süreçli eşzamanlı kullanım ve tam çevrimdışı mod kapsam dışıdır.
- Model atıflarının her iddiayı gerçekten desteklediği otomatik olarak doğrulanmaz.

## Gerçek Phi-4-mini testi

`python -m tests.smoke_local --with-llm`: **geçti**. Gerçek embedding modeli yeni PDF'yi ilk sırada getirdi; gerçek Microsoft Foundry Local / Phi-4-mini yanıtı **`[1] ORBIT-739`** oldu. Kaynak **`new_campaign.pdf — s. 1`** olarak döndü. Sonuç `.test-artifacts/smoke-result.json` dosyasında tutuluyor.

İlk deneme sandbox içinde Foundry katalog erişiminde Windows TLS/SSPI hatası verdi. İzin verilen sandbox dışı tekrar katalog erişimini sağladı ve Phi-4-mini'yi proje içindeki `.cache/foundry/models/` klasörüne indirerek testi tamamladı. Başka model sağlayıcısı kullanılmadı; sertifika doğrulaması kapatılmadı. Test belgeleri ayrı geçici klasör/veritabanında kullanıldı; ana bilgi tabanında yalnızca mevcut üç PDF bulunuyor.
