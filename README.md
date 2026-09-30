# Campaign RAG Assistant

PDF tabanlı reklam ve dijital pazarlama kaynaklarından yararlanarak soruları yanıtlayan yerel bir RAG uygulaması.

Yanıt üretimi **Microsoft Foundry Local + Phi-4-mini**, çok dilli semantik arama **Sentence Transformers**, kalıcı vektör indeksi ise **SQLite** ile gerçekleştirilir.

Uygulama, yüklenen PDF'lerden ilgili kaynak parçalarını bulur ve Phi-4-mini modeline yalnızca bu kaynakları vererek kaynak tabanlı Türkçe yanıt üretir.

## Özellikler

- Streamlit üzerinden bir veya birden fazla PDF yükleme
- PDF'lerin `data/documents/` içine güvenli kaydedilmesi
- Boş, bozuk, şifreli ve metin içermeyen PDF'ler için hata yönetimi
- SHA-256 ile aynı içeriğin tekrar yüklenmesini engelleme
- Aynı isimli fakat farklı içerikteki PDF'leri koruma
- Sayfa numarası taşıyan kaynak parçaları
- Çok dilli embedding tabanlı semantik retrieval
- `LUNA-582`, `ORBIT-739` gibi özel kodlarda birebir eşleşmeye öncelik verme
- SQLite tabanlı kalıcı indeks
- İndeksleme başarısız olduğunda önceki çalışan indeksi koruma
- Microsoft Foundry Local üzerinden cihaz üzerinde Phi-4-mini çalıştırma
- Kaynak dosya, sayfa ve retrieval benzerlik skorlarını arayüzde gösterme
- Model ve embedding modellerini ihtiyaç anında yükleyip süreç içinde tekrar kullanma
- Yetersiz kaynak bulunduğunda gereksiz LLM çağrısı yapmama

## Mimari

```text
PDF yükleme
    ↓
Dosya doğrulama + SHA-256 kontrolü
    ↓
data/documents/
    ↓
PDF sayfa metni
    ↓
Chunking
    ↓
Multilingual Sentence Transformer
    ↓
SQLite embedding index


Kullanıcı sorusu
    ↓
Sorgu analizi
    ├── Özel kod varsa exact-match önceliği
    └── Genel reklam sorularında query expansion
    ↓
Embedding
    ↓
Cosine similarity retrieval
    ↓
En ilgili kaynak parçaları
    ↓
Microsoft Foundry Local / Phi-4-mini
    ↓
Türkçe yanıt + numaralı kaynaklar + sayfa bilgisi