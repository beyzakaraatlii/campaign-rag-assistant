"""Local Streamlit UI for PDF ingestion, retrieval and grounded answers."""
import config  # Set project-local cache paths before loading model libraries.
import streamlit as st

from database import index_status, rebuild_database
from ingest import PDFValidationError
from llm import answer_from_results, setup_model, source_label
from retrieval import retrieve
from uploads import save_uploaded_pdf

st.set_page_config(page_title="Campaign RAG Assistant", page_icon="📣", layout="centered")

st.title("📣 Campaign RAG Assistant")
st.caption("Microsoft Foundry Local · Phi-4-mini · Çok dilli RAG · SQLite")
st.write("Kampanya sorularınızı kendi PDF kaynaklarınıza dayanarak yanıtlayın.")


@st.cache_resource(show_spinner=False)
def load_model():
    return setup_model()


with st.sidebar:
    st.header("Bilgi tabanı")
    st.caption("Metin içeren PDF · En fazla 25 MB / dosya · 500 sayfa")
    uploads = st.file_uploader("PDF kaynakları", type=["pdf"], accept_multiple_files=True)
    if st.button("PDF'leri kaydet ve indeksle", disabled=not uploads, use_container_width=True):
        accepted = False
        with st.spinner("PDF'ler kontrol ediliyor..."):
            for upload in uploads:
                try:
                    result = save_uploaded_pdf(upload.name, upload.getvalue())
                    accepted = True
                    if result.duplicate:
                        st.info(f"Zaten mevcut: {result.path.name}")
                    else:
                        st.success(f"Kaydedildi: {result.path.name}")
                except (PDFValidationError, ValueError, OSError) as exc:
                    st.error(f"{upload.name}: {exc}")
        if accepted:
            try:
                with st.spinner("Bilgi tabanı güncelleniyor..."):
                    count = rebuild_database()
                st.session_state.pop("response", None)
                st.success(f"Bilgi tabanı güncel: {count} metin parçası.")
            except Exception as exc:
                st.error("PDF kaydedildi ancak indeksleme tamamlanamadı. Mevcut indeks korundu. "
                         "Sorunu giderip yeniden indeksleyin.")
                with st.expander("Hata ayrıntısı"):
                    st.text(str(exc))

    if st.button("Bilgi tabanını yeniden indeksle", use_container_width=True):
        try:
            with st.spinner("PDF'ler yeniden indeksleniyor..."):
                count = rebuild_database()
            st.session_state.pop("response", None)
            st.success(f"İndeks güncellendi: {count} metin parçası.")
        except Exception as exc:
            st.error("Yeniden indeksleme başarısız. Mevcut indeks korundu.")
            with st.expander("Hata ayrıntısı"):
                st.text(str(exc))

    try:
        status = index_status()
        st.metric("PDF dosyası", status["documents"])
        st.metric("İndekslenen metin parçası", status["chunks"])
        if status["stale"]:
            st.warning("PDF klasörü ile indeks eşleşmiyor. Yeniden indeksleme gerekli; "
                       "sorular son başarılı indeks üzerinden yanıtlanır.")
        elif status["chunks"]:
            st.success("Bilgi tabanı güncel.")
        else:
            st.info("Başlamak için bir PDF yükleyin.")
    except Exception as exc:
        st.error("Bilgi tabanı durumu okunamadı.")
        st.text(str(exc))
    st.caption("Aynı içerik yeniden eklenmez. Aynı adlı farklı PDF ayrı adla saklanır.")
    st.caption("İlk kullanımda model dosyaları internetten indirilir ve proje içinde saklanır.")


with st.form("question_form"):
    question = st.text_area(
        "Kampanyan hakkında ne öğrenmek istiyorsun?",
        placeholder="Örnek: TikTok reklamlarında ilk saniyelerde dikkat nasıl çekilir?",
        height=120,
    )
    ask = st.form_submit_button("Yanıtla", type="primary", use_container_width=True)

if ask:
    st.session_state.pop("response", None)
    if not question.strip():
        st.warning("Lütfen önce bir soru yazın.")
    elif len(question) > 4000:
        st.warning("Soruyu 4000 karakterden kısa tutun.")
    else:
        try:
            with st.spinner("İlgili kaynaklar aranıyor..."):
                results = retrieve(question)
            # Empty retrieval never starts or downloads the generation model.
            if results:
                with st.spinner("Yerel model hazırlanıyor ve yanıt üretiliyor..."):
                    answer, _ = answer_from_results(load_model(), question, results)
            else:
                answer, _ = answer_from_results(None, question, [])
            st.session_state["response"] = {
                "question": question, "answer": answer, "results": results,
            }
        except Exception as exc:
            st.error("Yanıt hazırlanamadı. Model kurulumunu ve bilgi tabanını kontrol edip tekrar deneyin.")
            with st.expander("Hata ayrıntısı"):
                st.text(str(exc))

if "response" in st.session_state:
    response = st.session_state["response"]
    st.subheader("Yanıt")
    st.caption(f"Soru: {response['question']}")
    st.markdown(response["answer"])  # Model output is never rendered as unsafe HTML.
    if response["results"]:
        st.subheader("Modele verilen kaynaklar")
        st.caption("Bu liste retrieval sonuçlarıdır; her parçanın yanıtta kullanıldığını garanti etmez. "
                   "Benzerlik skoru doğruluk olasılığı değildir.")
        for i, result in enumerate(response["results"], 1):
            with st.expander(f"[{i}] {source_label(result)}"):
                st.caption(f"Benzerlik: {result['score']:.3f}")
                st.text(result["text"])

st.divider()
st.caption("Yanıtları kaynak parçalarıyla kontrol edin. Taranmış PDF'ler için OCR gerekir.")
