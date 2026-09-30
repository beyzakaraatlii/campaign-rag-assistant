import sqlite3
import tempfile
import unittest
from contextlib import closing
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import config
import database
import ingest
import llm
import retrieval
import uploads
import vector_store
from tests.helpers import pdf_bytes, fake_encode, RecordingModel


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        root = config.PROJECT_ROOT / '.test-artifacts'
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'documents'
        self.folder.mkdir()
        self.db = Path(self.temp.name) / 'index.db'

    def save(self, name='campaign.pdf', text='ZEBRA campaign code is ORBIT-739.'):
        return uploads.save_uploaded_pdf(name, pdf_bytes(text), self.folder)

    def rebuild(self):
        with patch('database.create_embeddings', side_effect=lambda cs: fake_encode([c['text'] for c in cs])):
            return database.rebuild_database(self.folder, self.db)

    def retrieve(self, query):
        with patch('retrieval.encode_texts', side_effect=fake_encode):
            return retrieval.retrieve(query, self.db)


class PDFTests(WorkspaceTest):
    def test_upload_and_page_metadata(self):
        result = uploads.save_uploaded_pdf('source.PDF', pdf_bytes('ZEBRA', '', 'coffee'), self.folder)
        self.assertEqual(result.path.suffix, '.pdf')
        self.assertEqual([p['page'] for p in ingest.load_documents(self.folder)], [1, 3])

    def test_invalid_pdfs_leave_no_files(self):
        for payload in (b'', b'not a pdf', b'%PDF-1.7\nbroken', pdf_bytes(''), pdf_bytes()):
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(ingest.PDFValidationError):
                    uploads.save_uploaded_pdf('bad.pdf', payload, self.folder)
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_encrypted_pdf(self):
        with self.assertRaisesRegex(ingest.PDFValidationError, 'Şifreli'):
            uploads.save_uploaded_pdf('locked.pdf', pdf_bytes('ZEBRA', encrypted=True), self.folder)

    def test_size_and_page_limits(self):
        with patch('ingest.MAX_PDF_BYTES', 10), self.assertRaises(ingest.PDFValidationError):
            self.save()
        with patch('ingest.MAX_PDF_PAGES', 1), self.assertRaises(ingest.PDFValidationError):
            ingest.extract_pages(pdf_bytes('one', 'two'))

    def test_unsafe_names_rejected(self):
        for name in ('../evil.pdf', '..\\evil.pdf', 'C:\\evil.pdf', '/evil.pdf', 'foo.pdf:bar', 'bad.txt'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                uploads.safe_pdf_name(name)
        self.assertEqual(uploads.safe_pdf_name('CON.pdf'), 'document_CON.pdf')
        self.assertNotIn('<', uploads.safe_pdf_name('<script>.pdf'))

    def test_outside_path_rejected_without_access(self):
        with self.assertRaises(ValueError):
            config.project_path(config.PROJECT_ROOT.parent / 'outside.pdf')

    def test_duplicate_content_different_name(self):
        first = self.save()
        second = self.save('renamed.pdf')
        self.assertTrue(second.duplicate)
        self.assertEqual(first.path, second.path)
        self.assertEqual(len(list(self.folder.iterdir())), 1)

    def test_same_name_different_content_keeps_both(self):
        first = self.save()
        before = first.path.read_bytes()
        second = self.save(text='coffee audience')
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(first.path.read_bytes(), before)

    def test_chunks_preserve_page_and_overlap(self):
        chunks = ingest.create_chunks([{'source': 's', 'page': 2, 'text': 'abcdefghijkl'}], 6, 2)
        self.assertEqual([c['text'] for c in chunks], ['abcdef', 'efghij', 'ijkl'])
        self.assertTrue(all(c['page'] == 2 for c in chunks))
        for size, overlap in ((0, 0), (10, 10), (10, -1)):
            with self.assertRaises(ValueError):
                ingest.create_chunks([], size, overlap)


class RAGTests(WorkspaceTest):
    def test_new_pdf_reaches_retrieval_and_answer_prompt(self):
        self.save('old.pdf', 'coffee audience')
        self.rebuild()
        self.assertEqual(self.retrieve('ZEBRA campaign code'), [])
        self.save('new.pdf')
        self.assertTrue(database.index_status(self.db, self.folder)['stale'])
        self.rebuild()
        results = self.retrieve('ZEBRA campaign code')
        self.assertEqual(results[0]['source'], 'new.pdf')
        model = RecordingModel()
        with patch('llm.retrieve', return_value=results):
            answer, sources = llm.answer_question(model, 'ZEBRA campaign code?')
        self.assertIn('ORBIT-739', answer)
        self.assertIn('ORBIT-739', model.messages[1]['content'])
        self.assertEqual(sources, ['new.pdf — s. 1'])
        self.assertFalse(database.index_status(self.db, self.folder)['stale'])

    def test_repeat_index_is_idempotent(self):
        self.save()
        self.assertEqual(self.rebuild(), self.rebuild())
        self.assertEqual(database.index_status(self.db, self.folder)['chunks'], 1)

    def test_embedding_failure_preserves_index(self):
        self.save()
        self.rebuild()
        before = self.db.read_bytes()
        self.save('new.pdf', 'coffee')
        with patch('database.create_embeddings', side_effect=RuntimeError('offline')):
            with self.assertRaises(RuntimeError):
                database.rebuild_database(self.folder, self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_invalid_embedding_count_preserves_index(self):
        self.save()
        self.rebuild()
        before = self.db.read_bytes()
        with patch('database.create_embeddings', return_value=[]), self.assertRaises(ValueError):
            database.rebuild_database(self.folder, self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_broken_existing_pdf_preserves_index(self):
        self.save()
        self.rebuild()
        before = self.db.read_bytes()
        (self.folder / 'broken.pdf').write_bytes(b'broken')
        with self.assertRaises(ingest.PDFValidationError):
            self.rebuild()
        self.assertEqual(self.db.read_bytes(), before)

    def test_sql_failure_rolls_back_deleted_chunks(self):
        self.save()
        self.rebuild()
        with closing(database.connect(self.db)) as c, c:
            c.execute("CREATE TRIGGER block_insert BEFORE INSERT ON chunks BEGIN SELECT RAISE(ABORT,'test'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.rebuild()
        self.assertEqual(self.retrieve('ZEBRA')[0]['source'], 'campaign.pdf')

    def test_missing_and_empty_index_do_not_load_models(self):
        with patch('retrieval.encode_texts') as encode:
            self.assertEqual(retrieval.retrieve('question', self.db), [])
            database.create_database(self.db)
            self.assertEqual(retrieval.retrieve('question', self.db), [])
            encode.assert_not_called()
        with patch('llm.setup_model') as model:
            self.assertEqual(llm.answer_from_results(None, 'question', []), (llm.NO_CONTEXT, []))
            model.assert_not_called()

    def test_empty_folder_cannot_erase_old_index(self):
        self.save()
        self.rebuild()
        before = self.db.read_bytes()
        empty = self.folder / 'empty'
        empty.mkdir()
        with self.assertRaises(ValueError):
            database.rebuild_database(empty, self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_legacy_schema_read_and_upgrade(self):
        with closing(database.connect(self.db)) as c, c:
            c.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY,source TEXT,text TEXT,embedding BLOB)')
            c.execute('INSERT INTO chunks VALUES (1,?,?,?)', ('legacy.pdf','ZEBRA',fake_encode(['ZEBRA'])[0].tobytes()))
        self.assertIsNone(self.retrieve('ZEBRA')[0]['page'])
        self.save()
        self.rebuild()
        self.assertEqual(self.retrieve('ZEBRA')[0]['page'], 1)

    def test_model_mismatch_is_actionable(self):
        self.save()
        self.rebuild()
        with closing(database.connect(self.db)) as c, c:
            c.execute("UPDATE metadata SET value='different' WHERE key='embedding_model'")
        with self.assertRaisesRegex(ValueError, 'yeniden indeksleyin'):
            self.retrieve('ZEBRA')

    def test_empty_question_and_empty_response(self):
        with self.assertRaises(ValueError):
            llm.answer_question(None, '  ')
        with self.assertRaisesRegex(RuntimeError, 'boş yanıt'):
            llm.answer_from_results(RecordingModel(''), 'question', [{'source':'s','text':'ZEBRA'}])


class CacheTests(unittest.TestCase):
    def test_embedding_model_shared(self):
        factory = Mock(return_value=SimpleNamespace(encode=lambda texts, **kwargs: fake_encode(texts)))
        vector_store.get_embedding_model.cache_clear()
        self.addCleanup(vector_store.get_embedding_model.cache_clear)
        with patch.dict('sys.modules', {'sentence_transformers': SimpleNamespace(SentenceTransformer=factory)}):
            vector_store.create_embeddings([{'text': 'ZEBRA'}])
            vector_store.encode_texts(['ZEBRA'])
        factory.assert_called_once()

    def test_foundry_model_loaded_once(self):
        model = Mock()
        manager = SimpleNamespace(catalog=SimpleNamespace(get_model=Mock(return_value=model)))
        with patch('llm._MODEL', None), patch('llm.get_manager', return_value=manager):
            self.assertIs(llm.setup_model(), llm.setup_model())
        model.download.assert_called_once()
        model.load.assert_called_once()


class StreamlitTests(WorkspaceTest):
    def test_upload_index_question_sources_and_rerun(self):
        from streamlit.testing.v1 import AppTest
        file = BytesIO(pdf_bytes('ZEBRA campaign code is ORBIT-739.'))
        file.name = 'uploaded.pdf'
        real_rebuild, real_status, real_save, real_retrieve = (
            database.rebuild_database, database.index_status, uploads.save_uploaded_pdf, retrieval.retrieve)
        with (patch('streamlit.file_uploader', return_value=[file]),
              patch('uploads.save_uploaded_pdf', side_effect=lambda n,b: real_save(n,b,self.folder)),
              patch('database.rebuild_database', side_effect=lambda: real_rebuild(self.folder,self.db)),
              patch('database.index_status', side_effect=lambda: real_status(self.db,self.folder)),
              patch('database.create_embeddings', side_effect=lambda cs: fake_encode([c['text'] for c in cs])),
              patch('retrieval.encode_texts', side_effect=fake_encode),
              patch('retrieval.retrieve', side_effect=lambda q: real_retrieve(q,self.db)),
              patch('llm.setup_model', return_value=RecordingModel())):
            app = AppTest.from_file(str(config.PROJECT_ROOT / 'app.py')).run()
            next(b for b in app.button if b.label == "PDF'leri kaydet ve indeksle").click().run()
            self.assertFalse(app.exception)
            self.assertTrue((self.folder / 'uploaded.pdf').exists())
            app.text_area[0].set_value('ZEBRA campaign code?')
            next(b for b in app.button if b.label == 'Yanıtla').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('ORBIT-739' in m.value for m in app.markdown))
            self.assertTrue(any('uploaded.pdf' in e.label for e in app.expander))
            app.run()
            self.assertTrue(any('ORBIT-739' in m.value for m in app.markdown))

    def test_blank_question_and_backend_failure(self):
        from streamlit.testing.v1 import AppTest
        with patch('database.index_status', return_value={'chunks':0,'documents':0,'stale':False}):
            app = AppTest.from_file(str(config.PROJECT_ROOT / 'app.py')).run()
            next(b for b in app.button if b.label == 'Yanıtla').click().run()
            self.assertTrue(app.warning)
            with patch('retrieval.retrieve', side_effect=RuntimeError('test retrieval failure')):
                app.text_area[0].set_value('question')
                next(b for b in app.button if b.label == 'Yanıtla').click().run()
            self.assertTrue(app.error)
            self.assertFalse(app.exception)


if __name__ == '__main__':
    unittest.main()
