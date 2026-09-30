"""Small generated fixtures; no user documents or downloaded test assets."""
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


def pdf_bytes(*texts, encrypted=False):
    writer = PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                     NameObject('/Subtype'): NameObject('/Type1'),
                                     NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({
                NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
            stream = DecodedStreamObject()
            escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            stream.set_data(f'BT /F1 12 Tf 40 740 Td ({escaped}) Tj ET'.encode('ascii'))
            page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('test-password')
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def fake_encode(texts):
    return np.asarray([[float('zebra' in t.lower()), float('coffee' in t.lower()), 0.01]
                       for t in texts], dtype=np.float32)


class RecordingModel:
    def __init__(self, answer='ZEBRA campaign code is ORBIT-739. [1]'):
        self.answer = answer
        self.messages = None
        self.settings = SimpleNamespace()

    def get_chat_client(self):
        return self

    def complete_streaming_chat(self, messages):
        self.messages = messages
        yield SimpleNamespace(choices=[])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=self.answer))])
