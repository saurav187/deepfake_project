import http.server
import io
import threading
import unittest
import zipfile

from pipeline_v3.modern_intake import TEST_FOLD, fold_of, norm_name, upstream_role
from pipeline_v3.remote_zip import RemoteZip
from pipeline_v3.speechfake_slice import blocked, parse


class Lineage(unittest.TestCase):
    def test_roles(self):
        libri = {"2803": ("M", "dev-clean", "x"), "1089": ("M", "test-clean", "y"),
                 "19": ("F", "train-clean-100", "z")}
        self.assertTrue(upstream_role("Libritts", "2803", libri).startswith("upstream_eval:dev-clean"))
        self.assertTrue(upstream_role("Libritts", "1089", libri).startswith("upstream_eval:test-clean"))
        self.assertEqual(upstream_role("Libritts", "19", libri), "librispeech:train-clean-100")
        self.assertTrue(upstream_role("Libritts", "7", libri).startswith("unresolved"))
        self.assertTrue(upstream_role("VCTK", "p228", libri).startswith("reserved"))
        self.assertEqual(upstream_role("VCTK", "p226", libri), "vctk:no_official_split")

    def test_fold_and_names(self):
        self.assertEqual(fold_of("VCTK", "p226"), fold_of("VCTK", "p226"))
        self.assertTrue(0 <= fold_of("VCTK", "p226") < 5)
        self.assertEqual(norm_name("Mary_Ann"), norm_name(" mary  ann "))

    def test_speechfake_parse(self):
        self.assertEqual(parse("VC_GPTSoVITS", "BD/GPTSoVITS_VC/VCTK/p239_003/p248_128.wav"),
                         ("p239", "p248_128", "p239_003"))
        self.assertEqual(parse("TTS_CosyVoice", "BD/CosyVoice/en_female/VCTK/p329_049.wav"),
                         ("stock:en_female", "p329_049", None))
        self.assertIsNone(parse("VC_OpenVoice", "BD/OpenVoice/LibriTTS/a/b.wav"))
        self.assertTrue(blocked("p228"))
        spk = next(s for s in (f"p{i}" for i in range(225, 380)) if fold_of("VCTK", s) == TEST_FOLD)
        self.assertTrue(blocked(spk))


class Remote(unittest.TestCase):
    def test_multipart_range_read(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("a.bin", b"A" * 5000, compress_type=zipfile.ZIP_STORED)
            z.writestr("b/c.txt", b"hello " * 300, compress_type=zipfile.ZIP_DEFLATED)
        data = buf.getvalue()
        cut = 3000  # boundary for the read_span check below
        parts = {"/p0": data[:cut], "/p1": data[cut:]}

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = parts.get(self.path) if self.path != "/all" else data
                a, b = self.headers["Range"].split("=")[1].split("-")
                chunk = body[int(a):int(b) + 1]
                self.send_response(206)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)

            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_port}"
        try:
            rz = RemoteZip([(base + "/all", len(data))])
            d = rz.directory()
            self.assertEqual(set(d), {"a.bin", "b/c.txt"})
            self.assertEqual(rz.read(d["b/c.txt"]), b"hello " * 300)
            # read_span crosses a part boundary
            rz2 = RemoteZip.__new__(RemoteZip)
            rz2.parts = [(base + "/p0", cut), (base + "/p1", len(data) - cut)]
            self.assertEqual(rz2.read_span(0, cut - 10, 20), data[cut - 10:cut + 10])
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
