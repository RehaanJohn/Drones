from email.message import Message
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import link_resolver


class LinkResolverTests(unittest.TestCase):
    def test_example_page_excludes_script_style_title_and_branding(self):
        html = '''<html><head><title>69</title><style>body {color: red}</style></head>
        <body><script>window.location.replace('');</script><div class="container"><p>69</p></div>
        <footer>Created with The QR Code Generator</footer></body></html>'''
        self.assertEqual(link_resolver.extract_page_text(html), '69')

    def test_main_content_and_entities_are_preserved(self):
        html = '<nav>Home</nav><main><p>Hello &amp; world</p><p>Second line</p><span hidden>secret</span><div style="display: none"><i style="color:red">hidden</i></div></main>'
        self.assertEqual(link_resolver.extract_page_text(html), 'Hello & world Second line')

    def test_only_web_urls_qualify(self):
        for value in ('69', 'file:///etc/passwd', 'javascript:alert(1)', 'https://'):
            self.assertFalse(link_resolver.is_web_url(value))
        self.assertTrue(link_resolver.is_web_url('https://qr-codes.io/zuabFE'))

    def test_private_addresses_and_redirects_are_rejected(self):
        address = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 80))]
        with patch.object(link_resolver.socket, 'getaddrinfo', return_value=address):
            with self.assertRaisesRegex(ValueError, 'private'):
                link_resolver.validate_public_url('http://example.com')
            with self.assertRaisesRegex(ValueError, 'private'):
                link_resolver.PublicRedirectHandler().redirect_request(None, None, 302, '', {}, 'http://127.0.0.1/')

    def test_fetch_text_json_unsupported_and_oversized_responses(self):
        cases = [('text/html', b'<main>69</main>', '69'),
                 ('text/plain', b'  69\n', '69'),
                 ('application/json', b'{"value": 69}', '{"value": 69}'),
                 ('image/png', b'fake image', None),
                 ('text/plain', b'x' * (link_resolver.MAX_RESPONSE_BYTES + 1), None)]
        for mime, data, expected in cases:
            with self.subTest(mime=mime, length=len(data)):
                response = Mock()
                response.__enter__ = Mock(return_value=response)
                response.__exit__ = Mock(return_value=False)
                response.url = 'https://example.com/final'
                response.headers = Message()
                response.headers['Content-Type'] = mime + '; charset=utf-8'
                response.read.return_value = data
                opener = Mock()
                opener.open.return_value = response
                with patch.object(link_resolver, 'validate_public_url'), patch.object(link_resolver.urllib.request, 'build_opener', return_value=opener):
                    result = link_resolver.resolve_url('https://example.com/start')
                if expected is None:
                    self.assertIn('error', result)
                else:
                    self.assertEqual(result['content'], expected)
                    self.assertEqual(result['url'], response.url)

    def test_network_error_is_reported_without_raising(self):
        with patch.object(link_resolver, 'validate_public_url', side_effect=TimeoutError('timed out')):
            self.assertEqual(link_resolver.resolve_url('https://example.com')['error'], 'timed out')

    def test_fetch_is_background_deduplicated_and_cached(self):
        gate = threading.Event()
        entered = threading.Event()

        def fetch(url):
            entered.set()
            gate.wait(2)
            return {'url': url, 'content': '69'}

        fetch = Mock(side_effect=fetch)
        resolver = link_resolver.LinkResolver(fetch=fetch)
        url = 'https://example.com'
        try:
            self.assertEqual(resolver.submit(url, 'capture-time'), [])
            self.assertTrue(entered.wait(2))
            self.assertFalse(gate.is_set())  # submit returned while HTTP is still waiting.
            self.assertEqual(resolver.submit(url, 'later-time'), [])
            gate.set()
            result = resolver.results.get(timeout=2)
            resolver.results.put(result)
            ready = resolver.poll()
            self.assertEqual(ready[0][1], 'capture-time')
            self.assertEqual(ready[0][2]['content'], '69')
            self.assertEqual(resolver.submit(url, 'new-capture')[0][1], 'new-capture')
            fetch.assert_called_once_with(url)
        finally:
            gate.set()
            resolver.close()


if __name__ == '__main__':
    unittest.main()
