"""Check the static GitHub Pages routes and the generated project evidence."""

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

PROJECT = Path(__file__).resolve().parents[1]
SITE = PROJECT.parents[1] / "docs"
BASE = "https://asshittu.github.io/Projects/"
CASE = "case-studies/bank-marketing.html"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.images = []
        self.ids = set()
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        fields = dict(attrs)
        if fields.get("id"):
            self.ids.add(fields["id"])
        if tag in {"a", "link"} and fields.get("href"):
            self.links.append(fields["href"])
        if tag in {"img", "script"} and fields.get("src"):
            self.links.append(fields["src"])
        if tag == "img":
            self.images.append(fields)


class StaticSiteTest(unittest.TestCase):
    def test_published_links_resolve_under_project_base(self):
        self.assertTrue((SITE / ".nojekyll").is_file())
        bank_case = Page(SITE / CASE)
        homepage = Page(SITE / "index.html")
        self.assertIn(CASE, homepage.links)
        host = urlparse(BASE).netloc
        for name, page in [("index.html", homepage), (CASE, bank_case)]:
            for link in page.links:
                with self.subTest(page=name, href=link):
                    url = urlparse(urljoin(BASE + name, link))
                    if url.netloc != host:
                        continue
                    self.assertTrue(url.path.startswith("/Projects/"), url.path)
                    destination = SITE / unquote(url.path.removeprefix("/Projects/"))
                    if destination.is_dir():
                        destination /= "index.html"
                    self.assertTrue(destination.is_file(), destination)
                    if url.fragment:
                        self.assertIn(unquote(url.fragment), Page(destination).ids)

    def test_case_images_match_pipeline_outputs(self):
        figure_dir = PROJECT / "reports" / "figures"
        figures = sorted(figure_dir.glob("*.svg"))
        self.assertEqual(len(figures), 5)
        images = {image["src"]: image for image in Page(SITE / CASE).images}
        for figure in figures:
            with self.subTest(figure=figure.name):
                local_path = Path("assets") / "bank-marketing" / figure.name
                self.assertTrue(images["../" + local_path.as_posix()]["alt"].strip())
                self.assertEqual((SITE / local_path).read_bytes(), figure.read_bytes())


if __name__ == "__main__":
    unittest.main()
