"""Simple Kivy OCR application using pytesseract and Pillow.

This application opens an image file, runs Tesseract OCR and displays the
recognized text. The UI uses Kivy.

Installation:
 - Install Tesseract for Windows (UB-Mannheim builds) and ensure `tesseract`
   is in PATH or at "C:\\Program Files\\Tesseract-OCR\\tesseract.exe".
 - Install Python dependencies from `requirements.txt`.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import re
import sys
import tempfile
from typing import Any, Callable, Dict, List, Tuple, cast

from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import pytesseract
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words

# Kivy imports
from kivy.app import App
from kivy.core.clipboard import Clipboard
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.properties import NumericProperty
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider


def ensure_tesseract_path() -> None:
    """If the typical UB-Mannheim Windows install exists, set the tesseract cmd.

    This makes the app work when tesseract is not already on PATH.
    """
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    candidates = [
        os.path.join(base_path, "tesseract", "tesseract.exe"),
        os.path.join(base_path, "tesseract.exe"),
        os.path.join(base_path, "Tesseract-OCR", "tesseract.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            tessdata_path = os.path.join(os.path.dirname(candidate), "tessdata")
            if os.path.isdir(tessdata_path):
                os.environ.setdefault("TESSDATA_PREFIX", tessdata_path + os.sep)
            return

    # 2) Fall back to default UB‑Mannheim install location on Windows
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        pytesseract.pytesseract.tesseract_cmd = default_path
        tessdata_path = os.path.join(os.path.dirname(default_path), "tessdata")
        if os.path.isdir(tessdata_path):
            os.environ.setdefault("TESSDATA_PREFIX", tessdata_path + os.sep)
        return


class OCRLogic:
    """Encapsulate OCR, preprocessing and search logic.

    Public methods are independent from UI toolkit.
    """

    def __init__(self) -> None:
        self.lang_map: Dict[str, str] = {"English (eng)": "eng", "Polish (pol)": "pol"}

    def preprocess_image(
        self,
        image: Image.Image,
        resize_pct: int = 100,
        grayscale: bool = False,
        threshold_enabled: bool = False,
        threshold_value: int = 128,
        brightness: float = 1.0,
        contrast: float = 1.0,
        rotate: int = 0,
        sharpen_enabled: bool = True,
    ) -> Image.Image:
        """Apply preprocessing to a PIL image and return the processed image.

        Operations:
        - resize by percent
        - convert to grayscale
        - apply binary threshold
        - optional rotate, brightness/contrast and sharpen
        """
        img = image
        try:
            # Resize
            pct = max(1, int(resize_pct))
            if pct != 100:
                new_w = max(1, int(img.width * pct / 100))
                new_h = max(1, int(img.height * pct / 100))
                resample_filter = getattr(Image, "LANCZOS", None)
                if resample_filter is None:
                    resample_filter = getattr(getattr(Image, "Resampling", None), "LANCZOS", None)
                if resample_filter is None:
                    resample_filter = getattr(Image, "BICUBIC", getattr(Image, "NEAREST", 1))
                img = img.resize((new_w, new_h), resample=resample_filter)

            # Grayscale / threshold
            if grayscale or threshold_enabled:
                img = ImageOps.grayscale(img)
            if threshold_enabled:
                thresh = int(threshold_value)
                img = img.point(lambda p: 255 if p > thresh else 0)

            # Rotate
            if rotate:
                img = img.rotate(float(rotate), expand=True)

            # Brightness / contrast enhancements
            try:
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(float(brightness))
                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(float(contrast))
            except Exception:
                # ignore enhancement errors
                pass

            # Sharpen (always apply small sharpen to help OCR)
            if sharpen_enabled:
                img = img.filter(ImageFilter.SHARPEN)
        except Exception as exc:
            print(f"Preprocessing failed: {exc}", file=sys.stderr)
            return image
        return img

    def _load_cache(self, cache_path: pathlib.Path) -> dict:
        """Load OCR cache from path, return empty dict on error."""
        if not cache_path.exists():
            return {}
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Could not read OCR cache, rebuilding: {exc}", file=sys.stderr)
            return {}

    def _save_cache(self, cache_path: pathlib.Path, cache: dict) -> None:
        """Save OCR cache (best-effort)."""
        try:
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(cache, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"Could not write OCR cache: {exc}", file=sys.stderr)

    def _ocr_image_to_text(self, path: str, lang_code: str, preprocess_opts: dict) -> str:
        """Run OCR on a single image with preprocessing; return text or empty string."""
        try:
            img = cast(Image.Image, Image.open(path))
            img = self.preprocess_image(img, **preprocess_opts)
            return pytesseract.image_to_string(img, lang=lang_code)
        except (OSError, pytesseract.TesseractError) as exc:
            print(f"OCR failed for {path}: {exc}", file=sys.stderr)
            return ""

    def _create_tokenizer_and_stoplist(self, language_code: str) -> Tuple[List[str], Callable[[str], List[str]]]:
        """Return (stoplist, tokenizer) for given language (try spaCy, then NLTK, else simple)."""
        # Build stoplist
        try:
            stoplist = list(get_stop_words("english") if language_code == "eng" else get_stop_words("polish"))
        except Exception:
            stoplist = ["the", "and", "is", "in", "to", "of", "a", "that", "it"] if language_code == "eng" else [
                "i", "w", "z", "na", "do", "się", "nie", "że", "to", "jest", "o", "a", "po", "jak", "dla"
            ]

        # spaCy tokenizer if available
        try:
            model = "en_core_web_sm" if language_code == "eng" else "pl_core_news_sm"
            nlp = spacy.load(model, disable=["ner", "parser"])

            def spacy_tokenizer(text: str) -> List[str]:
                doc = nlp(text)
                return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

            return stoplist, spacy_tokenizer
        except Exception:
            pass

        # NLTK-based tokenizers
        try:
            if language_code == "eng":
                try:
                    lemmatizer = WordNetLemmatizer()
                    _ = lemmatizer.lemmatize("test")

                    def nltk_eng_tokenizer(text: str) -> List[str]:
                        tokens = re.findall(r"\b\w+\b", text.lower())
                        return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]

                    return stoplist, nltk_eng_tokenizer
                except LookupError:
                    try:
                        nltk.download("wordnet", quiet=True)
                        lemmatizer = WordNetLemmatizer()

                        def nltk_eng_tokenizer(text: str) -> List[str]:
                            tokens = re.findall(r"\b\w+\b", text.lower())
                            return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]

                        return stoplist, nltk_eng_tokenizer
                    except Exception:
                        pass

                def simple_no_lemma(text: str) -> List[str]:
                    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t.isalpha()]

                    return stoplist, simple_no_lemma
            else:
                stemmer = SnowballStemmer("polish")

                def nltk_pol_tokenizer(text: str) -> List[str]:
                    tokens = re.findall(r"\b\w+\b", text.lower())
                    return [stemmer.stem(t) for t in tokens if t.isalpha()]

                return stoplist, nltk_pol_tokenizer
        except Exception:
            pass

        # Final fallback tokenizer
        def simple_tokenizer(text: str) -> List[str]:
            return [t.lower() for t in re.findall(r"\b\w+\b", text) if t.isalpha()]

        return stoplist, simple_tokenizer

    def _vectorize_and_score(self, texts: List[str], query: str, language_code: str) -> List[float]:
        """Compute TF-IDF and cosine similarity; return list of similarity scores."""
        stoplist, tokenizer = self._create_tokenizer_and_stoplist(language_code)
        if not any(t.strip() for t in texts):
            return []
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words=stoplist,
                ngram_range=(1, 2),
                min_df=1,
                tokenizer=tokenizer,
                token_pattern=None,
            )
            corpus_tfidf = vectorizer.fit_transform(texts)
            query_tfidf = vectorizer.transform([query])
            sims = cosine_similarity(query_tfidf, corpus_tfidf).flatten()
            return sims.tolist()
        except Exception as exc:
            print(f"Search vectorization error: {exc}", file=sys.stderr)
            return []


KV = """
<RootWidget>:
    orientation: "vertical"
    padding: 8
    spacing: 8

    BoxLayout:
        size_hint_y: None
        height: "40dp"
        spacing: 8
        Button:
            text: "Open Image..."
            on_release: root.open_image_dialog()
            font_size: app.font_size
        Button:
            text: "Copy Result"
            on_release: root.copy_result()
            font_size: app.font_size
        Button:
            text: "Search Similar..."
            on_release: root.search_similar()
            font_size: app.font_size
        Button:
            id: lang_button
            text: "English (eng) >"
            size_hint_x: None
            width: "160dp"
            on_release: root.select_language()
            font_size: app.font_size

    Label:
        id: status_label
        text: root.status_text
        size_hint_y: None
        height: "24dp"

    BoxLayout:
        orientation: "horizontal"
        spacing: 8

        BoxLayout:
            orientation: "vertical"
            size_hint_x: None
            width: "320dp"
            padding: 6
            spacing: 6

            Label:
                text: "Image edit settings"
                font_size: app.font_size * 1.05
                size_hint_y: None
                height: "28dp"
            # Move basic preprocess controls here: Resize, Grayscale, Threshold
            Label:
                text: "Resize (%)"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            BoxLayout:
                size_hint_y: None
                height: "36dp"
                spacing: 6
                Slider:
                    id: resize_slider
                    min: 25
                    max: 200
                    value: 100
                    orientation: "horizontal"
                    size_hint_x: 0.75
                Label:
                    id: resize_value
                    text: str(int(resize_slider.value)) + "%"
                    size_hint_x: 0.25
                    font_size: app.font_size

            Label:
                text: "Grayscale"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            BoxLayout:
                size_hint_y: None
                height: "36dp"
                Switch:
                    id: grayscale_switch
                    active: False

            Label:
                text: "Threshold"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            BoxLayout:
                size_hint_y: None
                height: "36dp"
                spacing: 6
                Switch:
                    id: threshold_switch
                    active: False
                    size_hint_x: 0.2
                Slider:
                    id: threshold_slider
                    min: 0
                    max: 255
                    value: 128
                    orientation: "horizontal"
                    size_hint_x: 0.6
                Label:
                    id: threshold_value
                    text: str(int(threshold_slider.value))
                    size_hint_x: 0.2
                    font_size: app.font_size
            Label:
                text: "Brightness"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            Slider:
                id: brightness_slider
                min: 0.5
                max: 1.5
                value: 1.0
            Label:
                id: brightness_value
                text: "{:.2f}".format(brightness_slider.value)
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"

            Label:
                text: "Contrast"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            Slider:
                id: contrast_slider
                min: 0.5
                max: 1.5
                value: 1.0
            Label:
                id: contrast_value
                text: "{:.2f}".format(contrast_slider.value)
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"

            Label:
                text: "Rotate"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            Slider:
                id: rotate_slider
                min: -180
                max: 180
                value: 0
            Label:
                id: rotate_value
                text: str(int(rotate_slider.value)) + "°"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"

            Label:
                text: "Sharpen"
                font_size: app.font_size
                size_hint_y: None
                height: "20dp"
            Switch:
                id: sharpen_switch
                active: True

            BoxLayout:
                size_hint_y: None
                height: "40dp"
                spacing: 8
                Button:
                    text: "Re-run OCR"
                    on_release: root.re_run()
                    font_size: app.font_size
                Button:
                    text: "Reset"
                    on_release: root.reset()
                    font_size: app.font_size

            BoxLayout:
                size_hint_y: None
                height: "36dp"
                spacing: 6
                Label:
                    text: "Font size"
                    size_hint_x: 0.6
                    font_size: app.font_size
                Slider:
                    min: 12
                    max: 28
                    value: app.font_size
                    on_value: app.font_size = int(self.value)

        BoxLayout:
            orientation: "vertical"
            spacing: 6
            TextInput:
                id: result_text
                text: root.result_text
                readonly: False
                multiline: True
                focus: False
                font_size: app.font_size * 1.2
                size_hint_y: 0.6

            BoxLayout:
                orientation: "vertical"
                size_hint_y: 0.4
                spacing: 4
                Label:
                    text: "Preview"
                    size_hint_y: None
                    height: "20dp"
                    font_size: app.font_size
                Image:
                    id: preview_image
                    source: ""
                    allow_stretch: True
                    keep_ratio: True
"""


class RootWidget(BoxLayout):
    """Root widget connecting UI controls to OCR logic."""

    status_text = StringProperty("Ready")
    result_text = StringProperty("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.logic = OCRLogic()

    def _show_popup(self, title: str, content_widget):
        popup = Popup(title=title, content=content_widget, size_hint=(0.9, 0.9))
        popup.open()
        return popup

    def _message(self, title: str, message: str) -> None:
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        content.add_widget(Label(text=message))
        popup = self._show_popup(title, content)
        btn = Button(text="OK", size_hint_y=None, height="40dp", on_release=lambda *_: popup.dismiss())
        content.add_widget(btn)

    def open_image_dialog(self) -> None:
        """Open a file chooser popup to select a single image and OCR it."""
        chooser = FileChooserListView(filters=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"], path=".")
        btn_box = BoxLayout(size_hint_y=None, height="40dp", spacing=8)
        container = BoxLayout(orientation="vertical")
        container.add_widget(chooser)
        container.add_widget(btn_box)
        popup = self._show_popup("Select image", container)

        def do_open(*_):
            selection = chooser.selection
            if not selection:
                self._message("No file", "Please select an image file.")
                return
            popup.dismiss()
            path = selection[0]
            self._last_image_path = path
            self.status_text = "Processing..."
            self._ocr_and_display(path)

        ok = Button(text="Open", on_release=do_open)
        cancel = Button(text="Cancel", on_release=lambda *_: popup.dismiss())
        btn_box.add_widget(ok)
        btn_box.add_widget(cancel)

    def _ocr_and_display(self, path: str) -> None:
        """Run OCR on selected path with current UI options and display text."""
        resize_pct = int(self.ids.resize_slider.value)
        grayscale = bool(self.ids.grayscale_switch.active)
        threshold_enabled = bool(self.ids.threshold_switch.active)
        threshold_value = int(self.ids.threshold_slider.value)
        lang_code = self.logic.lang_map.get(self._get_selected_language_label(), "eng")
        preprocess_opts = {
            "resize_pct": resize_pct,
            "grayscale": grayscale,
            "threshold_enabled": threshold_enabled,
            "threshold_value": threshold_value,
        }
        try:
            text = self.logic._ocr_image_to_text(path, lang_code, preprocess_opts)
            self.result_text = text
            self.ids.result_text.text = text
            # Update persistent preview area (if available)
            try:
                if getattr(self, "_last_image_path", None):
                    if getattr(self.ids, "preview_image", None) is not None:
                        self.ids.preview_image.source = path
                        try:
                            self.ids.preview_image.reload()
                        except Exception:
                            pass
            except Exception:
                pass
            self.status_text = f"Finished — {len(text)} characters"
        except Exception as exc:
            self._message("OCR Error", str(exc))
            self.status_text = "Error"
    def _process_and_ocr(self, path: str, preprocess_opts: dict, lang_code: str, update_preview: bool = True) -> None:
        """Process the ORIGINAL image with given preprocess options, update preview and OCR."""
        try:
            img = Image.open(path)
            processed_img = self.logic.preprocess_image(img, **preprocess_opts)

            tmp_path = None
            if update_preview:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
                        tmp_path = tf.name
                    try:
                        processed_img.save(tmp_path)
                    except Exception:
                        tmp_path = None
                except Exception:
                    tmp_path = None

                try:
                    if getattr(self.ids, "preview_image", None) is not None:
                        if tmp_path:
                            self.ids.preview_image.source = tmp_path
                        else:
                            self.ids.preview_image.source = path
                        try:
                            self.ids.preview_image.reload()
                        except Exception:
                            pass
                except Exception:
                    pass

            # Run OCR on processed image
            text = pytesseract.image_to_string(processed_img, lang=lang_code)
            self.result_text = text
            try:
                self.ids.result_text.text = text
            except Exception:
                pass
            self.status_text = f"Finished — {len(text)} characters"
        except Exception as exc:
            self._message("OCR Error", str(exc))
            self.status_text = "Error"

    def copy_result(self) -> None:
        """Copy extracted text to the clipboard."""
        text = self.ids.result_text.text.strip()
        if not text:
            self.status_text = "No text to copy"
            return
        try:
            Clipboard.copy(text)
            self.status_text = "Copied to clipboard"
        except Exception as exc:
            self._message("Clipboard Error", str(exc))
            self.status_text = "Error"

    def re_run(self) -> None:
        """Re-run OCR on the last opened image with current preprocessing options."""
        if not getattr(self, "_last_image_path", None):
            self._message("No image", "No image loaded to re-run OCR.")
            return
        self.status_text = "Processing..."
        # Build full preprocess options (same as Apply edits) and run OCR on original image
        preprocess_opts = {
            "resize_pct": int(self.ids.resize_slider.value),
            "grayscale": bool(self.ids.grayscale_switch.active),
            "threshold_enabled": bool(self.ids.threshold_switch.active),
            "threshold_value": int(self.ids.threshold_slider.value),
            "brightness": float(getattr(self.ids, "brightness_slider", None).value if getattr(self.ids, "brightness_slider", None) else 1.0),
            "contrast": float(getattr(self.ids, "contrast_slider", None).value if getattr(self.ids, "contrast_slider", None) else 1.0),
            "rotate": int(getattr(self.ids, "rotate_slider", None).value if getattr(self.ids, "rotate_slider", None) else 0),
            "sharpen_enabled": bool(getattr(self.ids, "sharpen_switch", None).active if getattr(self.ids, "sharpen_switch", None) else True),
        }
        lang_code = self.logic.lang_map.get(self._get_selected_language_label(), "eng")
        self._process_and_ocr(self._last_image_path, preprocess_opts, lang_code, update_preview=True)

    def apply_edits_and_ocr(self) -> None:
        """Apply current edit settings to last opened image and re-run OCR."""
        if not getattr(self, "_last_image_path", None):
            self._message("No image", "Open an image first (Open Image...)")
            return
        path = self._last_image_path
        preprocess_opts = {
            "resize_pct": int(self.ids.resize_slider.value),
            "grayscale": bool(self.ids.grayscale_switch.active),
            "threshold_enabled": bool(self.ids.threshold_switch.active),
            "threshold_value": int(self.ids.threshold_slider.value),
            "brightness": float(getattr(self.ids, "brightness_slider", None).value if getattr(self.ids, "brightness_slider", None) else 1.0),
            "contrast": float(getattr(self.ids, "contrast_slider", None).value if getattr(self.ids, "contrast_slider", None) else 1.0),
            "rotate": int(getattr(self.ids, "rotate_slider", None).value if getattr(self.ids, "rotate_slider", None) else 0),
            "sharpen_enabled": bool(getattr(self.ids, "sharpen_switch", None).active if getattr(self.ids, "sharpen_switch", None) else True),
        }
        lang_code = self.logic.lang_map.get(self._get_selected_language_label(), "eng")
        # Delegate to helper that handles processing, preview update and OCR
        self._process_and_ocr(path, preprocess_opts, lang_code, update_preview=True)

    def search_similar(self) -> None:
        """Choose a folder and query, then run similarity search over OCR texts."""
        # choose folder
        chooser = FileChooserListView(path=".", dirselect=True)
        btn_box = BoxLayout(size_hint_y=None, height="40dp", spacing=8)
        container = BoxLayout(orientation="vertical")
        container.add_widget(chooser)
        container.add_widget(btn_box)
        popup = self._show_popup("Select folder with images", container)

        def do_select(*_):
            selection = chooser.selection
            if not selection:
                self._message("No folder", "Please select a folder.")
                return
            popup.dismiss()
            folder = selection[0]
            self._ask_query_and_search(folder)

        ok = Button(text="Select", on_release=do_select)
        cancel = Button(text="Cancel", on_release=lambda *_: popup.dismiss())
        btn_box.add_widget(ok)
        btn_box.add_widget(cancel)

    def _ask_query_and_search(self, folder: str) -> None:
        """Ask for query in a popup and run search when user confirms."""
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        ti = TextInput(hint_text="Enter search query", multiline=False)
        btn_box = BoxLayout(size_hint_y=None, height="40dp", spacing=8)
        content.add_widget(ti)
        content.add_widget(btn_box)
        popup = self._show_popup("Search query", content)

        def do_search(*_):
            query = ti.text.strip()
            if not query:
                self._message("No query", "Please enter a query.")
                return
            popup.dismiss()
            self._run_folder_search(folder, query)

        ok = Button(text="Search", on_release=do_search)
        cancel = Button(text="Cancel", on_release=lambda *_: popup.dismiss())
        btn_box.add_widget(ok)
        btn_box.add_widget(cancel)

    def _run_folder_search(self, folder: str, query: str) -> None:
        """Collect images, OCR (with caching), compute similarities and show top results."""
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        image_paths: List[str] = []
        for ext in exts:
            image_paths.extend(glob.glob(str(pathlib.Path(folder) / ext)))
        image_paths = sorted(image_paths)
        if not image_paths:
            self._message("Search", "No image files found in folder.")
            return

        cache_path = pathlib.Path(folder) / ".ocr_cache.json"
        cache = self.logic._load_cache(cache_path)

        texts: List[str] = []
        paths_for_corpus: List[str] = []
        lang_code = self.logic.lang_map.get(self._get_selected_language_label(), "eng")
        preprocess_opts = {
            "resize_pct": int(self.ids.resize_slider.value),
            "grayscale": bool(self.ids.grayscale_switch.active),
            "threshold_enabled": bool(self.ids.threshold_switch.active),
            "threshold_value": int(self.ids.threshold_slider.value),
        }
        for p in image_paths:
            key = str(pathlib.Path(p).name)
            text = cache.get(key)
            if text is None:
                text = self.logic._ocr_image_to_text(p, lang_code, preprocess_opts)
                cache[key] = text
            texts.append(text or "")
            paths_for_corpus.append(p)

        self.logic._save_cache(cache_path, cache)

        sims = self.logic._vectorize_and_score(texts, query, lang_code)
        if not sims:
            self._message("Search", "No OCR text found in the selected folder or error building vocabulary.")
            return
        scored = sorted(zip(paths_for_corpus, sims), key=lambda x: x[1], reverse=True)
        # show a compact results popup
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        for path, score in scored[:20]:
            def _on_show_factory(p: str, s: float):
                def _handler(*_):
                    self._show_preview(p, cache.get(str(pathlib.Path(p).name), ""), s)

                return _handler

            handler = _on_show_factory(path, score)
            btn = Button(
                text=f"{pathlib.Path(path).name} — {score:.4f}",
                size_hint_y=None,
                height="40dp",
                on_release=handler,
            )
            content.add_widget(btn)
        popup = Popup(title="Search results", content=content, size_hint=(0.9, 0.9))
        popup.open()

    def _show_preview(self, path: str, text: str, score: float) -> None:
        """Show preview image and OCR text in a popup."""
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        try:
            # Use PIL to create a thumbnail and display path info + text
            img = Image.open(path)
            img.thumbnail((800, 600))
            # Title / path label: wrap long paths instead of overflowing popup
            path_label = Label(
                text=f"Score: {score:.4f} — {path}",
                size_hint_y=None,
                text_size=(780, None),
                halign="left",
                valign="middle",
            )
            path_label.texture_update()
            path_label.height = path_label.texture_size[1]
            content.add_widget(path_label)

            # OCR text wrapped inside a ScrollView to avoid overflow.
            ocr_label = Label(
                text=text or "(no OCR text)",
                size_hint_y=None,
                text_size=(760, None),
                halign="left",
                valign="top",
            )
            ocr_label.texture_update()
            ocr_label.height = max(200, ocr_label.texture_size[1])

            scroll = ScrollView(size_hint=(1, 0.65))
            scroll.add_widget(ocr_label)
            content.add_widget(scroll)

            # Slider to control scroll position (0 = bottom, 100 = top)
            slider = Slider(min=0, max=100, value=100, size_hint=(1, None), height="40dp")

            def _on_slider(_, val: float) -> None:
                try:
                    scroll.scroll_y = float(val) / 100.0
                except Exception:
                    pass

            slider.bind(value=_on_slider)
            content.add_widget(slider)
        except OSError:
            content.add_widget(Label(text="Cannot open image"))
        popup = Popup(title="Preview", content=content, size_hint=(0.9, 0.9))
        popup.open()

    def _get_selected_language_label(self) -> str:
        """Return the human-readable language label from the lang button text."""
        try:
            txt = self.ids.lang_button.text
            return txt.replace("▾", "").strip()
        except Exception:
            return "English (eng)"

    def set_language(self, label: str) -> None:
        """Set language label on the lang button."""
        try:
            self.ids.lang_button.text = f"{label} ▾"
        except Exception:
            pass

    def select_language(self) -> None:
        """Show a small popup to choose OCR language."""
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        en = Button(text="English (eng)", size_hint_y=None, height="40dp")
        pl = Button(text="Polish (pol)", size_hint_y=None, height="40dp")
        content.add_widget(en)
        content.add_widget(pl)
        popup = Popup(title="Select language", content=content, size_hint=(0.4, 0.4))

        def choose(label: str) -> None:
            self.set_language(label)
            popup.dismiss()

        en.bind(on_release=lambda *_: choose("English (eng)"))
        pl.bind(on_release=lambda *_: choose("Polish (pol)"))
        popup.open()

    def reset(self) -> None:
        """Reset UI controls to defaults and restore original preview and OCR."""
        if not getattr(self, "_last_image_path", None):
            self._message("No image", "Open an image first (Open Image...)")
            return
        # Reset UI controls to defaults matching KV initial values
        try:
            self.ids.resize_slider.value = 100
            if getattr(self.ids, "resize_value", None) is not None:
                self.ids.resize_value.text = str(int(self.ids.resize_slider.value)) + "%"
            self.ids.grayscale_switch.active = False
            self.ids.threshold_switch.active = False
            self.ids.threshold_slider.value = 128
            if getattr(self.ids, "threshold_value", None) is not None:
                self.ids.threshold_value.text = str(int(self.ids.threshold_slider.value))
            if getattr(self.ids, "brightness_slider", None) is not None:
                self.ids.brightness_slider.value = 1.0
            if getattr(self.ids, "brightness_value", None) is not None:
                self.ids.brightness_value.text = "{:.2f}".format(self.ids.brightness_slider.value)
            if getattr(self.ids, "contrast_slider", None) is not None:
                self.ids.contrast_slider.value = 1.0
            if getattr(self.ids, "contrast_value", None) is not None:
                self.ids.contrast_value.text = "{:.2f}".format(self.ids.contrast_slider.value)
            if getattr(self.ids, "rotate_slider", None) is not None:
                self.ids.rotate_slider.value = 0
            if getattr(self.ids, "rotate_value", None) is not None:
                self.ids.rotate_value.text = str(int(self.ids.rotate_slider.value)) + "°"
            if getattr(self.ids, "sharpen_switch", None) is not None:
                self.ids.sharpen_switch.active = True
        except Exception:
            pass

        # Restore preview to the original image and OCR with default options
        try:
            if getattr(self.ids, "preview_image", None) is not None:
                self.ids.preview_image.source = self._last_image_path
                try:
                    self.ids.preview_image.reload()
                except Exception:
                    pass
        except Exception:
            pass

        # Run OCR on original image with default preprocess opts
        preprocess_opts = {
            "resize_pct": 100,
            "grayscale": False,
            "threshold_enabled": False,
            "threshold_value": 128,
            "brightness": 1.0,
            "contrast": 1.0,
            "rotate": 0,
            "sharpen_enabled": True,
        }
        lang_code = self.logic.lang_map.get(self._get_selected_language_label(), "eng")
        self._process_and_ocr(self._last_image_path, preprocess_opts, lang_code, update_preview=True)


class OCRKivyApp(App):
    """Kivy App wrapper."""

    font_size = NumericProperty(14)

    def build(self) -> RootWidget:
        # adapt window size to screen where possible
        try:
            sw, sh = Window.system_size
            Window.size = (int(sw * 0.8), int(sh * 0.8))
        except Exception:
            # fallback: enlarge default window a bit
            try:
                Window.size = (900, 700)
            except Exception:
                pass
        # compute font size relative to window width
        try:
            w, h = Window.size
            self.font_size = max(12, min(18, int(w / 80)))
        except Exception:
            self.font_size = 14
        Builder.load_string(KV)
        return RootWidget()


def main() -> None:
    """Entry point for the application."""
    ensure_tesseract_path()
    OCRKivyApp().run()


if __name__ == "__main__":
    main()
