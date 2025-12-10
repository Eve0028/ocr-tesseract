"""Simple Tkinter OCR application using pytesseract and Pillow.

This application opens an image file, runs Tesseract OCR and displays the
recognized text in a scrollable text area. The UI is English.

Installation:
 - Install Tesseract for Windows (UB-Mannheim builds) and ensure `tesseract`
   is in PATH or at "C:\\Program Files\\Tesseract-OCR\\tesseract.exe".
 - Install Python dependencies from `requirements.txt`.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog
from typing import Dict, Any, cast
import glob
import json
import pathlib
import re

from PIL import Image, ImageOps, ImageFilter, ImageTk, UnidentifiedImageError
import pytesseract
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from stop_words import get_stop_words


def ensure_tesseract_path() -> None:
    """If the typical UB-Mannheim Windows install exists, set the tesseract cmd.

    This makes the app work when tesseract is not already on PATH.
    """
    # 1) If bundled with PyInstaller, files are extracted to sys._MEIPASS
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(__file__))
    # Common bundle layouts: project/tesseract/... or project/tesseract.exe
    candidates = [
        os.path.join(base_path, "tesseract", "tesseract.exe"),
        os.path.join(base_path, "tesseract.exe"),
        os.path.join(base_path, "Tesseract-OCR", "tesseract.exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            # Set TESSDATA_PREFIX if tessdata is bundled alongside
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

    # 3) Otherwise rely on system PATH (no change)


class OCRApp:
    """Tkinter GUI wrapper for running pytesseract on a chosen image.

    UI language is English. OCR language can be selected between
    Polish and English.
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("OCR — Load Image (Tesseract)")
        root.geometry("800x600")

        top_frame = tk.Frame(root)
        top_frame.pack(fill=tk.X, padx=8, pady=6)

        self.open_button = tk.Button(top_frame, text="Open Image...", command=self.select_image)
        self.open_button.pack(side=tk.LEFT)

        # Preprocessing controls
        self.resize_var = tk.IntVar(value=100)  # percent
        resize_label = tk.Label(top_frame, text="Resize %:")
        resize_label.pack(side=tk.LEFT, padx=(8, 2))
        self.resize_scale = tk.Scale(top_frame, from_=25, to=200, orient=tk.HORIZONTAL, variable=self.resize_var, length=120)
        self.resize_scale.pack(side=tk.LEFT)

        self.grayscale_var = tk.BooleanVar(value=False)
        self.gray_check = tk.Checkbutton(top_frame, text="Grayscale", variable=self.grayscale_var)
        self.gray_check.pack(side=tk.LEFT, padx=(8, 2))

        self.threshold_var = tk.BooleanVar(value=False)
        self.thresh_check = tk.Checkbutton(
            top_frame,
            text="Threshold",
            variable=self.threshold_var,
            command=self._on_threshold_toggle,
        )
        self.thresh_check.pack(side=tk.LEFT, padx=(8, 2))
        self.threshold_scale = tk.Scale(top_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=100)
        self.threshold_scale.set(128)
        self.threshold_scale.config(state=tk.DISABLED)
        self.threshold_scale.pack(side=tk.LEFT)

        self.copy_button = tk.Button(top_frame, text="Copy Result", command=self.copy_to_clipboard)
        self.copy_button.pack(side=tk.LEFT, padx=6)
        self.search_button = tk.Button(top_frame, text="Search Similar...", command=self.search_similar_images)
        self.search_button.pack(side=tk.LEFT, padx=(6, 0))

        # Language selector for OCR
        self.lang_map: Dict[str, str] = {"English (eng)": "eng", "Polish (pol)": "pol"}
        self.lang_var = tk.StringVar(value="English (eng)")
        lang_label = tk.Label(top_frame, text="OCR language:")
        lang_label.pack(side=tk.LEFT, padx=(12, 4))
        self.lang_menu = tk.OptionMenu(top_frame, self.lang_var, *self.lang_map.keys())
        self.lang_menu.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = tk.Label(top_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.RIGHT)

        self.output = scrolledtext.ScrolledText(root, wrap=tk.WORD)
        self.output.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    def _on_threshold_toggle(self) -> None:
        """Enable/disable threshold scale when checkbox changes."""
        state: Any = tk.NORMAL if self.threshold_var.get() else tk.DISABLED
        self.threshold_scale.config(state=state)

    def select_image(self) -> None:
        """Show file dialog, run OCR on selected image and display the text."""
        filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
        path: str = filedialog.askopenfilename(title="Select image file", filetypes=filetypes)
        if not path:
            return
        try:
            self.status_var.set("Processing...")
            image: Image.Image = Image.open(path)
            # Apply preprocessing according to GUI options
            image = self.preprocess_image(image)
            selected_label = self.lang_var.get()
            lang_code = self.lang_map.get(selected_label, "eng")
            # Run OCR with selected language (traineddata must be installed)
            text: str = pytesseract.image_to_string(image, lang=lang_code)
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, text)
            self.status_var.set(f"Finished — {len(text)} characters")
        except OSError as exc:
            # Image I/O errors (file not found / unreadable)
            messagebox.showerror("OCR Error", str(exc))
            self.status_var.set("Error")
        except pytesseract.TesseractError as exc:
            # Tesseract execution / processing errors
            messagebox.showerror("OCR Error", str(exc))
            self.status_var.set("Error")

    def copy_to_clipboard(self) -> None:
        """Copy extracted text to the system clipboard."""
        text = self.output.get("1.0", tk.END).strip()
        if not text:
            self.status_var.set("No text to copy")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Copied to clipboard")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply user-selected preprocessing to the image before OCR.

        Operations:
        - resize by percent
        - convert to grayscale
        - apply binary threshold
        """
        img = image
        try:
            # Resize
            pct = max(1, int(self.resize_var.get()))
            if pct != 100:
                new_w = max(1, int(img.width * pct / 100))
                new_h = max(1, int(img.height * pct / 100))
                # Choose resampling constant compatible with Pillow versions
                resample_filter = getattr(Image, "LANCZOS", None)
                if resample_filter is None:
                    resample_filter = getattr(getattr(Image, "Resampling", None), "LANCZOS", None)
                if resample_filter is None:
                    resample_filter = getattr(Image, "BICUBIC", getattr(Image, "NEAREST", 1))
                img = img.resize((new_w, new_h), resample=resample_filter)

            # Grayscale
            if self.grayscale_var.get():
                img = ImageOps.grayscale(img)

            # Optional threshold (convert to 'L' first)
            if self.threshold_var.get():
                if img.mode != "L":
                    img = ImageOps.grayscale(img)
                thresh = int(self.threshold_scale.get())
                img = img.point(lambda p: 255 if p > thresh else 0)

            # Small sharpen to help OCR
            img = img.filter(ImageFilter.SHARPEN)
        except (OSError, ValueError, AttributeError) as exc:
            # On preprocessing errors return original image and log to stderr
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
        except (TypeError, ValueError) as exc:
            print(f"Could not serialize OCR cache to JSON: {exc}", file=sys.stderr)

    def _ocr_image_to_text(self, path: str, lang_code: str) -> str:
        """Run OCR on a single image with preprocessing; return text or empty string."""
        try:
            img = cast(Image.Image, Image.open(path))
            img = self.preprocess_image(img)
            return pytesseract.image_to_string(img, lang=lang_code)
        except (OSError, pytesseract.TesseractError) as exc:
            print(f"OCR failed for {path}: {exc}", file=sys.stderr)
            return ""

    def _create_tokenizer_and_stoplist(self, language_code: str):
        """Return (stoplist, tokenizer) for given language (tries spaCy, NLTK, fallback)."""
        # stoplist
        try:
            stoplist = "english" if language_code == "eng" else get_stop_words("polish")
        except ImportError:
            stoplist = (
                ["english"]
                if language_code == "eng"
                else ["i", "w", "z", "na", "do", "się", "nie", "że", "to", "jest", "o", "a", "o", "po", "jak", "dla"]
            )

        # tokenizer: try spaCy
        try:
            model = "en_core_web_sm" if language_code == "eng" else "pl_core_news_sm"
            nlp = spacy.load(model, disable=["ner", "parser"])

            def spacy_tokenizer(text: str):
                doc = nlp(text)
                return [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

            return stoplist, spacy_tokenizer
        except (ImportError, OSError):
            pass

        # try NLTK
        try:
            if language_code == "eng":
                try:
                    lemmatizer = WordNetLemmatizer()
                    # sanity check
                    _ = lemmatizer.lemmatize("test")

                    def nltk_eng_tokenizer(text: str):
                        tokens = re.findall(r"\b\w+\b", text.lower())
                        return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]

                    return stoplist, nltk_eng_tokenizer
                except LookupError:
                    # try to download wordnet if user agrees
                    try:
                        if messagebox.askyesno("NLTK data missing", "WordNet data is missing. Download now?"):
                            try:
                                nltk.download("wordnet", quiet=True)
                                lemmatizer = WordNetLemmatizer()

                                def nltk_eng_tokenizer(text: str):
                                    tokens = re.findall(r"\b\w+\b", text.lower())
                                    return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha()]

                                return stoplist, nltk_eng_tokenizer
                            except Exception:  # pylint: disable=broad-except
                                messagebox.showerror("Download failed", "Could not download WordNet")
                    except (tk.TclError, RuntimeError):
                        pass

                    # fallback tokenizer
                    def simple_no_lemma(text: str):
                        return [t for t in re.findall(r"\b\w+\b", text.lower()) if t.isalpha()]

                    return stoplist, simple_no_lemma
            else:
                stemmer = SnowballStemmer("polish")

                def nltk_pol_tokenizer(text: str):
                    tokens = re.findall(r"\b\w+\b", text.lower())
                    return [stemmer.stem(t) for t in tokens if t.isalpha()]

                return stoplist, nltk_pol_tokenizer
        except ImportError:
            pass

        # final fallback
        def simple_tokenizer(text: str):
            return [t.lower() for t in re.findall(r"\b\w+\b", text) if t.isalpha()]

        return stoplist, simple_tokenizer

    def _vectorize_and_score(self, texts: list, query: str, language_code: str):
        """Compute TF-IDF and cosine similarity; return list of similarity scores."""
        # create tokenizer and stoplist
        stoplist, tokenizer = self._create_tokenizer_and_stoplist(language_code)
        if not any(t.strip() for t in texts):
            messagebox.showinfo("Search", "No OCR text found in the selected folder. Check OCR language/preprocessing.")
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
        except ValueError as exc:
            messagebox.showerror("Search Error", f"Could not build TF-IDF vocabulary: {exc}")
            return []
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Search Error", f"Unexpected error during vectorization: {exc}")
            return []

    def search_similar_images(self) -> None:
        """Search a folder for images whose OCRed text is similar to a user query.

        High-level flow:
        - choose folder, ask query
        - collect image paths, load cache
        - OCR missing entries, save cache
        - compute TF-IDF + cosine similarity and show results
        """
        folder_path = filedialog.askdirectory(title="Select folder with images to search")
        if not folder_path:
            return
        query = simpledialog.askstring("Query", "Enter text query to search for:")
        if not query:
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(str(pathlib.Path(folder_path) / ext)))
        image_paths = sorted(image_paths)
        if not image_paths:
            messagebox.showinfo("Search", "No image files found in folder.")
            return

        cache_path = pathlib.Path(folder_path) / ".ocr_cache.json"
        cache = self._load_cache(cache_path)

        texts = []
        paths_for_corpus = []
        lang_code = self.lang_map.get(self.lang_var.get(), "eng")
        for p in image_paths:
            key = str(pathlib.Path(p).name)
            text = cache.get(key)
            if text is None:
                text = self._ocr_image_to_text(p, lang_code)
                cache[key] = text
            texts.append(text if text is not None else "")
            paths_for_corpus.append(p)

        self._save_cache(cache_path, cache)

        sims = self._vectorize_and_score(texts, query, lang_code)
        if not sims:
            return
        scored = sorted(zip(paths_for_corpus, sims), key=lambda x: x[1], reverse=True)
        self._show_search_results(scored, cache)

    def _show_search_results(self, scored_results: list, cache: dict) -> None:
        """Display search results in a popup with preview and OCR text."""
        win = tk.Toplevel(self.root)
        win.title("Search Results")
        win.geometry("900x600")

        left = tk.Frame(win)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        right = tk.Frame(win)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        listbox = tk.Listbox(left, width=50)
        listbox.pack(fill=tk.Y, expand=True)
        for path, score in scored_results:
            listbox.insert(tk.END, f"{pathlib.Path(path).name} — {score:.4f}")

        preview_img_label = tk.Label(right)
        preview_img_label.pack(fill=tk.BOTH, expand=True)
        preview_text = scrolledtext.ScrolledText(right, height=10, wrap=tk.WORD)
        preview_text.pack(fill=tk.X, expand=False)

        def on_select(event: Any) -> None:
            sel = listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            path, score = scored_results[idx]
            try:
                img = cast(Image.Image, Image.open(path))
                img.thumbnail((600, 400))
                photo = ImageTk.PhotoImage(img)
                setattr(preview_img_label, "image", photo)  # keep ref
                preview_img_label.config(image=photo)
            except (UnidentifiedImageError, OSError):
                preview_img_label.config(image="", text="Cannot open image")
            # show OCR text from cache where possible
            key = str(pathlib.Path(path).name)
            text = cache.get(key, "")
            preview_text.delete("1.0", tk.END)
            preview_text.insert(tk.END, f"Score: {score:.4f}\n\n{text}")

        listbox.bind("<<ListboxSelect>>", on_select)


def main() -> None:
    """Entry point for the application."""
    ensure_tesseract_path()
    root = tk.Tk()
    OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
