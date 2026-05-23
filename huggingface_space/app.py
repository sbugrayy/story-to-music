"""
Story-to-Music — HuggingFace Space (Gradio demo)

Türkçe metin → Suno/Udio için müzik prompt + Türkçe şarkı sözü.
İki fine-tuned mT5-small modeli HuggingFace Hub'dan yüklenir.

GitHub: https://github.com/sbugrayy/story-to-music
"""

import json
import re
from collections import Counter

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Konfigürasyon ─────────────────────────────────────────────────────────────

ANALYZER_REPO = "bugrayildirim/story-to-music-analyzer"
LYRICIST_REPO = "bugrayildirim/story-to-music-lyricist"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lazy Model Loading ────────────────────────────────────────────────────────

_models = {}


def _load(repo_id: str, label: str):
    if repo_id in _models:
        return _models[repo_id]
    print(f"[{label}] yükleniyor: {repo_id}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        print(f"[{label}] tokenizer tamam, model indiriliyor...", flush=True)
        # NOT: low_cpu_mem_usage=True KULLANMA — mT5'in tied embeddings'i (shared.weight →
        # encoder.embed_tokens → decoder.embed_tokens → lm_head) meta tensor yükleme yoluyla
        # doğru kopyalanmıyor; sonuç untrained ağırlıklar + çok dilli gibberish çıktı.
        # HF Space 16 GB RAM, normal yükleme yeterli (~6 GB peak iki model için).
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            repo_id,
            torch_dtype=torch.float32,
        ).to(DEVICE)
        mdl.eval()
        try:
            mdl.gradient_checkpointing_disable()
        except Exception:
            pass
        _models[repo_id] = (tok, mdl)
        print(f"[{label}] hazır ({mdl.num_parameters():,} param)", flush=True)
        return tok, mdl
    except Exception as e:
        print(f"[{label}] YÜKLEME HATASI: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


# ── Analyzer ──────────────────────────────────────────────────────────────────

def analyze(text: str) -> dict:
    tok, mdl = _load(ANALYZER_REPO, "Analyzer")
    prompt = f"Şu metni analiz et ve müzik promptu üret: {text.strip()}"
    inputs = tok(prompt, return_tensors="pt", max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=192,
            num_beams=4,
            early_stopping=True,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True).strip()
    print(f"[Analyzer] RAW OUTPUT: {decoded[:600]!r}", flush=True)
    return _parse_metadata(decoded)


def _parse_metadata(raw: str) -> dict:
    try:
        return _normalize(json.loads(raw))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return _normalize(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass
    return _normalize({})


def _normalize(d: dict) -> dict:
    instruments = d.get("instruments", [])
    if isinstance(instruments, str):
        instruments = [i.strip() for i in instruments.split(",") if i.strip()]
    return {
        "emotion": str(d.get("emotion", "hüzün")),
        "energy": int(d.get("energy", 5)) if str(d.get("energy", "5")).strip() else 5,
        "bpm": int(d.get("bpm", 90)) if str(d.get("bpm", "90")).strip() else 90,
        "key": str(d.get("key", "A minor")),
        "instruments": list(instruments) if isinstance(instruments, list) else [],
        "vocal_style": str(d.get("vocal_style", "")),
        "suno_style_prompt": str(d.get("suno_style_prompt", "")),
    }


# ── Lyricist ──────────────────────────────────────────────────────────────────

_LYRICIST_GEN_KWARGS = dict(
    max_new_tokens=350,
    do_sample=True,
    num_beams=1,
    temperature=0.9,
    top_p=0.9,
    top_k=50,
    no_repeat_ngram_size=3,
    repetition_penalty=1.3,
)


def generate_lyrics(meta: dict) -> str:
    tok, mdl = _load(LYRICIST_REPO, "Lyricist")
    instr_str = ", ".join(meta["instruments"]) if meta["instruments"] else ""
    prompt = (
        f"Türkçe şarkı sözü yaz: "
        f"duygu={meta['emotion']}, "
        f"enerji={meta['energy']}, "
        f"tempo={meta['bpm']} BPM, "
        f"ton={meta['key']}, "
        f"enstrümanlar={instr_str}, "
        f"vokal={meta['vocal_style']}"
    )
    inputs = tok(prompt, return_tensors="pt", max_length=64, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = mdl.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **_LYRICIST_GEN_KWARGS,
        )
    raw = tok.decode(out[0], skip_special_tokens=True).strip()
    return _format_lyrics(raw)


def _format_lyrics(raw: str) -> str:
    raw = re.sub(r"\[Vers\w*\s*(\d*)\s*\]",
                 lambda m: f"[Verse{(' ' + m.group(1)) if m.group(1).strip() else ''}]",
                 raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Pre[\s-]?Chor\w*\]", "[Pre-Chorus]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Chor\w*\]", "[Chorus]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Brid\w*\]", "[Bridge]", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[Outr\w*\]", "[Outro]", raw, flags=re.IGNORECASE)
    formatted = re.sub(
        r"\s*(\[(?:Verse(?:\s*\d+)?|Chorus|Bridge|Outro|Pre-Chorus|Hook|Intro)\])\s*",
        r"\n\n\1\n", raw,
    )
    formatted = re.sub(r"[ \t]+", " ", formatted)
    lines = []
    for line in formatted.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^\[", line):
            lines.append(line)
            continue
        broken = re.sub(
            r"(?<=[a-zçğıöşü?!,.])  ?([A-ZÇĞİÖŞÜ][a-zçğıöşü])",
            r"\n\1", line,
        )
        for sub in broken.split("\n"):
            sub = sub.strip()
            if sub:
                lines.append(sub)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# ── Style Prompt Fallback ─────────────────────────────────────────────────────

_EMOTION_EN = {
    "hüzün": "melancholic", "öfke": "angry", "aşk": "romantic", "özlem": "yearning",
    "neşe": "joyful", "isyan": "rebellious", "korku": "fearful", "umut": "hopeful",
}
_INSTRUMENT_EN = {
    "ney": "ney flute", "keman": "violin", "ud": "oud", "saz": "saz",
    "bağlama": "baglama", "piyano": "piano", "gitar": "guitar",
    "elektro gitar": "electric guitar", "davul": "drums", "bas": "bass",
}
_TR_LIKELY = {"nostaljik", "romantik", "dramatik", "ne", "ve", "ile", "ney", "saz", "ud"}


def _looks_english(s: str) -> bool:
    if not s or len(s) < 15 or any(c in s for c in "çğıöşüÇĞİÖŞÜ"):
        return False
    words = set(re.findall(r"[a-z]+", s.lower()))
    if words & _TR_LIKELY:
        return False
    en = {"song", "music", "ballad", "vocal", "guitar", "violin", "piano", "drums",
          "slow", "fast", "moderate", "upbeat", "minor", "major", "bpm",
          "melancholic", "romantic", "turkish"}
    return bool(words & en)


def build_style_prompt(meta: dict) -> str:
    raw = meta["suno_style_prompt"].strip()
    if raw and _looks_english(raw):
        return raw
    emotion = _EMOTION_EN.get(meta["emotion"].lower(), meta["emotion"] or "atmospheric")
    energy = meta["energy"]
    if energy <= 3:
        energy_word = "slow, melancholic"
    elif energy <= 5:
        energy_word = "moderate tempo"
    elif energy <= 7:
        energy_word = "upbeat"
    else:
        energy_word = "energetic, intense"
    instr_en = [_INSTRUMENT_EN.get(i.lower(), i) for i in meta["instruments"]]
    instr_str = ", ".join(instr_en) if instr_en else "acoustic instruments"
    vocal = meta["vocal_style"].replace("erkek", "male").replace("kadın", "female")
    vocal = vocal.replace("dramatik", "dramatic").replace("yumuşak", "soft")
    vocal = vocal.replace("kısık", "raspy").replace("ağır", "deep")
    return f"Turkish {emotion} song, {energy_word}, {instr_str}, {meta['bpm']} BPM, {meta['key']}, {vocal} vocal"


# ── Title Suggestions ─────────────────────────────────────────────────────────

_STOP = {"verse", "chorus", "bridge", "outro", "intro", "hook", "pre",
         "için", "olan", "gibi", "ama", "veya", "ile", "kadar", "daha", "çok"}
_EMO_TITLES = {
    "hüzün": ["Sessiz Geceler", "Yarım Kalan"],
    "öfke": ["Kırgın", "Ses Verme"],
    "aşk": ["Seninle", "Bir Bakış"],
    "özlem": ["Uzaktan", "Hatıra"],
}


def suggest_titles(meta: dict, lyrics: str) -> list:
    clean = re.sub(r"\[[^\]]*\]", " ", lyrics)
    words = re.findall(r"[A-Za-zçğıöşüÇĞİÖŞÜ]+", clean)
    interesting = []
    seen = set()
    for w in words:
        wl = w.lower()
        if len(wl) >= 4 and wl not in seen and wl not in _STOP:
            seen.add(wl)
            interesting.append(w)
        if len(interesting) >= 4:
            break
    base = _EMO_TITLES.get(meta["emotion"].lower(), ["Yansıma", "Bir An"])
    candidates = list(base)
    if interesting:
        candidates.append(" ".join(interesting[:2]).title())
    return candidates[:3]


# ── Gradio Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(text: str, mode: str):
    if not text or not text.strip():
        return "❌ Lütfen bir Türkçe metin girin.", "", "", ""

    if len(text.split()) < 5:
        return "❌ Metin çok kısa (en az 5 kelime).", "", "", ""

    try:
        meta = analyze(text)
    except Exception as e:
        return f"❌ Analyzer hatası: {e}", "", "", ""

    style_prompt = build_style_prompt(meta)

    if mode == "prompt_only":
        return style_prompt, "", "", json.dumps(meta, ensure_ascii=False, indent=2)

    try:
        lyrics = generate_lyrics(meta)
    except Exception as e:
        return style_prompt, f"❌ Lyricist hatası: {e}", "", json.dumps(meta, ensure_ascii=False, indent=2)

    titles = suggest_titles(meta, lyrics)
    titles_str = "\n".join(f"• {t}" for t in titles)
    params_json = json.dumps(meta, ensure_ascii=False, indent=2)

    return style_prompt, lyrics, titles_str, params_json


# ── Gradio UI ─────────────────────────────────────────────────────────────────

EXAMPLE_TEXTS = [
    ["Bir zamanlar büyük bir aşk yaşamıştım. Sonra o aşk gitti, geride sadece anılar kaldı. "
     "Şimdi her gece pencereden bakıyorum, yağmurun sesi hatırlatıyor onu bana.", "custom"],
    ["Sokakta yürüyorum, kalabalığın ortasında ama yapayalnız. İnsanların yüzüne bakıyorum, "
     "kimse beni görmüyor. Şehrin gürültüsü içimdeki sessizliği bastıramıyor.", "custom"],
    ["Bugün hava çok güzeldi, güneş açtı. Yeni bir başlangıç hissi var içimde. "
     "Geçmişi ardımda bırakıp ileriye bakıyorum.", "custom"],
    ["Babaannemin kucağında büyüdüm. Onun anlattığı türküler hala kulağımda. "
     "Anadolu'nun toprağı, kokusu, yağmuru — hepsi orada saklı.", "custom"],
]

CUSTOM_CSS = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }

/* Markdown intro/footer: brute-force contrast (overrides any opacity rule) */
.prose, .prose *:not(a):not(code):not(pre) {
    color: var(--body-text-color) !important;
    opacity: 1 !important;
}
.prose a, .prose a * { color: var(--link-text-color) !important; opacity: 1 !important; }
.prose h1, .prose h2, .prose h3 { font-weight: 700 !important; }
.prose blockquote {
    border-left: 4px solid var(--primary-500);
    padding: 8px 14px;
    margin: 12px 0;
    background: var(--background-fill-secondary);
    border-radius: 6px;
}
.prose hr {
    border-color: var(--border-color-primary);
    opacity: 0.5 !important;
}

/* Component label chips → consistent purple brand */
.gradio-container .label-wrap > span,
.gradio-container span[data-testid="block-label"] {
    background: var(--primary-500) !important;
    color: #fff !important;
}
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#ffffff",
    body_background_fill_dark="#ffffff",
    body_text_color="#0f172a",
    body_text_color_dark="#0f172a",
    body_text_color_subdued="#475569",
    body_text_color_subdued_dark="#475569",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f8fafc",
    background_fill_secondary_dark="#f8fafc",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_label_background_fill="#7c3aed",
    block_label_background_fill_dark="#7c3aed",
    block_label_text_color="#ffffff",
    block_label_text_color_dark="#ffffff",
    block_title_text_color="#0f172a",
    block_title_text_color_dark="#0f172a",
    block_border_color="#e2e8f0",
    block_border_color_dark="#e2e8f0",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#cbd5e1",
    input_border_color_dark="#cbd5e1",
    input_placeholder_color="#94a3b8",
    input_placeholder_color_dark="#94a3b8",
    border_color_primary="#e2e8f0",
    border_color_primary_dark="#e2e8f0",
    link_text_color="#7c3aed",
    link_text_color_dark="#7c3aed",
    link_text_color_hover="#6d28d9",
    link_text_color_hover_dark="#6d28d9",
    color_accent_soft="#f5f3ff",
    color_accent_soft_dark="#f5f3ff",
)

with gr.Blocks(title="Story-to-Music", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.Markdown("""
    # 🎵 Story-to-Music

    Türkçe metni **Suno** veya **Udio** için hazır müzik promptu + yapılandırılmış şarkı sözüne çevirir.

    İki fine-tuned mT5-small modeli: [Analyzer](https://huggingface.co/bugrayildirim/story-to-music-analyzer) + [Lyricist](https://huggingface.co/bugrayildirim/story-to-music-lyricist).
    Detaylar ve MCP entegrasyonu için [GitHub](https://github.com/sbugrayy/story-to-music).

    > ⏱️ CPU'da bir üretim **~30-60 saniye** sürer. İlk denemede modeller indirilir (~2.5 GB), sonrası hızlı.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Türkçe metin",
                placeholder="Bir hikâye, konsept veya duygu anlatın (5-800 kelime)...",
                lines=6,
            )
            mode = gr.Radio(
                ["custom", "prompt_only"],
                value="custom",
                label="Mod",
                info="custom = tam çıktı (stil + sözler + başlıklar) · prompt_only = sadece tek satır Suno prompt",
            )
            submit_btn = gr.Button("🎵 Üret", variant="primary", size="lg")

        with gr.Column(scale=3):
            style_out = gr.Textbox(label="🎼 Suno/Udio Style Prompt", lines=2)
            lyrics_out = gr.Textbox(label="📝 Türkçe Şarkı Sözü", lines=18)
            with gr.Row():
                titles_out = gr.Textbox(label="✨ Başlık Önerileri", lines=4)
                params_out = gr.Code(label="Parametreler (JSON)", language="json")

    gr.Examples(
        examples=EXAMPLE_TEXTS,
        inputs=[text_input, mode],
        label="Örnek metinler — tıkla, dene",
    )

    gr.Markdown("""
    ---

    **Sınırlamalar:** mT5-small kapasite tavanı — gramer doğru, yapı temiz, ancak anlamsal akıcılık zayıf.
    Suno/Udio bunu kendi yorumuyla şarkıya çevirir; pratikte kullanılabilir. Detaylı limitations: [README](https://github.com/sbugrayy/story-to-music#limitations).

    Built by [@sbugrayy](https://github.com/sbugrayy) · MIT License
    """)

    submit_btn.click(
        fn=run_pipeline,
        inputs=[text_input, mode],
        outputs=[style_out, lyrics_out, titles_out, params_out],
    )


if __name__ == "__main__":
    demo.queue(max_size=10).launch()
