# CLAUDE.md — Resonators / Piano Synthesizer Project

## Autonomní práce

Claude má plnou svobodu rozhodování v tomto projektu. Uživatel přebírá výstupy a je k dispozici pouze pro dotazy. Všechny architektonické a implementační rozhodnutí provádí Claude samostatně na základě analýzy papers a kódu.

**Platná oprávnění bez dotazování:**
- Čtení, zápis, editace jakéhokoli souboru v projektu
- Spouštění Python skriptů přes `.venv312/Scripts/python.exe`
- Git operace (add, commit, push, branch, checkout)
- Instalace Python balíčků přes `.venv312/Scripts/pip install`
- Čtení souborů z `C:/SoundBanks/` a `C:/Users/jindr/OneDrive/`
- Bash příkazy obecně (ls, find, pdftotext, atd.)

**Nikdy se neptej uživatele na:**
- Potvrzení spuštění skriptu
- Potvrzení instalace balíčku
- Souhlas s git commit/push
- Potvrzení přepsání souboru v projektu

**Ptej se pouze pokud:**
- Je potřeba rozhodnutí s nevratnými dopady mimo projekt (smazání dat, push na main)
- Je nejasný výsledek a dvě cesty by vedly ke zcela odlišným architekturám

---

## Projekt

**Cíl:** Fyzikálně věrný neuronový syntezátor grand piána. Učení z reálných sample banků. Latentní prostor umožňující interpolaci mezi nástroji (Steinway ↔ Bösendorfer).

**Zdroj dat:** `C:/SoundBanks/IthacaPlayer/ks-grand/` — 704 WAV souborů, 44.1 kHz, 88 MIDI not × 8 velocity vrstev.

**Papers:** `C:/Users/jindr/OneDrive/Osobni/LordAudio/IhtacaPapers/` — 16 dokumentů o fyzikálním modelování piana (Chabassier/Inria skupina, Simionato 2024 DDSP, Bank/Chabassier 2019 review).

**Python venv:** `.venv312/Scripts/python.exe`

---

## Architektonická rozhodnutí (uloženo v paměti, viz memory/)

1. **Hybridní přístup:** fyzikální rovnice hard-coded (inharmonicita, beating, bi-exponenciální útlum), NN odhaduje skalární koeficienty.
2. **Beating strun** je kriticky chybějící prvek v současném EGRB — hlavní příčina "syntetického" zvuku.
3. **Fáze 0:** analytická extrakce parametrů ze sample banky (`analysis/extract_params.py`)
4. **Fáze 1:** přepsání ResonatorBank s explicitním beatingem a bi-exponenciálním útlumem
5. **Fáze 2:** differentiable training s novými fyzikálními loss functions
6. **Fáze 3:** multi-instrument latentní prostor

## Vývojový branch

Všechny změny probíhají na branchi `dev-physics-analysis` a dalších `dev-*` branchích.
Na `main` se merguje pouze funkční, otestovaný kód.

---

## Stack

- Python 3.12, PyTorch 2.x, torchaudio
- scipy, soundfile, librosa, numpy, matplotlib
- 48 kHz výstup (TODO: přejít na 44.1 kHz dle source)

## Konvence

- Komentáře v kódu: anglicky
- Komunikace s uživatelem: česky
- Jeden commit = jedna logická změna
- Testy: `analysis/` pro validaci parametrů, `generated/` pro audio výstupy
