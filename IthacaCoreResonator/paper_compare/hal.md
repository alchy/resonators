# Model-based digital pianos: from physics to sound synthesis
**Balazs Bank, Juliette Chabassier — IEEE Signal Processing Magazine, 2019**
*HAL Id: hal-01894219*

---

## Hlavní oblasti

Paper je přehledový článek pokrývající celý řetězec fyzikálního modelování grand piána od plných FEM/FDM simulací až po real-time DSP syntézu. Hlavní oblasti:

1. **Fyzika piana** — strunová disperse, inharmonicita, beating (dvě/tři struny, dvojitá polarizace), bi-exponenciální útlum, nelineární longitudinální vlnění (phantom partials), soundboard + zvukové záření.
2. **Komplexní fyzikální modely** — FEM/FDM v prostoru a čase (Chabassier/Inria), plná 3D simulace (~24 hod/300 CPU na 1 s audia).
3. **Digital waveguide syntéza** — efektivní modelování strun pomocí delay lines + reflection filter (Hr(z)) s allpass disperzí a lowpass ztrátovým filtrem.
4. **Modální syntéza strun** — paralelní banka 2. řádu IIR rezonátorů, jeden rezonátor = jeden vibračním mód, přímé mapování fyzikálních parametrů na koeficienty. Základ Pianoteq (Modartt, 2006).
5. **Nelineární longitudinální vibrace** — cross-product transversálních módů generuje longitudinální signál → "kovový" charakter bas. oktáv.
6. **Model kladívka** — 0D nelineární pružina + masa, diskretizace v čase.
7. **Soundboard a zvukové záření** — commuted synthesis, FDN/waveguide reverb, IIR filtr navržený z měřené impulsní odezvy desky.

---

## Principy paperu

### Strunová inharmonicita
Frekvence k-tého parciálu splňuje Euler-Bernoulli model tuhé struny:

```
f_k = f0 * k * sqrt(1 + B*k²)
```

kde `B = π³EI / (T0 L²)` je koeficient inharmonicity. Zásadní perceptuální vlastnost piana.

### Beating (rázy)
Pro jednu notu sní 2–3 struny mírně rozladěné. Modulace amplitudy (rázy) parciálu k vznikají superpozicí kosinusů na frekvencích `f_k ± Δf_k/2`. Odděleně od toho je dvojitá polarizace (ortogonální vibrační roviny jedné struny) dalším zdrojem beatingových efektů.

### Bi-exponenciální útlum
Dvoustupňový rozpad: rychlá fáze (coupling ke soundboardu) + pomalá fáze (rezonance dutin):

```
env_k(t) = A0_k * (a1 * exp(-t/τ1) + (1-a1) * exp(-t/τ2))
```

Perceptuálně kritické: ucho je velmi citlivé na doby doznívání.

### Modální syntéza (klíčová pro real-time)
Každý mód = second-order IIR resonator (Eq. 14 v paperu):

```
H_res,k(z) = b_k * z^{-1} / (1 + a1_k*z^{-1} + a2_k*z^{-2})
p_k = exp(j2π f_k/fs) * exp(-1/(τ_k*fs))
b_k = (A_k/fs) * Im{p_k}
a1_k = -2*Re{p_k},  a2_k = |p_k|²
```

Paralelní banka těchto IIR filtrů tvoří strunový model. Komplexita lineárně škáluje s počtem módů (bass ~100, treble ~5).

### Longitudinální vibrace (phantom partials)
Nelineární coupling: longitudinální mód `k` je buzen součinem transversálních módů `m`, `n` kde `k = m+n` nebo `k = |m-n|`. Klíčové pro "metalický" charakter bas tónů.

### Soundboard — commuted synthesis
Linearita systému umožňuje prohození pořadí bloků (hammer → string → soundboard ≡ soundboard impulse → string → hammer filter). Soundboard impulsní odezva může být modelována jako reverberation (FDN) nebo FIR/IIR filtr.

### Digital waveguide jako alternativa
Reflection filter `Hr(z) = Hl(z) * Hap(z)` (lowpass × allpass) sdružuje ztráty a disperzi celého round-tripu do jednoho filtru. Efektivní, ale hůře zvládá nelinearity (longitudinální vlnění).

### Model kladívka
0D: nelineární stiffness `F_h = K_h * y^P_h` (hysteretické). Diskretizace vyžaduje řešení delay-free loop každý sample.

---

## Srovnání se současnou implementací

| Princip z paperu | Stav v IthacaCoreResonator | Poznámka |
|---|---|---|
| **Inharmonicita** `f_k = k·f0·√(1+B·k²)` | **Implementováno** | `PartialParams.f_hz` přímo ukládá inharmonické frekvence extrahované z WAV; `B` parametr uložen v `NoteParams`. Vzorec použit v `analysis/extract_params.py`, ne v C++ real-time (tam jsou předspočítané f_hz). |
| **Bi-exponenciální útlum** `a1·exp(-t/τ1) + (1-a1)·exp(-t/τ2)` | **Implementováno** | `env1_[k] *= d1_[k]; env2_[k] *= d2_[k]; env = env1+env2` v `resonator_voice.cpp`. Plná shoda s paperem. Mono parciály degradují na single-exp. |
| **Beating — paralelní struny** `f_k ± Δf_k/2` | **Implementováno** | `STRING_SIGNS[n_strings-1]` × `beat_hz` v `noteOn()`. 2 struny: `±beat/2`, 3 struny: `-beat/2, 0, +beat/2`. Odpovídá papeovému modelu dvou/tří oscilátorů. |
| **Beating — dvojitá polarizace** | **Chybí** | Paper popisuje druhý zdroj beatingových efektů: ortogonální polarizace jedné struny. V implementaci se neuvažuje; všechen beating pochází pouze z mezistrunovéh rozladění. |
| **Modální syntéza (IIR rezonátory)** | **Částečně** | Implementace používá aditivní syntézu kosinus-oscilátor + multiplicativní obálka, nikoliv IIR rezonátor tvar `H_res,k(z)`. Výsledek je ekvivalentní pro lineární případ, ale schéma se liší: paper doporučuje IIR bank pro přesnou kontrolu τ a snadné rozšíření o longitudinální coupling. |
| **Nelineární longitudinální vibrace (phantom partials)** | **Chybí** | Žádný cross-product mezi transversálními módy. Bass tóny proto postrádají "kovový" charakter popsaný v paper Sec. 4.2. Identifikováno v `MEMORY.md` jako kriticky chybějící prvek. |
| **Parametry z analýzy reálných nahrávek** | **Implementováno** | `params.json` obsahuje extrahované `f_hz`, `A0`, `tau1`, `tau2`, `a1`, `beat_hz`, `beat_depth` pro všechny 88×8 not. Odpovídá doporučené workflow z paperu (Sec. 4.1). |
| **Reflection filter / allpass disperse** | **Nepoužito (design choice)** | Paper navrhuje allpass v digital waveguide pro přesné ladění partialů. Implementace místo toho ukládá přímo analytické `f_hz` — ekvivalentní přesnost, jiná architektura. |
| **Soundboard — commuted synthesis** | **Chybí** | Žádný soundboard model (FDN, FIR/IIR z měřené IR). `BiquadEQ` implementuje spektrální korekci ze „spectral_eq" části `params.json`, nikoliv fyzikální soundboard model. Toto je fundamentálně jiná věc: EQ koriguje dlouhodobé spektrum, ale nezachycuje časovou strukturu soundboard IR (úvodní „shock" zvuk, modální rozkmit). |
| **Model kladívka** `F_h = K_h·y^P_h` | **Chybí** | Hardcoded dynamika přes `vel_gamma` power-law nad MIDI velocity. Žádná explicitní simulace kladívkové hmoty, stiffness exponenty ani hysteretického kontaktu. Výsledkem je absence velocity-závislé spektrální změny barvy (tvrdší úder = více vyšších harmonik). |
| **Sustain pedál — sympathetic strings** | **Částečně** | Pedál odkládá note-off (implementováno), ale nevyvolává sympathetic resonance nevyrážených strun. Paper Sec. 4.1 popisuje vzájemné napájení waveguide modelů. |
| **Duplex stringing** | **Chybí** | Vibrující nevybuzené části strun (aliquot). Perceptuálně spíše méně důležité. |
| **Soundboard šum (attack transient)** | **Implementováno (proxy)** | `NoiseParams` (attack_tau_s, floor_rms, centroid_hz, spectral_slope_db_oct) + 1-pólový LP filtr pro filtrace bílého šumu. Nahrazuje fyzikální soundboard „shock" zvuk. Funkčně adekvátní aproximace. |
| **Stereo model (per-string panning)** | **Implementováno** | MIDI-závislý stereo tilt (bas vlevo, výška vpravo) + `pan_spread` šíře + Schroeder allpass dekorelace + M/S width. Odpovídá metodice paperu pro per-string routing. |
| **Frekvenčně závislý útlum** | **Implementováno** | Každý parciál má vlastní `tau1`, `tau2` → vyšší parciály mají kratší τ. Odpovídá paperu: „Damping is frequency dependent: upper partials decay faster." |
| **Velocity-závislá barva zvuku** | **Chybí (architektonicky)** | Různé `vel_layer` mají různé `A0` amplitudy; `vel_gamma` škáluje celkovou hlasitost. Ale spektrální změna barvy (měkký úder = více bas, tvrdý = více treble) není modelována přes hammer stiffness, pouze přes `A0` hodnoty v různých vel vrstvách. |

---

## Doporučení

### Priorita 1 — Longitudinální vibrace (phantom partials)
Paper (Sec. 4.2, ref. [17, 35, 42]) identifikuje nelineární longitudinální vibrace jako klíčový prvek pro realistické basy. Implementace: pro každý parciál k generovat longitudinální signál jako cross-product transversálních módů `m*n` kde `m+n=k` nebo `|m-n|=k`. V současné architektuře to znamená přidat druhý oscilátorový bank s frekvencemi podél longitudinální disperzní relace a budit ho součiny výstupů primárního banku.

### Priorita 2 — Soundboard impulse response model
Současná `BiquadEQ` zachycuje dlouhodobé spektrum, ale ne časovou strukturu soundboard. Paper doporučuje buď FDN reverb (efektivní) nebo IIR/FIR filtr z měřené IR. Přidání krátkého (50–200 ms) soundboard reverb tonu by zásadně zlepšilo "těleso" zvuku, zejména v mf/f dynamice.

### Priorita 3 — Fyzikální model kladívka
Velocity-závislá spektrální barva (hammer stiffness exponent P_h) chybí. Lze aproximovat bez plné simulace: parametricky škálovat počáteční amplitudy vyšších parciálů funkcí velocity, odvozenou z kladívkového kontaktního času. Jednodušší než simulace ODR kladívka, ale zachytí perceptuálně klíčový efekt.

### Priorita 4 — Dvojitá polarizace jako zdroj beatingů
Aktuálně beating = pouze mezistrunnové rozladění. Pro jednostrunné noty (nejnižší basy, `n_strings=1`) chybí jakýkoliv beating. Přidání druhého oscilátoru s mírně posunutou frekvencí modelujícího druhý polarizační mód by pokrylo tuto mezeru bez velké výpočetní náročnosti.

### Priorita 5 — Sympathetic strings při pedálu
Cross-excitation ostatních strun při stisku sustain pedálu. Lze aproximovat přidáním "halo" reverbového tónu laděného na harmonické základní tóniny při pedálu-down eventu.

---

## Shrnutí

Implementace `IthacaCoreResonator` správně implementuje tři z pěti fyzikálních pilířů popsaných paperem: inharmonicitu frekvencí parciálů, bi-exponenciální útlum a mezistrunnové rázy. Chybí fyzikální model soundboardu (nahrazen spektrální EQ — funkčně nedostatečné pro časovou strukturu zvuku), model kladívka (nahrazen power-law velocity curve), nelineární longitudinální vibrace (klíčové pro basy), a dvojitá polarizace. Architektura je vhodná pro rozšíření: přidání longitudinálního banku a soundboard IR filtru jsou kompatibilní s existující strukturou `ResonatorVoice`. Paper také potvrzuje, že analytická extrakce parametrů ze sample banky (approach zvolený v `analysis/extract_params.py`) je standardní a fyzikálně podložený postup, jak píše v Sec. 4.1.
