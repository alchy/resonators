# Time Domain Simulation of a Piano. Part 1: Model Description
> Juliette Chabassier, Antoine Chaigne, Patrick Joly
> ESAIM: Mathematical Modelling and Numerical Analysis, 2013, 48(05), pp. 1241–1278
> DOI: 10.1051/m2an/2013136 | hal-00913775v2

---

## Hlavní oblasti

Paper je první ze dvou článků (Part 2 se zabývá diskretizací a numerickou validací). Cílem je plná matematická formulace fyzikálního modelu grand piana — od počátečního úderu kladívka po šíření zvuku v okolním vzduchu. Důraz je kladen na zachování energetické konzistence modelu: každá rovnice, každá vazba splňuje energetickou identitu, a disipace je zavedena heuristicky, ale konzistentně.

Pět experimentálně motivovaných cílů modelu:

1. **Frekvenčně závislý útlum** — vyšší frekvence tlumeny rychleji
2. **Nelinearity** (piano vs. fortissimo) — geometrické nelinearity ve strunách
3. **Zvukový prekurzor** — podélné vlny struny přenesené přes kobylku do ozvučnice
4. **Inharmonicita** — f_n ≈ n·f₀·√(1 + B·n²) z tuhosti struny (Timoshenko beam)
5. **Fantomové parciály** — kombinační frekvence z nelineárního párování podélných a příčných vibrací

---

## Principy paperu

### 1. Model struny — nelineární tuhá struna (Nonlinear Stiff String)

Celý model struny je budován progresivně, od nejjednoduššího k nejkomplexnějšímu:

**a) d'Alembertova vlnová rovnice** (čistě harmonická, žádná inharmonicita)
ρA·∂²u/∂t² − T₀·∂²u/∂x² = 0

**b) Timoshenkův model předpjatého nosníku** (přidán úhel rotace průřezu ψ)
Přidá inharmonicitu: f_n ≈ n·f₀·√(1 + B·n²)
kde B = π²EI/(T₀L²)·(1 − T₀/EA)
Shear mody (frekvence fnS) jsou vždy nad 20 kHz — akusticky nevýznamné.
Timoshenko je preferován před Euler-Bernoulliho modelem: vyhýbá se 4. prostorovému derivátu, numericky pohodlnější.

**c) Geometricky exaktní model (GEM)** — podélné vibrace v_s
Bez zjednodušení o malých deformacích.
Nelinearita čistě geometrická (energetická hustota H(u,v) je nelineární v elongaci).
Klíčový efekt: příčná excitace kladívkem generuje podélné vibrace 2. řádu →
zdrojový mechanismus pro zvukový prekurzor a fantomové parciály.

**d) Plný model: kombinace Timoshenko + GEM** (tři neznámé: u_s, v_s, ψ_s)
Nelineárnost zachovává: příčné vlny f_n + podélné vlny f_n^ℓ (≈10× vyšší rychlost) + shear mody
Fantomové parciály vznikají z párování f_n + f_m^ℓ kombinacemi.

**e) Disipace ve strunách** — viskoelastický útlum
Dva empirické koeficienty R_u, ε_u na každou složku (příčnou, podélnou, torzní).
Výsledek: frekvenčně závislý útlum τ₁, τ₂ pro různé módy.

### 2. Model kladívka a interakce kladívko–struna

- Kladívko: tuhé dřevo + deformovatelná plsť, pohybuje se kolmo ke strunám
- Kontaktní síla (Hertz-like): F_i(t) = k_H · Φ_H(e_i) + r_H · d/dt[Φ_H(e_i)]
  kde e_i(t) je komprese plsti, p ∈ [1.5, 3.5] (závisí na kladívku)
- Člen r_H zajišťuje hysterezi (disipaci při dekomprimaci) a energetický útlum
- Kontakt může být bodový nebo distribuovaný (konvoluce s funkcí Δ_h)
- Počáteční podmínka: struna v klidu, kladívko přichází s rychlostí v_H (odpovídá MIDI velocity)
- Sbor strun (Ns = 1, 2, 3): každá struna má mírně odlišné T₀ → detuning → beating

### 3. Model ozvučnice (soundboard) — Reissner-Mindlin deska

- 2D deska s tloušťkou θ(x) proměnnou (6–9 mm), ortotropní dřevo (tenzor C)
- Tři neznámé: příčný posun u_p + dva úhly naklonění φ_p (2D verze Timoshenka)
- Frekvenčně závislý útlum: spektrální funkce f_d(λ) = α√λ + βλ + γ
  Klíčové: útlum je nelokalní operátor — nejde vyjádřit jako lokální PDE.
  V praxi se řeší modálním rozvojem (spektrální metoda).

### 4. Vazba struna–ozvučnice přes kobylku (bridge coupling)

- Struna svírá s ozvučnicí úhel α (není nulový!) — klíčový detail pro přenos podélných vibrací
- Kinematická podmínka: vertikální posun konce struny = posun bodu x_a na ozvučnici
- Síla přenesená na ozvučnici: F_p^i = cos(α)·T_i + sin(α)·T_i^ℓ
  → **podélná napnutost struny se přenáší do příčné síly na ozvučnici** (zdroj prekurzoru)
- Kobylka zjednodušena jako tuhé těleso (rigid bridge): pouze vertikální pohyb
- Energetická konzistence zachována: vazba splňuje d/dt(E_s + E_p) = −disipace

### 5. Akustické záření — 3D vlnová rovnice

- Vzduch kolem celého nástroje: rovnice pro tlak p a rychlostní pole V
- Rám piana (rim): dokonale tuhá překážka (Neumann podmínka: V·n = 0)
- Ozvučnice: fluid-structure coupling podmínka (normální rychlost vzduchu = rychlost desky)
- Výsledek: celé akustické pole v neomezené oblasti Ω (absorpční okrajové podmínky na umělé hranici)
- Úplný model zahrnuje i aerodynamiku nad deskou (zpětná vazba tlaku na desku)

### 6. Abstraktní hamiltonský rámec

Celý model je formulován jako abstraktní systém s energetickou hustotou H(p,q):
M·∂²q/∂t² + R·∂q/∂t − ∂_x(∂_p H(∂_x q, q)) + ∂_q H(∂_x q, q) = zdroj
Garantuje: konzistentní energetické odhady, stabilitu numerického schématu (Part 2), přidávání dalších fyzikálních jevů bez přepsání celé architektury.

---

## Srovnání se současnou implementací

| Princip z paperu | Stav v synth | Poznámka |
|---|---|---|
| **Inharmonicita f_n ≈ n·f₀·√(1+B·n²)** | Implementováno | Frekvence jsou předpočítány v `extract_params.py`, uloženy jako `f_hz` v `PartialParams`. Synth je načítá přímo — nevypočítává B za běhu. Fyzikálně ekvivalentní. |
| **Bi-exponenciální útlum** `a1·exp(-t/τ₁) + (1-a1)·exp(-t/τ₂)` | Implementováno | `d1_`, `d2_`, `env1_`, `env2_` v `resonator_voice.cpp`. Koeficienty τ₁, τ₂, a1 jsou fitovány z měření, takže frekvenční závislost útlumu (Cíl 1) je implicitně zachycena. |
| **Frekvenčně závislý útlum** (vyšší frekvence tlumeny rychleji) | Částečně | Zachyceno v datech τ₁(k), τ₂(k) na parciál — vyšší parciály mají typicky kratší τ. Ale synth neimplementuje explicitní viskoelastický model ani spectral damping funkci f_d(λ). |
| **Beating strun** (detuning sboru) | Implementováno | `beat_hz` v `PartialParams`, `STRING_SIGNS` ±beat/2 pro n=2,3 struny. Stereo pan dle MIDI polohy. Plně odpovídá paperu: "strings in a choir are slightly detuned". |
| **Geometrické nelinearity** (podélné vibrace) | Chybí | Paper: GEM model — nelineární párování příčných a podélných vln. Synth: čistě lineární aditivní syntéza bez podélných módů. Fantomové parciály ani zvukový prekurzor nejsou modelovány. |
| **Zvukový prekurzor** (sound precursor) | Chybí | Vyžaduje podélné vibrace struny (rychlost c_ℓ ≈ 10× vyšší) přenesené přes kobylku. V synth neexistuje žádný odpovídající mechanismus. |
| **Fantomové parciály** (phantom partials) | Chybí | Vznikají z nelineárních kombinačních frekvencí podélných a příčných módů: f_phantom = f_n ± f_m^ℓ. Synth generuje pouze inharmonické f_k. |
| **Hammer model — nelineární kontakt** (Hertz F = k·eᵖ) | Chybí | Synth neimplementuje fyziku kladívka vůbec. Velocity je mapována na amplitudu přes power law (vel/127)^γ — fenomonologicky, ne fyzikálně. Tvar transienty (attack) není fyzikálně modelován. |
| **Hystereze kladívka** (r_H člen) | Chybí | Disipace při odrazu kladívka chybí. Výsledek: tvar útokového tranzienta (tvrdost vs. měkkost útoky) není citlivý na sílu úderu způsobem, jakým je v reálném pianu. |
| **Ozvučnice (soundboard)** | Chybí | Synth nemá žádný model ozvučnice. Modální záření, frekvenčně závislý útlum desky, prostorová distribuce záření — vše chybí. |
| **Kobylka — přenos podélných vibrací** (úhel α) | Chybí | Synth nemá model kobylky. Přenos energie struna→vzduch probíhá přímou aditivní syntézou z parametrů. |
| **Spektrální EQ** (Long-Term Average Spectral correction) | Implementováno | `BiquadEQ` — 8-pásmový peaking EQ z 64-bodové křivky z `params.json`. Aproximuje účinek ozvučnice jako statický frekvenční filtr. Pragmatická náhrada chybějícího soundboard modelu. |
| **Akustické záření** (3D vlnová rovnice) | Chybí | Synth generuje přímý signál bez prostorového šíření. Není modelován žádný vzduchový prostor, rám piana, ani odrazová akustika. |
| **Sbor strun Ns=1/2/3** | Implementováno | `n_strings_` v `NoteParams`, `MAX_STRINGS=3`. Správná normalizace amplitudy `/n_strings`. |
| **Inicializace struny (klid)** | Implementováno | Náhodné počáteční fáze φ při každém `noteOn` — odpovídá neurčité počáteční poloze struny. |
| **Energetická konzistence modelu** | Netýká se | Synth je čistě fenomenologický — nepracuje s fyzikálními energetickými odhady. |
| **Double polarization** (horizontální příčné vibrace) | Chybí | Paper je zmiňuje jako rozšíření (5 neznámých místo 3). V synth neimplementováno. |

---

## Doporučení

### Vysoká priorita

1. **Šum jako proxy pro prekurzor a přechodný jev**
   Aktuální `NoiseParams` s `attack_tau_s` a `centroid_hz` plní roli přibližné náhražky za precursor + hammer noise. Parametr `centroid_hz` by mohl být fitován tak, aby spektrálně odpovídal longitudinal frekvencím dané noty (f₀^ℓ ≈ 10·f₀^⊥). Jde o minimální vylepšení s měřitelným efektem.

2. **Frekvenčně závislý útlum — explicitní validace**
   Ověřit, zda `τ₁(k)`, `τ₂(k)` ve fitovaných datech skutečně narůstají s klesajícím k (paper: vyšší parciály tlumeny rychleji). Pokud fit provedený v `extract_params.py` zachycuje tuto tendenci, model je částečně konzistentní s Cílem 1. Přidat diagnostický plot závislosti τ(k) pro vybrané noty.

3. **Fantomové parciály — přidání jako extra parciály**
   Kombinační frekvence f_phantom = f_n + f_m^ℓ kde f_m^ℓ = m·c_ℓ/(2L) a c_ℓ = √(E/ρ) ≈ 5000 m/s.
   Pro A0 jsou to přibližně k·475 Hz (longitudinal) kombinované s transverse spektrem.
   Lze přidat jako samostatné parciály s velmi krátkým τ (prekurzor decay ≈ 5–50 ms) a nízkou amplitudou.
   Implementačně: rozšíření `extract_params.py` o výpočet a uložení phantom partials do `params.json`.

4. **Hammer velocity → spektrální tvar (ne jen amplituda)**
   Paper: nelineárnost kontaktu F = k·eᵖ mění tvar kontaktní síly → spektrální obsah tranzienty závisí na velocity.
   Synth: vel_gain mění jen amplitudu, ne spektrální tvar.
   Minimum: velocity-závislé ladění τ₁ pro krátké parciály (fortissimo → kratší attack τ) nebo velocity-dependent harmonic_brightness.

### Střední priorita

5. **Kobylka angle α — uložit do params.json jako metadata**
   Úhel α určuje, jak silně se podélné vibrace projevují ve zvuku. Pro fyzikálně informované rozšíření synth by bylo vhodné mít α v datech.

6. **Ozvučnice jako konvoluce s impulsní odezvou**
   Nejpragmatičtější přiblížení k soundboard modelu bez PDE: měřená nebo syntetizovaná IRs desky.
   Aktuální biquad EQ zachycuje pouze statickou spektrální barvu, ne modální strukturu ozvučnice (rezonance).

### Nízká priorita

7. **Double polarization** (horizontální složka příčného pohybu strun)
   Klíčový jev pro „shimmer" vysokých strun. Implementačně: druhý set `beat_hz` v horizontální rovině s jiným útlumem. Vyžaduje rozšíření `PartialParams`.

8. **Geometrické nelinearity v plném rozsahu**
   Plná implementace GEM by vyžadovala simulaci PDE, nikoli aditivní syntézu — mimo rozsah současné RT architektury. Jen pro offline referenční syntézu.

---

## Shrnutí

ChabassierChaigneJoly (2013) je referenční fyzikální model nejkomplexnějšího typu: plný PDE model celého piana s prokázanou energetickou konzistencí a validací vůči měřením na Steinway D. Model pokrývá pět fyzikálních jevů (inharmonicita, frekvenčně závislý útlum, zvukový prekurzor, fantomové parciály, nelinearity piano/fortissimo), které jsou vzájemně provázány přes geometrické nelinearity struny.

Aktuální C++ synth implementuje správně inharmonicitu, bi-exponenciální útlum, beating sboru strun a normalizaci úrovně. Tyto složky jsou fyzikálně konzistentní s papery, ale jsou fitovány z měření (ne simulovány z prvních principů).

Největší mezery jsou tři fyzikální jevy, které paper identifikuje jako klíčové pro autentičnost pianového zvuku, a které synth zcela postrádá:

- **Zvukový prekurzor** (podélné vibrace → percussive high-freq click na začátku tónu)
- **Fantomové parciály** (kombinační frekvence posilující spektrální bohatost)
- **Velocity-dependent spektrální tvar** (nelineární kontakt kladívka → tvrdost vs. měkkost útoku)

Nejvýnosnější okamžité vylepšení: přidání phantom partials jako extra řádků do `params.json` (kratší τ, nižší amplituda, frekvence odvozené z longitudinal eigenfrequencies). Druhý krok: kalibrace `centroid_hz` šumu vůči longitudinal f₀^ℓ pro každou notu.
