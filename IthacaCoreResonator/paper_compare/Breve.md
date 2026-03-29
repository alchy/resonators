# Simuler le son d'un piano (Juliette Chabassier, 2013 — hal-00913678)

> Popularizační / přehledový článek shrnující výsledky Chabassierovy doktorské práce.
> Zdroj: Inria Saclay / ENSTA Palaiseau, spolupráce Patrick Joly a Antoine Chaigne.

---

## Hlavní oblasti

Článek popisuje komplexní fyzikální model celého piana jako mechanicko-akustického systému, kde každá komponenta a každá vazba mezi komponentami je modelována matematickými rovnicemi z první fyzikální principy. Jde o přehled, nikoliv o detailní algoritmický popis — detailní matematika je v disertaci (Chabassier 2012).

Hlavní témata:
1. Kompletní řetězec transduktoru: prsty → klaviatura → mechanismus → kladívko → struny → kobylka → ozvučnice → vzduch → uši
2. FEM numerická simulace celého 3D modelu (300 CPU, 24 hodin za 1 sekundu zvuku)
3. Validace simulace vůči měřením: tvar tlakové vlny, spektrogram, dvojitý útlum, beaty
4. Aplikace: pomoc výrobcům pian, generování fyzikálně plausibilních zvuků neexistujících nástrojů

---

## Principy paperu

### 1. Kompletní fyzikální řetězec (end-to-end)
Každá část piana má vlastní model:
- **Kladívko** — nelineární pružina s hysterezí (Hertz-like kontakt); akcelerace kladívka odpovídá měřené křivce
- **Struny** — rovnice vlnění pro příčný i podélný pohyb; inharmonicita vyplývá přirozeně z tuhosti struny (B koeficient)
- **Kobylka** — přenos energie ze struny na ozvučnici; vazbový bod
- **Ozvučnice (soundboard)** — 2D vlnění v desce; modální záření do vzduchu
- **Vzduch** — 3D vlnová rovnice kolem celého nástroje; odraz od skříně piana

### 2. Dvojitá dekadence (double decay)
Chabassier explicitně uvádí fenomén **"double decay"**: zvuk nejprve rychle klesá a pak mnohem pomaleji "bije" — je to důsledek:
- Různých rychlostí útlumu příčného a podélného pohybu struny
- Vazby více strun (uni/bi/tri-chord) s mírně různými frekvencemi → beating
- Odraz energie mezi strunou a ozvučnicí

### 3. Beaty strun (inter-string beating)
Tři struny noty nejsou dokonale laděny — malé frekvenční odchylky způsobují amplitudové modulace. Toto je v článku graficky potvrzeno na spektrogramu C#5: harmonické postupně "pulzují" v čase. Bez tohoto efektu zvuk zní synteticky.

### 4. Vazba ozvučnice
Ozvučnice není jen pasivní reproduktor — aktivně formuje spektrum. Přítomnost "hala" v nízkých frekvencích na začátku tónu (viditelné v článku na spektrogramu) je způsobena vlastními mody ozvučnice, které doznívají nezávisle od strun.

### 5. Fyzikální plausibilita jako percepční kritérium
Článek zdůrazňuje, že mozek přiřazuje fyzikálně plausibilní zvuky fyzické příčině — tj. nemusí jít o dokonalou kopii existujícího nástroje, stačí dodržet fyzikální zákony.

---

## Srovnání se současnou implementací

| Princip paperu | Stav v IthacaCoreResonator | Poznámka |
|---|---|---|
| **Inharmonicita strun** (B koeficient) | IMPLEMENTOVÁNO | `PartialParams.f_hz` je předpočítané `k * f0 * sqrt(1 + B*k^2)`; B je načten z `params.json` |
| **Bi-exponenciální útlum (double decay)** | IMPLEMENTOVÁNO | `tau1`, `tau2`, `a1` v `PartialParams`; `env1_[k]*d1_[k] + env2_[k]*d2_[k]` v smyčce |
| **Beaty strun (inter-string detuning)** | IMPLEMENTOVÁNO | `beat_hz` a `beat_depth` v `PartialParams`; `STRING_SIGNS[n_strings]` aplikuje `±beat/2` na frekvence strun; `beat_scale` v `SynthConfig` |
| **1/2/3 struny dle registru** | IMPLEMENTOVÁNO | `n_strings` = 1, 2 nebo 3; normalizace `/n_strings`; mapování bas=mono, střed=duo, výška=trio |
| **Kladívko — nelineární kontakt** | CHYBÍ | Paper modeluje Hertz kontakt s hysterezí; implementace předpokládá hotový spektrální snímek A0 amplitud z extrakce, nemodeluje dynamiku kladívka v reálném čase |
| **Ozvučnice — modální halo** | CHYBÍ | Paper popisuje samostatné dozvívání modů ozvučnice v nízkých frekvencích; implementace aplikuje pouze statický spektrální EQ (biquad), nikoli rezonující mody |
| **Kobylka — přenos energie** | CHYBÍ | Vazba struna↔kobylka ovlivňuje tvar dvojité dekadence; implementace učí `tau1/tau2` empiricky z dat, nemoděluje fyzikální příčinu |
| **Podélný pohyb struny** | CHYBÍ | Chabassier zahrnuje podélné vlnění (přispívá k phantom partials a k detailu útlumu); implementace pracuje jen s příčnými parcialy |
| **3D akustika vzduchu** | NEAPLIKUJE SE | Full FEM vzduchového pole není realistické pro RT; implementace správně abstrahuje na parametrický model |
| **Spektrální EQ (ozvučnice)** | ČÁSTEČNĚ | 8-pásmový peaking biquad s 64bodovou křivkou z `params.json`; zachytí průměrný frekvenční tvar ozvučnice, ale ne časový vývoj ani modální charakter |
| **Fyzikální plausibilita fází** | IMPLEMENTOVÁNO | Náhodné počáteční fáze při `noteOn`; odpovídá principu, že fáze nejsou perceptuálně kritické |
| **Velocity → amplituda** | IMPLEMENTOVÁNO | `vel_gain = (vel/127)^vel_gamma`; paper nezmiňuje explicitní MIDI mapování, ale princip odpovídá |
| **Percepce double decay** | IMPLEMENTOVÁNO | Bi-exp envelopy + beating dohromady reprodukují vizuální i sluchový fenomén popsaný v paperu |

---

## Doporučení

### Priorita 1 — Ozvučnicové mody (modální rezonátor)
Paper jasně ukazuje, že "halo" nízkofrekvenčního záření na začátku tónu pochází z vlastních módů ozvučnice, které jsou buzeny kobylkou a dozvívají nezávisle. Současná implementace tento efekt zcela ignoruje.

**Návrh implementace:** Přidat N=6–12 modálních rezonátorů (IIR biquad bandpass, jeden na mód ozvučnice) buzených stejným impulzem jako struny, s vlastní tau_soundboard > tau_strings. Frekvence a útlumy lze odhadnout z nízkofrekvenčního části params.json (mody se projevují jako energie pod f0 v prvních 50–200 ms).

### Priorita 2 — Phantom partials (podélný pohyb)
Podélné pohyby strun produkují parcialy na sudých násobcích f0 s odlišnou harmonickou strukturou. Jsou auditoricky nápadné zejména v bas-baritonové části. Implementace je nezachytí, protože při extrakci jsou zahrnuty do celkového spektra, ale jejich dynamické chování (rychlejší útlum než příčné parcialy) je odlišné.

**Návrh:** V extrakci `extract_params.py` detekovat skupiny partialů s anomálně rychlým útlumem a odlišnou amplitudou v párech — označit je příznakem `longitudinal=True` pro oddělené tau.

### Priorita 3 — Nelineární kladívko pro velocity modeling
Chabassierův model ukazuje, že spektrální obsah tónu se mění s velocity nelineárně (měkký úder = méně vyšších harmonických). Současná implementace aplikuje lineární `vel_gamma` na všechny amplitudy uniformně.

**Návrh:** Přidat `vel_brightness_scale` per-partial (vyšší k = strmější velocity závislost) odvozený z extrakce srovnáním velocity layers v `params.json`.

### Priorita 4 — Časová variabilita EQ
Spektrogram v paperu ukazuje, že frekvenční obsah se mění v čase (útlum vyšších harmonik je rychlejší). Biquad EQ je statický — je aplikován rovnoměrně přes celý tón.

**Poznámka:** Toto je do jisté míry zachyceno skrze `tau1/tau2` per-partial (každý partial má svůj útlum), takže jde o nižší prioritu. Problém nastane pouze pokud jsou `tau1/tau2` špatně odhadnuty pro vysoké harmonické.

---

## Shrnutí

Paper „Simuler le son d'un piano" je populárně-vědecký přehled Chabassierovy disertace (2012), nikoliv implementační návod. Popisuje fyzikálně kompletní FEM simulaci celého piana jako referenční standard. Klíčové fyzikální jevy jsou:

1. Inharmonicita strun (B koeficient) — **implementováno**
2. Dvojitá dekadence / bi-exponenciální útlum — **implementováno**
3. Beaty ze tří strun s odchylkami ladění — **implementováno**
4. Mody ozvučnice jako nezávislý rezonátor — **chybí**
5. Podélný pohyb strun (phantom partials) — **chybí**
6. Nelineární spektrální závislost na velocity — **chybí (zjednodušeno na uniformní vel_gamma)**

Implementace pokrývá tři kritické perceptuální fenomény (inharmonicita, double decay, beating) a korektně abstrahuje části, které v RT syntéze nemají smysl modelovat fyzikálně (3D akustika vzduchu). Největší chybějící prvek pro autenticitu je modální chování ozvučnice — zodpovídá za charakteristický "body" a "warmth" grand piána, který čistě aditivní syntéza bez modálního rezonátoru neprodukuje.
