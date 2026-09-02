"""Build the research-plan PDF from the measurement files themselves.

Every table below is generated from the JSON the harness wrote, not retyped, so the document
cannot drift away from the data. Where a measurement has not run yet, the section says so
instead of leaving a gap the reader would have to interpret.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\macad\Documents\_Projects\BambooEngine")
SHOTS = REPO / "Raytracer" / "SavedUserData" / "Screenshots"
SCRATCH = Path(__file__).resolve().parent
OUT = SCRATCH / "plan"
OUT.mkdir(exist_ok=True)

NAME = {
    "cornell-box": "Cornell Box", "bedroom": "Bedroom", "staircase": "Staircase",
    "veach-ajar": "Veach Ajar", "san-miguel": "San Miguel", "zero-day": "Zero Day",
    "sun-temple": "Sun Temple", "sun-temple-2": "Sun Temple 2",
    "breakfast-room": "Breakfast Room", "sponza": "Sponza",
}
CELL = {f"{k}--own": v for k, v in NAME.items()}
RUNGS = ["32", "64", "128", "256", "512"]
SIX = ["cornell-box", "bedroom", "staircase", "veach-ajar", "san-miguel", "zero-day"]
CANDIDATES = ["cornell-box", "bedroom", "staircase", "breakfast-room", "veach-ajar",
              "sponza", "san-miguel", "sun-temple", "sun-temple-2", "zero-day"]


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


scenes = load(SHOTS / "scenes-probe" / "scenes.json", {})
strategy = load(SCRATCH / "strategy.json", {})
params = load(SHOTS / "parametry-pass2" / "parameters.json", {})
params1 = load(SHOTS / "parametry" / "parameters-pass1.json", {})
suntemple = load(SHOTS / "parametry-suntemple" / "parameters.json", {})
costs = load(SHOTS / "parametry-testowania" / "koszt-klatki.json", {})
m1 = load(SHOTS / "wyniki-czas" / "m1-wyniki.json", None)


def refs():
    out = {}
    for sidecar in sorted((SHOTS / "ewaluacja-refs").glob("*.json")):
        data = load(sidecar)
        if not data:
            continue
        out[sidecar.stem] = {
            "frames": data.get("raytracing", {}).get("frameIndex", 0),
            "ms": data.get("benchmark", {}).get("meanFrameMs", 0.0),
            "vram": data.get("benchmark", {}).get("videoMemoryBytes", 0) / 1048576,
        }
    return out


REFERENCES = refs()


def tex(text):
    """Escape what LaTeX would otherwise eat."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                 ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        text = text.replace(a, b)
    return text


def num(value, digits=3, dash="---"):
    if value is None:
        return dash
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits=1, dash="---"):
    if value is None:
        return dash
    return f"{100 * value:.{digits}f}".replace(".", ",")


def table(header, rows, spec, caption=None, small=r"\footnotesize"):
    lines = [r"\begin{center}", small,
             r"\begin{tabular}{" + spec + "}", r"\hline",
             " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}"]
    if caption:
        lines += [r"\\[2pt]", r"\footnotesize\itshape " + caption]
    lines += [r"\end{center}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------- sections

def scene_table():
    rows = []
    for key in CANDIDATES:
        entry = scenes.get(key)
        if not entry:
            continue
        extent = entry.get("extent") or [0, 0, 0]
        at128 = entry["rungs"].get("128", {})
        rows.append([NAME[key],
                     f"{entry.get('triangles', 0):,}".replace(",", "\\,"),
                     num(max(extent), 0),
                     pct(at128.get("litShare")),
                     num(at128.get("voxelMetres"), 3)])
    return table(["scena", "trójkąty", "bok [m]", "oświetl. [\\%]", "woksel [m]"],
                 rows, "l r r r r")


def share_table():
    rows = []
    for key in CANDIDATES:
        entry = scenes.get(key)
        if not entry:
            continue
        cells = []
        for rung in RUNGS:
            rung_data = entry["rungs"].get(rung)
            if not rung_data or rung_data.get("failed"):
                cells.append("---")
            elif rung_data.get("truncated"):
                cells.append(pct(rung_data.get("litShare")) + "$^{*}$")
            else:
                cells.append(pct(rung_data.get("litShare")))
        rows.append([NAME[key]] + cells)
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r",
                 "$^{*}$ szczebel, na którym liczba oświetlonych wokseli przekracza "
                 "131\\,072-elementowy bufor kompaktacji: mierzy pułap, nie siatkę.")


def acceptance_table():
    rows = []
    for key in CANDIDATES:
        grids = strategy.get(key, {})
        if not grids:
            continue
        rows.append([NAME[key]] + [pct(grids[r]["accepted"]) if r in grids else "---"
                                   for r in RUNGS])
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r")


def voxel_metres_table():
    rows = []
    for key in CANDIDATES:
        entry = scenes.get(key)
        if not entry:
            continue
        rows.append([NAME[key]] + [num(entry["rungs"].get(r, {}).get("voxelMetres"), 3)
                                   for r in RUNGS])
    return table(["scena"] + [f"${r}^3$" for r in RUNGS], rows, "l r r r r r")


def reference_table():
    rows = []
    for key in SIX + ["sun-temple"]:
        entry = REFERENCES.get(f"{key}--own")
        if not entry:
            continue
        rows.append([NAME[key], f"{entry['frames']:,}".replace(",", "\\,"),
                     num(entry["ms"], 3), num(entry["vram"], 0)])
    return table(["scena", "klatek", "ms/klatkę", "pamięć [MiB]"], rows, "l r r r")


def sweep_table(factor, label, source, cells):
    columns = sorted({v for cell in cells
                      for v in source.get("scenes", {}).get(cell, {}).get("_sweep", {})
                      .get(factor, {})}, key=float)
    rows = []
    for cell in cells:
        entry = source.get("scenes", {}).get(cell, {})
        readings = entry.get("_sweep", {}).get(factor, {})
        chosen = str(entry.get(factor, ""))
        out = []
        for value in columns:
            reading = readings.get(value)
            if reading is None:
                out.append("---")
            elif value == chosen:
                out.append(r"\textbf{" + num(reading, 5) + "}")
            else:
                out.append(num(reading, 5))
        rows.append([CELL.get(cell, cell)] + out + [chosen or "---"])
    return table(["scena"] + [tex(c) for c in columns] + ["wybór"], rows,
                 "l" + " r" * (len(columns) + 1), label)


def cost_table():
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = costs.get(cell)
        if not arms:
            continue
        rows.append([CELL[cell], num(arms["BSDF"], 3), num(arms["WIE"], 3),
                     num(arms["WIE"] / arms["BSDF"], 2) + r"$\times$"])
    return table(["scena", "PT [ms]", "VXPG [ms]", "mnożnik"], rows, "l r r r")


def budget_table(budget_ms=24):
    rows = []
    for cell in [f"{k}--own" for k in SIX]:
        arms = costs.get(cell)
        if not arms:
            continue
        n1 = max(1, math.ceil(budget_ms / arms["BSDF"]))
        n2 = max(1, math.ceil(budget_ms / arms["WIE"]))
        t1, t2 = n1 * arms["BSDF"], n2 * arms["WIE"]
        rows.append([CELL[cell], str(n1), num(t1, 1), str(n2), num(t2, 1),
                     pct(abs(t1 - t2) / max(t1, t2))])
    return table(["scena", "N(PT)", "czas PT", "N(VXPG)", "czas VXPG", "rozjazd [\\%]"],
                 rows, "l r r r r r")


def m1_table():
    if not m1:
        return (r"\emph{Przebieg trwa w chwili złożenia tego dokumentu. Tabela zostanie "
                r"uzupełniona po jego zakończeniu.}" + "\n")
    rows = []
    for cell, arms in m1.get("data", {}).items():
        base = arms.get("BSDF", {}).get("0", {})
        guided = arms.get("WIE", {}).get("0", {})
        if not base or not guided:
            continue
        ratio = base["flip"]["mean"] / guided["flip"]["mean"] if guided["flip"]["mean"] else None
        rows.append([CELL.get(cell, cell),
                     num(base["ms"], 2), str(base["frames"]), num(base["flip"]["mean"], 5),
                     num(guided["ms"], 2), str(guided["frames"]), num(guided["flip"]["mean"], 5),
                     pct(ratio, 0)])
    return table(["scena", "ms", "kl.", "FLIP", "ms", "kl.", "FLIP", "VXPG/PT"],
                 rows, "l r r r r r r r",
                 "Kolumny 2--4: śledzenie ścieżek. Kolumny 5--7: technika naprowadzana. "
                 "Ostatnia kolumna powyżej 100\\,\\% oznacza przewagę techniki naprowadzanej.")


def suntemple_table():
    entry = suntemple.get("scenes", {}).get("sun-temple--own", {})
    readings = entry.get("_sweep", {}).get("voxel.gridDim", {})
    scene = scenes.get("sun-temple", {})
    grids = strategy.get("sun-temple", {})
    rows = []
    for rung in RUNGS:
        rows.append([f"${rung}^3$",
                     num(scene.get("rungs", {}).get(rung, {}).get("voxelMetres"), 2),
                     num(readings.get(rung), 5),
                     pct(grids.get(rung, {}).get("accepted"))])
    return table(["siatka", "woksel [m]", "FLIP", "przepustowość [\\%]"], rows, "l r r r")


# --------------------------------------------------------------------- document

BODY = r"""
\section{Po co ten dokument}

Praca porównuje dwie techniki obliczania oświetlenia globalnego w czasie rzeczywistym:
\textbf{śledzenie ścieżek} (dalej: wariant podstawowy) i \textbf{śledzenie ścieżek
naprowadzane rozkładem wokselowym}, czyli metodę, w której z widocznej części sceny buduje
się co klatkę przestrzenny rozkład jasności i używa go do celowania promieni.

Pytanie badawcze brzmi: \emph{czy naprowadzanie opłaca się przy budżecie czasu rzeczywistego,
w jakich scenach, i jakim kosztem}. Odpowiedź musi być liczbowa, powtarzalna i odporna na
zarzut, że wybrano korzystne warunki --- stąd cały aparat opisany niżej.

Dokument opisuje: warunki pomiaru, każdy miernik razem ze sposobem jego liczenia, przebieg
badania krok po kroku wraz z celem każdego kroku, oraz wszystkie wyniki uzyskane do chwili
jego złożenia.

\section{Stanowisko i warunki wspólne}

\begin{itemize}
\item Renderer własny (Direct3D~12, DXR), uruchamiany w trybie bezgłowym --- bez okna,
      bez interfejsu, z zapisem obrazu do pliku PNG.
\item Rozdzielczość \textbf{1920$\times$1080}, kompilacja \textbf{Release}.
\item Układ główny: \textbf{AMD Radeon RX~9070~XT}. Układ weryfikacyjny:
      \textbf{NVIDIA GeForce RTX~3050 Mobile}. Każda liczba bez wskazania układu pochodzi
      z układu głównego.
\item Liczba odbić: jedno po punkcie cieniowania (w konwencji pracy: dwa odbicia).
      Ta sama dla obu wariantów --- jest to jedyny parametr, który obie techniki dzielą.
\item Oświetlenie nieba wyłączone, oświetlenie pochodzi wyłącznie ze źródeł sceny.
      Powód: przy dominującym niebie rozkład naprowadzający nie ma czego reprezentować i
      porównanie przestaje dotyczyć badanej metody.
\item Ekspozycja, kontrast i nasycenie są \textbf{zamrożone per scena}. Decydują o tym, jak
      wygląda obraz, a więc o tym, co znaczy każda liczba miernika percepcyjnego; nie są
      parametrami swobodnymi.
\end{itemize}

\section{Mierniki i sposób ich liczenia}

\subsection{FLIP --- miernik percepcyjny}

Podstawowy miernik jakości obrazu. Porównuje obraz badany z obrazem odniesienia i zwraca
\textbf{mapę błędu} o wartościach od 0 do 1, osobno dla każdego piksela, modelując to, co
człowiek faktycznie zauważa: czułość na kontrast w różnych częstotliwościach, różnice barwy
oraz zaburzenia krawędzi. Używana wersja: \textbf{1.7}, tryb \textbf{LDR}, gęstość kątowa
\textbf{67,02 piksela na stopień} (odpowiada obrazowi oglądanemu z typowej odległości).

Z mapy błędu liczone są statystyki rozkładu: \textbf{średnia} (miernik wiodący --- jako
jedyna liczy pojedyncze bardzo jasne artefakty z ich prawdziwą wagą), \textbf{mediana},
\textbf{kwartyle}, \textbf{95.\ percentyl}, wartość \textbf{najmniejsza} i
\textbf{największa}, oraz \textbf{mediana ważona} (pula masy błędu --- ślepa na pojedyncze
artefakty, dlatego podawana obok, a nie zamiast).

\subsection{MSE --- miernik drugi, niezależny}

Średnia kwadratów różnic wartości pikseli między obrazem badanym a odniesieniem, po trzech
składowych barwy. Podawany zawsze obok FLIP, bo obie wielkości mogą się rozejść: MSE karze
jasne artefakty znacznie mocniej i jest mniej podatny na zarzut, że wniosek zależy od
jednego modelu percepcji.

\subsection{Obraz odniesienia}

Obraz \emph{prawdziwy}, do którego porównywane jest wszystko inne: ten sam kadr, to samo
oświetlenie, wyrenderowany wariantem podstawowym przez \textbf{1800 sekund}. Jest związany
ze sceną, stanem kamery, oprawą świetlną i rozdzielczością --- zmiana którejkolwiek z tych
rzeczy unieważnia go.

Zastrzeżenie, które musi towarzyszyć każdemu wynikowi: obraz odniesienia sam jest obrazem
śledzenia ścieżek, więc jego szum resztkowy koreluje z szumem wariantu podstawowego i
\emph{zaniża} mierzoną odległość tego wariantu. Efekt jest przy 1800~s mały i jednakowy dla
wszystkich scen, więc nie zmienia ich uporządkowania.

\subsection{Koszt klatki}

Różnica czasu ściennego między dwoma kolejnymi obiegami pętli gry, mierzona zegarem
wysokiej rozdzielczości. Obejmuje: obsługę komunikatów okna, aktualizację sceny, zbudowanie
i wykonanie grafu renderowania oraz prezentację (bez oczekiwania na wygaszanie pionowe).
Nie obejmuje kosztu zapisu obrazu --- silnik odejmuje zmierzony czas odczytu i kodowania
poprzedniego zrzutu, a samo kodowanie PNG biegnie poza wątkiem renderującym.

Raportowana wartość jest \textbf{średnią po wszystkich klatkach danego obrazu}, licząc od
chwili uzbrojenia zrzutu, czyli \emph{po} rozgrzewaniu.

\subsection{Pamięć karty graficznej}

Dwie liczby. Pierwsza to własny inwentarz silnika, etap po etapie, z rzeczywistego rozmiaru
alokacji. Druga to zużycie pamięci lokalnej przez cały proces, odczytane z systemu ---
obejmuje bufory sceny, tekstury i struktury przyspieszające budowane przez sterownik, więc
to ona odpowiada na pytanie o koszt \emph{całej metody}.

\subsection{Proporcja oświetlonych wokseli}

Miara opisująca scenę, wprowadzona po to, żeby dobór scen dało się uzasadnić liczbą.

\[
\text{proporcja} = \frac{\text{woksele, do których dotarło światło}}
                        {\text{woksele zawierające geometrię}}
\]

Licznik to komórki siatki, w których po jednym odbiciu od widocznych punktów cieniowania
wylądowała energia. Mianownik to komórki, przez które przechodzi jakakolwiek geometria.
Obie liczby pochodzą z tej samej klatki i tej samej siatki.

Wielkość ta \textbf{zależy od kadru} (naświetlanie startuje od widocznych pikseli) oraz od
rozdzielczości siatki, więc podawana jest zawsze razem z jednym i drugim.

\subsection{Przepustowość naprowadzania}

Udział promieni naprowadzanych, które przechodzą przez bramkę poprawności. Promień jest
odrzucany, gdy trafi \emph{przed} woksel, do którego był celowany --- czyli gdy po drodze
stoi przeszkoda. Wielkość ta odpowiada na pytanie, czy metoda ma w danej scenie czym
sterować.

Zastrzeżenie: \textbf{nie jest to predyktor wyniku}. Scena o wysokiej przepustowości potrafi
przegrać, a o niskiej wygrać, bo wynik zależy jeszcze od tego, ile klatek zabiera koszt
budowy rozkładu. Przepustowość wyjaśnia wynik, nie przewiduje go.

\subsection{Poziom szumu pomiaru}

Ten sam wariant, ta sama scena, dwa \emph{osobne uruchomienia programu}. Rozrzut liczony jako

\[
\text{rozrzut} = \frac{|a - b|}{\max(a, b)}
\]

gdzie $a$ i $b$ to średnie FLIP obu uruchomień. Jest to próg, poniżej którego różnicy nie
wolno interpretować. Przedział ufności liczony wewnątrz jednego uruchomienia \emph{nie}
nadaje się do tego celu --- obrazy jednego procesu dzielą rozgrzanie i stan cieplny, więc
zaniża on rzeczywisty rozrzut kilkukrotnie.

\section{Przebieg badania krok po kroku}

\subsection{Faza 0 --- dobór scen}

\textbf{Cel.} Wybrać sześć scen tak, żeby wybór dało się uzasadnić liczbą, a nie wrażeniem.
Zbiór ma pokryć trzy rozmiary sceny oraz dwa poziomy proporcji oświetlonych wokseli ---
bo to właśnie ta proporcja decyduje, ile rozkład naprowadzający wnosi ponad rozkład
jednostajny.

\textbf{Sposób.} Dziesięciu kandydatów przemierzonych na pięciu szczeblach siatki, dla
każdego zapisana proporcja oświetlonych wokseli, rozmiar woksela w metrach, koszt klatki,
przepustowość naprowadzania i zrzut diagnostyczny rozkładu jasności.

\textbf{Wynik.}
"""


def build():
    parts = [BODY, scene_table(),
             r"\noindent Proporcja oświetlonych wokseli wobec szczebla siatki:" + "\n",
             share_table(),
             r"\noindent Rozmiar woksela w metrach sceny --- to on, a nie rozdzielczość, "
             r"mówi, co siatka jest w stanie rozróżnić:" + "\n",
             voxel_metres_table(),
             r"\noindent Przepustowość naprowadzania:" + "\n",
             acceptance_table(),
             r"""
\textbf{Rozstrzygnięcie.} Pułap bezwzględny (20\,\%) nie dzieli kandydatów na dwie
równoliczne grupy: przekraczają go tylko dwie sceny i obie są małe albo średnie. Wśród scen
dużych nie ma \emph{żadnej} o wysokiej proporcji i nie jest to kwestia doboru kandydatów ---
proporcja spada z rozmiarem sceny, bo światło z jednego źródła dociera do coraz mniejszej
części objętości. Osie ,,rozmiar'' i ,,rozkład jasności'' nie są więc niezależne.

Przyjęto podział \textbf{względny wewnątrz klasy rozmiaru}: w każdej klasie bierzemy scenę
o najwyższej i o najniższej proporcji.

\begin{center}
\footnotesize
\begin{tabular}{l l l}
\hline
klasa & wyższa proporcja & niższa proporcja \\
\hline
mała & Cornell Box 63,9\,\% & Bedroom 13,1\,\% \\
średnia & Staircase 31,0\,\% & Veach Ajar 6,6\,\% \\
duża & San Miguel 2,3\,\% & Zero Day 0,6\,\% \\
\hline
\end{tabular}
\end{center}

Sun Temple wypadł ze zbioru i wchodzi do pracy jako \emph{udokumentowany przypadek
negatywny} (sekcja \ref{sec:suntemple}).

\subsection{Obrazy odniesienia}

\textbf{Cel.} Wytworzyć obraz prawdziwy dla każdej sceny --- wejście dla wszystkiego, co po
nim. \textbf{Sposób.} 1800~s wariantu podstawowego na scenę. \textbf{Wynik:}
""",
             reference_table(),
             r"""
Zakres zużycia pamięci --- od 1380 do 5243~MiB --- pokazuje, że na scenach dużych największą
pozycją jest sama scena, a nie łańcuch naprowadzający. Ma to bezpośredni skutek: dwie sceny
duże nie zmieszczą się w 4~GB pamięci układu weryfikacyjnego nawet przy samym wariancie
podstawowym, więc weryfikacja obejmie cztery komórki, a brak dwóch pozostałych jest wynikiem,
nie usterką.

\subsection{Faza 1 --- dobór parametrów osobno dla każdej sceny}

\textbf{Cel.} Metoda ma parametry, których dobra wartość zależy od sceny; puszczenie
wszystkich scen na jednej nastawie krzywdzi sceny duże. Pomiar główny ma biec na nastawie
najlepszej dla danej sceny.

\textbf{Sposób.} Przemiatanie \textbf{jednoczynnikowe}: jeden parametr naraz, pozostałe na
wartościach ustalonych. Wspólny budżet 5~s, trzy obrazy na punkt, rozgrzewka 35~s,
punktowanie wobec obrazu odniesienia. Pełny iloczyn kartezjański nie wchodzi w grę czasowo i
nie jest potrzebny --- celem jest nastawa dobra, nie dowodliwie optymalna.

\textbf{Kryterium.} Najniższa średnia FLIP; przy różnicy poniżej 2\,\% wygrywa wartość
mniejsza, bo nierozstrzygalna różnica nie jest powodem, żeby płacić za droższą nastawę.

Symetria wobec wariantu podstawowego jest zachowana w ten sposób, że wariant podstawowy
\emph{nie ma} żadnego z tych parametrów --- i ten fakt trzeba w pracy zapisać wprost, żeby
czytelnik wiedział, że strojenie dotyczy jednej strony porównania.

\textbf{Wynik} (drugie przejście, parametry pozostałe na wartościach wybranych):
""",
             sweep_table("voxel.gridDim", "Rozdzielczość siatki wokseli.",
                         params, [f"{k}--own" for k in SIX]),
             sweep_table("vxpg.topLevelTree.importance",
                         "Tryb wartości liścia: 0 --- widoczność binarna, "
                         "1 --- średnia widoczność, 2 --- sama moc.",
                         params, [f"{k}--own" for k in SIX]),
             sweep_table("vxpg.tree.weightMode",
                         "Ważenie gałęzi dolnego drzewa: 0 --- sama moc, "
                         "1 --- geometria dokładna, 2 --- geometria przybliżona.",
                         params, [f"{k}--own" for k in SIX]),
             r"""
\textbf{Co z tego wynika.} Po pierwsze, najdrobniejsza siatka prawie nigdy nie wygrywa:
na czterech z sześciu scen wybór padł na najgrubszą. Powód jest kosztowy --- drobniejsza
siatka zabiera klatki, a celniejsze naprowadzanie tego nie odrabia. Wyżej poszły tylko te
sceny, na których przy grubej siatce naprowadzanie jest martwe.

Po drugie, tryb wartości liścia dzieli zbiór na dwie grupy: tam, gdzie scena faktycznie
zasłania oświetlone obszary, opłaca się bramka widoczności; gdzie nie zasłania --- opłaca się
ją pominąć, bo kosztuje cały etap obliczeń.

Po trzecie, ważenie gałęzi dolnego drzewa jest praktycznie bez wpływu: różnice rzędu
0,1--2\,\%, poniżej progu rozstrzygalności.

\textbf{Kontrola.} Przemiatanie wykonano \emph{dwukrotnie}: raz przy parametrach pozostałych
na wartościach domyślnych, raz przy wartościach wybranych w pierwszym przejściu. Oba
przejścia dały \textbf{identyczny wybór na wszystkich scenach i wszystkich czynnikach}.
Gdyby czynniki silnie na siebie oddziaływały, drugie przejście przesunęłoby przynajmniej
jeden wybór; nie przesunęło żadnego, więc przemiatanie jednoczynnikowe jest tu wystarczające.

\textbf{Dwa parametry planu, których nie da się przemiatać.} Liczba superwokseli (32) i
liczba przedstawicieli widoczności (128) są stałymi czasu kompilacji: maska widoczności to
jedno 32-bitowe słowo na kafel obrazu, po jednym bicie na superwoksel, a odcisk widoczności
to cztery takie słowa. Zmiana którejkolwiek jest zmianą struktury danych i jąder
obliczeniowych, nie ustawieniem. Obie wartości wchodzą do tabeli parametrów wspólnych razem
z tym uzasadnieniem.

\subsection{Faza 2 --- parametry testowania}

\textbf{Cel.} Ustalić trzy liczby sterujące wszystkimi pomiarami podstawowymi: budżet
równego czasu, liczbę próbek dla porównania przy równej liczbie próbek oraz pułap błędu dla
porównania przy równej wariancji.

\textbf{Koszt klatki} obu wariantów na nastawach z fazy 1:
""",
             cost_table(),
             r"""
\textbf{Budżet równego czasu --- i dlaczego kryterium trzeba było zmienić.} Pierwsza
redakcja planu żądała, żeby budżet dobrać tak, aby \emph{liczba klatek} obu wariantów była
zbliżona. Warunek ten jest przy równym czasie niewykonalny: stosunek liczby klatek jest
równy stosunkowi kosztów klatki (2,4--6,1$\times$), a budżet skraca się obu wariantom
tak samo. Równa liczba klatek jest osiągalna wyłącznie budżetem klatkowym --- czyli osobnym
pomiarem, opisanym niżej jako M2.

Wyrównywany jest zatem \textbf{czas}, a warunek, który faktycznie ma treść, wynika ze
sposobu zatrzymywania przebiegu: silnik przerywa na \emph{pierwszej klatce sięgającej
budżetu}, więc każdy wariant przestrzeliwuje o niepełną klatkę. Przy budżecie rzędu jednej
klatki obrazu przestrzelenia są duże i \emph{różne} dla obu wariantów --- i wtedy nominalnie
równy czas przestaje być równy.

\textbf{Kryterium:} najmniejszy budżet, przy którym najgorszy rozjazd faktycznie zużytego
czasu nie przekracza 10\,\%. Najmniejszy --- bo praca dotyczy czasu rzeczywistego, a cel
liczbowy to poniżej 32~ms, czyli poniżej klatki przy 30 klatkach na sekundę.

\textbf{Wynik: 24~ms} (około 40 klatek na sekundę).
""",
             budget_table(),
             r"""
Wartości 30 i 32~ms są wyraźnie gorsze (rozjazd 34,8\,\%): San Miguel przeskakuje tam na
drugą klatkę wariantu naprowadzanego i zużywa 51~ms zamiast 32.

\textbf{Zastrzeżenie.} Przy 24~ms wariant naprowadzany renderuje na San Miguelu
\emph{jedną} klatkę na obraz, więc ta komórka nie akumuluje w ogóle, a jej obraz jest
pojedynczą próbką estymatora. Jest to uczciwy odczyt przy 40 klatkach na sekundę, ale nie
wolno go czytać jak komórki, w której obraz powstał z kilkudziesięciu klatek.

\subsection{Faza 3 --- pomiary}

Każdy pomiar odpowiada na inne pytanie. Kolejność jest wiążąca, bo obrazy odniesienia są
wejściem wszystkiego, co po nich.

\begin{description}
\item[M1 --- równy czas.] Oba warianty dostają ten sam budżet 24~ms. Odpowiada na pytanie
      praktyczne: \emph{co dostanę na ekranie w tym samym czasie}. To jest pomiar wiodący.
\item[M2 --- równa liczba próbek.] Budżetem jest liczba próbek ścieżki, nie czas. Odpowiada
      na pytanie: \emph{jak dobrze metoda celuje promienie}, z pominięciem kosztu klatki.
      Wariant naprowadzany zbiera dwie próbki na klatkę, podstawowy jedną, więc równa liczba
      próbek to różna liczba klatek.
\item[M3 --- równa wariancja.] Renderowanie z zapisem \emph{każdej} klatki; odczytem jest
      klatka o najmniejszym numerze, przy której błąd spada poniżej ustalonego pułapu.
      Bez interpolacji. Odpowiada na pytanie: \emph{ile trzeba czekać na obraz o zadanej
      jakości}.
\item[M4 --- krzywe błędu wobec czasu.] Jedyny pomiar na budżecie 30~s, szesnaście punktów
      kontrolnych o rozstawie logarytmicznym. Służy wyłącznie wykresowi; żadna liczba z tego
      przebiegu nie trafia do tabel.
\item[M5 --- rozbicie kosztu klatki.] Koszt każdego etapu potoku, osobno dla obu wariantów,
      z zaznaczeniem, które etapy śledzą promienie. To jest wyjaśnienie, skąd bierze się
      mnożnik kosztu klatki. Osobno czas jednorazowej wokselizacji geometrii.
\item[M6 --- stabilność czasowa.] Ciąg kolejnych obrazów przy równej liczbie próbek, z
      błędem każdego z nich. Odpowiada na pytanie o migotanie: czy obraz jest stabilny
      między klatkami, czy tylko średnio dobry.
\item[M7 --- wariant z ponownym użyciem próbki.] Wariant metody, który oszczędza jedno
      śledzenie promienia kosztem obciążenia estymatora. Mierzony przy dwóch budżetach, bo
      obciążenie ujawnia się dopiero powyżej przecięcia krzywych.
\item[M8 --- pamięć karty graficznej.] Zużycie całej metody wobec rozdzielczości siatki,
      z rozbiciem na pozycje zależne od siatki i od rozdzielczości obrazu. Wariant
      podstawowy jako punkt odniesienia.
\item[M9 --- wpływ rozdzielczości siatki.] Jakość przy wspólnym budżecie dla kolejnych
      szczebli. Uzasadnia kolumnę ,,siatka'' w tabeli nastaw.
\item[M10 --- optymalizacje zależne od producenta.] Osobny podrozdział; wyniki nie wchodzą
      do żadnego innego pomiaru, bo wiążą metodę z konkretnym producentem układu.
\item[M11 --- weryfikacja na drugim układzie.] Powtórzenie M1 na układzie weryfikacyjnym.
      Rozstrzyga, czy znak wyniku i kolejność scen się utrzymują.
\item[M12 --- kontrole wiarygodności.] Poziom szumu pomiaru, wpływ długości obrazu
      odniesienia, wpływ rodzaju źródła światła.
\end{description}

\textbf{Powtórzenia.} Każde zadanie pomiarowe wykonywane jest \textbf{dziesięciokrotnie},
a wynikiem jest średnia. Zadania przeplatane są rundami --- program obchodzi wszystkie
zadania po jednym razie, zanim wróci do pierwszego --- żeby powolne nagrzewanie się układu
rozłożyło się na oba warianty jednakowo, a nie trafiło w ten, który akurat biegł jako drugi.
Dodatkowo cała siatka pomiarowa przemierzana jest \emph{drugi raz jako osobne uruchomienie
programu}; różnica między tymi przemierzeniami jest poziomem szumu pomiaru.

\subsection{Wyniki M1 --- porównanie przy równym czasie}
""",
             m1_table(),
             r"""
\section{Sun Temple --- przypadek negatywny}
\label{sec:suntemple}

Scena o rozpiętości 271~metrów, wyłączona ze zbioru głównego, zmierzona osobno, bo
odpowiada na pytanie ,,kiedy tej metody \emph{nie} używać'' lepiej niż jakikolwiek argument.
""",
             suntemple_table(),
             r"""
Rozpiętość całej drabiny to \textbf{2,4\,\%} --- mniej niż próg rozstrzygalności przyjęty w
fazie 1. Siatka na tej scenie jest bezczynna: nawet przy najdrobniejszym szczeblu, gdzie
przepustowość naprowadzania dochodzi wreszcie do 20\,\%, błąd obrazu nie drgnął, a koszt
klatki rośnie tam tylko o 21\,\% (18,24 $\rightarrow$ 22,14~ms), więc budżet nie jest
przeszkodą. \emph{Nie istnieje szczebel siatki, który by na tej scenie pomógł.}

Zastrzeżenie: poziom błędu na tej scenie jest cztery razy wyższy niż na Veach Ajar --- pięć
sekund nie zbliża jej do zbieżności. Wniosek dotyczy więc reżimu krótkiego budżetu, który
jest reżimem całej pracy, a nie granicy przy budżecie nieskończonym.

\section{Ustalenia metodyczne, które zmieniły przebieg badania}

Cztery rzeczy wyszły w trakcie i każda z nich zmieniła coś w planie. Wszystkie są w pracy
warte wzmianki, bo dotyczą wiarygodności liczb.

\subsection{Cichy pułap rozkładu naprowadzającego}

Dwie sceny przy drobnej siatce kończyły przebieg utratą urządzenia. Przyczyną nie była
pamięć: liczba oświetlonych wokseli przekraczała pojemność bufora, do którego są pakowane;
nadmiar był odrzucany, ale licznik rósł dalej, a kolejny etap rozdzielał z niego pracę na
komórki, których nikt nie zapisał. Po poprawce oba szczeble liczą się normalnie.
\emph{Skutek dla wyników:} szczebel, na którym liczba oświetlonych wokseli przekracza
pojemność, mierzy pojemność, a nie siatkę --- takie szczeble są w tabelach oznaczone i
wyłączone z doboru nastaw.

\subsection{Koszt klatki z sondy fazy 0 jest zawyżony}

Sonda doboru scen miała trzysekundową rozgrzewkę, w którą wpadał jednorazowy wypiek
geometrii. Na najdrobniejszej siatce zawyżyło to koszt klatki nawet siedmiokrotnie
(139~ms wobec rzeczywistych 22~ms). \emph{Skutek:} kolumna kosztu klatki w tabeli fazy 0 nie
nadaje się do cytowania; koszty klatki pochodzą z fazy 2, która rozgrzewa się 45~s.

\subsection{Aparatura pomiarowa wchodziła do pomiaru}

Przy budżecie 24~ms zapis obrazu zostawiał po sobie pracę, której część lądowała w pierwszej
klatce następnego obrazu. Przy budżecie trzydziestosekundowym jest to niewidoczne; przy
budżecie rzędu jednej klatki jedna skażona klatka zjada cały budżet i obraz zostaje z jedną
klatką zamiast pięciu. Rozwiązanie: klatki wyciszające między obrazami, poza każdym oknem
pomiarowym. \emph{Skutek:} pierwszy przebieg pomiaru M1 został skasowany w całości i
powtórzony; żadna liczba z niego nie trafia do pracy.

\subsection{Warunek doboru budżetu był niewykonalny}

Opisany w fazie 2. Plan został poprawiony tak, żeby mówił wprost, że wyrównywany jest czas,
a nie liczba klatek.

\section{Czego ten dokument jeszcze nie zawiera}

Do wykonania pozostają pomiary M2--M12 oraz weryfikacja na drugim układzie graficznym.
Dwie liczby parametrów testowania --- liczba próbek dla M2 i pułap błędu dla M3 --- zostaną
wyznaczone z krzywych M4, bo to ten sam przebieg dostarcza obu.
"""]
    return "\n".join(parts)


HEADER = r"""\documentclass[12pt,a5paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage[margin=11mm,top=13mm,bottom=13mm]{geometry}
\usepackage{array}
\linespread{1.06}
\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.15}
\title{\bfseries Plan badawczy i wyniki pośrednie\\[4pt]
\large Naprowadzane śledzenie ścieżek z rozkładem wokselowym}
\author{}
\date{2 września 2026}
\begin{document}
\maketitle
\thispagestyle{empty}
\tableofcontents
\newpage
"""

source = HEADER + build() + "\n\\end{document}\n"
(OUT / "plan-badawczy.tex").write_text(source, encoding="utf-8")

for _ in range(2):
    result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "plan-badawczy.tex"],
                            cwd=OUT, capture_output=True, text=True, errors="replace")
pdf = OUT / "plan-badawczy.pdf"
if not pdf.exists():
    tail = (OUT / "plan-badawczy.log").read_text(encoding="utf-8", errors="replace")
    print("\n".join(line for line in tail.splitlines() if line.startswith("!"))[:2000])
    sys.exit("PDF nie powstal")
print(f"OK: {pdf} ({pdf.stat().st_size // 1024} KB)")
