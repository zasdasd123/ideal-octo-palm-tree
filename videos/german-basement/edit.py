"""Edit decisions for "German's basement": caption chunks and cue times, all derived from the VO word timings."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shots as SH  # noqa: E402

W, E, END, FPS = SH.W, SH.E, SH.END, SH.FPS
CUT = {s["name"]: s["t0"] for s in SH.timeline()}

# sentence -> [(caption text, number of spoken words it covers)]; *...* = yellow power words.
# Numbers are shown as digits, and no caption (or caption line) ends on a preposition.
CAPTIONS = {
    1: [("НАШ *ЭЛИТНЫЙ*", 2), ("ЕВРОПЕЙСКИЙ", 1), ("ДАТА-ЦЕНТР", 1), ("*В ГЕРМАНИИ!*", 2)],
    2: [("НУ, *БЛЯДЬ*,", 2), ("КОНКРЕТНО...", 1), ("*У ГЕРМАНА*", 2), ("*В ПОДВАЛЕ*", 2)],
    3: [("КЛИЕНТ *ИГОРЬ*", 2), ("ПЕРЕПЛАТИЛ", 1), ("ЗА НИЗКИЙ", 2), ("ЕВРОПЕЙСКИЙ *ПИНГ*", 2)],
    4: [("В ИГРЕ —", 2), ("*240*", 2, 1.9)],
    5: [("ИГОРЬ, *БЛЯДЬ*,", 2), ("ВКЛЮЧИ *ГОЛОВУ!*", 2)],
    6: [("ПАКЕТЫ ИДУТ", 2), ("ЧЕРЕЗ *БЕТОННУЮ*", 2), ("*ПЛИТУ*,", 1), ("ПОТОМ —", 1), ("ЧЕРЕЗ *АРГЕНТИНУ*", 2)],
    7: [("ЭТО ЕЩЁ,", 2), ("*НАХУЙ*,", 1), ("*БЫСТРО!*", 1)],
    8: [("ЗАТО КАКОЙ", 2), ("*РАЗГОН!*", 1)],
    9: [("СЕРВЕР СТОИТ", 2), ("ПРЯМО", 1), ("НА *СТИРАЛЬНОЙ*", 2), ("*МАШИНЕ*", 1)],
    10: [("ВЫХОДИТ", 1), ("НА *ОТЖИМ* —", 2), ("У ПРОЦЕССОРА", 2), ("ВКЛЮЧАЕТСЯ", 1), ("*ТУРБОБУСТ!*", 1)],
    11: [("ЧАСТОТЫ —", 1), ("*В КОСМОС!*", 2)],
    12: [("TPS...", 1), ("*В ПИЗДУ*", 2)],
    13: [("ВЛАЖНОСТЬ", 1), ("В ПОДВАЛЕ", 2), ("ВЫШЕ *80%?*", 3)],
    14: [("ЖЕЛЕЗО", 1), ("АВТОМАТИЧЕСКИ", 1), ("ПЕРЕХОДИТ", 1), ("НА *ЖИДКОСТНОЕ*", 2), ("*ОХЛАЖДЕНИЕ!*", 1)],
    15: [("КОНКУРЕНТЫ", 1), ("ГОВОРЯТ —", 1), ("*«ЗАТОПИЛО»*", 1)],
    16: [("*ЗАВИДУЮТ!*", 1)],
    17: [("И В ПОДДЕРЖКУ", 3), ("*НЕ НОЙТЕ!*", 2)],
    18: [("ТИКЕТ", 1), ("БЕЗ *«?»*", 3), ("СИСТЕМА", 1), ("*ЗАКРЫВАЕТ*", 1), ("САМА —", 1), ("КАК *ПОЛОЖИТЕЛЬНЫЙ*", 2), ("*ОТЗЫВ*", 1)],
    19: [("ИГОРЬ ПИШЕТ:", 2), ("*«ВЫ ТАМ ОХУЕЛИ!»*", 3)],
    20: [("ЗНАКА ВОПРОСА", 2), ("*НЕТ.*", 1)],
    21: [("СПАСИБО", 1), ("ЗА *5 ЗВЁЗД!*", 3)],
    22: [("ТЕСТОВЫЙ СЕРВЕР —", 2), ("*24 ЧАСА*", 3, 1.3), ("*БЕСПЛАТНО*", 1, 1.1)],
    23: [("ДОЛЬШЕ У НАС", 3), ("ВСЁ РАВНО", 2), ("НИЧЕГО", 1), ("*НЕ ЖИВЁТ*", 2)],
}
PREPOSITIONS = {"В", "НА", "ЗА", "У", "С", "К", "О", "ПО", "ИЗ", "ОТ", "ДО", "БЕЗ", "ПОД", "НАД", "ПРИ", "ПРО", "ДЛЯ",
                "ЧЕРЕЗ", "И", "А", "НО", "КАК", "НЕ"}
LEAD = 0.045      # captions land a touch before the word
SWEARS = ("БЛЯДЬ", "НАХУЙ", "ПИЗДУ", "ОХУЕЛИ")
# (time, sentence) of every swear word in the voiceover: they get a camera kick and a low thud
SWEAR_T = [(w["s"], w["sent"]) for w in SH.VO["words"] if any(s in w["w"].upper() for s in SWEARS)]


def captions():
    out = []
    for s in sorted(CAPTIONS):
        k = 0
        assert sum(c[1] for c in CAPTIONS[s]) == len(SH.SENT[s]), f"sentence {s}: caption words do not add up"
        for text, n, *opt in CAPTIONS[s]:
            last = text.replace("*", "").split()[-1].strip("—,.!?…:«»")      # "«?»" is a symbol, not a word
            assert last not in PREPOSITIONS, f"caption ends on a function word: {text}"
            out.append(dict(text=text, start=W(s, k) - LEAD, wend=E(s, k + n - 1), sent=s, scale=opt[0] if opt else 1.0,
                            strong=text.startswith("*") and text.rstrip(" —,.!?…:").endswith("*")))
            k += n
    for a, b in zip(out, out[1:]):
        a["end"] = b["start"] if b["start"] - a["wend"] < 0.45 else a["wend"] + 0.22
    out[-1]["end"] = out[-1]["wend"] + 0.22
    return out


# cue times (seconds on the final timeline)
C = dict(
    label_de=0.22,                        # "ЭЛИТНЫЙ ДАТА-ЦЕНТР / ГЕРМАНИЯ" tag
    label_basement=W(2, 4) - 0.03,        # "Германа": ПОДВАЛ ГЕРМАНА stamp
    lag=W(4, 0), stutter=W(4, 2),         # "В игре": rubber-banding starts; the beat stutters from "двести"
    freeze1=W(4, 3) + 0.02,               # "сорок": freeze-frame until the cut
    trace0=W(6, 0), trace1=W(6, 3), trace2=W(6, 7), trace3=W(7, 0) + 0.05,
    fast=W(7, 3),
    spin=W(10, 2), turbo=W(10, 6),
    tps=CUT["tps"], freeze2=W(12, 1),
    cool=W(14, 5),
    envy=W(16, 0),
    ticket=W(18, 0), closed=W(18, 5) + 0.10, positive=W(18, 8) + 0.16,
    igor_ticket=W(19, 0), typed0=W(19, 2), typed1=E(19) - 0.1, noq=W(20, 0), stars=W(21, 0) + 0.10,
    free=W(22, 2),
    dead=E(23) - 0.26,
    outro=CUT["outro"], logo=CUT["outro"] + 0.05, wordmark=W(24) + 0.22, badge=W(25) - 0.20,
    slogan0=W(25, 0), slogan1=W(25, 1), bye=E(25) + 0.12,
)
