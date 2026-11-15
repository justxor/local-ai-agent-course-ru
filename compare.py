"""Сравнение локальных моделей на одинаковых вопросах."""
import argparse
import json
from statistics import median
from time import perf_counter

from agent import run

CASES = [
    ("Назови дату релиза проекта Север. Найди заметку по словам «Север релиз».",
     ("18", "январ", "2027", "sever.md"), True),
    ("Сколько дней хранятся резервные копии? Ищи по запросу «хранения резервных копий».",
     ("35", "sever.md"), True),
    ("Проверь, указан ли бюджет проекта Север. Если сведений нет, напиши «нет данных».",
     ("нет данных",), True),
    ("Ответь одним словом «готово». Поиск не требуется.", ("готово",), False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat должен быть положительным")

    for model in args.models:
        durations = []
        passed = 0
        errors = 0
        for _ in range(args.repeat):
            for question, expected, needs_tool in CASES:
                start = perf_counter()
                try:
                    answer, calls = run(question, model)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    errors += 1
                    print(json.dumps({"model": model, "question": question, "error": str(exc)},
                                     ensure_ascii=False))
                    continue
                seconds = perf_counter() - start
                durations.append(seconds)
                tool_ok = calls == [] if not needs_tool else "search_notes" in calls
                ok = tool_ok and all(part in answer.casefold() for part in expected)
                passed += ok
                print(json.dumps({"model": model, "question": question, "pass": ok,
                                  "seconds": round(seconds, 2), "tools": calls,
                                  "answer": answer}, ensure_ascii=False))
        print(json.dumps({"model": model, "pass": passed, "total": len(CASES) * args.repeat,
                          "errors": errors, "median_seconds": round(median(durations), 2)
                          if durations else None}, ensure_ascii=False))


if __name__ == "__main__":
    main()
