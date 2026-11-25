"""FAQ duplicate ranking and calibration on cached E5 answer embeddings."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from faq_partial_detector import Corpus, embedding_cosines, load_embedding_cache, risks
from faq_partial_outputs import (POSITIVE, exact_report, load_benchmark, write_html,
                                 write_workbook)
from faq_report_rebuild import read_records


E5_SPLITS = ('подбор', 'проверка', 'контрольный аудит')
E5_GRID = (.75, .80, .85, .88, .90, .92, .94, .96, .98)
METHODS = {
    'Cosine E5': (False, False),
    'E5 и проверки · все продукты': (True, False),
    'E5 и проверки · внутри продукта': (True, True),
}


@dataclass
class E5Run:
    corpus: Corpus
    dense: tuple
    pool: dict
    scores: list
    top_k: int


def e5_rows(corpus, keys, pool, benchmark, dense):
    for key, feature in corpus.features(keys, dense):
        cards = tuple(corpus.cards[i] for i in key)
        same_product, relation, hard, caution = risks(*cards)
        sources = sorted(pool.get(key, ()))
        audit = benchmark.get(key, {})
        if hard:
            queue = 'существенное различие'
        elif relation != 'same' or caution:
            queue = 'сверка контекста'
        else:
            queue = 'приоритетная вычитка'
        yield dict(feature, cards=cards, flags=(same_product, relation, hard, caution),
                   source_key=key, audit=audit, sources=sources, queue=queue, signals='Cosine E5')


def e5_metrics(rows, cutoff, splits=E5_SPLITS):
    metrics = []
    for split in splits:
        cases = [r for r in rows if r['audit']['split'] == split
                 and r['audit']['decision'] != 'уточнить' and r['embedding'] is not None]
        if not cases:
            continue
        for method, (apply_risks, same_only) in METHODS.items():
            tp = fp = fn = 0
            for row in cases:
                same, _, hard, _ = row['flags']
                selected = (bool(row['sources']) and row['embedding'] >= cutoff
                            and (not apply_risks or not hard)
                            and (not same_only or same))
                positive = row['audit']['decision'] in POSITIVE
                tp += int(selected and positive)
                fp += int(selected and not positive)
                fn += int(not selected and positive)
            metrics.append((split, method, len(cases), tp, fp, fn,
                            tp / (tp + fp) if tp + fp else None,
                            tp / (tp + fn) if tp + fn else None))
    return metrics


def calibrate_e5(rows, same_product_only):
    """Select by F1 using only the tuning split, then report the comparison grid."""
    tuning = [r for r in rows if r['audit']['split'] == 'подбор'
              and r['audit']['decision'] != 'уточнить' and r['embedding'] is not None]
    labels = {r['audit']['decision'] in POSITIVE for r in tuning
              if not same_product_only or r['flags'][0]}
    cutoffs = sorted({r['embedding'] for r in tuning if r['sources']} | {.75, .80, .85, .90, .95, .98})
    grid = [(cut, *stats) for cut in cutoffs
            for stats in e5_metrics(tuning, cut, splits=('подбор',))]
    method = 'E5 и проверки · внутри продукта' if same_product_only else 'E5 и проверки · все продукты'
    ranked = [(2 * tp / (2 * tp + fp + fn), tp, -fp, cut)
              for cut, _, variant, _, tp, fp, fn, _, _ in grid if variant == method and tp]
    return (max(ranked)[-1] if len(labels) == 2 and ranked else None), grid


def prepare_e5(exports, embeddings, snapshot, *, top_k=12, block=96, expected_dimension=1024,
               require_full_coverage=False):
    missing = [str(path) for path in (*exports, embeddings, snapshot) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('Для запуска E5 нужны файлы: ' + ', '.join(missing))
    records, _ = read_records(exports)
    corpus = Corpus(records, lexical=False)
    try:
        dense = load_embedding_cache(embeddings, snapshot, corpus, expected_dimension=expected_dimension,
                                     require_full_coverage=require_full_coverage)
    except (ValueError, OSError) as exc:
        raise ValueError(f'Не удалось проверить кэш E5 и снимок строк: {exc}') from exc

    pool = corpus.pool(top_k=top_k, block=block, dense=dense, dense_search_floor=-1.)
    pairs = sorted(pool)
    scores = []
    for start in range(0, len(pairs), 1024):
        batch = pairs[start:start + 1024]
        scores.extend(zip(embedding_cosines(batch, dense), batch))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return E5Run(corpus, dense, pool, scores, top_k)


def product_match(corpus, pair):
    a, b = pair
    return bool(corpus.product[a] and corpus.product[a] == corpus.product[b])


def e5_threshold_counts(prepared):
    scopes = [(score, product_match(prepared.corpus, pair)) for score, pair in prepared.scores]
    return [(cut, sum(score >= cut and same for score, same in scopes),
             sum(score >= cut and not same for score, same in scopes)) for cut in E5_GRID]


def preview_by_band(scores, limit):
    edges = (-1., .80, .85, .90, .95, float('inf'))
    bands = [[item for item in scores if low <= item[0] < high]
             for low, high in zip(edges, edges[1:])]
    bands = [band for band in reversed(bands) if band]
    preview = []
    budget = limit
    for index, band in enumerate(bands):
        remaining = len(bands) - index
        size = min(len(band), (budget + remaining - 1) // remaining)
        if size == 1:
            preview.append(band[len(band) // 2])
        elif size > 1:
            preview.extend(band[i * (len(band) - 1) // (size - 1)] for i in range(size))
        budget -= size
    return preview


def export_e5(prepared, output_dir, *, benchmark_file=None, cutoff=None, same_product_only=False,
              preview_count=100):
    if cutoff is not None and not -1 <= cutoff <= 1:
        raise ValueError('Порог cosine E5 должен быть от −1 до 1.')
    if preview_count < 1:
        raise ValueError('Размер предварительной выборки должен быть положительным.')
    corpus, dense, pool = prepared.corpus, prepared.dense, prepared.pool
    benchmark, stale = load_benchmark(benchmark_file, corpus)
    eligible_labels = {key: value for key, value in benchmark.items()
                       if corpus.form[key[0]] != corpus.form[key[1]]
                       and all(index in dense[1] for index in key)}
    audited = list(e5_rows(corpus, eligible_labels, pool, benchmark, dense))
    calibrated, grid = calibrate_e5(audited, same_product_only)
    selected_cutoff = cutoff if cutoff is not None else calibrated
    metrics = e5_metrics(audited, selected_cutoff) if selected_cutoff is not None else []

    scores = [(score, pair) for score, pair in prepared.scores
              if (not same_product_only or product_match(corpus, pair))
              and (selected_cutoff is None or score >= selected_cutoff)]
    if selected_cutoff is None:
        scores = preview_by_band(scores, preview_count)
    rows = list(e5_rows(corpus, dict.fromkeys(pair for _, pair in scores), pool, benchmark, dense))
    for row in rows:
        if row['audit'].get('decision') in ('разные', 'конфликт'):
            row['queue'] = 'существенное различие'
    queue_order = {'приоритетная вычитка': 0, 'сверка контекста': 1, 'существенное различие': 2}
    rows.sort(key=lambda r: (queue_order[r['queue']], -r['embedding'], r['source_key']))
    counts = Counter(r['queue'] for r in rows)
    exact = exact_report(corpus)
    status = ('Просмотр по диапазонам для разметки, порог ещё не выбран' if selected_cutoff is None else
              'Задан вручную' if cutoff is not None else 'Подобран по F1 на части «подбор»')
    summary = [
        ('Метод поиска', 'Cosine E5 по сохранённым векторам ответов'),
        ('Активных карточек в выгрузках', len(corpus.cards)),
        ('Карточек с проверенным E5-кэшем', len(dense[1])),
        ('Карточек без совпавшего вектора', len(corpus.cards) - len(dense[1])),
        ('Лимит соседей на карточку и тип продукта', prepared.top_k),
        ('Пар в пуле соседей E5', len(pool)),
        ('Порог cosine E5', selected_cutoff),
        ('Статус порога', status),
        ('Приоритетная вычитка', counts['приоритетная вычитка']),
        ('Сверка контекста', counts['сверка контекста']),
        ('Существенные различия', counts['существенное различие']),
        ('Кандидаты между продуктами', sum(not r['flags'][0] and r['queue'] != 'существенное различие' for r in rows)),
        ('Пар разметки с актуальным текстом', len(audited)),
        ('Устаревших пар разметки', stale),
        ('Покрытие A и B', 'Доля общих взвешенных слов, дополнительный признак для проверки пары.'),
        ('Границы оценки', 'Метрики относятся к размеченной выборке и заданному лимиту соседей.'),
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in rows if r['queue'] != 'существенное различие']
    differences = [r for r in rows if r['queue'] == 'существенное различие']
    with TemporaryDirectory(dir=output) as temp:
        report, book = Path(temp) / 'faq_e5_report.html', Path(temp) / 'faq_e5_review.xlsx'
        write_html(report, summary, candidates, differences, exact, metrics, e5=True)
        write_workbook(book, summary, candidates, differences, audited, exact, metrics, [], grid,
                       e5=True, pool_counts=e5_threshold_counts(prepared))
        report.replace(output / report.name)
        book.replace(output / book.name)
    return summary


def run_e5(exports, embeddings, snapshot, output_dir, *, benchmark_file=None, top_k=12, block=96,
           cutoff=None, same_product_only=False, preview_count=100, expected_dimension=1024,
           require_full_coverage=False):
    prepared = prepare_e5(exports, embeddings, snapshot, top_k=top_k, block=block,
                          expected_dimension=expected_dimension, require_full_coverage=require_full_coverage)
    return export_e5(prepared, output_dir, benchmark_file=benchmark_file, cutoff=cutoff,
                     same_product_only=same_product_only, preview_count=preview_count)
