"""FAQ duplicate ranking and calibration on cached E5 answer embeddings."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from faq_partial_detector import Corpus, embedding_cosines, load_embedding_cache, risks
from faq_partial_outputs import (POSITIVE, exact_report, fingerprint, load_benchmark, write_html,
                                 write_workbook)
from faq_report_rebuild import read_records


E5_SPLITS = ('подбор', 'проверка', 'контрольный аудит')
E5_GRID = (.75, .80, .85, .88, .90, .92, .94, .96, .98)
E5_BASE = 'Cosine E5'
E5_RULES = 'E5 и проверки · все продукты'
E5_INSIDE = 'E5 и проверки · внутри продукта'
E5_COVERAGE = 'E5, проверки и вложенность · все продукты'
E5_COVERAGE_INSIDE = 'E5, проверки и вложенность · внутри продукта'
METHODS = (E5_BASE, E5_RULES, E5_INSIDE, E5_COVERAGE, E5_COVERAGE_INSIDE)
COVERAGE_MARGIN = .02


@dataclass
class E5Run:
    corpus: Corpus
    dense: tuple
    pool: dict
    scores: list
    top_k: int


def audit_card_key(ident, content_hash):
    return sha256(f'{ident}\n{content_hash}'.encode()).hexdigest()


def match_saved_audit(corpus, manifest):
    by_key = {audit_card_key(c.id, fingerprint(c)): i for i, c in enumerate(corpus.cards)}
    rows = {}
    for number, (a, b, split, decision) in enumerate(manifest, 1):
        if a in by_key and b in by_key:
            key = tuple(sorted((by_key[a], by_key[b])))
            rows[key] = dict(number=number, split=split, decision=decision,
                             source='Сохранённый аудит по отпечаткам')
    return rows, len(manifest) - len(rows)


def containment_support(row):
    return row['literal'] or (max(row['cover_a'], row['cover_b']) >= .93
                              and (row['same_question'] or min(row['cover_a'], row['cover_b']) >= .65))


def effective_cosine(row, method):
    cosine = row['embedding']
    if method in (E5_COVERAGE, E5_COVERAGE_INSIDE) and containment_support(row):
        return min(1., cosine + COVERAGE_MARGIN)
    return cosine


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
        for method in METHODS:
            tp = fp = fn = 0
            for row in cases:
                same, _, hard, _ = row['flags']
                selected = (bool(row['sources']) and effective_cosine(row, method) >= cutoff
                            and (method == E5_BASE or not hard)
                            and (method not in (E5_INSIDE, E5_COVERAGE_INSIDE) or same))
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
    cutoffs = sorted({effective_cosine(r, method) for r in tuning if r['sources']
                      for method in (E5_BASE, E5_COVERAGE)} | {.90, .94, .95, .96, .97, .98, .99})
    grid = [(cut, *stats) for cut in cutoffs
            for stats in e5_metrics(tuning, cut, splits=('подбор',))]
    base = E5_INSIDE if same_product_only else E5_RULES
    coverage = E5_COVERAGE_INSIDE if same_product_only else E5_COVERAGE
    ranked = [(2 * tp / (2 * tp + fp + fn), tp, -fp, variant == base, cut, variant)
              for cut, _, variant, _, tp, fp, fn, _, _ in grid if variant in (base, coverage) and tp]
    if len(labels) == 2 and ranked:
        _, _, _, _, cut, method = max(ranked)
        return cut, method, grid
    return None, base, grid


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


def evaluate_e5(prepared, *, benchmark_file=None, saved_audit=(), cutoff=None, same_product_only=False):
    if cutoff is not None and not -1 <= cutoff <= 1:
        raise ValueError('Порог cosine E5 должен быть от −1 до 1.')
    corpus, dense, pool = prepared.corpus, prepared.dense, prepared.pool
    benchmark, missed_audit = match_saved_audit(corpus, saved_audit) if saved_audit else ({}, 0)
    external, stale = load_benchmark(benchmark_file, corpus)
    benchmark.update(external)
    eligible_labels = {key: value for key, value in benchmark.items()
                       if corpus.form[key[0]] != corpus.form[key[1]]
                       and all(index in dense[1] for index in key)}
    audited = list(e5_rows(corpus, eligible_labels, pool, benchmark, dense))
    calibrated, policy, grid = calibrate_e5(audited, same_product_only)
    selected_cutoff = cutoff if cutoff is not None else calibrated
    metrics = e5_metrics(audited, selected_cutoff) if selected_cutoff is not None else []
    return dict(benchmark=benchmark, stale=stale, missed_audit=missed_audit, audited=audited,
                cutoff=selected_cutoff, policy=policy, grid=grid, metrics=metrics)


def export_e5(prepared, output_dir, *, benchmark_file=None, saved_audit=(), cutoff=None,
              same_product_only=False, preview_count=100):
    if preview_count < 1:
        raise ValueError('Размер предварительной выборки должен быть положительным.')
    corpus, dense, pool = prepared.corpus, prepared.dense, prepared.pool
    result = evaluate_e5(prepared, benchmark_file=benchmark_file, saved_audit=saved_audit,
                         cutoff=cutoff, same_product_only=same_product_only)
    benchmark, audited = result['benchmark'], result['audited']
    selected_cutoff, policy = result['cutoff'], result['policy']
    metrics, grid = result['metrics'], result['grid']

    coverage_enabled = policy in (E5_COVERAGE, E5_COVERAGE_INSIDE)
    floor = (selected_cutoff - COVERAGE_MARGIN if coverage_enabled and selected_cutoff is not None
             else selected_cutoff)
    scores = [(score, pair) for score, pair in prepared.scores
              if (not same_product_only or product_match(corpus, pair))
              and (floor is None or score >= floor)]
    if selected_cutoff is None:
        scores = preview_by_band(scores, preview_count)
    rows = list(e5_rows(corpus, dict.fromkeys(pair for _, pair in scores), pool, benchmark, dense))
    if coverage_enabled and selected_cutoff is not None:
        rows = [r for r in rows if effective_cosine(r, policy) >= selected_cutoff]
        for row in rows:
            if row['embedding'] < selected_cutoff:
                row['signals'] = 'Cosine E5 + вложенность'
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
        ('Базовый порог cosine E5', selected_cutoff),
        ('Правило отбора', policy),
        ('Статус порога', status),
        ('Приоритетная вычитка', counts['приоритетная вычитка']),
        ('Сверка контекста', counts['сверка контекста']),
        ('Существенные различия', counts['существенное различие']),
        ('Кандидаты между продуктами', sum(not r['flags'][0] and r['queue'] != 'существенное различие' for r in rows)),
        ('Пар разметки с актуальным текстом', len(audited)),
        ('Пар аудита без совпадения отпечатков', result['missed_audit']),
        ('Устаревших пар разметки', result['stale']),
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
