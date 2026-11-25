"""Benchmark validation, queue construction, and offline reports for FAQ review."""

from collections import Counter, defaultdict
import csv
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from faq_partial_detector import Thresholds, answer_key, qualify, risks


POSITIVE = {'дубль', 'покрытие'}
REVIEW = {'приоритетная вычитка', 'сверка контекста'}
SEVERITY = {'приоритетная вычитка': 0, 'сверка контекста': 1, 'существенное различие': 2}


def fingerprint(card):
    payload = [card.question, *card.alternatives, card.answer, *card.product_key]
    return sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def load_benchmark(path, corpus):
    if path is None:
        return {}, 0
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    good, stale = {}, 0
    for row in rows:
        a, b = row['id_a'], row['id_b']
        if (a not in corpus.index or b not in corpus.index
                or fingerprint(corpus.cards[corpus.index[a]]) != row['hash_a']
                or fingerprint(corpus.cards[corpus.index[b]]) != row['hash_b']):
            stale += 1
            continue
        key = tuple(sorted((corpus.index[a], corpus.index[b])))
        if key in good:
            raise ValueError(f'Повторная разметка пары {a} и {b}.')
        good[key] = row
    return good, stale


def search(corpus, pool, benchmark, thresholds, dense=None):
    combined = dict(pool)
    combined.update({key: set() for key in benchmark if key not in pool})
    found, benchmark_features = [], {}
    for key, feature in corpus.features(combined, dense):
        a, b = (corpus.cards[i] for i in key)
        if key not in benchmark and qualify(feature, (True, 'same', [], []), thresholds)[0] == 'ниже порога':
            continue
        flags = risks(a, b)
        row = dict(feature, cards=(a, b), flags=flags, sources=sorted(combined[key]), key=key)
        row['queue'], row['signals'] = qualify(feature, flags, thresholds)
        row['audit'] = benchmark.get(key, {})
        if key in benchmark:
            benchmark_features[key] = row
        if row['sources'] and row['queue'] in SEVERITY and answer_key(a.answer) != answer_key(b.answer):
            if row['audit'].get('decision') in ('разные', 'конфликт'):
                row = {**row, 'queue': 'существенное различие'}
            found.append(row)

    def order(row):
        a, b = row['cards']
        label = row['audit'].get('decision', '')
        audit_rank = 0 if label in POSITIVE else 2 if label in ('разные', 'конфликт') else 1
        score = .60 * row['answer'] + .22 * row['question'] + .18 * max(row['cover_a'], row['cover_b'])
        return SEVERITY[row['queue']], audit_rank, -score, a.id, b.id

    return sorted(found, key=order), benchmark_features


def benchmark_rows(features, cut, e5_only=False, splits=('подбор', 'проверка', 'контрольный аудит')):
    result = []
    for split in splits:
        rows = [r for r in features.values()
                if r['audit']['split'] == split and r['audit']['decision'] != 'уточнить'
                and answer_key(r['cards'][0].answer) != answer_key(r['cards'][1].answer)
                and (not e5_only or r['embedding'] is not None)]
        if not rows:
            continue
        for method in ('Только cosine ответа', 'Признаки · внутри продукта', 'Признаки · все продукты'):
            tp, fp, fn = 0, 0, 0
            for row in rows:
                if method == 'Только cosine ответа':
                    selected = bool(row['sources']) and row['answer'] >= cut.answer
                else:
                    status, _ = qualify(row, row['flags'], cut, method == 'Признаки · внутри продукта')
                    selected = bool(row['sources']) and status in REVIEW
                true = row['audit']['decision'] in POSITIVE
                tp += int(selected and true)
                fp += int(selected and not true)
                fn += int(not selected and true)
            result.append((split, method, len(rows), tp, fp, fn,
                           tp / (tp + fp) if tp + fp else None,
                           tp / (tp + fn) if tp + fn else None))
    return result


def lexical_sweep(features, current):
    from dataclasses import replace
    from itertools import product

    grid = product((.70, .76, .82), (.34, .46, .56), (.90, .93, .96))
    rows = []
    for answer, question, coverage in grid:
        cut = replace(current, answer=answer, question=question, coverage=coverage, embedding=None)
        stats = benchmark_rows(features, cut, splits=('подбор',))
        for item in stats:
            if item[1] == 'Признаки · все продукты':
                rows.append((answer, question, coverage, *item[2:]))
    return rows


def embedding_sweep(features, lexical_cut):
    """Compare E5 thresholds on the same labelled pairs with both vectors present."""
    from dataclasses import replace

    grid = (None, .80, .83, .86, .89, .92, .95, .97)
    result = []
    for cutoff in grid:
        run = benchmark_rows(features, replace(lexical_cut, embedding=cutoff), e5_only=True)
        result.extend((cutoff, *r) for r in run if r[1] == 'Признаки · все продукты')
    return result


def exact_report(corpus):
    grouped = defaultdict(list)
    for k, card in zip(corpus.form, corpus.cards):
        grouped[k].append(card)
    rows = []
    for cards in grouped.values():
        if len(cards) < 2:
            continue
        for card in cards:
            rows.append((len(rows) + 1, cards[0].id, len(cards), card.id, card.row['Продукт'],
                         card.row['Подпродукт'] or '', card.question, card.answer))
    return rows


def view_row(r):
    a, b = r['cards']
    same, relation, hard, soft = r['flags']
    audit = r['audit']
    queue = r['queue']
    if not r['sources']:
        queue = 'вне пула соседей'
    if answer_key(a.answer) == answer_key(b.answer):
        queue = 'совпадает после нормализации'
    return (queue, 'один продукт' if same else 'между продуктами или продукт пуст',
            r['answer'], r['question'], float(r['cover_a']), float(r['cover_b']), r['embedding'],
            r['signals'], ' | '.join([*hard, *soft]), audit.get('decision', ''),
            audit.get('comment', ''), audit.get('old_verdict', ''), a.id, b.id,
            a.row['Продукт'], a.row['Подпродукт'] or '', b.row['Продукт'], b.row['Подпродукт'] or '',
            a.question, b.question, a.answer, b.answer, ' | '.join(r['sources']), audit.get('source', ''))


PAIR_HEADER = ('Очередь', 'Сравнение', 'Cosine ответа (TF-IDF)', 'Cosine вопроса (TF-IDF)',
               'Доля A в B', 'Доля B в A', 'Cosine E5', 'Сигналы', 'Что проверить',
               'Решение по вычитанной паре', 'Комментарий', 'Старый вердикт ГигаЧата',
               'ID A', 'ID B', 'Продукт A', 'Подпродукт A', 'Продукт B', 'Подпродукт B',
               'Вопрос A', 'Вопрос B', 'Ответ A', 'Ответ B', 'Источник поиска', 'Источник разметки')
EXACT_HEADER = ('Строка', 'Группа', 'Размер', 'ID FAQ', 'Продукт', 'Подпродукт', 'Вопрос', 'Ответ')
METRIC_HEADER = ('Раздел', 'Метод', 'Размеченных пар', 'Найдено полезных', 'Лишних', 'Пропущено полезных',
                 'Доля полезных среди найденных', 'Доля найденных среди полезных в выборке')


def write_workbook(path, summaries, candidates, differences, audited, exact, metrics, lexical, sweep):
    wb = Workbook()
    wb.remove(wb.active)

    def sheet(name, header, data, widths, score_cols=(), row_height=34):
        ws = wb.create_sheet(name)
        ws.append(header)
        for row in data:
            ws.append(row)
            for cell in ws[ws.max_row]:
                if isinstance(cell.value, str):
                    cell.data_type = 's'
            ws.row_dimensions[ws.max_row].height = row_height
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        ws.sheet_view.showGridLines = False
        ws.row_dimensions[1].height = 34
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='233F53')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(vertical='center', wrap_text=True)
        for n, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(n)].width = width
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                if cell.column in score_cols:
                    cell.number_format = '0.000'
                cell.alignment = Alignment(vertical='top', wrap_text=True)
        return ws

    sheet('Сводка', ('Показатель', 'Значение'), summaries, (43, 98))
    sizes = (23, 23, 18, 18, 13, 13, 13, 32, 52, 20, 50, 53,
             34, 34, 24, 24, 24, 24, 47, 47, 65, 65, 26, 24)
    sheet('Неполные дубли', PAIR_HEADER, map(view_row, candidates), sizes, range(3, 8), 70)
    sheet('Существенные различия', PAIR_HEADER, map(view_row, differences), sizes, range(3, 8), 58)
    sheet('Аудит и разметка', PAIR_HEADER, map(view_row, audited), sizes, range(3, 8), 68)
    sheet('Полные совпадения', EXACT_HEADER, exact, (12, 35, 12, 35, 24, 25, 53, 86), (), 46)
    sheet('Проверка порогов', METRIC_HEADER, metrics, (18, 39, 16, 18, 14, 21, 24, 24), (7, 8))
    if lexical:
        sheet('Сетка TF-IDF', ('Порог ответа', 'Порог вопроса', 'Порог покрытия', *METRIC_HEADER[2:]),
              lexical, (16, 16, 17, 16, 18, 14, 21, 24, 24), (1, 2, 3, 8, 9))
    if sweep:
        sheet('Пороги E5', ('Порог E5', *METRIC_HEADER), sweep,
              (14, 18, 39, 16, 18, 14, 21, 24, 24), (1, 8, 9))
    wb.save(path)


def write_html(path, summaries, candidates, differences, exact, metrics):
    def esc(x):
        return escape(str(x), quote=True)

    def pair_table(rows):
        out = ['<table><thead><tr><th>Пара и ответы</th><th>Очередь</th>'
               '<th>Cosine ответа / вопроса</th><th>Покрытие A / B</th><th>Замечания</th></tr></thead><tbody>']
        for r in rows:
            a, b = r['cards']
            same, _, hard, soft = r['flags']
            audit = r['audit']
            ids = esc(a.id + ' · ' + b.id)
            product = esc(' / '.join((a.row['Продукт'] or '—', b.row['Продукт'] or '—')))
            notes = ' · '.join(filter(None, [*hard, *soft, audit.get('comment', '')]))
            e5 = '' if r['embedding'] is None else f'<br>E5 {r["embedding"]:.3f}'
            info = esc(audit.get('decision', ''))
            out.append(
                f'<tr data-scope="{int(same)}" data-queue="{esc(r["queue"])}"><td><small>{product}</small>'
                f'<details><summary>{esc(a.question)}<br>{esc(b.question)}</summary>'
                f'<p><small>{ids}</small></p><p><b>Ответ A</b><br>{esc(a.answer)}</p>'
                f'<p><b>Ответ B</b><br>{esc(b.answer)}</p></details></td>'
                f'<td>{esc(r["queue"])}<br><small>{info}</small></td>'
                f'<td>{r["answer"]:.3f} / {r["question"]:.3f}{e5}</td>'
                f'<td>{r["cover_a"]:.3f} / {r["cover_b"]:.3f}</td><td>{esc(notes)}</td></tr>')
        out.append('</tbody></table>')
        return ''.join(out)

    def simple_table(header, rows, attrs=False):
        return ('<table><thead><tr>' + ''.join(f'<th>{esc(x)}</th>' for x in header) +
                '</tr></thead><tbody>' + ''.join(
                    '<tr' + (' data-scope="" data-queue=""' if attrs else '') + '>' + ''.join(
                        f'<td>{esc(x if x is not None else "—")}</td>' for x in row) + '</tr>' for row in rows) +
                '</tbody></table>')

    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width, initial-scale=1"><title>Неполные дубли FAQ</title>',
            """<style>
:root{color-scheme:light;--ink:#19334a;--muted:#516175;--line:#d9e2ea;--bg:#f4f7fa;--accent:#067385}
*{box-sizing:border-box}body{max-width:1440px;margin:auto;padding:24px;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif}h1{font-size:clamp(28px,5vw,44px);margin:0}h2{margin:.6em 0}
nav,.tools{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
button,input,select{font:inherit;border:1px solid #aabac8;border-radius:7px;padding:8px 12px;background:white;color:var(--ink)}
button{cursor:pointer}button[aria-pressed=true]{background:var(--accent);color:white;border-color:var(--accent)}
input{flex:1;min-width:18ch}small{color:var(--muted)}
table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:9px 12px;
vertical-align:top;border-bottom:1px solid var(--line)}th{background:#e4edf4}td:first-child{min-width:16ch}
summary{cursor:pointer;font-weight:650}details p{white-space:pre-wrap;max-height:28em;overflow:auto}section{overflow:auto}
[hidden]{display:none!important}
</style>""",
            '<h1>Неполные дубли FAQ ЕЛЧ</h1><p>Совпадения и покрытие ответов для редакторской проверки.</p>',
            '<nav><button data-tab="candidates" aria-pressed="true">Кандидаты</button>'
            '<button data-tab="differences" aria-pressed="false">Существенные различия</button>'
            '<button data-tab="exact" aria-pressed="false">Полные совпадения</button>'
            '<button data-tab="quality" aria-pressed="false">Проверка порогов</button></nav>',
            '<div class="tools"><input id="search" aria-label="Поиск" placeholder="Поиск по вопросу, ответу, продукту или ID">'
            '<select id="scope" aria-label="Продукт"><option value="">Все продукты</option>'
            '<option value="1">Внутри продукта</option><option value="0">Между продуктами или продукт пуст</option></select>'
            '<select id="queue" aria-label="Очередь"><option value="">Вся очередь</option>'
            '<option>приоритетная вычитка</option><option>сверка контекста</option></select></div>',
            '<p id="shown" aria-live="polite"></p>',
            '<section id="candidates">' + pair_table(candidates) + '</section>',
            '<section id="differences" hidden>' + pair_table(differences) + '</section>',
            '<section id="exact" hidden>' + simple_table(EXACT_HEADER, exact, True) + '</section>',
            '<section id="quality" hidden><h2>Сводка запуска</h2>' + simple_table(('Показатель','Значение'), summaries)
            + '<h2>Вычитанная выборка</h2>' + simple_table(METRIC_HEADER,
                [(*r[:6], *(None if x is None else f'{x:.1%}' for x in r[6:])) for r in metrics]) + '</section>',
            """<script>
const byId = id => document.getElementById(id);
let tab = 'candidates';
function refresh() {
  byId('shown').hidden = tab === 'quality';
  if (tab === 'quality') return;
  const q = byId('search').value.toLocaleLowerCase('ru').trim();
  const scope = byId('scope').value, queue = byId('queue').value;
  let n = 0;
  for (const row of byId(tab).querySelectorAll('tbody tr')) {
    row.hidden = (q && !row.textContent.toLocaleLowerCase('ru').includes(q))
      || (tab !== 'exact' && scope && row.dataset.scope !== scope)
      || (tab === 'candidates' && queue && row.dataset.queue !== queue);
    if (!row.hidden) n++;
  }
  byId('shown').textContent = 'Показано строк: ' + n;
}
for (const button of document.querySelectorAll('[data-tab]')) {
  button.addEventListener('click', () => {
    byId(tab).hidden = true;
    tab = button.dataset.tab;
    byId(tab).hidden = false;
    for (const item of document.querySelectorAll('[data-tab]')) {
      item.setAttribute('aria-pressed', String(item === button));
    }
    byId('search').disabled = tab === 'quality';
    byId('scope').disabled = tab === 'quality' || tab === 'exact';
    byId('queue').disabled = tab !== 'candidates';
    refresh();
  });
}
for (const id of ['search', 'scope', 'queue']) byId(id).addEventListener('input', refresh);
refresh();
</script></html>"""]
    Path(path).write_text(''.join(body), encoding='utf-8')


def export_review(output_dir, corpus, found, benchmark_features, metrics, lexical, sweep, thresholds, pool_size, stale):
    candidates = [r for r in found if r['queue'] in REVIEW]
    differences = [r for r in found if r['queue'] == 'существенное различие']
    counts = Counter(r['queue'] for r in candidates)
    exact = exact_report(corpus)
    audited = list(benchmark_features.values())
    audited.sort(key=lambda r: (r['audit']['source'] != 'прежний HTML', int(r['audit']['number'])))
    summaries = [
        ('Активных FAQ ЕЛЧ', len(corpus.cards)),
        ('Пар в пуле соседей', pool_size),
        ('Приоритетная вычитка', counts['приоритетная вычитка']),
        ('Сверка контекста', counts['сверка контекста']),
        ('Кандидатов между продуктами / с пустым продуктом', sum(not r['flags'][0] for r in candidates)),
        ('Существенное различие', len(differences)),
        ('Сохранённых вердиктов ГигаЧата', sum(bool(r['audit'].get('old_verdict')) for r in audited)),
        ('Групп полных совпадений', len({row[1] for row in exact})),
        ('Устаревших пар разметки', stale),
        ('Cosine ответа / вопроса', f'{thresholds.answer:.2f} / {thresholds.question:.2f}'),
        ('Покрытие / порог по вопросу', f'{thresholds.coverage:.2f} / {thresholds.containment_question:.2f}'),
        ('Высокий cosine ответа', thresholds.strong_answer),
        ('Cosine ответа при общей формулировке вопроса', thresholds.same_question_answer),
        ('Порог cosine E5', thresholds.embedding if thresholds.embedding is not None else 'не используется'),
        ('Покрытие A в B', 'Доля совпадающих взвешенных слов A относительно всех слов A. Это признак, направление проверяется по тексту.'),
        ('Метрики', 'Вычитанные выборки для аудита, без общих ID между разделами. Контрольный аудит отобран по очередям. Метрики по всей базе неизвестны.'),
        ('Полные совпадения', 'Нормализация пробелов, регистра, ё/е и оформления кавычек. Условие и вопрос проверяются отдельно.'),
        ('Действие', 'Кандидаты для проверки. Автоматических объединений и изменений в FAQ нет.'),
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as temp:
        temp = Path(temp)
        report, book = temp/'faq_partial_report.html', temp/'faq_partial_review.xlsx'
        write_html(report, summaries, candidates, differences, exact, metrics)
        write_workbook(book, summaries, candidates, differences, audited, exact, metrics, lexical, sweep)
        report.replace(output_dir/report.name)
        book.replace(output_dir/book.name)
    return summaries
