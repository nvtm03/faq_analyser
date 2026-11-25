"""Refresh the FAQ HTML report against two Excel exports and a prior HTML report.

Requires openpyxl and lxml. No network or model calls.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from lxml import etree, html
from openpyxl import load_workbook


KINDS = {
    1: 'Дословно одинаковые ответы',
    2: 'Кандидаты на вложенность',
    3: 'Похожие ответы',
    4: 'Разные продукты или программы',
    5: 'Классификация не подтверждена',
    6: 'Различия только в пробелах',
}
CONTEXT_FIELDS = ('Продукт', 'Подпродукт', 'Тематика', 'Основной канал',
                  'Дополнительный канал', 'Дата, время включения', 'Дата, время отключения')
MAIN, EXTRA = 'Основной канал', 'Дополнительный канал'
LEGACY = 'Перенесено из предыдущего отчёта'
SAME, DIFFERENT, UNKNOWN = 'same', 'different', 'unknown'
INSURANCE_PROGRAM = re.compile(r'\bзащит[аеуы]\s+(дома|денег)\b', re.I)
DEPOSIT_CLOSE = re.compile(r'^как\s+(?:досрочно\s+)?закрыть\s+вклад\s+[«"]([^»"]+)[»"]\s*\??$', re.I)
NUMBERS = re.compile(r'\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|\d+(?:[.,:/-]\d+)*')
NEGATION = re.compile(r'\b(?:не|нет|нельзя|без)\b', re.I)


def clean(value):
    return '' if value is None else str(value)


def norm(value):
    return ' '.join(clean(value).split())


def numbers(answer):
    return {re.sub(r'\s', '', token).replace(',', '.')
            for token in NUMBERS.findall(answer)}


def differences(a, b):
    links = {field for field in a.row.keys() | b.row.keys() if field and field.startswith('Ссылка')}
    fields = [field for field in (*CONTEXT_FIELDS, *sorted(links))
              if norm(a.row.get(field)) != norm(b.row.get(field))]
    if numbers(a.answer) != numbers(b.answer):
        fields.append('Числа в тексте')
    if Counter(NEGATION.findall(a.answer.casefold())) != Counter(NEGATION.findall(b.answer.casefold())):
        fields.append('Отрицания в тексте')
    return fields


@dataclass(frozen=True)
class FAQ:
    id: str
    question: str
    answer: str
    alternatives: tuple
    source: str
    row: dict

    @property
    def product_key(self):
        return tuple(norm(self.row.get(field)).casefold() for field in CONTEXT_FIELDS[:2])


def context(a, b):
    ap, asub = a.product_key
    bp, bsub = b.product_key
    if ap and bp and ap != bp:
        return DIFFERENT, 'Разные продукты'
    if not ap or not bp or not asub or not bsub:
        return UNKNOWN, 'У одной или обеих карточек не заполнен продукт или подпродукт'
    if asub != bsub:
        return DIFFERENT, 'Разные подпродукты'
    program = None
    if (ap, asub) == ('страхование', 'злс'):
        program = INSURANCE_PROGRAM
    elif ap == 'вклады и счета':
        program = DEPOSIT_CLOSE
    if program:
        left, right = (program.search(norm(card.question)) for card in (a, b))
        if left and right and norm(left[1]).casefold() != norm(right[1]).casefold():
            return DIFFERENT, 'Разные программы внутри подпродукта'
    return SAME, 'Совпадают продукт и подпродукт'


def read_records(paths):
    records, seen = {}, set()
    counts = Counter()
    required = {'Идентификатор', 'Вопрос', 'Ответ', 'Продукт', 'Подпродукт', 'Статус', MAIN, EXTRA}
    for path in paths:
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = iter(book['Выгрузка'].values)
            fields = next(rows)
            if missing := required.difference(fields):
                raise ValueError(f'{path.name}: отсутствуют поля {sorted(missing)}')
            alt_fields = [f for f in fields if f and f.startswith('Альтернативный вопрос')]
            for line, values in enumerate(rows, 2):
                if all(value is None for value in values):
                    continue
                counts['Строк в выгрузках'] += 1
                row = dict(zip(fields, values))
                ident = norm(row['Идентификатор'])
                if not ident or ident in seen:
                    raise ValueError(f'{path.name}:{line}: пустой или повторяющийся ID FAQ')
                seen.add(ident)
                main, extra = ('ЕЛЧ' in {norm(x) for x in clean(row[key]).split(',')}
                               for key in (MAIN, EXTRA))
                if norm(row['Статус']) != 'Активно' or not (main or extra):
                    continue
                if not norm(row['Вопрос']) or not norm(row['Ответ']):
                    raise ValueError(f'{path.name}:{line}: пустой вопрос или ответ')
                source = 'main' if main else 'additional'
                counts[source] += 1
                records[ident] = FAQ(ident, clean(row['Вопрос']), clean(row['Ответ']),
                                     tuple(clean(row[f]) for f in alt_fields if norm(row[f])),
                                     source, row)
        finally:
            book.close()
    return records, counts


@dataclass
class Member:
    faq: FAQ
    verdict: str = ''


@dataclass
class Group:
    kind: int
    anchor: FAQ
    members: list
    previous_id: str = ''


def exact_groups(records, whitespace_only=False):
    indexed = defaultdict(list)
    for faq in records.values():
        key = norm(faq.answer) if whitespace_only else faq.answer
        indexed[key].append(faq)
    groups = []
    for matches in indexed.values():
        if len(matches) < 2:
            continue
        if whitespace_only and len({faq.answer for faq in matches}) < 2:
            continue
        # A display reference with the most question variants. It has no editorial status.
        ranked = sorted(matches, key=lambda faq: (-len(faq.alternatives), faq.id))
        groups.append(Group(6 if whitespace_only else 1, ranked[0], [Member(faq) for faq in ranked[1:]]))
    return sorted(groups, key=lambda group: (-len(group.members), group.anchor.id))


def same_snapshot(card, faq):
    return (norm(card.xpath('string(./p[@class="q"])')) == norm(faq.question)
            and norm(card.xpath('string(./p[@class="a"])')) == norm(faq.answer))


def legacy_groups(doc, records, stats):
    groups, seen = [], set()
    for section in doc.xpath('//section[@class="group"]'):
        kind = int(section.get('data-kind'))
        stats[f'Групп в старом HTML, тип {kind}'] += 1
        if kind not in (2, 3, 4):
            continue
        cards = section.xpath('./article')
        references = [card for card in cards if card.get('data-id') == card.get('data-main')]
        if len(references) != 1:
            raise ValueError('В группе старого HTML ожидается одна опорная карточка')
        reference = references[0]
        anchor = records.get(reference.get('data-id'))
        if not anchor or not same_snapshot(reference, anchor):
            stats['Сравнений без актуальной опорной карточки'] += len(cards) - 1
            continue
        partitions = defaultdict(list)
        for card in cards:
            if card is reference:
                continue
            if card.get('data-main') != anchor.id:
                raise ValueError('Ссылка на опорную карточку разошлась с группой')
            faq = records.get(card.get('data-id'))
            if not faq or not same_snapshot(card, faq):
                stats['Сравнений без актуальной второй карточки'] += 1
                continue
            if norm(faq.answer) == norm(anchor.answer):
                stats['Сравнений учтено среди текстовых совпадений'] += 1
                continue
            relation, _ = context(anchor, faq)
            if relation == SAME and kind == 4:
                stats['Сравнений из раздела разных продуктов без подтверждённого различия'] += 1
                continue
            target = {DIFFERENT: 4, UNKNOWN: 5}.get(relation, kind)
            key = target, *sorted((anchor.id, faq.id))
            if key in seen:
                stats['Повторных сравнений в старом HTML'] += 1
                continue
            seen.add(key)
            verdict = norm(card.xpath('string(./div[starts-with(@class, "llm ")])'))
            partitions[target].append(Member(faq, verdict))
        for target, members in partitions.items():
            groups.append(Group(target, anchor, members, reference.get('data-group')))
    return groups


def add(parent, tag, text='', **attrs):
    node = etree.SubElement(parent, tag, attrs)
    node.text = text
    return node


def exact_reason(a, b):
    q_a, q_b = norm(a.question), norm(b.question)
    if q_a == q_b:
        return 'Совпадают основной вопрос и ответ'
    matches = []
    if q_a in {norm(q) for q in b.alternatives}:
        matches.append('вопрос опорной есть в альтернативных этой карточки')
    if q_b in {norm(q) for q in a.alternatives}:
        matches.append('вопрос этой карточки есть в альтернативных опорной')
    return 'Совпадает ответ. ' + ('; '.join(matches) if matches else 'Основные вопросы различаются')


def render_card(section, group_id, group, member):
    faq, anchor = member.faq, group.anchor
    reference = faq.id == anchor.id
    relation, context_label = context(anchor, faq)
    recommendation = 'опорная' if reference else 'проверить'
    if group.kind == 4 or (group.kind in (1, 6) and relation == DIFFERENT):
        recommendation = 'оставить'
    card = add(section, 'article', **{
        'class': f'card rec-{recommendation}', 'data-group': str(group_id),
        'data-kind': str(group.kind), 'data-kind-name': KINDS[group.kind],
        'data-rec': recommendation, 'data-main': anchor.id, 'data-id': faq.id,
        'data-src': faq.source, 'data-prod': norm(faq.row.get('Продукт')),
        'data-sub': norm(faq.row.get('Подпродукт')),
        'data-text': norm(' '.join((faq.id, faq.question, *faq.alternatives, faq.answer))).lower(),
    })
    header = add(card, 'header')
    add(header, 'span', ' · '.join(norm(faq.row.get(f)) or '—' for f in CONTEXT_FIELDS[:2]),
        **{'class': 'meta'})
    add(header, 'span', faq.source, **{'class': 'meta'})
    add(header, 'code', faq.id)
    add(card, 'p', faq.question, **{'class': 'q'})
    alts = ' | '.join(faq.alternatives)
    add(card, 'p', 'Альтернативные вопросы: ' + (alts or '—'),
        **{'class': 'alts', 'data-raw': alts})
    add(card, 'p', faq.answer, **{'class': 'a'})
    details = add(card, 'div', **{'class': 'why'})
    if reference:
        reason = 'Опорная карточка для сравнения'
    elif group.kind in (1, 6):
        reason = f'{exact_reason(anchor, faq)}. {context_label}.'
        if anchor.answer != faq.answer:
            reason = reason.replace('ответ', 'ответ после выравнивания пробелов', 1)
        if relation == DIFFERENT:
            reason += ' Сохранить обе карточки.'
    elif group.kind == 4:
        reason = f'{context_label}. Оставить обе карточки.'
    elif group.kind == 5:
        reason = f'{context_label}. Уточнить классификацию перед решением.'
    else:
        reason = f'{LEGACY}. Сверить, отвечает ли каждая карточка на вопрос другой.'
    add(details, 'p', reason)
    if not reference:
        flags = differences(faq, anchor)
        if flags:
            add(details, 'p', 'Различия: ' + ', '.join(flags))
        left, right = numbers(faq.answer), numbers(anchor.answer)
        if left != right:
            marker = 'check' if relation == SAME else 'info'
            action = {
                SAME: 'Сверить числовые значения и условия.',
                DIFFERENT: 'Карточки относятся к разным продуктам или программам.',
                UNKNOWN: 'Сначала уточнить классификацию.',
            }[relation]
            add(card, 'div',
                'Числовые значения различаются. '
                f'Только здесь: {", ".join(sorted(left - right)) or "—"}. '
                f'Только в опорной: {", ".join(sorted(right - left)) or "—"}. {action}',
                **{'class': f'nmark {marker}'})
    if member.verdict and group.kind in (2, 3):
        add(card, 'div', 'Исторический вердикт · ' + member.verdict, **{'class': 'llm'})
    return card


def regenerate(doc, groups, counts, stats):
    by_kind = Counter(g.kind for g in groups)
    unique = len({card.id for g in groups for card in (g.anchor, *(m.faq for m in g.members))})
    appearances = sum(len(g.members) + 1 for g in groups)
    main = doc.xpath('//main')[0]
    for section in main.xpath('./section[@class="group"]'):
        main.remove(section)
    lead = main.xpath('./p[@class="lead"]')[0]
    lead.clear()
    lead.set('class', 'lead')
    lead.text = (f'{counts["main"] + counts["additional"]} активных FAQ ЕЛЧ '
                 f'({counts["main"]} основной канал, {counts["additional"]} дополнительный). '
                 f'В отчёте {len(groups)} групп и {unique} уникальных FAQ. '
                 'Вложенность и похожесть перенесены из прежнего отчёта для карточек, '
                 'чьи вопросы и ответы сверены с текущими выгрузками.')
    main.xpath('./h1')[0].text = 'Проверка совпадений в FAQ ЕЛЧ'
    main.xpath('./h2')[0].text = 'Карточки для сравнения'
    doc.xpath('//title')[0].text = 'Проверка совпадений в FAQ ЕЛЧ'
    main.remove(main.xpath('./div[@class="stats"]')[0])
    tiles = main.xpath('./div[@id="tiles"]')[0]
    tiles[:] = []
    all_tile = add(tiles, 'div', **{'class': 'tile active', 'data-kind': ''})
    add(all_tile, 'b', str(len(groups)))
    add(all_tile, 'span', 'Все группы')
    for kind, label in KINDS.items():
        box = add(tiles, 'div', **{'class': 'tile', 'data-kind': str(kind)})
        add(box, 'b', str(by_kind[kind]))
        add(box, 'span', label)
    note = main.xpath('./div[@class="note"]')[0]
    note[:] = []
    add(note, 'p', 'Опорная карточка служит для сравнения. Решение об объединении принимает редактор.')
    add(note, 'p', 'Группы с различиями в пробелах могут включать дословно одинаковые карточки.')
    add(note, 'p',
        'Разные продукты, подпродукты и явно различимые программы сохраняются. '
        'При незаполненной классификации требуется уточнение. '
        'Числовые различия в сопоставимых карточках выделены красным.')
    add(note, 'p',
        'Исторические вердикты ГигаЧата приложены для вычитки. '
        'У похожих текстов могут различаться операции, порядок действий и условия. '
        'Числовая подсветка сравнивает текстовые значения и может включать нумерацию.')
    rec_filter = main.xpath('.//select[@id="rec-filter"]')[0]
    rec_filter[:] = []
    for value, label in (('', 'Все карточки'), ('проверить', 'Проверить'),
                         ('оставить', 'Оставить'), ('опорная', 'Опорная')):
        add(rec_filter, 'option', label, value=value)
    product_filter = main.xpath('.//select[@id="prod-filter"]')[0]
    product_filter[:] = []
    add(product_filter, 'option', 'Все продукты', value='')
    products = sorted({norm(card.row.get('Продукт')) for group in groups
                       for card in (group.anchor, *(member.faq for member in group.members))})
    for product in filter(None, products):
        add(product_filter, 'option', product, value=product)
    main.xpath('.//*[@id="shown"]')[0].text = str(appearances)
    main.xpath('.//*[@id="shown-groups"]')[0].text = str(len(groups))
    script = main.xpath('./script|../script')[0]
    for group_id, group in enumerate(sorted(groups, key=lambda g: g.kind), 1):
        section = etree.Element('section', {'class': 'group', 'data-kind': str(group.kind),
                                            'data-previous-group': group.previous_id})
        main.insert(len(main) - (script.getparent() is main), section)
        heading = add(section, 'h3')
        add(heading, 'span', KINDS[group.kind], **{'class': f'kbadge k{group.kind}'}).tail = (
            f' Группа {group_id} · {len(group.members) + 1} карточек')
        render_card(section, group_id, group, Member(group.anchor))
        for member in group.members:
            render_card(section, group_id, group, member)

    # Keep the original filters and expandable text. Update CSV headers and treat input as text.
    js = script.text.replace("'Основная карточка'", "'Опорная карточка'")
    js = js.replace(",'Каналы для отписки'", "")
    js = js.replace("    const metas=[...c.querySelectorAll('header .meta')].map(x=>x.textContent.trim());\n", "")
    js = js.replace("const prod=(metas[0]||'').split(' · ');", "")
    js = js.replace("prod[0]||'',prod[1]||''", "c.dataset.prod,c.dataset.sub")
    js = js.replace("why?(why.querySelector('p')||{textContent:''}).textContent.trim():'',\n"
                    "      (why&&why.querySelector('b'))?why.querySelector('b').textContent:''",
                    "why?why.textContent.trim():''")
    js = js.replace("""String(v).replace(/"/g,'""')""",
                    """String(v).replace(/^[\\s]*[=+@-]/, match => "'" + match).replace(/"/g,'""')""")
    js = js.replace("tiles.forEach(x=>x.classList.remove('active'));apply();",
                    "tiles.forEach(x=>x.classList.toggle('active',!x.dataset.kind));apply();")
    js = js.replace("x.dataset.kind===kind", "x.dataset.kind===(kind||'')")
    js = js.replace(
        "    g.classList.toggle('hide',!vis);",
        "    g.classList.toggle('hide',!vis);\n"
        "    if(vis) g.querySelector('.card').classList.remove('hide');",
    )
    script.text = js
    stats.update({f'Групп в новом HTML, тип {k}': by_kind[k] for k in KINDS})
    stats.update({'Уникальных FAQ в отчёте': unique, 'Показов карточек в отчёте': appearances,
                  'Дословных пар': sum(len(g.members) * (len(g.members) + 1) // 2
                                       for g in groups if g.kind == 1),
                  'Пар с перенесённым вердиктом ГигаЧата': sum(
                      bool(m.verdict) for g in groups if g.kind in (2, 3) for m in g.members)})


def verify(markup, records):
    doc = html.fromstring(markup)
    declared = {int(box.get('data-kind')): int(box.xpath('string(./b)'))
                for box in doc.xpath('//*[@id="tiles"]/div[@data-kind!=""]')}
    sections = doc.xpath('//section[@class="group"]')
    observed = Counter()
    for section in sections:
        kind = int(section.get('data-kind'))
        observed[kind] += 1
        cards = section.xpath('./article')
        anchor_id = cards[0].get('data-id')
        assert len(cards) >= 2
        assert len({card.get('data-id') for card in cards}) == len(cards)
        if kind == 6:
            assert len({records[c.get('data-id')].answer for c in cards}) > 1
        for card in cards[1:]:
            assert card.get('data-main') == anchor_id
            a, b = records[anchor_id], records[card.get('data-id')]
            relation, _ = context(a, b)
            if kind == 1:
                assert a.answer == b.answer
                assert cards[0].xpath('string(./p[@class="a"])') == card.xpath('string(./p[@class="a"])')
            if kind == 6:
                assert norm(a.answer) == norm(b.answer)
            if kind in (2, 3):
                assert relation == SAME
            if kind == 4:
                assert relation == DIFFERENT and card.get('data-rec') == 'оставить'
            if kind == 5:
                assert relation == UNKNOWN
            if card.xpath('./div[@class="nmark check"]'):
                assert relation == SAME and card.get('data-rec') == 'проверить'
    assert observed == Counter(declared)
    assert int(doc.xpath('string(//*[@id="shown-groups"])')) == len(sections)
    assert int(doc.xpath('string(//*[@id="shown"])')) == len(doc.xpath('//section/article'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', nargs=2, type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in [*args.exports, args.previous]}:
        parser.error('Укажите отдельный путь для обновлённого отчёта.')

    records, counts = read_records(args.exports)
    stats = Counter(counts)
    doc = html.fromstring(args.previous.read_text(encoding='utf-8-sig'))
    groups = (exact_groups(records) + exact_groups(records, whitespace_only=True)
              + legacy_groups(doc, records, stats))
    regenerate(doc, groups, counts, stats)
    markup = html.tostring(doc, encoding='unicode', doctype='<!doctype html>')
    verify(markup, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=args.output.parent, delete=False) as out:
        temp_path = Path(out.name)
    try:
        temp_path.write_text(markup, encoding='utf-8')
        temp_path.replace(args.output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
