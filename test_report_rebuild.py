from collections import Counter
from unittest import TestCase

from lxml import etree, html
from faq_report_rebuild import DIFFERENT, SAME, UNKNOWN, FAQ, context, exact_groups, legacy_groups


def faq(ident, product, subproduct, question='Вопрос', answer='Ответ'):
    return FAQ(ident, question, answer, (), 'main', {'Продукт': product, 'Подпродукт': subproduct})


def old_group(anchor, member, kind):
    section = html.fromstring(f'<main><section class="group" data-kind="{kind}"></section></main>')[0]
    for item in (anchor, member):
        card = html.Element('article', {'data-id': item.id, 'data-main': anchor.id, 'data-group': '1'})
        etree.SubElement(card, 'p', {'class': 'q'}).text = item.question
        etree.SubElement(card, 'p', {'class': 'a'}).text = item.answer
        section.append(card)
    return section.getparent()


class ReportRebuildTests(TestCase):
    def test_known_insurance_programs_in_same_subproduct_distinct(self):
        a = faq('a', 'Страхование', 'ЗЛС', 'Срок по Защите дома', '12 рабочих дней')
        b = faq('b', 'Страхование', 'ЗЛС', 'Срок по Защите денег', '24 рабочих дня')
        self.assertEqual(context(a, b)[0], DIFFERENT)
        groups = legacy_groups(old_group(a, b, 3), {'a': a, 'b': b}, Counter())
        self.assertEqual([g.kind for g in groups], [4])

    def test_missing_subproduct_does_not_join_semantic_group(self):
        a = faq('a', 'Карты', 'Кредитная', answer='Первый ответ')
        b = faq('b', 'Карты', '', answer='Второй ответ')
        self.assertEqual(context(a, b)[0], UNKNOWN)
        self.assertEqual(context(b, b)[0], UNKNOWN)
        groups = legacy_groups(old_group(a, b, 2), {'a': a, 'b': b}, Counter())
        self.assertEqual([g.kind for g in groups], [5])

    def test_both_product_levels_checked(self):
        a = faq('a', 'Карты', 'Кредитная')
        self.assertEqual(context(a, faq('b', ' Карты ', 'Кредитная'))[0], SAME)
        self.assertEqual(context(a, faq('b', 'Карты', 'Дебетовая'))[0], DIFFERENT)
        self.assertEqual(context(a, faq('b', 'Кредиты', 'Кредитная'))[0], DIFFERENT)

    def test_named_deposit_template(self):
        a = faq('a', 'Вклады и счета', 'Закрытие', 'Как досрочно закрыть вклад «А»?')
        b = faq('b', 'Вклады и счета', 'Закрытие', 'Как досрочно закрыть вклад «Б»?')
        self.assertEqual(context(a, b)[0], DIFFERENT)

    def test_stale_candidate_and_its_verdict_are_dropped(self):
        a = faq('a', 'Карты', 'Кредитная', answer='Первый ответ')
        b = faq('b', 'Карты', 'Кредитная', answer='Второй ответ')
        changed = faq('b', 'Карты', 'Кредитная', answer='Новый ответ')
        stats = Counter()
        self.assertFalse(legacy_groups(old_group(a, b, 3), {'a': a, 'b': changed}, stats))
        self.assertEqual(stats['Сравнений без актуальной второй карточки'], 1)

    def test_exact_group_has_equal_raw_answers_across_products(self):
        records = {a.id: a for a in (
            faq('a', 'Карты', 'Кредитная', answer='Один ответ'),
            faq('b', 'Кредиты', 'Потребительский', answer='Один ответ'),
            faq('c', 'Карты', 'Дебетовая', answer='Один  ответ'),
            faq('d', 'Карты', 'Дебетовая', answer='один ответ'),
        )}
        groups = exact_groups(records)
        self.assertEqual(len(groups), 1)
        self.assertEqual((groups[0].anchor.id, groups[0].members[0].faq.id), ('a', 'b'))
        whitespace = exact_groups(records, whitespace_only=True)
        self.assertEqual(len(whitespace), 1)
        self.assertEqual(whitespace[0].kind, 6)
        self.assertEqual([m.faq.id for m in whitespace[0].members], ['b', 'c'])
