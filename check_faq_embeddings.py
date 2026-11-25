"""Check and optionally score existing FAQ answer vectors. No embedding model is used."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID = 'Идентификатор'
ANSWER = 'Ответ'


def indexed(frame, label):
    result = frame[[ID, ANSWER]].copy().fillna('')
    result[ID] = result[ID].astype(str).str.strip()
    result[ANSWER] = result[ANSWER].astype(str).str.strip()
    if result[ID].eq('').any() or result[ID].duplicated().any():
        raise ValueError(f'{label}: пустой или неуникальный ID')
    return result.set_index(ID)[ANSWER]


def read_vectors(path, row_count):
    try:
        vectors = np.load(path, mmap_mode='r', allow_pickle=False)
    except (ValueError, OSError) as exc:
        raise ValueError(f'{path.name}: NPY не читается целиком ({exc})') from exc
    if vectors.shape != (row_count, 1024) or vectors.dtype.kind != 'f':
        raise ValueError(f'Ожидалось {row_count} строк по 1024 числа. Получено {vectors.shape}.')
    norms = np.empty(row_count, dtype=np.float64)
    for start in range(0, row_count, 512):
        block = vectors[start:start + 512]
        norm = np.linalg.norm(block.astype(np.float64), axis=1)
        if not np.isfinite(norm).all() or (norm == 0).any():
            raise ValueError(f'Некорректные векторы в диапазоне от строки {start + 1}.')
        norms[start:start + len(block)] = norm
    return vectors, norms


def score_pairs(input_path, output_path, ready, valid_ids, vectors, norms):
    picked = pd.read_excel(input_path, sheet_name='Кандидаты', dtype=str, keep_default_na=False)
    pairs = picked.loc[picked['Тип'].isin(['Кандидаты на вложенность', 'Похожие ответы']),
                       ['Группа HTML', 'Тип', 'ID A', 'ID B']].copy()
    rows_by_id = {ident: i for i, ident in enumerate(ready.index) if ident in valid_ids}
    row_a = pairs['ID A'].map(rows_by_id)
    row_b = pairs['ID B'].map(rows_by_id)
    verified = row_a.notna() & row_b.notna()
    i, j = row_a[verified].astype(int).to_numpy(), row_b[verified].astype(int).to_numpy()
    pairs['Статус'] = 'Нет подтверждённой пары векторов'
    pairs.loc[verified, 'Статус'] = 'Cosine рассчитан'
    pairs['Cosine'] = np.nan
    pairs.loc[verified, 'Cosine'] = np.clip(
        np.einsum('ij,ij->i', vectors[i].astype(np.float64), vectors[j]) / (norms[i] * norms[j]),
        -1, 1)
    if output_path.exists():
        raise FileExistsError(f'{output_path} уже существует. Укажите другое имя.')
    with output_path.open('x', encoding='utf-8-sig', newline='') as out:
        pairs.to_csv(out, sep=';', index=False)
    return int(verified.sum()), len(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True,
                        help='Доверенный локальный pickle из того же запуска или CSV с ID и исходным ответом в порядке векторов.')
    parser.add_argument('--exports', nargs=2, type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--score-csv', type=Path)
    args = parser.parse_args()
    if bool(args.review) != bool(args.score_csv):
        parser.error('--review и --score-csv нужны вместе.')
    if args.snapshot.suffix.lower() == '.csv':
        frame = pd.read_csv(args.snapshot, dtype=str, keep_default_na=False)
    elif args.snapshot.suffix.lower() == '.pkl':
        # Only the snapshot saved by the original local encoding job is supported.
        frame = pd.read_pickle(args.snapshot)
    else:
        parser.error('Поддерживаются .pkl и .csv.')
    ready = indexed(frame, 'Снимок')
    if ready.empty or ready.eq('').any():
        raise ValueError('Снимок пуст или содержит пустой ответ.')
    current = indexed(pd.concat([
        pd.read_excel(path, sheet_name='Выгрузка', dtype=str, keep_default_na=False)
        for path in args.exports
    ], ignore_index=True), 'Текущие выгрузки')
    vectors, norms = read_vectors(args.embeddings, len(ready))
    shared = current.index.intersection(ready.index)
    unchanged = current.loc[shared].eq(ready.loc[shared])
    valid_ids = unchanged.index[unchanged]
    result = {
        'Строк в NPY и снимке': len(ready),
        'Строк в текущих выгрузках': len(current),
        'Совпадают ID и ответ': len(valid_ids),
        'Ответ поменялся у того же ID': int((~unchanged).sum()),
        'ID из выгрузок отсутствуют в снимке': len(current) - len(shared),
        'ID из снимка отсутствуют в выгрузках': len(ready) - len(shared),
    }
    if args.review:
        scored, total = score_pairs(args.review, args.score_csv, ready, valid_ids, vectors, norms)
        result['Проверено cosine пар из Excel'] = scored
        result['Всего выбранных пар из Excel'] = total
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
