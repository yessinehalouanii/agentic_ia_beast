import pandas as pd

def _is_dict_like(x) -> bool:
    return isinstance(x, dict)

def _is_list_of_dicts(x) -> bool:
    return isinstance(x, list) and any(isinstance(i, dict) for i in (x or []))

def _iter_dotted_paths(value, base="", max_depth=3):
    if max_depth <= 0 or value is None: return
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{base}.{k}" if base else k
            yield path
            yield from _iter_dotted_paths(v, path, max_depth - 1)
    elif isinstance(value, list):
        first = next((i for i in value if isinstance(i, (dict, list))), None)
        if first is not None:
            yield from _iter_dotted_paths(first, base, max_depth - 1)

def detect_nested_schema(df: pd.DataFrame, sample_rows: int = 200) -> list[str]:
    if df is None or df.empty: return []
    sample = df.head(sample_rows)
    dotted = set()
    for col in df.columns:
        for v in sample[col]:
            if _is_dict_like(v) or _is_list_of_dicts(v):
                for path in _iter_dotted_paths(v, base="", max_depth=3):
                    dotted.add(f"{col}.{path}" if not path.startswith(f"{col}.") else path)
                break
    return sorted(dotted)

def _read_nested(obj, path_tokens: list[str]):
    cur = obj
    for tok in path_tokens:
        if cur is None: return None
        if isinstance(cur, list):
            cur = next((i for i in cur if isinstance(i, (dict, list))), None)
            if cur is None: return None
        if isinstance(cur, dict):
            cur = cur.get(tok, None)
        else:
            return None
    return cur

def get_or_make_dotted_series(df: pd.DataFrame, dotted: str) -> str | None:
    if not dotted or "." not in dotted: return None
    parts = dotted.split(".")
    head, tail = parts[0], parts[1:]
    if head not in df.columns: return None
    new_col = "__nested__" + "__".join(parts)
    if new_col in df.columns: return new_col
    try:
        df[new_col] = df[head].apply(lambda x: _read_nested(x, tail))
        return new_col
    except Exception:
        return None
