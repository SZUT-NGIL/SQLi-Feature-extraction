from __future__ import annotations

import re
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from .hdcan import advanced_preprocess, smart_recursive_decode


warnings.filterwarnings("ignore")


NUMERIC_FEATURE_COLUMNS = [
    "qlen",
    "wcount",
    "sq",
    "dq",
    "puncts",
    "comments",
    "spaces",
    "logic",
    "arith",
    "alpha",
    "sqlkw",
    "sqlfunc",
]

SQL_KEYWORDS = {
    "alter", "analyze", "begin", "call", "commit", "create", "delete", "drop", "execute",
    "explain", "fetch", "grant", "handler", "insert", "into", "join", "lock", "optimize",
    "prepare", "repair", "replace", "rollback", "savepoint", "select", "set", "start",
    "truncate", "union", "update", "values",
    "all", "distinct", "exists", "fetch", "from", "group", "having", "in", "index",
    "inner", "left", "limit", "offset", "on", "order", "outer", "right", "top",
    "view", "where", "with", "recursive",
    "case", "when", "then", "else", "end", "if", "between", "like", "regexp", "rlike",
    "and", "or", "xor", "not", "is", "as", "asc", "desc",
    "null", "true", "false", "binary",
    "by",
}

# 保持与原始实现完全一致；其中 convert/isnull 在旧代码中因缺少逗号被拼成了一个 token。
SQL_FUNCTIONS = {
    "ascii", "char", "char_length", "charset", "concat", "concat_ws", "conv", "crc32",
    "elt", "hex", "instr", "json_quote", "lcase", "left", "length", "load_file",
    "locate", "lower", "lpad", "ltrim", "mid", "oct", "ord", "position", "quote",
    "repeat", "replace", "regexp_instr", "regexp_like", "regexp_replace",
    "regexp_substr", "right", "rpad", "rtrim", "substr", "substring",
    "substring_index", "trim", "ucase", "upper", "weight_string",
    "acos", "avg", "benchmark", "ceil", "ceiling", "conv", "count", "exp", "floor",
    "greatest", "least", "log", "log10", "log2", "pow", "power", "rand", "round",
    "sin", "sqrt", "sum",
    "json_array", "json_contains", "json_contains_path", "json_depth", "json_extract",
    "json_keys", "json_merge", "json_object", "json_objectagg", "json_quote",
    "json_storage_size", "json_type", "json_valid", "json_value",
    "extractvalue", "updatexml",
    "geometrycollectionfromtext", "mbrcontains", "point", "polygonfromtext",
    "st_area", "st_astext", "st_distance_sphere", "st_geomfromtext",
    "database", "current_user", "user", "version", "@@hostname", "@@version",
    "benchmark", "sleep", "delay", "waitfor",
    "aes_encrypt", "crc32", "encode", "md5", "sha1", "sha2", "uncompress",
    "get_format", "get_lock", "row_count", "savepoint",
    "execute_immediate", "prepare", "deallocate_prepare",
    "coercibility", "collation", "convert_tz", "decode", "encode", "if", "ifnull",
    "convertisnull", "nullif", "raise_error",
    "regexp_substring", "chr", "repeat", "crypt_key", "dbms_utility", "sqlid_to_sqlhash",
    "pg_sleep", "make_set", "dbms_pipe", "receive_message", "randomblob", "xmltype",
    "procedure", "analyse",
}

WORD_RE = re.compile(r"\w+")
SINGLE_QUOTE_RE = re.compile(r"'")
DOUBLE_QUOTE_RE = re.compile(r'"')
PUNCT_RE = re.compile(r"[!#$%&,.:;<=>?@\[\\\]^_`{|}~]")
COMMENT_RE = re.compile(r"(--|#|/\*)")
SPACE_RE = re.compile(r"\s+")
LOGIC_RE = re.compile(r"\bnot\b|\band\b|\bor\b|\bxor\b")
ARITH_RE = re.compile(r"(?<!/)\*(?!\*)|(?<!\*)/(?!\*)|[+\-<>]=?")
ALPHA_RE = re.compile(r"[a-zA-Z]")


def _normalized_tokens(query: str) -> list[str]:
    return advanced_preprocess(query).split()


def _count_sql_keywords(tokens: list[str]) -> int:
    hits = [token for token in tokens if token in SQL_KEYWORDS]
    if len(tokens) == 1 and len(hits) == 1:
        return 0
    return len(hits)


def _count_sql_functions(tokens: list[str]) -> int:
    return sum(1 for token in tokens if token in SQL_FUNCTIONS)


def _extract_feature_row_from_decoded_query(decoded_query: str) -> list[int]:
    # 单次预处理后复用 token，避免 wcount/sqlkw/sqlfunc 重复走完整归一化流程。
    tokens = _normalized_tokens(decoded_query)
    return [
        len(decoded_query),
        len(tokens),
        len(SINGLE_QUOTE_RE.findall(decoded_query)),
        len(DOUBLE_QUOTE_RE.findall(decoded_query)),
        len(PUNCT_RE.findall(decoded_query)),
        len(COMMENT_RE.findall(decoded_query)),
        len(SPACE_RE.findall(decoded_query)),
        len(LOGIC_RE.findall(decoded_query)),
        len(ARITH_RE.findall(decoded_query)),
        len(ALPHA_RE.findall(decoded_query)),
        _count_sql_keywords(tokens),
        _count_sql_functions(tokens),
    ]


def cnt_sqlkw(q):
    return _count_sql_keywords(_normalized_tokens(q))


def cnt_sqlfunc(q):
    return _count_sql_functions(_normalized_tokens(q))


def extract_struct_features(data):
    decoded_queries = [smart_recursive_decode(str(query)).lower() for query in data["Query"].tolist()]
    feature_rows = [_extract_feature_row_from_decoded_query(query) for query in decoded_queries]
    feature_array = np.asarray(feature_rows, dtype=np.int64)

    data = data.copy()
    data["Query"] = decoded_queries
    for index, column in enumerate(NUMERIC_FEATURE_COLUMNS):
        data[column] = feature_array[:, index]
    return data


def standardize_and_combine_features(x_train, x_test, num_cols):
    X_train_num = x_train[num_cols].values
    X_test_num = x_test[num_cols].values

    scaler = StandardScaler()
    scaler.fit(X_train_num)

    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)
    return X_train_num_scaled, X_test_num_scaled, scaler


def extract_struct_features_single(q):
    decoded_query = smart_recursive_decode(q).lower()
    return _extract_feature_row_from_decoded_query(decoded_query)
