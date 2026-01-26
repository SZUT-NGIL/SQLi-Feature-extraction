import re, urllib.parse, base64, hashlib, json, html, csv
from sqlglot import parse_one
from sqlglot.errors import ParseError
from sqlglot.expressions import (
    Identifier, Table, Literal,
    Select, Insert, Update, Delete, Union,
    Boolean
)
from sql_metadata import Parser

###############################
# 全局 SQL 关键词与排序（降序，便于贪婪匹配融合关键词）
###############################
SQL_KEYWORDS = [
    'select', 'union', 'from', 'where', 'insert', 'update', 'delete',
    'drop', 'create', 'table', 'values', 'into', 'set', 'join', 'having',
    'order', 'group', 'limit'
]
KW_SORTED = sorted(SQL_KEYWORDS, key=lambda x: len(x), reverse=True)

###############################
# 1. 多层解码 + 深层规范化
###############################
def _is_probable_hex(s: str) -> bool:
    t = re.sub(r'\s+', '', s)  # 允许中间有空白
    return len(t) % 2 == 0 and re.fullmatch(r'[0-9A-Fa-f]+', t) is not None

def _hex_to_str(s: str) -> str:
    t = re.sub(r'\s+', '', s)
    b = bytes.fromhex(t)
    # 严格按 UTF-8 解码，避免把任意二进制误当文本
    return b.decode('utf-8', errors='strict')

def smart_recursive_decode(s, max_rounds=10):
    for _ in range(max_rounds):
        original = s

        # 1) URL 解码
        s_url = urllib.parse.unquote(s)
        if s_url != s:
            s = s_url
            continue

        # 2) Base64 解码（严格模式）
        try:
            tmp = s.strip().replace(' ', '').replace('\n', '').replace('\r', '')
            if len(tmp) % 4 == 0:
                decoded_b64 = base64.b64decode(tmp, validate=True).decode('utf-8', errors='strict')
                if decoded_b64 != s:
                    s = decoded_b64
                    continue
        except Exception:
            pass

        # 3) %uXXXX 解码
        s_unicode = re.sub(r'%u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)
        if s_unicode != s:
            s = s_unicode
            continue

        # 4) \xHH 解码
        if _is_probable_hex(s):
            try:
                decoded_hex = _hex_to_str(s)
                if decoded_hex != s:
                    s = decoded_hex
                    continue
            except Exception:
                pass
        s_hex = re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), s)
        if s_hex != s:
            s = s_hex
            continue

        # 5) HTML 实体解码
        s_html = html.unescape(s)
        if s_html != s:
            s = s_html
            continue

        if s == original:
            break
    return s

def normalize_attack_payload(text, max_rounds=10):
    """
    对攻击载荷进行深层规范化，采用迭代式处理确保同时处理以下两种混淆：
      - 内部标记干扰（Intra-token Obfuscation）：允许在每个 SQL 关键字的字母间插入任意非单词字符，
        如将 "s%e%l%e%c%t" 恢复为 "select"。
      - 包装型注释混淆（Annotation-based Obfuscation）：对包裹在注释或特殊符号中的关键词进行提取，
        如将 "/!select!/"、"/*!50000select*/" 或 "/*select*/" 统一转换为 "select"。
    最后，对处理后的文本进一步清洗（压缩空白、分词）并输出标准化文本。
    """
    # 将输入转换为小写
    text = text.lower()
    
    # 迭代处理，确保对可能混合出现的各种混淆形式进行逐轮修正
    for _ in range(max_rounds):
        prev_text = text
        
        # 类型A处理：恢复插入非单词字符的混淆关键词
        for kw in SQL_KEYWORDS:
            # 构造正则表达式：在每个字母间允许出现任意数量的非单词字符
            pattern = rf"{kw[0]}" + "".join(f"[^\w]*{c}" for c in kw[1:])
            text = re.sub(pattern, kw, text, flags=re.IGNORECASE)
        
        # 类型B处理：提取包装型注释混淆中的关键词
        special_patterns = [
            (r'/*\s!([a-z0-9_]+)\s*/', r' \1 '),  
            # 针对诸如 /*!50000select*/、/!*select*/、/*select*/ 这类模式，
            # 其中允许可选的感叹号、版本号以及前后空白字符的存在，
            # 提取出包含小写字母、数字和下划线构成的关键词
            (r'/\*+!?\s*(?:\d+\s*)?([a-z0-9_]+).*?\*/', r' \1 '),
        ]
        for pat, rep in special_patterns:
            text = re.sub(pat, rep, text, flags=re.I | re.S | re.M)

        # 清除不具提取意义的干扰内容：压缩多余空白字符
        text = re.sub(r'\s+', ' ', text, flags=re.IGNORECASE | re.DOTALL)
        
        # 如果本轮处理文本没有变化，则认为达到稳定状态，终止迭代
        if text == prev_text:
            break

    # 分词处理：按照非字母、数字或下划线进行切分，保留有效的关键词
    tokens = re.split(r'[^a-z0-9_]+', text)
    text = ' '.join(token for token in tokens if token)
    # 最后压缩空格并返回最终文本
    text = re.sub(r'\s+', ' ', text).strip()
    return text

###############################
# AST 占位符替换
###############################
def placeholder_ast(expr):
    for node in expr.walk():
        if isinstance(node, (Identifier, Table)):
            node.set("this", "<ID>")
        elif isinstance(node, Literal):
            node.set("this", "<LIT>")
    return expr

###############################
# 指纹计算
###############################
def fingerprint_ast(expr):
    s = json.dumps(expr.to_dict(), sort_keys=True)
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def fingerprint_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

###############################
# SQL 识别关键字 & 运算符
###############################
strong_ops = [
    r'union\s+select', r'sleep\s*\(', r'extractvalue\s*\(',
    r'updatexml\s*\(', r'load_file\s*\('
]
strong_pattern = re.compile("|".join(strong_ops), re.IGNORECASE)

kw_list = {
    'select','delete','order','fetch','join','avg','count','sum','rows',
    'mid','ascii','ord','limit','offset','concat','group','left',
    'sleep','exp','hex','union','insert','drop','case','then','cast',
    'chr','else','numeric','end','char','declare','if','set','waitfor','delay',
    'like','exists','null','true','false','binary','int','varchar','numeric','regexp_substring',
    'substring','substring_index','length','char_length','locate','position',
    'instr','replace','regexp','regexp_like','regexp_instr','regexp_replace',
    'repeat','lpad','rpad','ltrim','rtrim','trim','upper','lower',
    'dbms_utilit','extractvalue','load_file','database','xmltype','updatexml','get_host_address'
}
kw_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in kw_list) + r')\b', re.IGNORECASE)
op_pattern = re.compile(r'[=<>(),]')

def is_valid_sql_ast(expr):
    # 根节点类型判断
    root = expr.__class__.__name__.upper()
    if root in {"SELECT", "INSERT", "UPDATE", "DELETE", "UNION"}:
        return True
    # 子节点中包含 'BOOLEAN' 或 'EQ'/'NEQ' 等比较词即视为合法 SQL AST
    for node in expr.walk():
        name = node.__class__.__name__.upper()
        if "BOOLEAN" in name or name in {"EQ", "NEQ", "GT", "LT", "GTE", "LTE"}:
            return True
    return False

###############################
# 生成指纹：识别 SQL 还是 文本
###############################
def generate_fingerprint(q_clean):
    # 1. 剥注释
    q = re.sub(r'(#|--).*$', '', q_clean).strip()

    # 2. 尝试用 sql-metadata 判断表名，捕获不支持的 query_type
    try:
        parser = Parser(q)
        has_table = bool(parser.tables)
    except ValueError:
        has_table = False

    # 3. 其他 SQL 判定逻辑（强关键词 or 关键词+运算符）
    if strong_pattern.search(q):
        candidate = True
    else:
        candidate = has_table or (bool(kw_pattern.search(q)) and bool(op_pattern.search(q)))

    # 4. 如果是 SQL 尝试 AST 处理，否则回退到文本哈希
    if candidate:
        try:
            ast = parse_one(q, error_level='ignore')
            if is_valid_sql_ast(ast):
                return fingerprint_ast(placeholder_ast(ast))
        except ParseError:
            pass

    return fingerprint_text(q)

###############################
# 去重：Fingerprint + Label
###############################
def dedupe_by_fingerprint(data_items):
    seen = set()
    unique = []
    for item in data_items:
        q_clean = clean_query(item['Query'])
        fp = generate_fingerprint(q_clean)
        key = (fp, item['Label'])
        if key not in seen:
            seen.add(key)
            unique.append({'Query': q_clean, 'Label': item['Label']})
    return unique

# 工具函数
def clean_query(q):
    """
    对 Query 同时做多层解码 + 规范化.
    """
    step1 = smart_recursive_decode(q)
    step2 = normalize_attack_payload(step1)
    return step2

def advanced_preprocess(query_str):
    """
    1. 多层 URL 与 Base64 解码
    2. 深层规范化：移除注释、归一化空白、统一各种 SQL 关键词变体
    3. 拆分并过滤 token（仅保留长度在 [2,50] 且非纯数字的 token）
    返回以空格拼接后的字符串，供 BoW 特征提取使用。
    """
    decoded = smart_recursive_decode(query_str)
    normed  = normalize_attack_payload(decoded)
    tokens  = re.split(r'[^a-zA-Z0-9_]+', normed)
    tokens  = [t for t in tokens if t and not t.isdigit() and 2 <= len(t) <= 50]
    return " ".join(tokens)

def compute_word_count(query):
    """
    对输入的 query 进行多层解码、规范化与拆分，
    然后使用正则表达式提取所有连续的字母数字（包括下划线）序列，
    返回这些 token 的个数，作为词数。
    """
    preprocessed = advanced_preprocess(query)
    # 提取所有连续的字母、数字和下划线序列
    tokens = re.findall(r'\w+', preprocessed)
    return len(tokens)

def unify_for_kwmatch(q):
    splitted = advanced_preprocess(q).split()
    return splitted