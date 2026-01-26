# 12个特征提取
import warnings
warnings.filterwarnings("ignore")
import re

from utils.hdcan import *
from utils.hfes import *

from sklearn.preprocessing import StandardScaler

def cnt_sqlkw(q):
    kw_list = {
    # DML / DDL / 事务
    'alter','analyze','begin','call','commit','create','delete','drop','execute',
    'explain','fetch','grant','handler','insert','into','join','lock','optimize',
    'prepare','repair','replace','rollback','savepoint','select','set','start',
    'truncate','union','update','values',
    # 查询修饰 / 语法结构
    'all','distinct','exists','fetch','from','group','having','in','index',
    'inner','left','limit','offset','on','order','outer','right','top',
    'view','where','with','recursive',
    # 控制流 / 条件
    'case','when','then','else','end','if','between','like','regexp','rlike',
    # 逻辑与比较
    'and','or','xor','not','is','as','asc','desc',
    # 常量
    'null','true','false','binary',

    # 额外的
    'by'
    }
    tokens = unify_for_kwmatch(q)
#     return sum(1 for w in tokens if w in kw_list)
    hits = [w for w in tokens if w in kw_list]
    if len(tokens) == 1 and len(hits) == 1:
        return 0
    return len(hits)

def cnt_sqlfunc(q):
    func_list = {
        # 字符串 / 编码
        'ascii','char','char_length','charset','concat','concat_ws','conv','crc32',
        'elt','hex','instr','json_quote','lcase','left','length','load_file',
        'locate','lower','lpad','ltrim','mid','oct','ord','position','quote',
        'repeat','replace','regexp_instr','regexp_like','regexp_replace',
        'regexp_substr','right','rpad','rtrim','substr','substring',
        'substring_index','trim','ucase','upper','weight_string',
        # 数值 / 计算
        'acos','avg','benchmark','ceil','ceiling','conv','count','exp','floor',
        'greatest','least','log','log10','log2','pow','power','rand','round',
        'sin','sqrt','sum',
        # JSON
        'json_array','json_contains','json_contains_path','json_depth','json_extract',
        'json_keys','json_merge','json_object','json_objectagg','json_quote',
        'json_storage_size','json_type','json_valid','json_value',
        # XML / 报错利用
        'extractvalue','updatexml',
        # 地理 / GIS
        'geometrycollectionfromtext','mbrcontains','point','polygonfromtext',
        'st_area','st_astext','st_distance_sphere','st_geomfromtext',
        # 信息函数（泄露）
        'database','current_user','user','version','@@hostname','@@version',
        # 时间 / 延时
        'benchmark','sleep','delay','waitfor',
        # 加密 / 压缩
        'aes_encrypt','crc32','encode','md5','sha1','sha2','uncompress',
        # 事务 / 系统
        'get_format','get_lock','row_count','savepoint',
        # 动态 SQL
        'execute_immediate','prepare','deallocate_prepare',
        # 其他系统级（容易被滥用）
        'coercibility','collation','convert_tz','decode','encode','if','ifnull','convert'
        'isnull','nullif','raise_error',

        # 额外的
        'regexp_substring','chr','repeat','crypt_key','dbms_utility','sqlid_to_sqlhash','pg_sleep',
        'make_set','dbms_pipe','receive_message','randomblob','xmltype','procedure', 'analyse'
    }
    tokens = unify_for_kwmatch(q)
    return sum(1 for w in tokens if w in func_list)

# HFES特征
def extract_struct_features(data):
    data['Query'] = data['Query'].astype(str).apply(lambda s: smart_recursive_decode(s).lower())
    data['qlen']   = data['Query'].apply(len)
    data['wcount'] = data['Query'].apply(compute_word_count)
    

    def cnt_sq(x):     return len(re.findall(r"'", x))
    def cnt_dq(x):     return len(re.findall(r'"', x))
    def cnt_punc(x):   return len(re.findall(r"[!#$%&,.:;<=>?@\[\\\]^_`{|}~]", x))
    def cnt_comments(x): return len(re.findall(r'(--|#|/\*)', x))
    def cnt_spaces(x): return len(re.findall(r'\s+', x))
    def cnt_logic(x):   return len(re.findall(r'\bnot\b|\band\b|\bor\b|\bxor\b', x))
    def cnt_arith(x):  return len(re.findall(r'(?<!/)\*(?!\*)|(?<!\*)/(?!\*)|[+\-<>]=?', x))
    def cnt_alpha(x):  return len(re.findall(r'[a-zA-Z]', x))
    def cnt_digit(x):  return len(re.findall(r'[0-9]', x))

    data['sq']       = data['Query'].apply(cnt_sq)
    data['dq']       = data['Query'].apply(cnt_dq)
    data['puncts']   = data['Query'].apply(cnt_punc)
    data['comments'] = data['Query'].apply(cnt_comments)
    data['spaces']   = data['Query'].apply(cnt_spaces)
    data['logic']    = data['Query'].apply(cnt_logic)
    data['arith']    = data['Query'].apply(cnt_arith)
    data['alpha']    = data['Query'].apply(cnt_alpha)
    data['sqlkw']    = data['Query'].apply(cnt_sqlkw)
    data['sqlfunc']     = data['Query'].apply(cnt_sqlfunc)


    return data

def standardize_and_combine_features(x_train, x_test, num_cols):
    X_train_num = x_train[num_cols].values
    X_test_num  = x_test[num_cols].values

    scaler = StandardScaler()
    scaler.fit(X_train_num)

    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled  = scaler.transform(X_test_num)
    return X_train_num_scaled, X_test_num_scaled, scaler

def extract_struct_features_single(q):
    """
    针对单条 Query 字符串做特征提取
    """
    # 将输入字符串转换为小写
    q = smart_recursive_decode(q).lower()
    # 利用 compute_word_count 函数计算预处理后的 token 数量
    wcount = compute_word_count(q)

    def cnt_sq(x):    return len(re.findall(r"'", x))
    def cnt_dq(x):    return len(re.findall(r'"', x))
    def cnt_punc(x):  return len(re.findall(r"[!#$%&,.:;<=>?@\[\\\]^_`{|}~]", x))
    def cnt_comments(x): return len(re.findall(r'(--|#|/\*)', x))
    def cnt_spaces(x):return len(re.findall(r'\s+', x))
    def cnt_logic(x):   return len(re.findall(r'\bnot\b|\band\b|\bor\b|\bxor\b', x))
    # def cnt_arith(x): return len(re.findall(r'(?<!/)\*(?!\*)|(?<!\*)/(?!\*)|(?:(?<=\s)|^)[+-](?=\s|$)', x))
    def cnt_arith(x): return len(re.findall(r'(?<!/)\*(?!\*)|(?<!\*)/(?!\*)|[+\-<>]=?', x))
    def cnt_alpha(x): return len(re.findall(r'[a-zA-Z]', x))
    def cnt_digit(x): return len(re.findall(r'[0-9]', x))

    def cnt_symlogic(x):
        return len(re.findall(r'(<>|!=|>=|<=|\|\||&&|\^)',x))
    
    feats = []
    feats.append(len(q))         # qlen
    feats.append(wcount)               # wcount
    feats.append(cnt_sq(q))      # sq
    feats.append(cnt_dq(q))      # dq
    feats.append(cnt_punc(q))    # puncts
    feats.append(cnt_comments(q))     # slc
    feats.append(cnt_spaces(q))  # spaces
    feats.append(cnt_logic(q))   # logic
    feats.append(cnt_arith(q))   # arith
    feats.append(cnt_alpha(q))   # alpha)
    feats.append(cnt_sqlkw(q))   # sqlkw
    feats.append(cnt_sqlfunc(q))

    return feats
