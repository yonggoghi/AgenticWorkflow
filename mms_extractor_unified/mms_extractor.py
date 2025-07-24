
# %%
from concurrent.futures import ThreadPoolExecutor
import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import re
# from pygments import highlight
# from pygments.lexers import JsonLexer
# from pygments.formatters import HtmlFormatter
# from IPython.display import HTML
import pandas as pd
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI
from typing import List, Tuple, Union, Dict, Any
import ast
from rapidfuzz import fuzz, process
import re
import json
import glob
import os
from config import settings
from kiwipiepy import Kiwi
from joblib import Parallel, delayed
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import difflib
from dotenv import load_dotenv
import cx_Oracle


pd.set_option('display.max_colwidth', 500)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Helper functions
def dataframe_to_markdown_prompt(df, max_rows=None):
    if max_rows is not None and len(df) > max_rows:
        display_df = df.head(max_rows)
        truncation_note = f"\n[Note: Only showing first {max_rows} of {len(df)} rows]"
    else:
        display_df = df
        truncation_note = ""
    df_markdown = display_df.to_markdown()
    prompt = f"\n\n    {df_markdown}\n    {truncation_note}\n\n    "
    return prompt

def clean_segment(segment):
    segment = segment.strip()
    if len(segment) >= 2 and segment[0] in ['"', "'"] and segment[-1] == segment[0]:
        q = segment[0]
        inner = segment[1:-1].replace(q, '')
        return q + inner + q
    return segment

def split_key_value(text):
    in_quote = False
    quote_char = ''
    for i, char in enumerate(text):
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
        elif char == ':' and not in_quote:
            return text[:i], text[i+1:]
    return text, ''

def split_outside_quotes(text, delimiter=','):
    parts = []
    current = []
    in_quote = False
    quote_char = ''
    for char in text:
        if char in ['"', "'"]:
            if in_quote:
                if char == quote_char:
                    in_quote = False
                    quote_char = ''
            else:
                in_quote = True
                quote_char = char
            current.append(char)
        elif char == delimiter and not in_quote:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())
    return parts

def clean_ill_structured_json(text):
    parts = split_outside_quotes(text, delimiter=',')
    cleaned_parts = []
    for part in parts:
        key, value = split_key_value(part)
        key_clean = clean_segment(key)
        value_clean = clean_segment(value) if value.strip() != "" else ""
        if value_clean:
            cleaned_parts.append(f"{key_clean}: {value_clean}")
        else:
            cleaned_parts.append(key_clean)
    return ', '.join(cleaned_parts)

def repair_json(broken_json):
    json_str = broken_json
    json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1 "\2":', json_str)
    parts = json_str.split('"')
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r':\s*([a-zA-Z0-9_]+)(?=\s*[,\]\}])', r': "\1"', parts[i])
    json_str = '"'.join(parts)
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    return json_str

def extract_json_objects(text):
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    result = []
    for match in re.finditer(pattern, text):
        potential_json = match.group(0)
        try:
            json_obj = ast.literal_eval(clean_ill_structured_json(repair_json(potential_json)))
            result.append(json_obj)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            pass
    return result

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fuzzy_similarities(text, entities):
    results = []
    for entity in entities:
        scores = {
            'ratio': fuzz.ratio(text, entity) / 100,
            'partial_ratio': fuzz.partial_ratio(text, entity) / 100,
            'token_sort_ratio': fuzz.token_sort_ratio(text, entity) / 100,
            'token_set_ratio': fuzz.token_set_ratio(text, entity) / 100
        }
        max_score = max(scores.values())
        results.append((entity, max_score))
    return results

def get_fuzzy_similarities(args_dict):
    text = args_dict['text']
    entities = args_dict['entities']
    threshold = args_dict['threshold']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']
    text_processed = preprocess_text(text.lower())
    similarities = fuzzy_similarities(text_processed, entities)
    filtered_results = [
        {
            text_col_nm: text,
            item_col_nm: entity, 
            "sim": score
        } 
        for entity, score in similarities 
        if score >= threshold
    ]
    return filtered_results

def parallel_fuzzy_similarity(texts, entities, threshold=0.5, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None):
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    if batch_size is None:
        batch_size = max(1, len(entities) // (n_jobs * 2))
    batches = []
    for text in texts:
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batches.append({"text": text, "entities": batch, "threshold": threshold, "text_col_nm": text_col_nm, "item_col_nm": item_col_nm})
    with Parallel(n_jobs=n_jobs) as parallel:
        batch_results = parallel(delayed(get_fuzzy_similarities)(args) for args in batches)
    return pd.DataFrame(sum(batch_results, []))

def longest_common_subsequence_ratio(s1, s2, normalizaton_value):
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_len = lcs_length(s1, s2)
    if normalizaton_value == 'max':
        max_len = max(len(s1), len(s2))
        return lcs_len / max_len if max_len > 0 else 1.0
    elif normalizaton_value == 'min':
        min_len = min(len(s1), len(s2))
        return lcs_len / min_len if min_len > 0 else 1.0
    elif normalizaton_value == 's1':
        return lcs_len / len(s1) if len(s1) > 0 else 1.0
    elif normalizaton_value == 's2':
        return lcs_len / len(s2) if len(s2) > 0 else 1.0
    else:
        raise ValueError(f"Invalid normalization value: {normalizaton_value}")

def sequence_matcher_similarity(s1, s2, normalizaton_value):
    matcher = difflib.SequenceMatcher(None, s1, s2)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    normalization_length = min(len(s1), len(s2))
    if normalizaton_value == 'max':
        normalization_length = max(len(s1), len(s2))
    elif normalizaton_value == 's1':
        normalization_length = len(s1)
    elif normalizaton_value == 's2':
        normalization_length = len(s2)
    if normalization_length == 0: 
        return 0.0
    return matches / normalization_length

def substring_aware_similarity(s1, s2, normalizaton_value):
    if s1 in s2 or s2 in s1:
        shorter = min(s1, s2, key=len)
        longer = max(s1, s2, key=len)
        base_score = len(shorter) / len(longer)
        return min(0.95 + base_score * 0.05, 1.0)
    return longest_common_subsequence_ratio(s1, s2, normalizaton_value)

def token_sequence_similarity(s1, s2, normalizaton_value, separator_pattern=r'[\s_\-]+'):
    tokens1 = [t for t in re.split(separator_pattern, s1.strip()) if t]
    tokens2 = [t for t in re.split(separator_pattern, s2.strip()) if t]
    if not tokens1 or not tokens2:
        return 0.0
    def token_lcs_length(t1, t2):
        m, n = len(t1), len(t2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t1[i-1] == t2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    lcs_tokens = token_lcs_length(tokens1, tokens2)
    normalization_tokens = max(len(tokens1), len(tokens2))
    if normalizaton_value == 'min':
        normalization_tokens = min(len(tokens1), len(tokens2))
    elif normalizaton_value == 's1':
        normalization_tokens = len(tokens1)
    elif normalizaton_value == 's2':
        normalization_tokens = len(tokens2)
    return lcs_tokens / normalization_tokens  

def combined_sequence_similarity(s1, s2, weights=None, normalizaton_value='max'):
    if weights is None:
        weights = {'substring': 0.4, 'sequence_matcher': 0.4, 'token_sequence': 0.2}
    similarities = {
        'substring': substring_aware_similarity(s1, s2, normalizaton_value),
        'sequence_matcher': sequence_matcher_similarity(s1, s2, normalizaton_value),
        'token_sequence': token_sequence_similarity(s1, s2, normalizaton_value)
    }
    return sum(similarities[key] * weights[key] for key in weights), similarities

def calculate_seq_similarity(args_dict):
    sent_item_batch = args_dict['sent_item_batch']
    text_col_nm = args_dict['text_col_nm']
    item_col_nm = args_dict['item_col_nm']
    normalizaton_value = args_dict['normalizaton_value']
    results = []
    for sent_item in sent_item_batch:
        sent = sent_item[text_col_nm]
        item = sent_item[item_col_nm]
        try:
            sent_processed = preprocess_text(sent.lower())
            item_processed = preprocess_text(item.lower())
            similarity = combined_sequence_similarity(sent_processed, item_processed, normalizaton_value=normalizaton_value)[0]
            results.append({text_col_nm:sent, item_col_nm:item, "sim":similarity})
        except Exception as e:
            print(f"Error processing {item}: {e}")
            results.append({text_col_nm:sent, item_col_nm:item, "sim":0.0})
    return results

def parallel_seq_similarity(sent_item_pdf, text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=None, batch_size=None, normalizaton_value='s2'):
    if n_jobs is None:
        n_jobs = min(os.cpu_count()-1, 8)
    if batch_size is None:
        batch_size = max(1, sent_item_pdf.shape[0] // (n_jobs * 2))
    batches = []
    for i in range(0, sent_item_pdf.shape[0], batch_size):
        batch = sent_item_pdf.iloc[i:i + batch_size].to_dict(orient='records')
        batches.append({"sent_item_batch": batch, 'text_col_nm': text_col_nm, 'item_col_nm': item_col_nm, 'normalizaton_value': normalizaton_value})
    with Parallel(n_jobs=n_jobs) as parallel:
        batch_results = parallel(delayed(calculate_seq_similarity)(args) for args in batches)
    return pd.DataFrame(sum(batch_results, []))

def load_sentence_transformer(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")
    model = SentenceTransformer(model_path).to(device)
    print(f"Model loaded on {device}")
    return model

# Kiwi helper classes
class Token:
    def __init__(self, form, tag, start, length):
        self.form = form
        self.tag = tag
        self.start = start
        self.len = length

class Sentence:
    def __init__(self, text, start, end, tokens, subs=None):
        self.text = text
        self.start = start
        self.end = end
        self.tokens = tokens
        self.subs = subs or []

def filter_text_by_exc_patterns(sentence, exc_tag_patterns):
    individual_tags = set()
    sequences = []
    for pattern in exc_tag_patterns:
        if isinstance(pattern, list):
            if len(pattern) == 1:
                individual_tags.add(pattern[0])
            else:
                sequences.append(pattern)
        else:
            individual_tags.add(pattern)
    tokens_to_exclude = set()
    for i, token in enumerate(sentence.tokens):
        if token.tag in individual_tags:
            tokens_to_exclude.add(i)
    for sequence in sequences:
        seq_len = len(sequence)
        for i in range(len(sentence.tokens) - seq_len + 1):
            if all(sentence.tokens[i + j].tag == sequence[j] for j in range(seq_len)):
                for j in range(seq_len):
                    tokens_to_exclude.add(i + j)
    result_chars = list(sentence.text)
    for i, token in enumerate(sentence.tokens):
        if i in tokens_to_exclude:
            start_pos = token.start - sentence.start
            end_pos = start_pos + token.len
            for j in range(start_pos, end_pos):
                if j < len(result_chars) and result_chars[j] != ' ':
                    result_chars[j] = ' '
    filtered_text = ''.join(result_chars)
    return re.sub(r'\s+', ' ', filtered_text)

def filter_specific_terms(strings: List[str]) -> List[str]:
    unique_strings = list(set(strings))
    unique_strings.sort(key=len, reverse=True)
    filtered = []
    for s in unique_strings:
        if not any(s in other for other in filtered):
            filtered.append(s)
    return filtered

def convert_df_to_json_list(df):
    result = []
    grouped = df.groupby('item_name_in_msg')
    for item_name_in_msg, group in grouped:
        item_dict = {
            'item_name_in_msg': item_name_in_msg,
            'item_in_voca': []
        }
        item_nm_groups = group.groupby('item_nm')
        for item_nm, item_group in item_nm_groups:
            item_ids = list(item_group['item_id'].unique())
            voca_item = {
                'item_nm': item_nm,
                'item_id': item_ids
            }
            item_dict['item_in_voca'].append(voca_item)
        result.append(item_dict)
    return result


class MMSExtractor:
    def __init__(self, model_path='./models/ko-sbert-nli', data_dir='./data/', offer_info_data_src='local'):
        self.data_dir = data_dir
        self.offer_info_data_src = offer_info_data_src  # 'local' or 'db'
        self.product_info_extraction_mode = 'llm'
        self.entity_extraction_mode = 'logic'
        self.num_cand_pgms = 5
        
        # Load environment variables
        load_dotenv()
        
        self._initialize_device()
        self._initialize_llm()
        self._initialize_embedding_model(model_path)
        self._initialize_kiwi()
        self._load_data()

    def _initialize_device(self):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

    def _initialize_llm(self):
        self.llm_gem3 = ChatOpenAI(
            temperature=0,
            openai_api_key=settings.API_CONFIG.llm_api_key,
            openai_api_base=settings.API_CONFIG.llm_api_url,
            model='skt/gemma3-12b-it',
            max_tokens=4000
        )
        print("Initialized LLM: skt/gemma3-12b-it")

    def _initialize_embedding_model(self, model_path):
        self.emb_model = load_sentence_transformer(model_path, self.device)

    def _initialize_kiwi(self):
        self.kiwi = Kiwi()
        self.exc_tag_patterns = [
            ['SN', 'NNB'], ['W_SERIAL'], ['JKO'], ['W_URL'], ['W_EMAIL'], ['XSV', 'EC'], 
            ['VV', 'EC'], ['VCP', 'ETM'], ['XSA', 'ETM'], ['VV', 'ETN'], ['W_SERIAL'],
            ['W_URL'], ['JKO'], ['SSO'], ['SSC'], ['SW'], ['SF'], ['SP'], ['SS'], ['SE'],
            ['SO'], ['SB'], ['SH'], ['W_HASHTAG']
        ]
        print("Initialized Kiwi morphological analyzer.")

    def _load_data(self):
        print("Loading data...")
        
        if self.offer_info_data_src == "local":
            item_pdf_raw = pd.read_csv(os.path.join(self.data_dir, "item_info_all_250527.csv"))
            self.item_pdf_all = item_pdf_raw.drop_duplicates(['item_nm','item_id'])[['item_nm','item_id','item_desc','domain']].copy()
            self.item_pdf_all['item_ctg'] = None
            self.item_pdf_all['item_emb_vec'] = None
            self.item_pdf_all['ofer_cd'] = self.item_pdf_all['item_id']
            self.item_pdf_all['oper_dt_hms'] = '20250101000000'
            self.item_pdf_all = self.item_pdf_all.rename(columns={c:c.lower() for c in self.item_pdf_all.columns})
        elif self.offer_info_data_src == "db":
            # DB 접속 정보
            username = os.getenv("DB_USERNAME")
            password = os.getenv("DB_PASSWORD")
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT")
            service_name = os.getenv("DB_NAME")
            
            # DB 연결
            dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
            conn = cx_Oracle.connect(user=username, password=password, dsn=dsn, encoding="UTF-8")
            
            # 데이터 조회 (적절한 조건 설정)
            sql = "SELECT * FROM TCAM_RC_OFER_MST WHERE ROWNUM <= 1000000"
            self.item_pdf_all = pd.read_sql(sql, conn)
            conn.close()
            
            # 컬럼명을 소문자로 변환
            self.item_pdf_all = self.item_pdf_all.rename(columns={c:c.lower() for c in self.item_pdf_all.columns})

        alias_pdf = pd.read_csv(os.path.join(self.data_dir, "alias_rules.csv"))
        alia_rule_set = list(zip(alias_pdf['alias_1'], alias_pdf['alias_2']))

        def apply_alias_rule(item_nm):
            item_nm_list = [item_nm]
            for r in alia_rule_set:
                if r[0] in item_nm:
                    item_nm_list.append(item_nm.replace(r[0], r[1]))
                if r[1] in item_nm:
                    item_nm_list.append(item_nm.replace(r[1], r[0]))
            return item_nm_list

        self.item_pdf_all['item_nm_alias'] = self.item_pdf_all['item_nm'].apply(apply_alias_rule)
        self.item_pdf_all = self.item_pdf_all.explode('item_nm_alias')
        
        user_defined_entity = ['AIA Vitality' , '부스트 파크 건대입구' , 'Boost Park 건대입구']
        item_pdf_ext = pd.DataFrame([{'item_nm':e,'item_id':e,'item_desc':e, 'domain':'user_defined', 'start_dt':20250101, 'end_dt':99991231, 'rank':1, 'item_nm_alias':e} for e in user_defined_entity])
        # self.item_pdf_all = pd.concat([self.item_pdf_all,item_pdf_ext]) # This was causing issues with missing columns in original code
        
        self.stop_item_names = pd.read_csv(os.path.join(self.data_dir, "stop_words.csv"))['stop_words'].to_list()

        for w in self.item_pdf_all['item_nm_alias'].unique():
            self.kiwi.add_user_word(w, "NNP")

        self.pgm_pdf = pd.read_csv(os.path.join(self.data_dir, "pgm_tag_ext_250516.csv"))
        self.clue_embeddings = self.emb_model.encode(
            self.pgm_pdf[["pgm_nm","clue_tag"]].apply(lambda x: preprocess_text(x['pgm_nm'].lower())+" "+x['clue_tag'].lower(), axis=1).tolist(),
            convert_to_tensor=True, show_progress_bar=True
        )

        self.org_pdf = pd.read_csv(os.path.join(self.data_dir, "org_info_all_250605.csv"), encoding='cp949')
        self.org_pdf['sub_org_cd'] = self.org_pdf['sub_org_cd'].apply(lambda x: str(x).zfill(4))
        print("Data loading complete.")

    def extract_entities_from_kiwi(self, mms_msg):
        sentences = sum(self.kiwi.split_into_sents(re.split(r"_+", mms_msg), return_tokens=True, return_sub_sents=True), [])
        sentences_all = []
        for sent in sentences:
            if sent.subs:
                sentences_all.extend(sent.subs)
            else:
                sentences_all.append(sent)
        
        sentence_list = [filter_text_by_exc_patterns(sent, self.exc_tag_patterns) for sent in sentences_all]

        result_msg = self.kiwi.tokenize(mms_msg, normalize_coda=True, z_coda=False, split_complex=False)
        entities_from_kiwi = [token.form for token in result_msg if token.tag == 'NNP' and token.form not in self.stop_item_names+['-'] and len(token.form)>=2 and not token.form.lower() in self.stop_item_names]
        entities_from_kiwi = filter_specific_terms(entities_from_kiwi)
        print("추출된 개체명 (Kiwi):", list(set(entities_from_kiwi)))

        similarities_fuzzy = parallel_fuzzy_similarity(
            sentence_list, self.item_pdf_all['item_nm_alias'].unique(), threshold=0.4,
            text_col_nm='sent', item_col_nm='item_nm_alias', n_jobs=6, batch_size=30
        )
        if similarities_fuzzy.empty:
            cand_item_list = entities_from_kiwi
            extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[['item_nm','item_nm_alias','item_id']].groupby(["item_nm"])['item_id'].apply(list).reset_index()
            return cand_item_list, extra_item_pdf

        similarities_seq = parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy, text_col_nm='sent', item_col_nm='item_nm_alias',
            n_jobs=6, batch_size=100
        )
        cand_items = similarities_seq.query("sim>=0.7 and item_nm_alias.str.contains('', case=False) and item_nm_alias not in @self.stop_item_names")
        entities_from_kiwi_pdf = self.item_pdf_all.query("item_nm_alias in @entities_from_kiwi")[['item_nm','item_nm_alias']]
        entities_from_kiwi_pdf['sim'] = 1.0

        cand_item_pdf = pd.concat([cand_items, entities_from_kiwi_pdf])
        cand_item_list = cand_item_pdf.sort_values('sim', ascending=False).groupby(["item_nm_alias"])['sim'].max().reset_index(name='final_sim').sort_values('final_sim', ascending=False).query("final_sim>=0.2")['item_nm_alias'].unique()
        extra_item_pdf = self.item_pdf_all.query("item_nm_alias in @cand_item_list")[['item_nm','item_nm_alias','item_id']].groupby(["item_nm"])['item_id'].apply(list).reset_index()

        return cand_item_list, extra_item_pdf

    def extract_entities_by_logic(self, cand_entities):
        similarities_fuzzy = parallel_fuzzy_similarity(
            cand_entities, self.item_pdf_all['item_nm_alias'].unique(), threshold=0.8,
            text_col_nm='item_name_in_msg', item_col_nm='item_nm_alias', n_jobs=6, batch_size=30
        )
        if similarities_fuzzy.empty:
            return pd.DataFrame()
        
        cand_entities_sim = parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy, text_col_nm='item_name_in_msg', item_col_nm='item_nm_alias',
            n_jobs=6, batch_size=30, normalizaton_value='s1'
        ).rename(columns={'sim':'sim_s1'}).merge(parallel_seq_similarity(
            sent_item_pdf=similarities_fuzzy, text_col_nm='item_name_in_msg', item_col_nm='item_nm_alias',
            n_jobs=6, batch_size=30, normalizaton_value='s2'
        ).rename(columns={'sim':'sim_s2'}), on=['item_name_in_msg','item_nm_alias']).groupby(['item_name_in_msg','item_nm_alias'])[['sim_s1','sim_s2']].apply(lambda x: x['sim_s1'].sum() + x['sim_s2'].sum()).reset_index(name='sim')
        return cand_entities_sim

    def process_message(self, mms_msg):
        print(f"Processing message: {mms_msg[:100]}...")
        msg = mms_msg.strip()
        cand_item_list, extra_item_pdf = self.extract_entities_from_kiwi(msg)
        
        mms_embedding = self.emb_model.encode([msg.lower()], convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(mms_embedding, self.clue_embeddings, dim=1).cpu().numpy()
        
        pgm_pdf_tmp = self.pgm_pdf.copy()
        pgm_pdf_tmp['sim'] = similarities
        pgm_pdf_tmp = pgm_pdf_tmp.sort_values('sim', ascending=False)
        pgm_cand_info = "\n\t".join(pgm_pdf_tmp.iloc[:self.num_cand_pgms][['pgm_nm','clue_tag']].apply(lambda x: re.sub(r'\[.*?\]', '', x['pgm_nm'])+" : "+x['clue_tag'], axis=1).to_list())
        rag_context = f"\n### 광고 분류 기준 정보 ###\n\t{pgm_cand_info}" if self.num_cand_pgms > 0 else ""

        prd_ext_guide = "* 상품 추출시 정확도(precision) 보다는 재현율(recall)에 중심을 두어라."
        if len(cand_item_list)>0 and self.product_info_extraction_mode == 'rag':
             rag_context += f"\n\n### 후보 상품 이름 목록 ###\n\t{cand_item_list}"
             prd_ext_guide += "\n* 후보 상품 이름 목록에 포함된 상품 이름은 참고하여 Product 정보를 추출하라."
        
        schema_prd = {
            "title": '광고 제목. 광고의 핵심 주제와 가치 제안을 명확하게 설명할 수 있도록 생성',
            'purpose': '광고의 주요 목적을 다음 중에서 선택(복수 가능): [상품 가입 유도, 대리점/매장 방문 유도, 웹/앱 접속 유도, 이벤트 응모 유도, 혜택 안내, 쿠폰 제공 안내, 경품 제공 안내, 수신 거부 안내, 기타 정보 제공]',
            'product': {'type': 'array', 'items': {'name': '광고하는 제품이나 서비스 이름', 'action': '고객에게 기대하는 행동: [구매, 가입, 사용, 방문, 참여, 코드입력, 쿠폰다운로드, 기타] 중에서 선택'}},
            'channel': {'type': 'array', 'items': {'properties': {'type': '[URL, 전화번호, 앱, 대리점] 중에서 선택', 'value': '실제 URL, 전화번호, 앱 이름, 대리점 이름 등 구체적 정보', 'action': '채널 목적: [가입, 추가 정보, 문의, 수신, 수신 거부] 중에서 선택'}}},
            'pgm': {'type': 'array', 'description': '아래 광고 분류 기준 정보에서 선택. 메세지 내용과 광고 분류 기준을 참고하여, 광고 메세지에 가장 부합하는 2개의 pgm_nm을 적합도 순서대로 제공'},
        }
        
        prompt = f"""
        아래 광고 메시지에서 광고 목적과 상품 이름을 추출해 주세요.
        ### 광고 메시지 ###
        {msg}
        ### 추출 작업 순서 ###
        1. 광고 목적을 먼저 파악한다.
        2. 파악된 목적에 기반하여 Main 상품을 추출한다.
        3. 추출한 Main 상품에 관련되는 Sub 상품을 추출한다.
        4. 추출된 상품 정보를 고려하여 채널 정보를 제공한다.
        ### 추출 작업 가이드 ###
        * 상품 추출시 정확도(precision) 보다는 재현율(recall)에 중심을 두어라.
        * 광고 목적에 대리점 방문이 포함되어 있으면 대리점 채널 정보를 제공해라.
        {prd_ext_guide}
        * Only generate the json object, do not include any other text to save as a json file
        아래와 같은 스키마로 결과를 제공해 주세요.
        {json.dumps(schema_prd, indent=4, ensure_ascii=False)}
        {rag_context}
        """

        result_json_text = self.llm_gem3.invoke(prompt).content
        json_objects_list = extract_json_objects(result_json_text)
        if not json_objects_list:
            print("LLM did not return a valid JSON object.")
            return {}
        
        json_objects = json_objects_list[0]
        
        if self.entity_extraction_mode == 'logic':
            product_items = json_objects.get('product', [])
            if isinstance(product_items, dict): # Handle cases where LLM returns a dict instead of list
                product_items = product_items.get('items', [])
            cand_entities = [item['name'] for item in product_items]
            similarities_fuzzy = self.extract_entities_by_logic(cand_entities)
        else: # llm mode (not fully implemented in original script but placeholder here)
            similarities_fuzzy = pd.DataFrame()

        final_result = json_objects.copy()
        
        print("Entity from LLM:", [x['name'] for x in product_items])

        if not similarities_fuzzy.empty:
            high_sim_items = similarities_fuzzy.query('sim >= 1.5')['item_nm_alias'].unique()
            filtered_similarities = similarities_fuzzy[
                (similarities_fuzzy['item_nm_alias'].isin(high_sim_items)) &
                (~similarities_fuzzy['item_nm_alias'].str.contains('test', case=False)) &
                (~similarities_fuzzy['item_name_in_msg'].isin(self.stop_item_names))
            ]
            product_tag = convert_df_to_json_list(self.item_pdf_all.merge(filtered_similarities, on=['item_nm_alias']))
            final_result['product'] = product_tag
        else:
            product_items = json_objects.get('product', [])
            if isinstance(product_items, dict):
                product_items = product_items.get('items', [])
            final_result['product'] = [{'item_name_in_msg':d['name'], 'item_in_voca':[{'item_name_in_voca':d['name'], 'item_id': ['#']}]} for d in product_items if d.get('name') and d['name'] not in self.stop_item_names]

        if self.num_cand_pgms > 0 and 'pgm' in json_objects and isinstance(json_objects['pgm'], list):
            pgm_json = self.pgm_pdf[self.pgm_pdf['pgm_nm'].apply(lambda x: re.sub(r'\[.*?\]', '', x) in ' '.join(json_objects['pgm']))][['pgm_nm','pgm_id']].to_dict('records')
            final_result['pgm'] = pgm_json

        channel_tag = []
        channel_items = json_objects.get('channel', [])
        if isinstance(channel_items, dict):
            channel_items = channel_items.get('items', [])

        for d in channel_items:
            if d.get('type') == '대리점' and d.get('value'):
                org_pdf_cand = parallel_fuzzy_similarity(
                    [preprocess_text(d['value'].lower())], self.org_pdf['org_abbr_nm'].unique(), threshold=0.5,
                    text_col_nm='org_nm_in_msg', item_col_nm='org_abbr_nm', n_jobs=6, batch_size=100
                ).drop('org_nm_in_msg', axis=1)

                if not org_pdf_cand.empty:
                    org_pdf_cand = self.org_pdf.merge(org_pdf_cand, on=['org_abbr_nm'])
                    org_pdf_cand['sim'] = org_pdf_cand.apply(lambda x: combined_sequence_similarity(d['value'], x['org_nm'])[0], axis=1).round(5)
                    org_pdf_tmp = org_pdf_cand.query("org_cd.str.startswith('D') & sim >= 0.7", engine='python').sort_values('sim', ascending=False)
                    if org_pdf_tmp.empty:
                        org_pdf_tmp = org_pdf_cand.query("sim>=0.7").sort_values('sim', ascending=False)
                    
                    if not org_pdf_tmp.empty:
                        org_pdf_tmp['rank'] = org_pdf_tmp['sim'].rank(method='dense',ascending=False)
                        org_pdf_tmp['org_cd_full'] = org_pdf_tmp.apply(lambda x: x['org_cd']+x['sub_org_cd'], axis=1)
                        org_info = org_pdf_tmp.query("rank==1").groupby('org_nm')['org_cd_full'].apply(list).reset_index(name='org_cd').to_dict('records')
                        d['store_info'] = org_info
                    else:
                        d['store_info'] = []
                else:
                    d['store_info'] = []
            else:
                d['store_info'] = []
            channel_tag.append(d)

        final_result['channel'] = channel_tag
        return final_result

if __name__ == '__main__':
    
    # Make sure to have the correct data files in ./data/ and models in ./models/
    # And a config/settings.py file with your API keys
    
    # Set offer_info_data_src to "db" to use database, "local" to use CSV files
    offer_info_data_src = "local"  # Change to "db" if you want to use database
    
    extractor = MMSExtractor(data_dir='./data', offer_info_data_src=offer_info_data_src)
    
    test_text = """
    [SK텔레콤] ZEM폰 포켓몬에디션3 안내
    (광고)[SKT] 우리 아이 첫 번째 스마트폰, ZEM 키즈폰__#04 고객님, 안녕하세요!
    우리 아이 스마트폰 고민 중이셨다면, 자녀 스마트폰 관리 앱 ZEM이 설치된 SKT만의 안전한 키즈폰,
    ZEM폰 포켓몬에디션3으로 우리 아이 취향을 저격해 보세요!
    신학기를 맞이하여 SK텔레콤 공식 인증 대리점에서 풍성한 혜택을 제공해 드리고 있습니다!
    ■ 주요 기능
    1. 실시간 위치 조회
    2. 모르는 회선 자동 차단
    3. 스마트폰 사용 시간 제한
    4. IP68 방수 방진
    5. 수업 시간 자동 무음모드
    6. 유해 콘텐츠 차단
    ■ 가까운 SK텔레콤 공식 인증 대리점 찾기
    http://t-mms.kr/t.do?m=#61&s=30684&a=&u=https://bit.ly/3yQF2hx
    ■ 문의 : SKT 고객센터(1558, 무료)
    무료 수신거부 1504
    """

    result = extractor.process_message(test_text)
    
    print("\n" + "="*40)
    print("Final Extracted Information")
    print("="*40)
    print(json.dumps(result, indent=4, ensure_ascii=False)) 