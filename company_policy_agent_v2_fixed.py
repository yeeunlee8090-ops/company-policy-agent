#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os, base64, glob, html, datetime, hashlib, uuid, csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, TextLoader

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="사내 정책 RAG 에이전트", layout="wide")
DATA_DIR = "./rag_data"
SAMPLE_DOCS_DIR = "./sample_docs"
LOG_FILE = "./qa_logs.csv"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_DOCS_DIR, exist_ok=True)
CSV_CACHE = {}

# 형태소 분석기
try:
    from konlpy.tag import Okt
    okt = Okt()
except Exception:
    okt = None
    st.warning("⚠️ 형태소 분석기(KoNLPy)를 사용할 수 없습니다. 키워드 추출은 비활성화됩니다.")

# 관계 매핑
relation_map = {
    "장인": "배우자 부모", "장모": "배우자 부모", "시어머니": "배우자 부모", "시아버지": "배우자 부모",
    "시부모": "배우자 부모", "시부모님": "배우자 부모",
    "친정어머니": "배우자 부모", "친정아버지": "배우자 부모",
    "아버지": "부모", "어머니": "부모", "부모님": "부모",
    "배우자": "배우자", "남편": "배우자", "아내": "배우자",
    "자녀": "자녀", "아들": "자녀", "딸": "자녀"
}
def extract_relation(text):
    for k, v in relation_map.items():
        if k in text:
            return v
    return None

# =========================
# 세션 상태
# =========================
if "llm_choice" not in st.session_state:
    st.session_state.llm_choice = "Ollama (로컬)"
if "openai_key" not in st.session_state:
    st.session_state.openai_key = None

# =========================
# 헬퍼 함수
# =========================
def get_db_dir(use_ollama: bool):
    db_dir = f"./vector_db_{'ollama' if use_ollama else 'openai'}"
    os.makedirs(db_dir, exist_ok=True)
    return db_dir

def make_doc_hash(doc: Document):
    idx = str(doc.metadata.get('chunk_index') if doc.metadata.get('chunk_index') is not None else uuid.uuid4())
    key = f"{doc.metadata.get('ID','')}_{idx}_{doc.metadata.get('is_augmented',False)}"
    return hashlib.md5(key.encode()).hexdigest()

# **복원된 augment_question**
def augment_question(question, base_meta):
    if not question or not question.strip():
        return []
    synonyms = {"경조금": ["지원금", "경조사비"], "휴가": ["휴일", "휴무"]}
    aug_questions = [question]
    for key, values in synonyms.items():
        for v in values:
            if key in question:
                aug_questions.append(question.replace(key, v))
    if "?" in question:
        aug_questions.append(question.replace("?", " 알려주세요."))
    docs = []
    for q in set(aug_questions):
        meta = base_meta.copy()
        meta["is_augmented"] = True
        meta["chunk_index"] = str(uuid.uuid4())
        meta["parent_id"] = str(base_meta.get("ID"))
        new_doc = Document(page_content=q, metadata=meta)
        new_doc.metadata['doc_hash'] = make_doc_hash(new_doc)
        docs.append(new_doc)
    return docs

def load_and_split(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    docs = []
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            if 'ID' not in df.columns or '답변' not in df.columns:
                st.error(f"{os.path.basename(file_path)}: 'ID'와 '답변' 컬럼이 필요합니다.")
                return []
            CSV_CACHE[file_path] = df  # 업로드 시 캐싱
            for _, row in df.iterrows():
                content = row.get("답변", "")
                metadata = {
                    "질문": row.get("질문", ""),
                    "ID": str(row.get("ID", "")),
                    "조항": row.get("조항", ""),
                    "카테고리": row.get("카테고리", ""),
                    "대상": str(row.get("대상", "")) if '대상' in df.columns else "",
                    "is_augmented": False,
                }
                docs.append(Document(page_content=str(content), metadata=metadata))
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {e}")
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = idx
        chunk.metadata['doc_hash'] = make_doc_hash(chunk)
    return chunks

def build_vector_db(docs, embedding_model, persist_dir):
    if os.listdir(persist_dir):
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        existing_data = vectordb.get() or {}
        metas = existing_data.get('metadatas') or []
        existing_hashes = {m.get('doc_hash') for m in metas if isinstance(m, dict)}
        new_docs = [d for d in docs if d.metadata.get('doc_hash') not in existing_hashes]
        if new_docs:
            vectordb.add_documents(new_docs)
    else:
        Chroma.from_documents(docs, embedding_model, persist_directory=persist_dir)

def get_embedding(use_ollama=True, openai_api_key=None):
    return OllamaEmbeddings(model="llama3") if use_ollama else OpenAIEmbeddings(openai_api_key=openai_api_key)

def log_qa(question, answer, score, faq_id, is_augmented):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", encoding="utf-8-sig", newline='') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["timestamp","question","answer","score","faq_id","is_augmented"])
        writer.writerow([now, question, answer, score, faq_id, is_augmented])

def make_download_button(file_path):
    if not file_path: return ""
    with open(file_path, "rb") as f: data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">
        <button style="background-color:#007bff;color:white;padding:6px 14px;border:none;border-radius:5px;font-weight:500;cursor:pointer;">경조사 신청서 샘플 다운받기</button></a>"""

def make_link_button():
    return f"""<a href="https://www.notion.so/23838859e65c8049bb8ae5dc6540889b?source=copy_link" target="_blank">
        <button style="background-color:#28a745;color:white;padding:6px 14px;border:none;border-radius:5px;font-weight:500;cursor:pointer;">경조사 신청하기</button></a>"""

def extract_keywords(text):
    if okt:
        nouns = okt.nouns(text)
        weighted = {n: len(n) * nouns.count(n) for n in set(nouns)}
        return sorted(weighted.items(), key=lambda x: x[1], reverse=True)[:5]
    return []

def render_answer(answer, faq_id, score, chunk_idx, keywords):
    chunk_display = "증강" if chunk_idx and isinstance(chunk_idx, str) and len(str(chunk_idx)) > 5 else chunk_idx
    base_id = faq_id.split('-')[0] if faq_id else ''
    sample_files = glob.glob(os.path.join(SAMPLE_DOCS_DIR, f"{base_id}*.docx"))
    sample_file_path = sample_files[0] if sample_files else None
    safe_answer = html.escape(answer).replace("\n", "<br>")
    safe_answer = safe_answer.replace(html.escape("[경조사 신청서 샘플 다운받기]"), f"""<div style="display:inline-block; margin-right:8px;">{make_download_button(sample_file_path)}</div>""")
    safe_answer = safe_answer.replace(html.escape("[경조사 신청하기]"), f"""<div style="display:inline-block; margin-right:8px;">{make_link_button()}</div>""")
    st.markdown(
        f"""
        <div style="background-color:#f0f9ff;padding:16px;border-radius:8px;line-height:1.5; font-size:15px;">
            {safe_answer}
        </div>
        <div style="background-color:#f8f9fa;padding:8px;border-radius:5px;margin-top:8px; font-size:14px;">
            <b>유사도 점수:</b> {round(score,3)} |
            <b>청킹 위치:</b> {chunk_display} |
            <b>키워드:</b> {', '.join([f"{k}({v})" for k,v in keywords]) if keywords else '없음'}
        </div>""",
        unsafe_allow_html=True
    )

# =========================
# UI
# =========================
st.title("🏢 사내 정책 RAG 에이전트 (Ollama + OpenAI)")
tab1, tab2, tab3 = st.tabs(["📂 데이터 업로드", "💬 질의응답", "📈 성능 평가"])

with tab1:
    st.subheader("문서 업로드 & 임베딩")
    uploaded_files = st.file_uploader("문서를 업로드하세요 (PDF, CSV, TXT)", type=["pdf","csv","txt"], accept_multiple_files=True)
    st.session_state.llm_choice = st.radio("LLM 선택", ["Ollama (로컬)", "OpenAI API"], index=0)
    st.session_state.openai_key = st.text_input("OpenAI API 키 (OpenAI 선택 시)", type="password") if st.session_state.llm_choice == "OpenAI API" else None
    if st.button("🚀 업로드 & 임베딩 실행"):
        if st.session_state.llm_choice == "OpenAI API" and not st.session_state.openai_key:
            st.error("⚠️ OpenAI API 키를 입력해주세요.")
            st.stop()
        if not uploaded_files:
            st.warning("업로드할 문서를 선택하세요!")
        else:
            total_docs, total_aug = 0, 0
            with st.spinner("문서 처리 및 증강 중..."):
                db_dir = get_db_dir(st.session_state.llm_choice == "Ollama (로컬)")
                embedding_model = get_embedding(st.session_state.llm_choice == "Ollama (로컬)", st.session_state.openai_key)
                for file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, file.name)
                    with open(file_path, "wb") as f: f.write(file.read())
                    docs = load_and_split(file_path)
                    if not docs: continue
                    aug_docs = []
                    for d in docs:
                        if d.metadata.get("질문","").strip():
                            aug_docs += augment_question(d.metadata.get("질문"), d.metadata)
                    build_vector_db(docs + aug_docs, embedding_model, db_dir)
                    total_docs += len(docs)
                    total_aug += len(aug_docs)
                st.success(f"✅ 문서 업로드 및 증강 완료! (원본 {total_docs}, 증강 후 {total_docs+total_aug})")

with tab2:
    st.subheader("질문하기 (RAG 기반)")
    query = st.text_input("질문을 입력하세요", placeholder="예: 장인어른이 돌아가셨어요. 경조금은?")
    if st.button("🔍 검색 & 답변"):
        # 캐시 비어있으면 CSV 다시 로드
        if not CSV_CACHE:
            for file in os.listdir(DATA_DIR):
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(DATA_DIR, file), encoding="utf-8-sig")
                    CSV_CACHE[file] = df

        db_dir = get_db_dir(st.session_state.llm_choice == "Ollama (로컬)")
        if not os.listdir(db_dir):
            st.error("데이터베이스가 비어 있습니다. 먼저 문서를 업로드하고 임베딩하세요.")
            st.stop()
        with st.spinner("검색 중..."):
            relation = extract_relation(query)
            found_answer, target_id = None, None

            # 1단계: 관계 매칭
            if relation:
                for df in CSV_CACHE.values():
                    match = df[df['대상'] == relation]
                    if not match.empty:
                        found_answer = str(match.iloc[0]['답변'])
                        target_id = str(match.iloc[0]['ID'])
                        break

            # 2단계: 벡터 검색
            if not found_answer:
                embedding_model = get_embedding(st.session_state.llm_choice == "Ollama (로컬)", st.session_state.openai_key)
                vectordb = Chroma(persist_directory=db_dir, embedding_function=embedding_model)
                docs = vectordb.similarity_search_with_score(query, k=1)
                if docs:
                    top_doc, score = docs[0]
                    target_id = str(top_doc.metadata.get("parent_id") if top_doc.metadata.get("is_augmented") else top_doc.metadata.get("ID"))
                    for df in CSV_CACHE.values():
                        match = df[df['ID'].astype(str) == target_id]
                        if not match.empty:
                            found_answer = str(match.iloc[0]['답변'])
                            break
                    if not found_answer:
                        found_answer = top_doc.page_content.strip() or "⚠️ 관련 답변을 찾을 수 없습니다."

            if found_answer:
                keywords = extract_keywords(query)
                render_answer(found_answer, target_id or "-", 1.0, "-", keywords)
                log_qa(query, found_answer, 1.0, target_id or "-", False)
            else:
                st.warning("관련 FAQ를 찾을 수 없습니다.")

with tab3:
    st.subheader("성능 평가")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        if not df.empty:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df['is_augmented'] = df['is_augmented'].map({'True': True, 'False': False}).fillna(False)
            avg_score = df['score'].mean()
            match_rate = (df['score'] >= 0.8).mean() * 100
            st.metric("전체 평균 유사도", f"{avg_score:.3f}")
            st.metric("전체 매칭률(0.8 이상)", f"{match_rate:.1f}%")
            if 'is_augmented' in df.columns:
                before = df[df['is_augmented']==False]['score']
                after = df[df['is_augmented']==True]['score']
                comp = pd.DataFrame({
                    '유형': ['원본','증강'],
                    '평균 유사도': [before.mean() if not before.empty else 0, after.mean() if not after.empty else 0]
                })
                if not comp.empty:
                    st.bar_chart(comp.set_index('유형'))
            if not df['score'].dropna().empty:
                st.line_chart(df['score'].dropna())
            st.dataframe(df.tail(20))
            csv_data = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("CSV 다운로드", csv_data, "qa_logs.csv")
    else:
        st.info("아직 로그가 없습니다. 질문을 입력해보세요.")
